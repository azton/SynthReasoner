#!/usr/bin/env python3
"""
Clean reasoning trace files by removing samples with ARGO_USER errors.

Loads JSON files, removes traces containing "ARGO_USER=your_username" error text,
and saves the cleaned versions.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def has_argo_user_error(trace: Dict[str, Any]) -> bool:
    """Check if the trace contains ARGO_USER error text."""
    # Check common fields where the error might appear
    fields_to_check = [
        trace.get('reasoning_trace', ''),
        trace.get('final_answer', ''),
        trace.get('user_interaction', {}).get('question', '')
    ]
    
    for field in fields_to_check:
        if isinstance(field, str) and 'ARGO_USER=your_username' in field:
            return True
    
    return False


def clean_reasoning_trace_file(input_file: str, output_file: str = None) -> Dict[str, int]:
    """Clean a single reasoning trace file."""
    if output_file is None:
        output_file = input_file  # Overwrite original
    
    print(f"Processing {input_file}...")
    
    # Load the file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both direct list and nested structure with 'questions' key
    if isinstance(data, list):
        traces = data
        is_nested = False
    elif isinstance(data, dict) and 'questions' in data:
        traces = data['questions']
        is_nested = True
    else:
        raise ValueError(f"Unexpected JSON structure in {input_file}")
    
    original_count = len(traces)
    print(f"  Original traces: {original_count}")
    
    # Filter out traces with ARGO_USER errors
    clean_traces = []
    argo_errors = 0
    
    for trace in traces:
        if has_argo_user_error(trace):
            argo_errors += 1
        else:
            clean_traces.append(trace)
    
    print(f"  Clean traces: {len(clean_traces)}")
    print(f"  Removed ARGO_USER errors: {argo_errors}")
    
    # Prepare output data
    if is_nested:
        # Preserve original structure but update traces and metadata
        clean_data = data.copy()
        clean_data['questions'] = clean_traces
        
        # Update metadata if it exists
        if 'metadata' in clean_data:
            clean_data['metadata']['total_traces_generated'] = len(clean_traces)
            clean_data['metadata']['argo_errors_removed'] = argo_errors
            clean_data['metadata']['original_count'] = original_count
    else:
        clean_data = clean_traces
    
    # Save the cleaned file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved cleaned file to {output_file}")
    
    return {
        'original_count': original_count,
        'clean_count': len(clean_traces),
        'argo_errors': argo_errors
    }


def main():
    parser = argparse.ArgumentParser(description="Clean reasoning trace files by removing ARGO_USER errors")
    
    parser.add_argument(
        "input_files",
        nargs='+',
        help="Input JSON files to clean"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for cleaned files (default: overwrite originals)"
    )
    
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to add to output filenames (e.g., '_cleaned')"
    )
    
    args = parser.parse_args()
    
    # Process each file
    total_stats = {
        'files_processed': 0,
        'total_original': 0,
        'total_clean': 0,
        'total_argo_errors': 0
    }
    
    for input_file in args.input_files:
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Warning: {input_file} does not exist, skipping")
            continue
        
        # Determine output file
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{input_path.stem}{args.suffix}{input_path.suffix}"
        else:
            if args.suffix:
                output_file = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"
            else:
                output_file = input_file  # Overwrite original
        
        try:
            stats = clean_reasoning_trace_file(input_file, str(output_file))
            
            total_stats['files_processed'] += 1
            total_stats['total_original'] += stats['original_count']
            total_stats['total_clean'] += stats['clean_count']
            total_stats['total_argo_errors'] += stats['argo_errors']
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total original traces: {total_stats['total_original']}")
    print(f"Total clean traces: {total_stats['total_clean']}")
    print(f"Total ARGO_USER errors removed: {total_stats['total_argo_errors']}")
    if total_stats['total_original'] > 0:
        error_rate = (total_stats['total_argo_errors'] / total_stats['total_original']) * 100
        print(f"Error rate: {error_rate:.1f}%")


if __name__ == "__main__":
    main()