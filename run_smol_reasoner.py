#!/usr/bin/env python3
"""
Command line runner for SmolSyntheticReasoner

Processes JSONL files with text data and generates reasoning traces using 
the multi-agent system with support for both Argo and Gemini models.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from smol_synthetic_reasoner import SmolSyntheticReasoner
    from llm_interface import get_available_models
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure smol_synthetic_reasoner.py and llm_interface.py are in the same directory")
    sys.exit(1)

try:
    from smolagents import CodeAgent, tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    print("Warning: smolagents not available. Install with: pip install smolagents")
    SMOLAGENTS_AVAILABLE = False


class ProgressiveOutputWriter:
    """Handles progressive writing of results as they're generated"""
    
    def __init__(self, output_path: str, model_name: str, interface: str):
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.interface = interface
        self.results_written = 0
        self.start_time = datetime.now()
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize files
        self.results_file = self.output_path.with_suffix('.jsonl')
        self.metadata_file = self.output_path.with_suffix('.meta.json')
        self.progress_file = self.output_path.with_suffix('.progress.txt')
        
        # Initialize metadata
        self.metadata = {
            "generation_timestamp": self.start_time.isoformat(),
            "model_used": model_name,
            "interface_used": interface,
            "status": "running",
            "total_samples_requested": 0,
            "samples_completed": 0,
            "samples_failed": 0,
            "results_file": str(self.results_file.name),
            "average_token_count": 0,
            "total_tokens": 0,
            "average_processing_time": 0,
            "errors": []
        }
        
        # Write initial metadata
        self._write_metadata()
        
        # Write initial progress
        self._write_progress()
        
        print(f"📁 Progressive output initialized:")
        print(f"   Results: {self.results_file}")
        print(f"   Metadata: {self.metadata_file}")
        print(f"   Progress: {self.progress_file}")
    
    def set_total_samples(self, total: int):
        """Set the total number of samples to process"""
        self.metadata["total_samples_requested"] = total
        self._write_metadata()
    
    def write_result(self, result: Dict[str, Any], sample_index: int):
        """Write a single result to the progressive output"""
        
        # Add sample metadata
        result["sample_index"] = sample_index
        result["completion_timestamp"] = datetime.now().isoformat()
        
        # Write to JSONL file
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Update metadata
        self.results_written += 1
        self.metadata["samples_completed"] = self.results_written
        
        # Update token statistics
        token_count = result.get("token_count", 0)
        self.metadata["total_tokens"] += token_count
        if self.results_written > 0:
            self.metadata["average_token_count"] = self.metadata["total_tokens"] / self.results_written
        
        # Update timing
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.results_written > 0:
            self.metadata["average_processing_time"] = elapsed / self.results_written
        
        self._write_metadata()
        self._write_progress()
        
        print(f"✅ Result {self.results_written} written to {self.results_file}")
    
    def record_error(self, error: str, sample_index: int):
        """Record an error for a sample"""
        self.metadata["samples_failed"] += 1
        self.metadata["errors"].append({
            "sample_index": sample_index,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
        
        self._write_metadata()
        self._write_progress()
    
    def finalize(self, consolidate: bool = True):
        """Finalize the progressive output"""
        
        # Update final metadata
        self.metadata["status"] = "completed"
        self.metadata["completion_timestamp"] = datetime.now().isoformat()
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.metadata["total_processing_time"] = elapsed
        
        self._write_metadata()
        self._write_progress()
        
        print(f"\n📊 Progressive output completed:")
        print(f"   Total results: {self.results_written}")
        print(f"   Total failures: {self.metadata['samples_failed']}")
        print(f"   Average tokens: {self.metadata['average_token_count']:.1f}")
        print(f"   Total time: {elapsed:.1f}s")
        
        # Optionally consolidate into single JSON file
        if consolidate and self.results_written > 0:
            self._consolidate_results()
    
    def _write_metadata(self):
        """Write metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _write_progress(self):
        """Write progress summary to text file"""
        total = self.metadata["total_samples_requested"]
        completed = self.metadata["samples_completed"]
        failed = self.metadata["samples_failed"]
        
        progress_text = f"""Smol Reasoner Progress Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Status: {self.metadata['status']}
Model: {self.model_name} ({self.interface})

Progress: {completed}/{total} samples completed
Failures: {failed} samples failed
Success Rate: {(completed/(completed+failed)*100) if (completed+failed) > 0 else 0:.1f}%

Average Tokens: {self.metadata['average_token_count']:.1f}
Total Tokens: {self.metadata['total_tokens']}
Average Time per Sample: {self.metadata['average_processing_time']:.1f}s

Files:
- Results: {self.results_file}
- Metadata: {self.metadata_file}
- Progress: {self.progress_file}
"""
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            f.write(progress_text)
    
    def _consolidate_results(self):
        """Consolidate JSONL results into single JSON file"""
        consolidated_file = self.output_path
        
        # Read all results from JSONL
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        
        # Create consolidated output
        consolidated_data = {
            "metadata": self.metadata,
            "reasoning_traces": results
        }
        
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        print(f"📦 Consolidated results written to {consolidated_file}")


def load_texts_from_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load texts from JSONL file"""
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                if 'text' in data and data['text'].strip():
                    texts.append(data['text'])
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON on line {i+1}")
                continue
    
    return texts


def convert_reasoning_trace_to_dict(trace) -> Dict[str, Any]:
    """Convert ReasoningTrace object to dictionary for JSON serialization"""
    return {
        "question": {
            "stem": trace.question.stem,
            "topic": trace.question.topic,
            "difficulty": trace.question.difficulty,
            "quality_score": trace.question.quality_score
        },
        "choices": [
            {
                "id": choice.id,
                "text": choice.text,
                "reasoning": choice.reasoning,
                "confidence": choice.confidence
            }
            for choice in trace.choices
        ],
        "selected_choice": trace.selected_choice,
        "reasoning_chain": trace.reasoning_chain,
        "token_count": trace.token_count,
        "metadata": trace.metadata
    }


def main():
    """Main function to run the reasoner with progressive output"""
    
    parser = argparse.ArgumentParser(description="Generate reasoning traces using multi-agent system")
    
    # Required arguments
    parser.add_argument("input_file", help="Path to JSONL file with text data")
    parser.add_argument("output_file", help="Path to output JSON file")
    
    # Model selection
    parser.add_argument("--model", "-m", default="gemini-2.5-flash", 
                       help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--interface", "-i", choices=["auto", "argo", "gemini"], 
                       default="auto", help="Interface to use (default: auto)")
    
    # Processing options
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--num-questions", "-q", type=int, default=5,
                       help="Number of questions to generate per sample (default: 5)")
    
    # API configuration
    parser.add_argument("--api-key", help="API key for Gemini models")
    parser.add_argument("--user-name", help="User name for Argo models")
    
    # Progressive output options
    parser.add_argument("--no-consolidate", action="store_true",
                       help="Don't consolidate results into single JSON file")
    
    # Display options
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        print("Available models:")
        models = get_available_models()
        for interface, model_list in models.items():
            print(f"\n{interface.upper()} Interface:")
            for model in model_list:
                print(f"  - {model}")
        return
    
    # Check if smolagents is available
    if not SMOLAGENTS_AVAILABLE:
        print("Error: smolagents not available. Install with: pip install smolagents")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)
    
    # Load texts from JSONL file
    print(f"📂 Loading texts from {args.input_file}")
    texts = load_texts_from_jsonl(args.input_file, args.max_samples)
    
    if not texts:
        print("Error: No valid texts found in input file")
        sys.exit(1)
    
    print(f"📊 Loaded {len(texts)} texts")
    
    # Initialize progressive output writer
    output_writer = ProgressiveOutputWriter(
        args.output_file, 
        args.model, 
        args.interface
    )
    output_writer.set_total_samples(len(texts))
    
    # Initialize reasoner
    print(f"🚀 Initializing reasoner with {args.model} via {args.interface}")
    
    kwargs = {}
    if args.api_key:
        kwargs['api_key'] = args.api_key
    if args.user_name:
        kwargs['user_name'] = args.user_name
    
    try:
        reasoner = SmolSyntheticReasoner(
            model_name=args.model, 
            interface=args.interface, 
            **kwargs
        )
    except Exception as e:
        print(f"Error initializing reasoner: {e}")
        output_writer.record_error(f"Initialization failed: {e}", -1)
        sys.exit(1)
    
    # Process texts with progressive output
    successful_results = 0
    
    for i, text in enumerate(texts):
        print(f"\n🔄 Processing text {i+1}/{len(texts)}")
        print(f"📝 Text length: {len(text)} characters")
        
        try:
            # Generate reasoning trace using proper smolagents
            trace = reasoner.generate_reasoning_trace(text, args.num_questions)
            
            # Convert to dict for JSON serialization
            result = convert_reasoning_trace_to_dict(trace)
            
            # Write result immediately
            output_writer.write_result(result, i)
            successful_results += 1
            
            if args.verbose:
                print(f"   Token count: {result['token_count']}")
                print(f"   Question: {result['question'].get('stem', 'N/A')[:100]}...")
                
        except Exception as e:
            print(f"❌ Error processing text {i+1}: {e}")
            output_writer.record_error(str(e), i)
            
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Finalize progressive output
    print(f"\n🏁 Processing complete!")
    output_writer.finalize(consolidate=not args.no_consolidate)
    
    if successful_results == 0:
        print("❌ No results generated")
        sys.exit(1)
    else:
        print(f"✅ Generated {successful_results} reasoning traces")


if __name__ == "__main__":
    main() 