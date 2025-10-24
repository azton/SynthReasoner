#!/usr/bin/env python3
"""
Convert reasoning trace datafiles into instruct tuning datasets.

Reads JSON files created by llm_synthetic_reasoner.py and creates a JSONL file
where each line is a JSON entry in the format required by HuggingFace TRL SFT trainer.
"""

import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


# System prompts that instruct the model to reason before responding
SYSTEM_PROMPTS = [
    "You are a helpful assistant that thinks step by step. Before providing your final answer, work through your reasoning in <think> ... </think> tags.",
    
    "You are an expert assistant. When answering questions, first analyze the problem carefully in <think> ... </think> tags, then provide your response.",
    
    "Please think through this question systematically. Show your reasoning process in <think> ... </think> tags before giving your final answer.",
    
    "You are a thoughtful AI assistant. Break down complex problems by reasoning through them step-by-step in <think> ... </think> tags before responding.",
    
    "Before answering, please think carefully about the question. Use <think> ... </think> tags to show your reasoning process, then provide a clear final answer.",
    
    "You are an analytical assistant. For each question, first work through your thinking in <think> ... </think> tags, considering different aspects and possibilities before giving your response.",
    
    "Take time to reason through this question carefully. Show your thought process in <think> ... </think> tags, then provide a comprehensive answer.",
    
    "You are a reasoning-focused assistant. Always think through problems step by step in <think> ... </think> tags before providing your final answer.",
    
    "Please approach this question methodically. Use <think> ... </think> tags to show your analysis and reasoning before giving your response.",
    
    "You are an AI that reasons carefully before responding. Work through your thinking in <think> ... </think> tags, then provide a clear and helpful answer."
]


def extract_reasoning_from_trace(trace: str) -> str:
    """Extract reasoning content from <thought> tags if present, otherwise return the whole trace."""
    # Look for content between <thought> and </thought> tags
    thought_match = re.search(r'<thought>(.*?)</thought>', trace, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    else:
        # No thought tags found, return the whole trace
        return trace.strip()


def ensure_thought_tags(reasoning: str) -> str:
    """Ensure reasoning is wrapped in <thought> tags."""
    reasoning = reasoning.strip()
    if not reasoning.startswith('<thought>'):
        reasoning = f'<thought>\n{reasoning}\n</thought>'
    return reasoning


def load_reasoning_traces(file_path: str) -> List[Dict[str, Any]]:
    """Load reasoning traces from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both direct list and nested structure with 'questions' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        raise ValueError(f"Unexpected JSON structure in {file_path}")


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


def convert_trace_to_instruct_format(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a reasoning trace to instruct tuning format."""
    # Skip traces with ARGO_USER errors
    if has_argo_user_error(trace):
        raise ValueError("Trace contains ARGO_USER error - skipping")
    
    # Extract the question
    question = trace['user_interaction']['question']
    
    # Extract reasoning from trace
    reasoning_trace = trace['reasoning_trace']
    reasoning = extract_reasoning_from_trace(reasoning_trace)
    
    # Ensure reasoning is wrapped in thought tags
    reasoning_with_tags = ensure_thought_tags(reasoning)
    
    # Get final answer
    final_answer = trace['final_answer']
    
    # Combine reasoning and final answer for assistant response
    assistant_response = f"{reasoning_with_tags}\n\n{final_answer}"
    
    # Select a random system prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)
    
    # Create the instruct format
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def process_files(input_files: List[str], output_file: str, max_samples: int = None):
    """Process input files and convert to instruct dataset."""
    all_instruct_data = []
    total_traces = 0
    skipped_argo_errors = 0
    other_errors = 0
    
    for file_path in input_files:
        print(f"Processing {file_path}...")
        try:
            traces = load_reasoning_traces(file_path)
            print(f"  Loaded {len(traces)} traces")
            total_traces += len(traces)
            
            for trace in traces:
                if max_samples and len(all_instruct_data) >= max_samples:
                    break
                    
                try:
                    instruct_data = convert_trace_to_instruct_format(trace)
                    all_instruct_data.append(instruct_data)
                except ValueError as e:
                    if "ARGO_USER error" in str(e):
                        skipped_argo_errors += 1
                    else:
                        other_errors += 1
                        print(f"  Warning: Failed to convert trace: {e}")
                    continue
                except Exception as e:
                    other_errors += 1
                    print(f"  Warning: Failed to convert trace: {e}")
                    continue
            
            if max_samples and len(all_instruct_data) >= max_samples:
                break
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    print(f"\nProcessing summary:")
    print(f"  Total traces loaded: {total_traces}")
    print(f"  Successfully converted: {len(all_instruct_data)}")
    print(f"  Skipped (ARGO_USER errors): {skipped_argo_errors}")
    print(f"  Skipped (other errors): {other_errors}")
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_instruct_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved instruct dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert reasoning traces to instruct tuning dataset")
    
    parser.add_argument(
        "input_files",
        nargs='+',
        help="Input JSON files containing reasoning traces"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default="instruct_dataset.jsonl",
        help="Output JSONL file (default: instruct_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for file_path in args.input_files:
        if not Path(file_path).exists():
            print(f"Error: Input file {file_path} does not exist")
            return
    
    process_files(args.input_files, args.output, args.max_samples)


if __name__ == "__main__":
    main()