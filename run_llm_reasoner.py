#!/usr/bin/env python3
"""
CLI for LLM Synthetic Reasoner
"""

import argparse
import sys
from pathlib import Path

try:
    from .llm_synthetic_reasoner import LLMSyntheticReasoner
    from .llm_interface import get_available_models
except ImportError:
    from llm_synthetic_reasoner import LLMSyntheticReasoner
    from llm_interface import get_available_models


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning chains using LLMs")
    
    parser.add_argument("input_file", help="Input JSONL file with text data")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file for results")
    # Get available models
    available_models = get_available_models()
    all_models = []
    for interface_models in available_models.values():
        all_models.extend(interface_models)
    
    parser.add_argument("-m", "--model", default="gemini-2.5-flash", 
                       choices=all_models,
                       help="Model to use (auto-detects interface)")
    parser.add_argument("-n", "--num-questions", type=int, default=5,
                       help="Number of initial MCQs to generate per text sample")
    parser.add_argument("-s", "--min-score", type=float, default=7.0,
                       help="Minimum quality score (1-10) to accept MCQs")
    parser.add_argument("--max-samples", type=int, help="Maximum number of text samples to process")
    parser.add_argument("-i", "--interface", choices=["auto", "argo", "gemini"], default="auto",
                       help="Interface to use (auto-detects based on model)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_TEST_KEY env var)")
    parser.add_argument("--user-name", help="Argo username (or set ARGO_USER env var)")
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Initialize reasoner
    print(f"Initializing LLM Synthetic Reasoner with model: {args.model}")
    try:
        reasoner = LLMSyntheticReasoner(
            model_name=args.model, 
            interface=args.interface,
            api_key=args.api_key,
            user_name=args.user_name
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Process file
    try:
        results = reasoner.process_jsonl_file(
            file_path=args.input_file,
            output_path=args.output,
            max_samples=args.max_samples,
            num_initial_questions=args.num_questions,
            min_score=args.min_score
        )
        
        print(f"\\n✅ Success!")
        print(f"Generated {len(results)} high-quality questions")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()