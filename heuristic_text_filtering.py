import json
import pandas as pd
import re
import argparse
from collections import Counter
import numpy as np
import hashlib
import os

def load_predictive_power_data(file_path):
    """Load the term predictive power data from CSV."""
    return pd.read_csv(file_path, index_col=0)

def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def generate_passage_id(source_file, doc_index, passage_index):
    """Generate a unique ID for a passage."""
    # Create a unique string combining source file, doc index, and passage index
    id_string = f"{source_file}_{doc_index}_{passage_index}"
    
    # Generate a 16-character hash as the ID
    return hashlib.md5(id_string.encode()).hexdigest()[:16]

def is_references_or_acknowledgements(passage):
    """
    Simple check if a passage appears to be a references or acknowledgements section
    based on its header.
    
    Returns True if it should be excluded, False otherwise.
    """
    if not passage or not isinstance(passage, str):
        return False
    
    # Look at the first line of the passage to check for section headers
    first_line = passage.split('\n')[0].lower()
    
    # Simple list of section headers to exclude
    excluded_headers = [
        'references', 
        'bibliography', 
        'works cited', 
        'literature cited',
        'acknowledgements',
        'acknowledgments',
        'funding'
    ]
    
    # Check if any of the excluded headers appear in the first line
    for header in excluded_headers:
        # Check for header at start of passage or after a # marking
        if (header in first_line and 
            (first_line.startswith(header) or 
             re.search(r'#\s*' + re.escape(header), first_line))):
            return True
    
    return False

def split_into_passages(text):
    """Split text into passages based on # delimiter."""
    # Use regex to match # that has a space after it (avoid matching #tags)
    passages = re.split(r'# ', text)
    
    # First element might not start with # so check and fix if needed
    if passages and not passages[0].strip():
        passages = passages[1:]
    
    # Add the # back to passage starts (except possibly the first one if it didn't have one)
    for i in range(1, len(passages)):
        passages[i] = '# ' + passages[i]
    
    # If first element didn't start with #, don't add it
    if passages and text.strip().startswith('# '):
        passages[0] = '# ' + passages[0]
        
    return [p for p in passages if p.strip()]  # Filter out empty passages

def count_logical_terms(text, term_list):
    """Count occurrences of logical terms in text."""
    if not text or not isinstance(text, str):
        return {}
    
    text = text.lower()
    counts = {}
    
    for term in term_list:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        counts[term] = len(re.findall(pattern, text))
    
    return counts

def calculate_passage_low_rating_probability(passage, term_predictive_power, threshold=0.5):
    """
    Calculate probability that a passage has a low rating based on terms it contains.
    
    Uses a naive Bayes-inspired approach combining the probabilities from each term.
    """
    # Count logical terms in the passage
    term_list = term_predictive_power.index.tolist()
    term_counts = count_logical_terms(passage, term_list)
    
    # Find terms that appear in the passage
    present_terms = [term for term, count in term_counts.items() if count > 0]
    
    if not present_terms:
        # If no terms are present, use the base rate
        return 1.0  # Consider passages with no logical terms as 100% likely to be low-rated
    
    # Get probabilities for present terms
    term_probs = []
    weights = []
    
    for term in present_terms:
        if term in term_predictive_power.index:
            prob = term_predictive_power.loc[term, 'probability_low_rating']
            count = term_predictive_power.loc[term, 'occurrence_count']
            weight = np.log(count)  # Weight by log of occurrence count
            
            term_probs.append(prob)
            weights.append(weight)
    
    if not term_probs:
        return 1.0  # Default to 100% if no known terms found
    
    # Calculate weighted average probability
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, use simple average
        average_prob = sum(term_probs) / len(term_probs)
    else:
        # Weighted average
        average_prob = sum(p * w for p, w in zip(term_probs, weights)) / total_weight
    
    return average_prob

def process_file(input_file, term_predictive_power, args):
    """Process a single file and return statistics and filtered passages."""
    # Load input data for this file
    print(f"Processing file: {input_file}")
    input_data = load_jsonl(input_file)
    
    if args.sample and args.sample < len(input_data):
        print(f"Sampling {args.sample} documents from this file")
        input_data = input_data[:args.sample]
    
    # Process each document in this file
    file_passages = []
    file_probabilities = []
    file_saved_passages = []
    
    for doc_index, doc in enumerate(input_data):
        if "text" not in doc:
            print(f"Warning: Document missing 'text' field: {str(doc)[:100]}...")
            continue
        # Check if any of the specified keywords are in the first 3000 characters
        if args.limit_keywords:
            document_start = doc["text"][:3000].lower()
            # Split the comma-separated string into a list of keywords
            keywords = [kw.strip().lower() for kw in args.limit_keywords.split(',')]
            if not any(keyword in document_start for keyword in keywords):
                continue
        passages = split_into_passages(doc["text"])
        
        for passage_index, passage in enumerate(passages):
            if len(passage) < args.min_length:
                continue
                
            # Skip references and acknowledgements sections
            if is_references_or_acknowledgements(passage):
                continue
                
            # Calculate probability
            prob_low_rated = calculate_passage_low_rating_probability(
                passage, term_predictive_power
            )
            file_probabilities.append(prob_low_rated)
            
            # Generate unique ID for this passage
            passage_id = generate_passage_id(input_file, doc_index, passage_index)
            
            # Create passage object
            passage_obj = {
                "id": passage_id,
                "text": passage,
                "probability_low_rated": prob_low_rated,
                "source_file": input_file,
                "doc_index": doc_index,
                "passage_index": passage_index
            }
            file_passages.append(passage_obj)
            
            # If probability is below threshold, save for evaluation
            if prob_low_rated < args.threshold:
                file_saved_passages.append(passage_obj)
    
    return {
        "passages": file_passages,
        "probabilities": file_probabilities,
        "saved_passages": file_saved_passages
    }

def main():
    import glob
    
    parser = argparse.ArgumentParser(description='Filter passages that likely need evaluation.')
    parser.add_argument('--input_pattern', default = "../synthetic_pipeline/data/*.jsonl", 
                        help='Path pattern to input JSONL files with text field (e.g., "data/*.jsonl")')
    parser.add_argument('--predictive_power_file', default='logical_term_analysis/term_predictive_power.csv',
                        help='Path to term_predictive_power.csv file')
    parser.add_argument('--output_file', default='heuristic_filtered_texts.jsonl',
                        help='Path to output JSONL file for passages requiring evaluation')
    parser.add_argument('--limit_keywords', type=str, default=None,
                        help='Comma-separated list of keywords to limit processing to (e.g., "dark energy,hubble constant,ΛCDM")')
    parser.add_argument('--threshold', type=float, default=0.14,
                        help='Threshold below which a passage is considered unlikely to be low-rated')
    parser.add_argument('--min_length', type=int, default=150,
                        help='Minimum character length for passages to be considered')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample this many passages to process per file (for testing)')
    parser.add_argument('--show_histograms', action='store_true',
                        help='Show histograms of probabilities')
    parser.add_argument('--low_rating_threshold', type=int, default=3,
                        help='Threshold for defining low rating (passages rated <= this value)')
    parser.add_argument('--no_terms_probability', type=float, default=0.5,
                        help='Probability to assign to passages with no logical terms (default: 0.5)')
    
    args = parser.parse_args()
    
    # Update the default probability for passages with no terms
    global calculate_passage_low_rating_probability
    original_function = calculate_passage_low_rating_probability
    
    def wrapped_function(passage, term_predictive_power, threshold=0.5):
        # Count logical terms in the passage
        term_list = term_predictive_power.index.tolist()
        term_counts = count_logical_terms(passage, term_list)
        
        # Find terms that appear in the passage
        present_terms = [term for term, count in term_counts.items() if count > 0]
        
        if not present_terms:
            # If no terms are present, use the specified probability
            return args.no_terms_probability
        
        # Otherwise use the original function logic
        return original_function(passage, term_predictive_power, threshold)
    
    # Replace the function with our wrapped version
    calculate_passage_low_rating_probability = wrapped_function
    
    # Load predictive power data
    print(f"Loading term predictive power data from {args.predictive_power_file}")
    term_predictive_power = load_predictive_power_data(args.predictive_power_file)
    
    # Find all files matching the pattern
    input_files = glob.glob(args.input_pattern)
    if not input_files:
        print(f"Error: No files found matching pattern '{args.input_pattern}'")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process files one by one
    all_probabilities = []
    total_passages = 0
    saved_passages = []
    excluded_references = 0
    
    # Open the output file once for streaming output
    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        for i, input_file in enumerate(input_files):
            print(f"Processing file {i+1}/{len(input_files)}: {input_file}")
            
            # Process this file
            results = process_file(input_file, term_predictive_power, args)
            
            # Accumulate statistics
            all_probabilities.extend(results["probabilities"])
            total_passages += len(results["passages"])
            
            # Write saved passages directly to output file
            for passage in results["saved_passages"]:
                out_file.write(json.dumps(passage) + '\n')
            
            # Keep count of saved passages
            saved_passages.extend(results["saved_passages"])
            
            # Report progress
            print(f"  - Found {len(results['saved_passages'])} passages to save from this file")
            print(f"  - Running total: {len(saved_passages)}/{total_passages} passages saved")
    
    # Show summary statistics
    print(f"\nCompleted processing {len(input_files)} files")
    print(f"Processed {total_passages} total passages")
    print(f"Saved {len(saved_passages)} passages ({len(saved_passages)/total_passages*100:.1f}%) with low probability of being low-rated (< {args.threshold})")
    
    # Show histogram if requested
    if args.show_histograms and all_probabilities:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.hist(all_probabilities, bins=20)
            plt.title(f"Distribution of Low Rating Probabilities (rating <= {args.low_rating_threshold})")
            plt.xlabel("Probability of Low Rating")
            plt.ylabel("Count")
            plt.savefig("probability_histogram.png")
            print("Saved probability histogram to probability_histogram.png")
            
            # Show another histogram with threshold marked
            plt.figure(figsize=(10, 6))
            plt.hist(all_probabilities, bins=20, alpha=0.7)
            plt.axvline(x=args.threshold, color='red', linestyle='--')
            plt.text(args.threshold+0.02, plt.ylim()[1]*0.9, f'Threshold: {args.threshold}', color='red')
            plt.title("Distribution with Threshold")
            plt.xlabel("Probability of Low Rating")
            plt.ylabel("Count")
            plt.savefig("probability_histogram_with_threshold.png")
            print("Saved threshold histogram to probability_histogram_with_threshold.png")
            
        except ImportError:
            print("Could not generate histograms: matplotlib not available")

if __name__ == "__main__":
    main()