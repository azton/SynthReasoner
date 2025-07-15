# SynthReasoner: Synthetic Reasoning Trace Generator

An advanced tool for generating realistic reasoning traces from scientific text using Large Language Models (LLMs) via Argo server. This system creates high-quality question-answer pairs with authentic human-like reasoning patterns for training AI models.

## New Features

### 🚀 **Instruct Tuning Dataset Conversion**
- Convert reasoning traces to HuggingFace TRL SFT trainer format
- Convert MCQ datasets to natural reasoning traces
- Automatic filtering of corrupted samples (ARGO_USER errors)
- 10 diverse system prompts for reasoning instruction
- JSONL output ready for fine-tuning

### 🔄 **Improved Resume Logic** 
- Smart paper-level resume based on "Scientific Paper N" titles
- MPI-aware consistent paper numbering across ranks
- Skip already processed papers automatically
- Robust incremental processing

### 🧹 **Data Cleaning Tools**
- Batch cleaning of reasoning trace files
- Remove corrupted ARGO_USER error samples
- Preserve file structure and metadata
- Processing statistics and error reporting

## Overview

This tool processes scientific papers and generates:
- **Realistic user questions** across different difficulty levels and question types
- **Authentic reasoning traces** that show natural thinking patterns with uncertainties, corrections, and meta-cognition
- **Quality-controlled outputs** with optional LLM judge evaluation
- **Training-ready datasets** with answer grading for reinforcement learning

## Features

### Core Capabilities
- **Multi-stage reasoning generation** with question analysis, hypothesis generation, evidence evaluation, and method critique
- **Authentic thinking patterns** including uncertainty markers, self-corrections, and natural speech patterns
- **Quality control** with LLM judges (Claude Sonnet 3.7 & GPT-4o) to ensure high-quality traces
- **Answer grading** across multiple dimensions for RL training
- **Incremental saving** with resume capability for long-running jobs
- **Temperature variation** to generate diverse reasoning traces for the same question

### Question Types Supported
- **Clarification** (beginner): Basic understanding questions
- **Critical** (intermediate): Challenging methodology and conclusions
- **Interpretation** (intermediate): Meaning and significance questions
- **Comparison** (advanced): Relating to other research
- **Application** (intermediate): Practical applications
- **Mechanism** (advanced): How things work
- **Limitation** (advanced): Weaknesses and constraints

## Installation

### Requirements
- Python 3.8+
- Access to Argo server for LLM inference
- Required Python packages (see dependencies)

### Dependencies
```bash
# Install required packages
pip install asyncio argparse json pathlib dataclasses typing
```

### Setup
1. Ensure you have access to the Argo server
2. Place your scientific text data in JSONL format with 'text' field
3. Configure the LLM interface (default: Claude Sonnet 4 via Argo)

## Usage

### Basic Usage

```bash
# Generate 5 traces for testing (default settings)
python llm_synthetic_reasoner_v2.py --max-traces 5

# Process all samples in dataset
python llm_synthetic_reasoner_v2.py

# Use different model and input file
python llm_synthetic_reasoner_v2.py --model gpt4o --input-file my_data.jsonl --max-traces 10
```

### Quality Control

```bash
# Enable quality checking (slower but higher quality)
python llm_synthetic_reasoner_v2.py --quality-check --min-quality-score 6.5 --max-traces 5

# Grade answers for RL training
python llm_synthetic_reasoner_v2.py --grade-answers --max-traces 5
```

### Advanced Options

```bash
# Generate multiple traces per question using temperature variation
python llm_synthetic_reasoner_v2.py --traces-per-question 3 --max-traces 5

# Start fresh (ignore existing output)
python llm_synthetic_reasoner_v2.py --no-resume --max-traces 5

# Resume from previous run (default behavior)
python llm_synthetic_reasoner_v2.py --max-traces 10
```

### MPI Parallel Processing

```bash
# Run with MPI for parallel processing across multiple nodes
mpiexec -np 12 python -u llm_synthetic_reasoner.py \
  --model gpt41 \
  --output cosmo-paper-traces/gpt41_fulltrace.json \
  --traces-per-sample 2 \
  --grade-answers \
  --resume
```

### Data Cleaning

```bash
# Clean all reasoning trace files (remove ARGO_USER errors)
python clean_reasoning_traces.py cosmo-paper-traces/*.json

# Clean with backup to separate directory
python clean_reasoning_traces.py cosmo-paper-traces/*.json --output-dir cleaned_traces --suffix _clean
```

### Convert to Instruct Dataset

```bash
# Convert reasoning traces to instruct tuning format
python convert_to_instruct_dataset.py cosmo-paper-traces/*.json -o instruct_dataset.jsonl

# Convert MCQ datasets to natural reasoning traces
python convert_mcq_reasoning.py --input mcq_data.jsonl --output mcq_reasoning.jsonl

# Limit number of samples for testing
python convert_to_instruct_dataset.py cosmo-paper-traces/gpt41_fulltrace_rank0.json --max-samples 1000
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | LLM model to use | `claudesonnet4` |
| `--input-file` | Input JSONL file path | `../data/heuristic_filtered_cosmo_limited.jsonl` |
| `--max-traces` | Maximum traces to generate | `None` (all) |
| `--output` | Output file path | `reasoning_training_data.json` |
| `--traces-per-sample` | Traces per text sample | `1` |
| `--traces-per-question` | Traces per question (temp variation) | `1` |
| `--resume` | Resume from existing output | `True` |
| `--no-resume` | Start fresh, ignore existing output | `False` |
| `--grade-answers` | Enable answer grading for RL | `False` |
| `--quality-check` | Enable quality checking | `False` |
| `--min-quality-score` | Minimum quality score (1-10) | `6.0` |

## Input Format

The script expects a JSONL file where each line contains a JSON object with a 'text' field:

```json
{"text": "This study investigates the effects of dark matter on galaxy formation..."}
{"text": "Our methodology involved analyzing 1,000 galaxy samples using..."}
{"text": "The results demonstrate a significant correlation between..."}
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "generation_timestamp": "2024-01-01T12:00:00",
    "total_traces_generated": 50,
    "status": "completed"
  },
  "questions": [
    {
      "source_paper": {
        "title": "Scientific Paper 1",
        "domain": "General Science"
      },
      "user_interaction": {
        "question": "What are the main limitations of this experimental approach?",
        "question_type": "limitation",
        "difficulty_level": "advanced"
      },
      "reasoning_trace": "<thought>The user is asking about limitations... Let me think through this carefully...</thought>",
      "final_answer": "The main limitations of this approach include...",
      "answer_grades": {
        "question_alignment": 0.85,
        "scientific_accuracy": 0.90,
        "completeness": 0.80,
        "overall_score": 0.85
      },
      "metadata": {
        "authenticity_score": 0.75,
        "complexity_level": "high",
        "reasoning_patterns": ["evidence_evaluation", "self_correction"]
      }
    }
  ]
}
```

## Workflow

### Stage 1: Question Generation
- Analyzes scientific passages to understand content and context
- Generates 6-8 diverse questions across different types and difficulty levels
- Ensures questions are self-contained and appropriate for target audience

### Stage 2: Context Analysis
- Analyzes what each question requires in terms of reasoning
- Identifies relevant paper sections and potential misconceptions
- Determines appropriate technical depth for user level

### Stage 3: Reasoning Generation
- **Targeted Hypotheses**: Multiple ways to interpret and approach the question
- **Evidence Evaluation**: Assesses paper's evidence from question perspective
- **Method Critique**: Evaluates methodology relevant to the question
- **User Perspective**: Considers what would be most helpful for the user

### Stage 4: Trace Assembly
- Weaves reasoning fragments into natural, flowing thought process
- Shows authentic thinking patterns with uncertainties and corrections
- Maintains scientific accuracy while adding human-like authenticity

### Stage 5: Quality Control (Optional)
- LLM judges evaluate traces across multiple dimensions
- Ensures minimum quality threshold before acceptance
- Provides feedback for iterative improvement

### Stage 6: Answer Grading (Optional)
- Evaluates final answers across 6 dimensions:
  - Question alignment
  - Scientific accuracy
  - Completeness
  - Uncertainty calibration
  - Clarity and structure
  - Evidence usage

## Key Features

### Authenticity Enhancement
- Adds natural speech patterns ("hmm", "let me think", "actually")
- Includes realistic corrections and self-doubt
- Shows meta-cognitive awareness and uncertainty
- Captures the messy, error-prone nature of human thinking

### Quality Control
- Uses Claude Sonnet 3.7 and GPT-4o as judges
- Evaluates traces before acceptance
- Configurable quality thresholds
- Retry mechanism for failed quality checks

### Resume Capability
- Saves progress incrementally after each trace
- Can resume from interruptions
- Atomic file operations prevent corruption
- Maintains generation metadata

## Performance Considerations

- **Quality checking**: Significantly slower but produces higher quality traces
- **Answer grading**: Adds processing time but provides RL training signals
- **Multiple traces per question**: Uses temperature variation for diversity
- **Incremental saving**: Prevents data loss during long runs

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Check Argo server accessibility
   - Verify model name spelling
   - Ensure proper authentication

2. **Quality Check Failures**
   - Lower `--min-quality-score` threshold
   - Disable quality checking for faster generation
   - Check judge model availability

3. **Memory Issues**
   - Reduce `--traces-per-sample` or `--traces-per-question`
   - Process smaller batches with `--max-traces`
   - Enable incremental saving

4. **Resume Not Working**
   - Check output file permissions
   - Verify JSON format integrity
   - Use `--no-resume` to start fresh

### Error Handling
- Graceful failure with informative error messages
- Retry mechanism for transient failures
- Continues processing remaining samples after individual failures
- Preserves partial results with incremental saving

## Examples

### Generate High-Quality Dataset
```bash
python llm_synthetic_reasoner_v2.py \
  --model claudesonnet4 \
  --input-file scientific_papers.jsonl \
  --quality-check \
  --min-quality-score 7.0 \
  --grade-answers \
  --max-traces 100 \
  --output high_quality_traces.json
```

### Create Question Variations
```bash
python llm_synthetic_reasoner_v2.py \
  --traces-per-question 3 \
  --max-traces 10 \
  --output question_variations.json
```

### Quick Testing
```bash
python llm_synthetic_reasoner_v2.py \
  --max-traces 3 \
  --output test_traces.json
```

## Tools Included

### 1. `llm_synthetic_reasoner.py`
Main reasoning trace generator with MPI support and improved resume logic.

### 2. `convert_to_instruct_dataset.py` 
Convert reasoning traces to HuggingFace TRL SFT trainer format:
```bash
python convert_to_instruct_dataset.py [input_files] -o output.jsonl --max-samples N
```

### 3. `convert_mcq_reasoning.py`
Convert MCQ datasets to natural reasoning traces with fluid exploration:
```bash
# Convert MCQ with multiple choices
python convert_mcq_reasoning.py --input mcq_data.jsonl --output reasoning_traces.jsonl --model gpt4o

# Convert open-ended questions with ground truth
python convert_mcq_reasoning.py --input open_ended.jsonl --answer-field "correct_answer" --model gpt4o

# Generate multiple reasoning variations per question
python convert_mcq_reasoning.py --input data.jsonl --traces-per-question 3 --max-questions 50
```

**Key Features:**
- **Fluid Reasoning**: Explores possibilities naturally without referencing "Choice A/B/C"
- **Hidden Choices**: Answer choices guide reasoning but don't appear in user input
- **Natural Answers**: Final answers state actual content instead of meaningless letters
- **Dual Support**: Handles both traditional MCQ and open-ended questions with ground truth
- **Instruct Ready**: Outputs HuggingFace TRL SFT trainer format

### 4. `clean_reasoning_traces.py`
Clean reasoning trace files by removing corrupted samples:
```bash
python clean_reasoning_traces.py [input_files] --output-dir [dir] --suffix [suffix]
```

### 5. Helper Scripts
- `llm_interface.py` - Unified LLM client interface
- `llm_judge_evaluator.py` - Quality evaluation system
- `gemini_client.py` - Gemini API integration

## Recent Updates

- ✅ **MCQ Reasoning Generation**: Convert MCQ datasets to natural reasoning traces with fluid exploration
- ✅ **Smart Resume Logic**: Papers are tracked by title, skipping already processed ones
- ✅ **MPI Paper Numbering**: Consistent paper IDs across all MPI ranks  
- ✅ **Error Filtering**: Automatic detection and removal of ARGO_USER errors
- ✅ **Instruct Format**: Ready-to-use datasets for SFT training
- ✅ **Data Cleaning**: Batch processing tools for large datasets

## License

This project is part of the SynthReasoner framework for generating synthetic reasoning data.