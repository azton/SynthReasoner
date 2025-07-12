# SmolSyntheticReasoner

A multi-agent system for generating high-quality reasoning chains from scientific text using both Argo and Gemini models.

## Features

- **Multi-Agent Architecture**: Uses 4 specialized agents (Question Writer, Solution Proposer, Final Solution Writer, Critic)
- **Nested Agent Groups**: Each main agent has an internal critic for quality assurance
- **Model Support**: Both Argo models (GPT-4, Claude, etc.) and Gemini models
- **Progressive Output**: Real-time results with metadata and error tracking
- **Terminal GUI**: Beautiful terminal interface for monitoring progress
- **Robust Error Handling**: Graceful failure handling with partial result recovery
- **Token Limit Management**: Multi-part generation to avoid API limits

## Installation

```bash
# Install dependencies
pip install -r requirements_gui.txt

# Or install individually
pip install rich requests
```

## Usage

### Terminal GUI Mode (Recommended)

The GUI provides a real-time terminal interface to monitor progress:

```bash
# Run with GUI
python run_smol_reasoner_gui.py input.jsonl output --model gemini25flash

# With custom parameters
python run_smol_reasoner_gui.py input.jsonl output \
    --model gemini25flash \
    --max-samples 10 \
    --num-questions 3 \
    --api-key YOUR_API_KEY

# List available models
python run_smol_reasoner_gui.py --list-models
```

**GUI Features:**
- Real-time progress monitoring
- Live statistics (success rate, token counts, processing time)
- Recent activity log
- Error tracking and display
- Graceful shutdown with Ctrl+C
- Detailed logging to files

**Output Files:**
- `output.jsonl` - Progressive results (one per line)
- `output.detailed.log` - Detailed processing logs
- `output.meta.json` - Metadata and statistics

### Command Line Mode

For headless operation or integration with other tools:

```bash
# Basic usage
python run_smol_reasoner.py input.jsonl output.json --model gemini25flash

# With progressive output
python run_smol_reasoner.py input.jsonl output.json \
    --model gemini25flash \
    --max-samples 5 \
    --num-questions 3
```

## Input Format

Create a JSONL file with text samples:

```jsonl
{"text": "Quantum entanglement is a phenomenon where..."}
{"text": "Dark matter is one of the most mysterious..."}
{"text": "The process of photosynthesis involves..."}
```

## Model Support

### Argo Models (via UnifiedLLMClient)
- `gpt4o` - GPT-4 Omni
- `claudesonnet4` - Claude 3.5 Sonnet
- `gpt4turbo` - GPT-4 Turbo
- `claude3haiku` - Claude 3 Haiku

### Gemini Models
- `gemini25flash` - Gemini 2.5 Flash (default)
- `gemini20flash` - Gemini 2.0 Flash
- `gemini15pro` - Gemini 1.5 Pro

## Architecture

### Multi-Agent System

The system uses 4 specialized agents working together:

1. **Question Writer Group**
   - Writer: Generates diverse, challenging questions
   - Internal Critic: Evaluates question quality
   - Feedback Loop: Up to 3 iterations for refinement

2. **Solution Proposer Group**
   - Proposer: Creates multiple plausible answer choices
   - Internal Critic: Evaluates choice quality and diversity
   - Feedback Loop: Ensures high-quality options

3. **Final Solution Writer Group**
   - Writer: Generates detailed reasoning chains (2000-4000 words)
   - Internal Critic: Evaluates reasoning quality and authenticity
   - Multi-part Generation: Breaks long responses into manageable chunks

4. **Orchestrator**
   - Selects best questions from validated candidates
   - Coordinates between agent groups
   - Makes final quality decisions

### Reasoning Chain Generation

The system generates human-like reasoning traces with:
- Natural thinking progression
- Self-correction phrases ("Wait, let me reconsider...")
- Uncertainty expressions ("I'm not entirely sure...")
- Bias checking ("Am I missing something here?")
- User awareness ("The user is asking about...")

## Output Structure

### Progressive Output (.jsonl)
Each line contains a complete reasoning trace:

```json
{
  "question": {
    "stem": "What is the primary mechanism...",
    "topic": "Physics",
    "difficulty": "Medium"
  },
  "choices": [
    {
      "id": "A",
      "text": "Quantum tunneling effect",
      "reasoning": "Based on the principles...",
      "confidence": 0.8
    }
  ],
  "reasoning_chain": "Looking at this question, I need to think through...",
  "token_count": 3247,
  "metadata": {
    "model_used": "gemini25flash",
    "interface_used": "gemini",
    "completion_timestamp": "2024-01-15T10:30:00"
  }
}
```

### Metadata (.meta.json)
Contains processing statistics and configuration:

```json
{
  "generation_timestamp": "2024-01-15T10:00:00",
  "model_used": "gemini25flash",
  "total_samples": 50,
  "completed_samples": 47,
  "failed_samples": 3,
  "success_rate": 94.0,
  "total_tokens": 152840,
  "average_tokens": 3252.0,
  "total_processing_time": 1847.3,
  "average_time_per_sample": 36.9
}
```

## Error Handling

The system includes robust error handling:
- Token limit management with multi-part generation
- Graceful failure with partial result recovery
- Detailed error logging
- Automatic fallback strategies
- Progress preservation on interruption

## Configuration

### API Keys

Set environment variables:
```bash
export GEMINI_TEST_KEY="your_gemini_api_key"
export ARGO_USER_NAME="your_argo_username"
```

Or pass via command line:
```bash
python run_smol_reasoner_gui.py input.jsonl output \
    --api-key YOUR_GEMINI_KEY \
    --user-name YOUR_ARGO_USERNAME
```

### Parameters

- `--max-samples` - Limit number of samples to process
- `--num-questions` - Questions generated per sample (default: 5)
- `--max-workers` - Parallel processing threads (default: 1)
- `--interface` - Force specific interface (auto/argo/gemini)

## Examples

### Basic Usage with GUI
```bash
# Process 10 samples with GUI monitoring
python run_smol_reasoner_gui.py sample_texts.jsonl results \
    --model gemini25flash \
    --max-samples 10 \
    --num-questions 3
```

### Production Processing
```bash
# Process large dataset with detailed logging
python run_smol_reasoner_gui.py large_dataset.jsonl production_results \
    --model claudesonnet4 \
    --interface argo \
    --max-samples 1000 \
    --user-name your_username
```

### Quick Testing
```bash
# Test with small sample
python run_smol_reasoner_gui.py sample_texts.jsonl test_output \
    --model gemini25flash \
    --max-samples 3 \
    --num-questions 2
```

## Troubleshooting

### Common Issues

1. **Rich library not found**
   ```bash
   pip install rich
   ```

2. **API key errors**
   - Check environment variables
   - Verify API key validity
   - Ensure proper permissions

3. **Token limit errors**
   - System automatically handles with multi-part generation
   - Check detailed logs for specific error messages

4. **Memory issues**
   - Reduce `--max-workers` to 1
   - Process smaller batches with `--max-samples`

### GUI Controls

- **Ctrl+C**: Graceful shutdown (preserves progress)
- **Terminal resize**: GUI automatically adapts
- **Interruption**: Progress is saved to files

## File Structure

```
├── run_smol_reasoner_gui.py      # GUI interface
├── run_smol_reasoner.py          # Command line interface
├── smol_synthetic_reasoner.py    # Core multi-agent system
├── llm_interface.py              # Unified LLM client
├── gemini_client.py              # Gemini API client
├── sample_texts.jsonl            # Example input data
├── requirements_gui.txt          # GUI dependencies
└── README_smol_reasoner.md       # This file
```

## Contributing

The system is designed to be modular and extensible:
- Add new agents by extending the base classes
- Implement new model interfaces in the UnifiedLLMClient
- Customize reasoning prompts in the agent system prompts
- Extend the GUI with additional monitoring features

## License

This project is part of the TPC synthetic reasoning system. 