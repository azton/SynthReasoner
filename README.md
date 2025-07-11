# LLM Synthetic Reasoning Chain Generator 🧠⚡

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean, LLM-based system for generating high-quality multiple-choice questions and detailed reasoning chains from scientific text. Perfect for creating training data for reasoning models like DeepSeek-R1.

## 🔄 **Architecture**

1. **LLM generates MCQs** from scientific text (multiple candidates)
2. **LLM-as-judge evaluates** MCQ quality (scores 1-10)
3. **Keep only high-scoring MCQs** (≥7/10 by default)
4. **LLM generates detailed reasoning chains** for accepted MCQs

## 🚀 **Key Features**

- **Dual Interface Support**: Argo models + Gemini models
- **Quality Filtering**: Only keeps high-quality questions (≥7/10 score)
- **Deep Reasoning Traces**: Suitable for training reasoning models
- **Self-contained Questions**: No references to "the text" or source material
- **JSON Output**: Consolidated format with metadata

## 📋 **Core Components**

### 1. **LLM Interface** (`llm_interface.py`)
- Unified interface supporting both Argo and Gemini
- Auto-detection of model interface
- Handles authentication and API calls

### 2. **Synthetic Reasoner** (`llm_synthetic_reasoner.py`)
- Main generator class
- MCQ creation, evaluation, and reasoning chain generation
- Quality filtering and batch processing

### 3. **CLI Tool** (`run_llm_reasoner.py`)
- Command-line interface for processing JSONL files
- Supports both interfaces with auto-detection

## 🚀 **Quick Start**

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-synthetic-reasoner.git
cd llm-synthetic-reasoner

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_TEST_KEY=your_gemini_api_key
# Optional: export ARGO_USER=your_argo_username
```

### Basic Usage

```bash
# Basic usage with Gemini 2.5 Flash (default)
python run_llm_reasoner.py data.jsonl -o output.json

# With specific model
python run_llm_reasoner.py data.jsonl -o output.json -m gemini-1.5-pro

# Process limited samples for testing
python run_llm_reasoner.py data.jsonl -o output.json --max-samples 10

# With quality threshold and more questions per sample
python run_llm_reasoner.py data.jsonl -o output.json -n 5 -s 8.0
```

### Python API

```python
from llm_synthetic_reasoner import LLMSyntheticReasoner

# Initialize with Gemini 2.5 Flash
reasoner = LLMSyntheticReasoner(model_name="gemini-2.5-flash")

# Generate from text
sample_text = "Your scientific text here..."
results = reasoner.process_text(sample_text, num_initial_questions=3, min_score=7.0)

# Process JSONL file
results = reasoner.process_jsonl_file("input.jsonl", "output.json", max_samples=50)
```

## 📊 **Output Format**

```json
{
  "metadata": {
    "generation_timestamp": "2025-07-11T15:59:29",
    "total_samples_processed": 2,
    "total_questions_generated": 3,
    "model_used": "gemini-1.5-flash",
    "interface_used": "gemini",
    "source_file": "data.jsonl"
  },
  "questions": [
    {
      "question": {
        "stem": "Which instrument provided data on...",
        "choices": [...],
        "correct_answer": "A",
        "explanation": "..."
      },
      "evaluation": {
        "score": 8.0,
        "strengths": [...],
        "weaknesses": [...],
        "recommendation": "accept"
      },
      "reasoning_chain": {
        "thinking_process": "Initial analysis...",
        "step_by_step_analysis": "Examining each choice...",
        "verification_and_checking": "Double-checking work...",
        "final_conclusion": "The answer is A because...",
        "confidence_assessment": "High confidence (95%)..."
      },
      "source_text": "Original text...",
      "model_used": "gemini-1.5-flash"
    }
  ]
}
```

## 🎯 **Quality Features**

### MCQ Quality Criteria
- **Clarity**: Well-formed, unambiguous questions
- **Relevance**: Tests important concepts from source
- **Difficulty**: Appropriately challenging
- **Distractors**: Plausible but clearly incorrect choices
- **Accuracy**: Correct answer is well-supported

### Reasoning Chain Quality
- **Thinking Process**: Like `<think>` tokens in advanced models
- **Step-by-Step Analysis**: Systematic evaluation of each choice
- **Verification**: Double-checking and bias detection
- **Self-Correction**: Explicit reconsideration of reasoning
- **Confidence Assessment**: Uncertainty quantification

## 🔧 **Available Models**

### Gemini Models (via API)
- `gemini-2.5-flash` (default, free tier)
- `gemini-1.5-flash`
- `gemini-1.5-pro`
- `gemini-2.0-flash-exp`

### Argo Models (via Argo API)
- `gpt4o`, `claudesonnet4`, `claudeopus4`
- `gemini25flash`, `gemini25pro`
- `gpt4`, `gpt4turbo`, `gpto1preview`

## 🚦 **Environment Setup**

### For Gemini
```bash
export GEMINI_TEST_KEY=your_api_key
```

### For Argo
```bash
export ARGO_USER=your_username
```

## 📈 **Performance**

- **Generation Speed**: ~30-60 seconds per high-quality question
- **Quality Rate**: ~60-80% of generated MCQs pass quality threshold
- **Reasoning Depth**: 2,000-4,000 characters per reasoning chain
- **Success Rate**: >95% successful MCQ generation

## 🎯 **Training Ready**

The generated reasoning chains are specifically designed for training reasoning models:

- **Complete thinking processes** similar to Deepseek-R1 style
- **Self-correction mechanisms** and doubt expression
- **Bias detection** and systematic verification
- **Uncertainty quantification** and confidence assessment
- **Alternative consideration** of edge cases

This makes the output ideal for bootstrapping reasoning model training with high-quality synthetic data.

## 📝 **Examples**

See the [examples/](examples/) directory for detailed usage examples:

- `example_usage.py` - Comprehensive examples showing all features
- Basic text processing
- Batch JSONL file processing  
- Different model interfaces
- Quality threshold comparisons

## 🧪 **Development**

### Running Tests

```bash
# Run example usage
python examples/example_usage.py

# Test specific functionality
python -c "from llm_synthetic_reasoner import LLMSyntheticReasoner; print('✅ Import successful')"
```

### Project Structure

```
llm-synthetic-reasoner/
├── llm_synthetic_reasoner.py    # Main reasoner class
├── llm_interface.py             # Unified LLM interface
├── gemini_client.py             # Gemini API client
├── argo.py                      # Argo models interface
├── run_llm_reasoner.py          # CLI tool
├── examples/                    # Usage examples
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Built for high-quality reasoning model training
- Supports both Argo and Gemini LLM interfaces
- Designed for scientific text processing and educational content generation