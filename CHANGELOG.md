# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-07-11

### Added
- Initial release of LLM Synthetic Reasoning Chain Generator
- Support for Gemini models (2.5 Flash, 1.5 Flash, 1.5 Pro, 2.0 Flash Exp)
- Support for Argo models (GPT-4o, Claude Sonnet 4, etc.)
- Unified LLM interface with auto-detection
- LLM-as-judge quality evaluation system (1-10 scoring)
- High-quality reasoning chain generation suitable for training reasoning models
- CLI tool for batch processing JSONL files
- Python API for programmatic usage
- Consolidated JSON output format with metadata
- Quality filtering (configurable minimum score threshold)
- Support for both scientific text processing and general content

### Features
- **LLM-powered MCQ generation**: Creates multiple-choice questions from scientific text
- **Quality evaluation**: LLM-as-judge scores questions 1-10, only keeps high-quality ones
- **Detailed reasoning chains**: Generates comprehensive reasoning traces with:
  - Thinking process (like `<think>` tokens)
  - Step-by-step analysis of each choice
  - Verification and double-checking
  - Final conclusion with complete justification
  - Confidence assessment and uncertainty quantification
  - Alternative considerations and edge cases
- **Dual interface support**: Works with both Argo and Gemini APIs
- **Batch processing**: Process large JSONL files efficiently
- **Configurable parameters**: Adjustable quality thresholds, number of questions, etc.

### Technical Details
- Python 3.8+ compatibility
- Minimal dependencies (requests, smolagents for Argo)
- Clean package structure with proper imports
- MIT License
- Comprehensive documentation and examples

### Default Settings
- Model: Gemini 2.5 Flash (free tier)
- Quality threshold: 7.0/10
- Questions per sample: 5 initial → ~2-3 accepted
- Output format: Consolidated JSON with metadata