#!/usr/bin/env python3
"""
Convert Multiple Choice Questions to Reasoning Traces

Takes MCQ data in eleuther-ai evaluation harness format and generates reasoning traces
that lead to the correct answer. Reuses machinery from llm_synthetic_reasoner.py.

Usage examples:
    # Convert MCQ file to reasoning traces
    python convert_mcq_reasoning.py --input mcq_data.jsonl --output mcq_reasoning.jsonl
    
    # Specify correct answer field and model
    python convert_mcq_reasoning.py --input data.jsonl --answer-field "correct" --model gpt4o
    
    # Generate multiple reasoning variations per question
    python convert_mcq_reasoning.py --input data.jsonl --traces-per-question 3
"""

import asyncio
import random
import argparse
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import components from llm_synthetic_reasoner
try:
    from llm_synthetic_reasoner import LLMInterface, ArgoLLMAdapter, ReasoningFragment
    from llm_interface import UnifiedLLMClient
except ImportError:
    # Fallback imports
    import sys
    sys.path.append('.')
    from llm_synthetic_reasoner import LLMInterface, ArgoLLMAdapter, ReasoningFragment
    from llm_interface import UnifiedLLMClient

# System prompts for instruct tuning (copied from convert_to_instruct_dataset.py)
SYSTEM_PROMPTS = [
    "You are a helpful assistant that thinks step by step. Before providing your final answer, work through your reasoning in <thought> tags.",
    
    "You are an expert assistant. When answering questions, first analyze the problem carefully in <thought> tags, then provide your response.",
    
    "Please think through this question systematically. Show your reasoning process in <thought> tags before giving your final answer.",
    
    "You are a thoughtful AI assistant. Break down complex problems by reasoning through them step-by-step in <thought> tags before responding.",
    
    "Before answering, please think carefully about the question. Use <thought> tags to show your reasoning process, then provide a clear final answer.",
    
    "You are an analytical assistant. For each question, first work through your thinking in <thought> tags, considering different aspects and possibilities before giving your response.",
    
    "Take time to reason through this question carefully. Show your thought process in <thought> tags, then provide a comprehensive answer.",
    
    "You are a reasoning-focused assistant. Always think through problems step by step in <thought> tags before providing your final answer.",
    
    "Please approach this question methodically. Use <thought> tags to show your analysis and reasoning before giving your response.",
    
    "You are an AI that reasons carefully before responding. Work through your thinking in <thought> tags, then provide a clear and helpful answer."
]


@dataclass
class MCQQuestion:
    """Multiple choice question structure"""
    question: str
    choices: List[str]  # List of answer choices
    correct_answer: str  # The correct answer (letter or text)
    context: str = ""  # Optional context/passage
    question_id: str = ""  # Optional question identifier
    domain: str = "General"  # Subject domain


class MCQReasoningGenerator:
    """Generate reasoning traces for multiple choice questions"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
        # Reuse reasoning patterns from synthetic reasoner
        self.logical_rigor_markers = [
            "therefore", "however", "given that", "this leads to", "consequently",
            "alternatively", "nevertheless", "furthermore", "specifically", "precisely",
            "in contrast", "as a result", "let me analyze", "considering", "examining"
        ]
    
    async def generate_mcq_reasoning_trace(self, mcq: MCQQuestion) -> Dict[str, Any]:
        """Generate a complete reasoning trace for an MCQ leading to the correct answer."""
        
        start_time = time.time()
        print(f"  🚀 Generating reasoning for MCQ: {mcq.question[:60]}...")
        
        # Stage 1: Analyze the question and choices
        stage_start = time.time()
        analysis = await self._analyze_mcq_structure(mcq)
        print(f"     Question analysis: {time.time() - stage_start:.1f}s")
        
        # Stage 2: Generate reasoning fragments
        stage_start = time.time()
        fragments = await asyncio.gather(
            self._evaluate_each_choice(mcq),
            self._identify_key_concepts(mcq),
            self._apply_domain_knowledge(mcq)
        )
        print(f"     Fragment generation: {time.time() - stage_start:.1f}s")
        
        # Stage 3: Assemble coherent reasoning trace
        stage_start = time.time()
        reasoning_trace = await self._assemble_mcq_reasoning(mcq, analysis, fragments)
        print(f"     Reasoning assembly: {time.time() - stage_start:.1f}s")
        
        # Stage 4: Generate final answer
        stage_start = time.time()
        final_answer = await self._generate_final_mcq_answer(mcq, reasoning_trace)
        print(f"     Answer generation: {time.time() - stage_start:.1f}s")
        
        total_time = time.time() - start_time
        print(f"  📊 Total MCQ reasoning time: {total_time:.1f}s")
        
        return {
            'question_data': {
                'question': mcq.question,
                'choices': mcq.choices,
                'correct_answer': mcq.correct_answer,
                'context': mcq.context,
                'domain': mcq.domain,
                'question_id': mcq.question_id
            },
            'reasoning_trace': reasoning_trace,
            'final_answer': final_answer,
            'metadata': {
                'logical_rigor_score': self._calculate_logical_rigor_score(reasoning_trace),
                'complexity_level': self._assess_complexity(reasoning_trace),
                'reasoning_patterns': self._identify_patterns(reasoning_trace),
                'generation_time': round(total_time, 2)
            }
        }
    
    async def _analyze_mcq_structure(self, mcq: MCQQuestion) -> str:
        """Analyze the MCQ structure and identify what makes the correct answer right."""
        
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(mcq.choices)])
        
        prompt = f"""
Analyze this multiple choice question to understand what reasoning approach would lead to the correct answer.

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

Choices:
{choices_text}

Correct Answer: {mcq.correct_answer}

Think about:
1. What key concepts or knowledge areas does this question test?
2. What makes the correct answer right and the other options wrong?
3. What reasoning steps would naturally lead someone to the correct choice?
4. What common misconceptions might lead to wrong answers?
5. What domain-specific knowledge is required?

Provide analysis that will help generate authentic reasoning leading to the correct answer.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    async def _evaluate_each_choice(self, mcq: MCQQuestion) -> ReasoningFragment:
        """Generate reasoning that evaluates choices or works toward the answer."""
        
        if len(mcq.choices) > 1:
            # Traditional MCQ with multiple choices - but reason fluidly without referencing letters
            choices_text = "\n".join([f"- {choice}" for choice in mcq.choices])
            
            prompt = f"""
Think through the possible answers to this question, exploring different possibilities naturally:

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

Possible answers to consider:
{choices_text}

Explore these possibilities naturally in your thinking. Show your reasoning as you:
- Consider what each possibility means and implies
- Think about which ones seem plausible initially
- Notice why some ideas might be appealing but ultimately incorrect
- Reason through what makes the correct answer stand out

Think fluidly: "One possibility is that... but this seems problematic because... Another idea is that... this makes more sense because..." 

Avoid referencing specific choice letters or numbers. Instead, refer to the ideas themselves naturally.
"""
        else:
            # Open-ended question with ground truth answer
            prompt = f"""
Think through how to approach this question, working toward the correct answer:

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

Ground Truth Answer: {mcq.correct_answer}

Show your reasoning process as you:
- Break down what the question is asking
- Consider what knowledge or approach is needed
- Think through the steps to reach the answer
- Work systematically toward the solution

Express this as natural internal thought showing your problem-solving process.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='choice_evaluation' if len(mcq.choices) > 1 else 'problem_solving',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _identify_key_concepts(self, mcq: MCQQuestion) -> ReasoningFragment:
        """Identify and reason about key concepts needed to answer the question."""
        
        prompt = f"""
Identify and think through the key concepts needed to answer this question correctly:

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

Think about:
- What fundamental concepts or principles apply here?
- What knowledge is essential to distinguish right from wrong answers?
- How do these concepts relate to each other?
- What background understanding is assumed?

Show your thinking as you work through the conceptual foundation needed to answer this question.
Express this as natural internal thought, showing how you're connecting ideas.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='concept_identification',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _apply_domain_knowledge(self, mcq: MCQQuestion) -> ReasoningFragment:
        """Apply domain-specific knowledge and reasoning patterns."""
        
        prompt = f"""
Apply relevant domain knowledge to reason through this question:

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

Domain: {mcq.domain}

Think about:
- What domain-specific principles or rules apply?
- Are there standard approaches or methodologies relevant here?
- What experience or knowledge from this field helps?
- How would an expert in this domain approach this question?

Show your thinking as you apply domain expertise to work toward the answer.
Maintain a natural internal dialogue, as if you're drawing on your knowledge in this area.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='domain_application',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _assemble_mcq_reasoning(
        self, 
        mcq: MCQQuestion, 
        analysis: str, 
        fragments: List[ReasoningFragment]
    ) -> str:
        """Assemble fragments into a coherent reasoning trace leading to correct answer."""
        
        if len(mcq.choices) > 1:
            # Traditional MCQ format - but create fluid reasoning
            choices_text = "\n".join([f"- {choice}" for choice in mcq.choices])
            choices_section = f"""
Possible answers to consider:
{choices_text}
"""
            reasoning_focus = """- Natural exploration of different possibilities without referencing choice letters
- Fluid consideration of ideas: "One possibility is...", "Another idea is...", "This seems problematic because..."
- Organic comparison between concepts and approaches"""
        else:
            # Open-ended question format
            choices_section = f"Expected Answer: {mcq.correct_answer}"
            reasoning_focus = "- Systematic problem-solving approach\n- Step-by-step derivation of the answer"
        
        prompt = f"""
Create a natural, flowing reasoning trace that leads to the correct answer for this question.

Question: {mcq.question}

{f"Context: {mcq.context}" if mcq.context else ""}

{choices_section}

Question Analysis: {analysis}

Reasoning Components:
1. Problem approach: {fragments[0].content}
2. Key concepts: {fragments[1].content}
3. Domain knowledge: {fragments[2].content}

Create a structured <thought>...</thought> section that demonstrates:
- Natural progression of thinking about the question
{reasoning_focus}
- Application of relevant knowledge and concepts
- Logical reasoning that leads to the correct answer
- Moments of consideration, analysis, and decision-making
- Authentic thought patterns including some uncertainty and self-correction

IMPORTANT: For multiple choice questions, avoid referencing "Choice A", "Option B", etc. Instead, refer to the concepts naturally: "One possibility is that mitochondria handle protein synthesis, but this doesn't align with... Another idea is energy production, which makes sense because..."

Weave the reasoning components into a flowing expert thought process that feels completely natural and organic.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    async def _generate_final_mcq_answer(self, mcq: MCQQuestion, reasoning_trace: str) -> str:
        """Generate a clear final answer based on the reasoning trace."""
        
        if len(mcq.choices) > 1:
            # Traditional MCQ format - but answer with content, not letter
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(mcq.choices)])
            correct_choice_content = mcq.choices[ord(mcq.correct_answer.upper()) - ord('A')] if mcq.correct_answer.isalpha() else mcq.correct_answer
            
            prompt = f"""
Based on the reasoning trace below, provide a clear, concise final answer for this question.

Question: {mcq.question}

Available choices:
{choices_text}

Correct answer: {correct_choice_content}

Reasoning: {reasoning_trace}

Provide a final answer that:
- States the correct answer directly (the actual content, not a letter)
- Briefly summarizes the key reasoning
- Is confident and decisive
- Connects to the reasoning shown
- Does NOT reference choice letters since the user didn't see any choices

Format: "The answer is [actual content]. [Brief explanation based on reasoning]"
"""
        else:
            # Open-ended question format
            prompt = f"""
Based on the reasoning trace below, provide a clear, comprehensive final answer for this question.

Question: {mcq.question}

Reasoning: {reasoning_trace}

Provide a final answer that:
- Directly addresses the question asked
- Summarizes the key points from your reasoning
- Is well-structured and comprehensive
- Demonstrates understanding of the topic
- Does not reference answer choices (since this is an open-ended question)

Provide your answer in a natural, flowing format that directly responds to the question.
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return response.strip()
    
    # Reuse utility methods from synthetic reasoner
    def _estimate_confidence(self, text: str) -> float:
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'not sure', 'unclear', 'might']
        certainty_words = ['definitely', 'clearly', 'obviously', 'certain', 'sure']
        
        uncertainty_count = sum(text.lower().count(word) for word in uncertainty_words)
        certainty_count = sum(text.lower().count(word) for word in certainty_words)
        
        return max(0.1, min(0.9, 0.5 + 0.1 * (certainty_count - uncertainty_count)))
    
    def _find_logical_markers(self, text: str) -> List[str]:
        markers = []
        for marker in self.logical_rigor_markers:
            if marker in text.lower():
                markers.append(marker)
        return markers
    
    def _calculate_logical_rigor_score(self, trace: str) -> float:
        patterns = [
            'therefore', 'however', 'given that', 'this leads to', 'consequently',
            'specifically', 'let me analyze', 'considering', 'examining', 'systematic'
        ]
        score = sum(1 for pattern in patterns if pattern in trace.lower())
        return min(1.0, score / len(patterns))
    
    def _assess_complexity(self, trace: str) -> str:
        word_count = len(trace.split())
        if word_count < 150:
            return 'low'
        elif word_count < 300:
            return 'medium'
        else:
            return 'high'
    
    def _identify_patterns(self, trace: str) -> List[str]:
        patterns = []
        if any(word in trace.lower() for word in ['choice', 'option', 'alternative']):
            patterns.append('choice_evaluation')
        if any(word in trace.lower() for word in ['concept', 'principle', 'knowledge']):
            patterns.append('concept_application')
        if any(word in trace.lower() for word in ['analyze', 'examine', 'consider']):
            patterns.append('systematic_analysis')
        if any(word in trace.lower() for word in ['wait', 'actually', 'reconsider']):
            patterns.append('self_correction')
        return patterns


def parse_mcq_data(data: Dict[str, Any], answer_field: str = "answer") -> MCQQuestion:
    """Parse MCQ data from various formats into standardized MCQQuestion."""
    
    # Extract question text
    question = data.get("question", data.get("prompt", ""))
    
    # Extract choices - handle multiple formats
    choices = []
    if "choices" in data:
        if isinstance(data["choices"], list):
            choices = data["choices"]
        elif isinstance(data["choices"], dict):
            # Handle {"A": "choice1", "B": "choice2"} format
            choices = list(data["choices"].values())
    elif "options" in data:
        choices = data["options"]
    else:
        # Look for individual choice fields
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in data:
                choices.append(data[letter])
    
    # Extract correct answer
    correct_answer = data.get(answer_field, data.get("correct", data.get("label", "")))
    
    # If no choices found but we have an answer, treat as open-ended question
    if not choices and correct_answer:
        choices = [correct_answer]  # Single "choice" which is the ground truth
    
    # Extract optional fields
    context = data.get("context", data.get("passage", ""))
    question_id = data.get("id", data.get("question_id", ""))
    domain = data.get("domain", data.get("subject", "General"))
    
    return MCQQuestion(
        question=question,
        choices=choices,
        correct_answer=str(correct_answer),
        context=context,
        question_id=question_id,
        domain=domain
    )


def convert_to_instruct_format(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MCQ reasoning trace to instruct tuning format."""
    
    # Build the question WITHOUT choices in user prompt
    question = trace_data['question_data']['question']
    choices = trace_data['question_data']['choices']
    
    # Always format as just the question - no answer choices shown to user
    question_section = f"Question: {question}"
    
    # Add context if available
    context = trace_data['question_data'].get('context', '')
    if context:
        full_question = f"Context: {context}\n\n{question_section}"
    else:
        full_question = question_section
    
    # Get reasoning and answer
    reasoning_trace = trace_data['reasoning_trace']
    final_answer = trace_data['final_answer']
    
    # Ensure reasoning is wrapped in thought tags
    if not reasoning_trace.strip().startswith('<thought>'):
        reasoning_trace = f"<thought>\n{reasoning_trace.strip()}\n</thought>"
    
    # Combine reasoning and final answer
    assistant_response = f"{reasoning_trace}\n\n{final_answer}"
    
    # Select random system prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": assistant_response}
        ]
    }


async def process_mcq_file(
    input_file: str,
    output_file: str,
    model: str = "claudesonnet4",
    answer_field: str = "answer",
    traces_per_question: int = 1,
    max_questions: int = None,
    output_format: str = "instruct"
):
    """Process an MCQ file and generate reasoning traces."""
    
    print(f"Processing MCQ file: {input_file}")
    print(f"Model: {model}")
    print(f"Answer field: {answer_field}")
    print(f"Traces per question: {traces_per_question}")
    print(f"Output format: {output_format}")
    print("-" * 50)
    
    # Initialize LLM
    llm = ArgoLLMAdapter(model)
    generator = MCQReasoningGenerator(llm)
    
    # Process input file
    all_traces = []
    questions_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_questions and questions_processed >= max_questions:
                break
                
            try:
                data = json.loads(line.strip())
                mcq = parse_mcq_data(data, answer_field)
                
                print(f"Processing question {line_num}: {mcq.question[:60]}...")
                
                # Generate multiple traces if requested
                for trace_idx in range(traces_per_question):
                    if traces_per_question > 1:
                        print(f"  Generating trace {trace_idx + 1}/{traces_per_question}")
                    
                    try:
                        trace = await generator.generate_mcq_reasoning_trace(mcq)
                        trace['line_number'] = line_num
                        trace['trace_index'] = trace_idx
                        
                        if output_format == "instruct":
                            instruct_data = convert_to_instruct_format(trace)
                            all_traces.append(instruct_data)
                        else:
                            all_traces.append(trace)
                        
                        print(f"  ✅ Generated reasoning trace")
                        
                    except Exception as e:
                        print(f"  ❌ Error generating trace: {e}")
                        continue
                
                questions_processed += 1
                
            except Exception as e:
                print(f"  ❌ Error processing line {line_num}: {e}")
                continue
    
    # Save results
    print(f"\nSaving {len(all_traces)} traces to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if output_format == "instruct":
            # Save as JSONL for instruct tuning
            for trace in all_traces:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        else:
            # Save as structured JSON
            output_data = {
                "metadata": {
                    "total_questions_processed": questions_processed,
                    "total_traces_generated": len(all_traces),
                    "model_used": model,
                    "traces_per_question": traces_per_question
                },
                "traces": all_traces
            }
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Completed processing {questions_processed} questions → {len(all_traces)} traces")


def main():
    parser = argparse.ArgumentParser(description="Convert MCQ data to reasoning traces")
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with MCQ data"
    )
    
    parser.add_argument(
        "--output",
        default="mcq_reasoning_traces.jsonl",
        help="Output file for reasoning traces (default: mcq_reasoning_traces.jsonl)"
    )
    
    parser.add_argument(
        "--model",
        default="claudesonnet4",
        help="LLM model to use (default: claudesonnet4)"
    )
    
    parser.add_argument(
        "--answer-field",
        default="answer",
        help="Field name containing correct answer (default: answer)"
    )
    
    parser.add_argument(
        "--traces-per-question",
        type=int,
        default=1,
        help="Number of reasoning traces per question (default: 1)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["instruct", "structured"],
        default="instruct",
        help="Output format: instruct (JSONL) or structured (JSON) (default: instruct)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Run processing
    asyncio.run(process_mcq_file(
        args.input,
        args.output,
        args.model,
        args.answer_field,
        args.traces_per_question,
        args.max_questions,
        args.output_format
    ))


if __name__ == "__main__":
    main()