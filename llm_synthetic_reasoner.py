#!/usr/bin/env python3
"""
LLM-powered Synthetic Reasoning Chain Generator

Uses LLMs to generate high-quality MCQs and reasoning chains from scientific text.
Architecture:
1. LLM generates multiple MCQs from text
2. LLM-as-judge evaluates MCQ quality (scores 1-10)
3. Keep only high-scoring MCQs (>=7/10)
4. LLM generates detailed reasoning chains for good MCQs
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .llm_interface import UnifiedLLMClient
except ImportError:
    from llm_interface import UnifiedLLMClient


@dataclass
class MCQChoice:
    id: str
    text: str
    is_correct: bool


@dataclass
class MCQuestion:
    stem: str
    choices: List[MCQChoice]
    correct_answer: str
    explanation: str


@dataclass
class MCQEvaluation:
    score: float  # 1-10
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str  # "accept" or "reject"


@dataclass
class ReasoningChain:
    thinking_process: str
    step_by_step_analysis: str
    verification_and_checking: str
    final_conclusion: str
    confidence_assessment: str


class LLMSyntheticReasoner:
    """LLM-powered synthetic reasoner supporting both Argo and Gemini models"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", interface: str = "auto", **kwargs):
        """Initialize with specified model and interface"""
        self.model = UnifiedLLMClient(model_name=model_name, interface=interface, **kwargs)
        self.model_name = model_name
        self.interface = self.model.interface
        
    def generate_mcqs(self, text: str, num_questions: int = 3) -> List[MCQuestion]:
        """Generate multiple-choice questions from scientific text"""
        
        prompt = f"""You are an expert at creating high-quality multiple-choice questions from scientific text.

TEXT TO ANALYZE:
{text}

TASK: Create {num_questions} multiple-choice questions based on this scientific text.

REQUIREMENTS:
- Each question should test understanding, analysis, or recall of key scientific concepts
- Each question should have exactly 4 answer choices (A, B, C, D)
- Exactly one choice should be correct, three should be plausible but incorrect
- Questions should be well-formed, clear, and grammatically correct
- Focus on the most important concepts from the text
- Ensure that the question is self-contained and does not require knowledge from the text to answer
- Do not reference "the text" or "the source material" in the question stem--the question should be able to stand on its own
- Ensure that no information in the question indicates the correct answer

OUTPUT FORMAT:
Return a JSON array where each question has this structure:
{{
  "stem": "The question text ending with a question mark?",
  "choices": [
    {{"id": "A", "text": "First choice", "is_correct": true}},
    {{"id": "B", "text": "Second choice", "is_correct": false}},
    {{"id": "C", "text": "Third choice", "is_correct": false}},
    {{"id": "D", "text": "Fourth choice", "is_correct": false}}
  ],
  "correct_answer": "A",
  "explanation": "Brief explanation of why A is correct"
}}

Generate {num_questions} high-quality questions now:"""

        try:
            response = self.model.generate([{"role": "user", "content": prompt}])
            content = response
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                print(f"Warning: Could not find JSON array in response")
                return []
            
            questions_data = json.loads(json_match.group())
            questions = []
            
            for q_data in questions_data:
                choices = []
                for choice_data in q_data.get('choices', []):
                    choices.append(MCQChoice(
                        id=choice_data['id'],
                        text=choice_data['text'],
                        is_correct=choice_data['is_correct']
                    ))
                
                questions.append(MCQuestion(
                    stem=q_data['stem'],
                    choices=choices,
                    correct_answer=q_data['correct_answer'],
                    explanation=q_data.get('explanation', '')
                ))
            
            return questions
            
        except Exception as e:
            print(f"Error generating MCQs: {e}")
            return []
    
    def evaluate_mcq(self, question: MCQuestion, source_text: str) -> MCQEvaluation:
        """Use LLM-as-judge to evaluate MCQ quality"""
        
        choices_text = "\\n".join([f"{c.id}. {c.text}" for c in question.choices])
        correct_answer = next(c for c in question.choices if c.is_correct)
        
        prompt = f"""You are an expert educational assessment evaluator. Rate the quality of this multiple-choice question.

SOURCE TEXT:
{source_text[:1000]}{"..." if len(source_text) > 1000 else ""}

QUESTION TO EVALUATE:
Stem: {question.stem}
Choices:
{choices_text}
Correct Answer: {correct_answer.id}. {correct_answer.text}

EVALUATION CRITERIA (Rate 1-10):
1. Clarity: Is the question clearly worded and unambiguous?
2. Relevance: Does it test important concepts from the source text?
3. Difficulty: Is it appropriately challenging (not too easy/hard)?
4. Distractors: Are the incorrect choices plausible but clearly wrong?
5. Accuracy: Is the correct answer actually correct and well-supported?

Provide your evaluation in this JSON format:
{{
  "score": 7.5,
  "strengths": ["Clear question stem", "Good distractors"],
  "weaknesses": ["Could be more specific"],
  "recommendation": "accept"
}}

Score should be 1-10 (decimals allowed).
Recommendation should be "accept" (score ≥7) or "reject" (score <7).

Your evaluation:"""

        try:
            response = self.model.generate([{"role": "user", "content": prompt}])
            content = response
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                # Fallback: parse manually
                return MCQEvaluation(score=5.0, strengths=[], weaknesses=["Could not parse evaluation"], recommendation="reject")
            
            eval_data = json.loads(json_match.group())
            
            return MCQEvaluation(
                score=float(eval_data.get('score', 5.0)),
                strengths=eval_data.get('strengths', []),
                weaknesses=eval_data.get('weaknesses', []),
                recommendation=eval_data.get('recommendation', 'reject')
            )
            
        except Exception as e:
            print(f"Error evaluating MCQ: {e}")
            return MCQEvaluation(score=5.0, strengths=[], weaknesses=[f"Evaluation error: {e}"], recommendation="reject")
    
    def generate_reasoning_chain(self, question: MCQuestion, source_text: str) -> ReasoningChain:
        """Generate detailed reasoning chain for a high-quality MCQ"""
        
        choices_text = "\\n".join([f"{c.id}. {c.text}" for c in question.choices])
        
        prompt = f"""You are an expert reasoning model. Generate a detailed reasoning chain for solving this multiple-choice question, similar to the internal thinking process of advanced AI models like Deepseek-R1.

SOURCE TEXT:
{source_text}

QUESTION:
{question.stem}

CHOICES:
{choices_text}

TASK: Generate a complete reasoning chain with these components:

1. THINKING PROCESS: Initial analysis of the question and source material (like <think> tokens)
2. STEP-BY-STEP ANALYSIS: Systematic evaluation of each answer choice
3. VERIFICATION AND CHECKING: Double-check reasoning, consider biases, validate logic
4. FINAL CONCLUSION: Definitive answer with complete justification
5. CONFIDENCE ASSESSMENT: Certainty level and potential uncertainties

Make this reasoning trace suitable for training reasoning models. Include:
- Self-correction and doubt ("Wait, let me reconsider...")
- Bias checking ("Am I falling for confirmation bias?")
- Multiple verification steps
- Complete justification for every claim
- Explicit uncertainty quantification

OUTPUT FORMAT - Return JSON:
{{
  "thinking_process": "Initial thoughts and question analysis...",
  "step_by_step_analysis": "Detailed analysis of each choice...",
  "verification_and_checking": "Double-checking work, bias considerations...",
  "final_conclusion": "Final answer with complete justification...",
  "confidence_assessment": "Confidence level and uncertainty factors..."
}}

Generate the reasoning chain:"""

        try:
            response = self.model.generate([{"role": "user", "content": prompt}])
            content = response
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                # Create fallback reasoning
                return self._create_fallback_reasoning(question, source_text)
            
            reasoning_data = json.loads(json_match.group())
            
            return ReasoningChain(
                thinking_process=reasoning_data.get('thinking_process', ''),
                step_by_step_analysis=reasoning_data.get('step_by_step_analysis', ''),
                verification_and_checking=reasoning_data.get('verification_and_checking', ''),
                final_conclusion=reasoning_data.get('final_conclusion', ''),
                confidence_assessment=reasoning_data.get('confidence_assessment', '')
            )
            
        except Exception as e:
            print(f"Error generating reasoning chain: {e}")
            return self._create_fallback_reasoning(question, source_text)
    
    def process_text(self, text: str, num_initial_questions: int = 5, min_score: float = 7.0) -> List[Dict[str, Any]]:
        """Complete pipeline: generate MCQs, evaluate, create reasoning chains for good ones"""
        
        results = []
        
        print(f"Generating {num_initial_questions} initial MCQs...")
        mcqs = self.generate_mcqs(text, num_initial_questions)
        
        if not mcqs:
            print("No MCQs generated")
            return results
        
        print(f"Generated {len(mcqs)} MCQs, evaluating quality...")
        
        for i, mcq in enumerate(mcqs, 1):
            print(f"Evaluating MCQ {i}/{len(mcqs)}...")
            evaluation = self.evaluate_mcq(mcq, text)
            
            print(f"  Score: {evaluation.score}/10")
            print(f"  Recommendation: {evaluation.recommendation}")
            
            if evaluation.score >= min_score and evaluation.recommendation == "accept":
                print(f"  ✅ Accepted! Generating reasoning chain...")
                reasoning = self.generate_reasoning_chain(mcq, text)
                
                result = {
                    "question": {
                        "stem": mcq.stem,
                        "choices": [{"id": c.id, "text": c.text, "is_correct": c.is_correct} for c in mcq.choices],
                        "correct_answer": mcq.correct_answer,
                        "explanation": mcq.explanation
                    },
                    "evaluation": {
                        "score": evaluation.score,
                        "strengths": evaluation.strengths,
                        "weaknesses": evaluation.weaknesses,
                        "recommendation": evaluation.recommendation
                    },
                    "reasoning_chain": {
                        "thinking_process": reasoning.thinking_process,
                        "step_by_step_analysis": reasoning.step_by_step_analysis,
                        "verification_and_checking": reasoning.verification_and_checking,
                        "final_conclusion": reasoning.final_conclusion,
                        "confidence_assessment": reasoning.confidence_assessment
                    },
                    "source_text": text,
                    "model_used": self.model_name
                }
                
                results.append(result)
                print(f"  ✅ Complete reasoning chain generated")
            else:
                print(f"  ❌ Rejected (score {evaluation.score} < {min_score})")
        
        print(f"\\nFinal results: {len(results)} high-quality questions with reasoning chains")
        return results
    
    def process_jsonl_file(self, file_path: str, output_path: str, max_samples: Optional[int] = None, **kwargs):
        """Process JSONL file and save results"""
        
        file_path = Path(file_path)
        output_path = Path(output_path)
        
        print(f"Processing JSONL file: {file_path}")
        
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text'].strip():
                        samples.append(data['text'])
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {i+1}")
                    continue
        
        print(f"Loaded {len(samples)} text samples")
        
        all_results = []
        for i, text in enumerate(samples):
            print(f"\\nProcessing sample {i+1}/{len(samples)}...")
            print(f"Text length: {len(text)} characters")
            
            results = self.process_text(text, **kwargs)
            
            for result in results:
                result['sample_index'] = i
                all_results.append(result)
        
        # Save results
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_samples_processed": len(samples),
                "total_questions_generated": len(all_results),
                "model_used": self.model_name,
                "interface_used": self.interface,
                "source_file": str(file_path),
                "generation_parameters": kwargs
            },
            "questions": all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\\n✅ Results saved to: {output_path}")
        print(f"Generated {len(all_results)} high-quality questions with reasoning chains")
        
        return all_results
    
    
    def _create_fallback_reasoning(self, question: MCQuestion, source_text: str) -> ReasoningChain:
        """Create fallback reasoning when LLM generation fails"""
        correct_choice = next(c for c in question.choices if c.is_correct)
        
        return ReasoningChain(
            thinking_process=f"I need to analyze the question: {question.stem}\\n\\nLooking at the source text, I can identify key information relevant to this question.",
            step_by_step_analysis=f"Examining each choice:\\nA. {question.choices[0].text}\\nB. {question.choices[1].text}\\nC. {question.choices[2].text}\\nD. {question.choices[3].text}\\n\\nThe correct answer appears to be {correct_choice.id}.",
            verification_and_checking=f"Double-checking: {correct_choice.text} aligns with the information in the source text.",
            final_conclusion=f"The answer is {correct_choice.id}: {correct_choice.text}",
            confidence_assessment="Moderate confidence based on source text alignment."
        )


def main():
    """Example usage"""
    
    # Test with sample text
    sample_text = """
    Dark matter is one of the most mysterious components of the universe. Unlike ordinary matter, 
    dark matter does not emit, absorb, or reflect electromagnetic radiation, making it invisible 
    to telescopes. However, its presence is inferred from its gravitational effects on visible 
    matter, such as the rotation curves of galaxies and gravitational lensing of distant objects. 
    Current estimates suggest that dark matter makes up approximately 27% of the universe's 
    total mass-energy content, significantly more than the 5% contributed by ordinary matter.
    """
    
    # Initialize reasoner
    reasoner = LLMSyntheticReasoner(model_name="gemini25flash")
    
    # Process text
    results = reasoner.process_text(sample_text, num_initial_questions=3, min_score=6.0)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\\n=== Question {i} ===")
        print(f"Stem: {result['question']['stem']}")
        print(f"Score: {result['evaluation']['score']}/10")
        print(f"Reasoning length: {len(result['reasoning_chain']['thinking_process'])} chars")


if __name__ == "__main__":
    main()