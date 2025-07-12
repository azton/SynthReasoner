#!/usr/bin/env python3
"""
Smolagents-based Multi-Agent Synthetic Reasoning Chain Generator

A multi-agent system that generates high-quality reasoning chains from scientific text.
Uses smolagents framework with proper tools, models, and orchestration.
"""

import json
import re
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from smolagents import CodeAgent, tool, Model, ChatMessage, OpenAIServerModel
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    print("Warning: smolagents not installed. Install with: pip install smolagents")
    SMOLAGENTS_AVAILABLE = False
    # Fallback for testing
    class CodeAgent:
        def __init__(self, **kwargs): pass
        def run(self, task): return "Mock response"
    def tool(func): return func
    class Model: pass
    class ChatMessage: pass
    class OpenAIServerModel: pass

# Custom Argo model for non-Gemini models
class ArgoModel(Model):
    """Custom model for Argo interface"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__()
        self.model_id = model_id
        # Import here to avoid circular imports
        try:
            from llm_interface import UnifiedLLMClient
            self.client = UnifiedLLMClient(model_name=model_id, interface="argo", **kwargs)
        except ImportError:
            raise ImportError("llm_interface not available for Argo models")
    
    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """Generate response using Argo interface"""
        # Convert ChatMessage objects to dict format
        converted_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                converted_messages.append({
                    'role': msg.role,
                    'content': str(msg.content)
                })
            elif isinstance(msg, dict):
                converted_messages.append(msg)
            else:
                converted_messages.append({
                    'role': 'user',
                    'content': str(msg)
                })
        
        try:
            response = self.client.generate(converted_messages, **kwargs)
            return ChatMessage(role="assistant", content=response)
        except Exception as e:
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")


@dataclass
class Question:
    stem: str
    topic: str
    difficulty: str
    quality_score: float


@dataclass
class Choice:
    id: str
    text: str
    reasoning: str
    confidence: float


@dataclass
class ReasoningTrace:
    question: Question
    choices: List[Choice]
    selected_choice: str
    reasoning_chain: str
    token_count: int
    metadata: Dict[str, Any]


# Tools for the agents
@tool
def format_questions_for_critic(questions_json: str) -> str:
    """Format questions for critic evaluation.
    
    Args:
        questions_json: JSON string containing list of questions
        
    Returns:
        Formatted string for critic agent
    """
    try:
        questions = json.loads(questions_json)
        formatted = "QUESTIONS TO EVALUATE:\n\n"
        for i, q in enumerate(questions, 1):
            formatted += f"{i}. Question: {q.get('stem', '')}\n"
            formatted += f"   Topic: {q.get('topic', '')}\n"
            formatted += f"   Difficulty: {q.get('difficulty', '')}\n\n"
        
        formatted += "\nINSTRUCTIONS:\n"
        formatted += "Evaluate each question based on intellectual depth, clarity, appropriateness of difficulty, potential for generating meaningful reasoning, originality, and alignment with source material.\n\n"
        formatted += "For each question provide:\n"
        formatted += "- Detailed critique explaining strengths and weaknesses\n"
        formatted += "- Quality score (0-10) with justification\n"
        formatted += "- Final recommendation (accept/reject/revise)\n\n"
        formatted += "Select the highest quality question and return it as JSON with format:\n"
        formatted += '{"stem": "question text", "topic": "topic", "difficulty": "Easy/Medium/Hard", "quality_score": score, "critique": "detailed evaluation"}'
        
        return formatted
    except Exception as e:
        return f"Error formatting questions: {str(e)}"


@tool
def format_question_for_proposer(question_json: str, source_text: str) -> str:
    """Format question and source text for solution proposer.
    
    Args:
        question_json: JSON string containing the question
        source_text: Original source text for context
        
    Returns:
        Formatted string for solution proposer
    """
    try:
        question = json.loads(question_json)
        formatted = f"SOURCE TEXT:\n{source_text}\n\n"
        formatted += f"QUESTION TO CREATE SOLUTIONS FOR:\n"
        formatted += f"Question: {question.get('stem', '')}\n"
        formatted += f"Topic: {question.get('topic', '')}\n"
        formatted += f"Difficulty: {question.get('difficulty', '')}\n\n"
        
        formatted += "INSTRUCTIONS:\n"
        formatted += "Generate exactly 4 answer choices (A, B, C, D) that are:\n"
        formatted += "- Plausible to someone without deep expertise\n"
        formatted += "- Include both correct and incorrect but reasonable options\n"
        formatted += "- Show different reasoning paths and approaches\n"
        formatted += "- Test different aspects of understanding\n"
        formatted += "- Vary in sophistication and approach\n\n"
        formatted += "One choice should be clearly correct based on the source text.\n\n"
        formatted += "Return as JSON array with format:\n"
        formatted += '[{"id": "A", "text": "answer text", "reasoning": "brief explanation", "confidence": 0.8}, ...]'
        
        return formatted
    except Exception as e:
        return f"Error formatting question: {str(e)}"


@tool
def format_solutions_for_critic(solutions_json: str, question_json: str) -> str:
    """Format solutions for critic evaluation.
    
    Args:
        solutions_json: JSON string containing proposed solutions
        question_json: JSON string containing the question
        
    Returns:
        Formatted string for solution critic
    """
    try:
        solutions = json.loads(solutions_json)
        question = json.loads(question_json)
        
        formatted = f"QUESTION: {question.get('stem', '')}\n\n"
        formatted += "PROPOSED SOLUTIONS TO EVALUATE:\n\n"
        for i, solution in enumerate(solutions, 1):
            formatted += f"Solution {i} ({solution.get('id', '')}):\n"
            formatted += f"Text: {solution.get('text', '')}\n"
            formatted += f"Reasoning: {solution.get('reasoning', '')}\n"
            formatted += f"Confidence: {solution.get('confidence', 0)}\n\n"
        
        formatted += "INSTRUCTIONS:\n"
        formatted += "Evaluate each solution based on:\n"
        formatted += "- Accuracy relative to source material\n"
        formatted += "- Quality of reasoning and logic\n"
        formatted += "- Completeness of explanation\n"
        formatted += "- Appropriateness as distractor or correct answer\n"
        formatted += "- Potential for generating rich reasoning traces\n\n"
        formatted += "For each solution provide:\n"
        formatted += "- Detailed critique of strengths and weaknesses\n"
        formatted += "- Identification of logical gaps or errors\n"
        formatted += "- Assessment of educational value\n"
        formatted += "- Recommendation (approve/revise/reject)\n\n"
        formatted += "Select the best set of solutions and return as JSON array:\n"
        formatted += '[{"id": "A", "text": "text", "reasoning": "reasoning", "confidence": 0.8, "critique": "evaluation"}, ...]'
        
        return formatted
    except Exception as e:
        return f"Error formatting solutions: {str(e)}"


@tool
def format_for_reasoner(question_json: str, choices_json: str, source_text: str) -> str:
    """Format question, choices, and source text for final reasoner.
    
    Args:
        question_json: JSON string containing the question
        choices_json: JSON string containing the answer choices
        source_text: Original source text for context
        
    Returns:
        Formatted string for final reasoner
    """
    try:
        question = json.loads(question_json)
        choices = json.loads(choices_json)
        
        formatted = f"SOURCE TEXT:\n{source_text}\n\n"
        formatted += f"QUESTION:\n{question.get('stem', '')}\n\n"
        formatted += "ANSWER CHOICES:\n"
        for choice in choices:
            formatted += f"{choice.get('id', '')}: {choice.get('text', '')}\n"
        
        formatted += "\nINSTRUCTIONS:\n"
        formatted += "Create a detailed reasoning trace (2000-4000 words) that:\n"
        formatted += "- Shows natural thinking progression with self-correction\n"
        formatted += "- Includes phrases like 'Wait, let me reconsider...', 'Actually, thinking about this more...', 'Hmm, the user wants me to...'\n"
        formatted += "- Demonstrates bias checking and verification steps\n"
        formatted += "- Builds comprehensive justifications for each choice\n"
        formatted += "- Expresses uncertainty and confidence appropriately\n"
        formatted += "- Feels like the internal monologue of a careful, thoughtful problem-solver\n\n"
        formatted += "Structure your reasoning by:\n"
        formatted += "1. Initial analysis of the question and source material\n"
        formatted += "2. Systematic evaluation of each answer choice\n"
        formatted += "3. Self-correction and reconsideration phases\n"
        formatted += "4. Final selection with confidence assessment\n\n"
        formatted += "Return as JSON with format:\n"
        formatted += '{"question": question_object, "choices": choices_array, "selected_choice": "A", "reasoning_chain": "full_reasoning_text", "token_count": word_count}'
        
        return formatted
    except Exception as e:
        return f"Error formatting for reasoner: {str(e)}"


@tool
def save_reasoning_trace(trace_json: str, output_file: str) -> str:
    """Save a reasoning trace to file.
    
    Args:
        trace_json: JSON string containing the reasoning trace
        output_file: Path to save the trace
        
    Returns:
        Status message
    """
    try:
        trace = json.loads(trace_json)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        
        return f"Reasoning trace saved to {output_file}"
    except Exception as e:
        return f"Error saving trace: {str(e)}"


class SmolSyntheticReasoner:
    """Multi-agent synthetic reasoner using smolagents framework"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", interface: str = "auto", **kwargs):
        """Initialize the multi-agent system"""
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents not available. Install with: pip install smolagents")
        
        self.model_name = model_name
        self.interface = self._determine_interface(model_name, interface)
        self.model = self._create_model(model_name, self.interface, **kwargs)
        
        # Initialize specialized agents
        self.question_writer = self._create_question_writer()
        self.question_critic = self._create_question_critic()
        self.solution_proposer = self._create_solution_proposer()
        self.solution_critic = self._create_solution_critic()
        self.final_solution_writer = self._create_final_solution_writer()
        
        # Create main orchestrator agent
        self.orchestrator = self._create_orchestrator()
    
    def _determine_interface(self, model_name: str, interface: str) -> str:
        """Determine which interface to use based on model name"""
        if interface != "auto":
            return interface
        
        # Gemini models use OpenAI-compatible interface
        if any(gemini in model_name.lower() for gemini in ["gemini", "2.5", "1.5"]):
            return "gemini"
        
        # Other models use Argo interface
        return "argo"
    
    def _create_model(self, model_name: str, interface: str, **kwargs) -> Model:
        """Create appropriate model based on interface"""
        if interface == "gemini":
            # Use OpenAI compatibility layer for Gemini models
            api_key = kwargs.get('api_key') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter.")
            
            return OpenAIServerModel(
                model_id=model_name,
                api_key=api_key,
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                **{k: v for k, v in kwargs.items() if k != 'api_key'}
            )
        else:
            # Use custom Argo model
            return ArgoModel(model_id=model_name, **kwargs)
    
    def _create_question_writer(self) -> CodeAgent:
        """Create question writer agent"""
        return CodeAgent(
            model=self.model,
            tools=[],
            max_steps=5,
            verbosity_level=1,
            name="question_writer",
            description="""Generate diverse, challenging questions from scientific text.

Your task is to create thought-provoking questions that:
- Test deep understanding, not just recall
- Require analytical thinking and reasoning
- Have multiple plausible approaches
- Are intellectually engaging
- Cover different aspects of the source material

Generate questions at varying difficulty levels and different cognitive demands.
Focus on creativity and intellectual depth rather than following rigid templates.

Always return your response as a JSON array of question objects with:
- stem: The question text
- topic: The main topic area
- difficulty: Easy/Medium/Hard

DO NOT include quality scores - that is the job of the question critic.

Example format:
[
  {
    "stem": "What are the implications of quantum entanglement for...",
    "topic": "quantum physics", 
    "difficulty": "hard"
  }
]
"""
        )
    
    def _create_question_critic(self) -> CodeAgent:
        """Create question critic agent"""
        return CodeAgent(
            model=self.model,
            tools=[format_questions_for_critic],
            max_steps=5,
            verbosity_level=1,
            name="question_critic",
            description="""Evaluate and rate questions for quality, selecting only the best ones.

Your task is to critically assess questions based on:
- Intellectual depth and rigor
- Clarity and precision of language
- Appropriateness of difficulty level
- Potential for generating meaningful reasoning
- Originality and creativity
- Alignment with source material

For each question, provide:
- Detailed critique explaining strengths and weaknesses
- Quality score (0-10) with justification - this is YOUR responsibility, not the writer's
- Specific suggestions for improvement if needed
- Final recommendation (accept/reject/revise)

Select only the highest quality question that will produce the most valuable reasoning trace.
Return the selected question as JSON with your assigned quality score and detailed evaluation.
"""
        )
    
    def _create_solution_proposer(self) -> CodeAgent:
        """Create solution proposer agent"""
        return CodeAgent(
            model=self.model,
            tools=[format_question_for_proposer],
            max_steps=5,
            verbosity_level=1,
            name="solution_proposer",
            description="""Create multiple plausible answer choices for complex questions.

Your task is to generate diverse solution approaches that:
- Include both correct and incorrect but reasonable options
- Show different reasoning paths
- Test different aspects of understanding
- Are all plausible to someone without deep expertise
- Vary in sophistication and approach

Create choices that would genuinely challenge a reasoning model, not obvious distractors.

Generate exactly 4 answer choices (A, B, C, D) as JSON with:
- id: Choice letter (A, B, C, D)
- text: The answer choice text
- reasoning: Brief explanation of the reasoning behind this choice
- confidence: Estimated confidence level (0.0-1.0)

One choice should be clearly correct based on the source text, while others should be plausible but incorrect.
"""
        )
    
    def _create_solution_critic(self) -> CodeAgent:
        """Create solution critic agent"""
        return CodeAgent(
            model=self.model,
            tools=[format_solutions_for_critic],
            max_steps=7,
            verbosity_level=1,
            name="solution_critic",
            description="""Evaluate proposed solutions and expand on their logic.

Your task is to critically analyze each proposed solution for:
- Accuracy relative to source material
- Quality of reasoning and logic
- Completeness of explanation
- Appropriateness as distractor or correct answer
- Potential for generating rich reasoning traces

For each solution:
- Provide detailed critique of strengths and weaknesses
- Identify logical gaps or errors
- Suggest improvements to reasoning
- Assess educational value

Select the best set of solutions that will create the most valuable reasoning experience.
If solutions need refinement, provide specific guidance to the proposer.
Return approved solutions as JSON array with your evaluation notes.
"""
        )
    
    def _create_final_solution_writer(self) -> CodeAgent:
        """Create final solution writer agent"""
        return CodeAgent(
            model=self.model,
            tools=[format_for_reasoner],
            max_steps=10,
            verbosity_level=1,
            name="final_solution_writer",
            description="""Create detailed, human-like reasoning traces.

Your task is to synthesize questions and choices into coherent reasoning traces that:
- Show natural thinking progression with self-correction
- Include phrases like "Wait, let me reconsider...", "Actually, thinking about this more...", "Hmm, the user wants me to..."
- Demonstrate bias checking and verification steps
- Build comprehensive justifications
- Express uncertainty and confidence appropriately
- Aim for 2000-4000 words

Create reasoning that feels like the internal monologue of a careful, thoughtful problem-solver.

Return the complete reasoning trace as a JSON object with:
- question: The original question object
- choices: Array of choice objects
- selected_choice: The letter of the chosen answer
- reasoning_chain: The full reasoning text (2000-4000 words)
- token_count: Word count of reasoning
"""
        )
    
    def _create_orchestrator(self) -> CodeAgent:
        """Create main orchestrator agent"""
        return CodeAgent(
            model=self.model,
            tools=[save_reasoning_trace],
            max_steps=25,
            verbosity_level=1,
            managed_agents=[
                self.question_writer,
                self.question_critic,
                self.solution_proposer,
                self.solution_critic,
                self.final_solution_writer
            ],
            name="orchestrator",
            description="""Main orchestrator that coordinates the 5-step multi-agent reasoning process.

Your responsibilities - follow this exact workflow:

1. QUESTION GENERATION: Ask question_writer to generate multiple diverse questions from source text
2. QUESTION CRITICISM: Ask question_critic to evaluate and select the best question
3. SOLUTION PROPOSAL: Ask solution_proposer to create multiple plausible answer choices
4. SOLUTION CRITICISM: Ask solution_critic to evaluate proposed solutions and refine them
5. REASONING CHAIN: Ask final_solution_writer to create detailed reasoning trace

Detailed Process:
1. question_writer generates 5 diverse questions from the source text
2. question_critic evaluates all questions and selects the highest quality one
3. solution_proposer creates 4 answer choices (A,B,C,D) for the selected question
4. solution_critic evaluates the proposed solutions, potentially asking for refinements
5. final_solution_writer creates a detailed 2000-4000 word reasoning trace
6. Save the complete result using save_reasoning_trace tool

Ensure each step is completed before moving to the next. The final output must be a complete reasoning trace with question, choices, and detailed reasoning chain.
"""
        )
    
    def generate_reasoning_trace(self, seed_text: str, num_questions: int = 5) -> ReasoningTrace:
        """Generate a complete reasoning trace from seed text"""
        
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents not available")
        
        # Create the orchestration task
        task = f"""
        Generate a high-quality reasoning trace using the 5-step multi-agent process:
        
        SOURCE TEXT:
        {seed_text}
        
        FOLLOW THIS EXACT 5-STEP PROCESS:
        1. QUESTION GENERATION: Ask question_writer to generate {num_questions} diverse, challenging questions
        2. QUESTION CRITICISM: Ask question_critic to evaluate all questions and select the best one
        3. SOLUTION PROPOSAL: Ask solution_proposer to create 4 plausible answer choices (A,B,C,D)
        4. SOLUTION CRITICISM: Ask solution_critic to evaluate proposed solutions and approve/refine them
        5. REASONING CHAIN: Ask final_solution_writer to create detailed 2000-4000 word reasoning trace
        
        Each step must be completed before proceeding to the next. The final output should be a complete 
        reasoning trace with question, choices, and detailed reasoning suitable for training reasoning models.
        """
        
        try:
            # Run the orchestrator
            response = self.orchestrator.run(task=task, reset=True)
            
            # Parse the response to extract reasoning trace
            return self._parse_response_to_trace(response, seed_text)
            
        except Exception as e:
            print(f"Error generating reasoning trace: {e}")
            return self._create_fallback_trace(seed_text)
    
    def _parse_response_to_trace(self, response: str, seed_text: str) -> ReasoningTrace:
        """Parse orchestrator response to ReasoningTrace object"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Extract components
                question_data = data.get('question', {})
                choices_data = data.get('choices', [])
                reasoning_chain = data.get('reasoning_chain', '')
                
                question = Question(
                    stem=question_data.get('stem', 'Generated question'),
                    topic=question_data.get('topic', 'General'),
                    difficulty=question_data.get('difficulty', 'Medium'),
                    quality_score=question_data.get('quality_score', 5.0)
                )
                
                choices = [Choice(
                    id=choice.get('id', 'A'),
                    text=choice.get('text', 'Generated choice'),
                    reasoning=choice.get('reasoning', 'Generated reasoning'),
                    confidence=choice.get('confidence', 0.5)
                ) for choice in choices_data]
                
                return ReasoningTrace(
                    question=question,
                    choices=choices,
                    selected_choice=choices[0].id if choices else 'A',
                    reasoning_chain=reasoning_chain,
                    token_count=len(reasoning_chain.split()),
                    metadata={
                        'model_used': self.model_name,
                        'interface_used': self.interface,
                        'generation_timestamp': datetime.now().isoformat(),
                        'source_text_length': len(seed_text)
                    }
                )
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return self._create_fallback_trace(seed_text)
    
    def _create_fallback_trace(self, seed_text: str) -> ReasoningTrace:
        """Create fallback reasoning trace when generation fails"""
        return ReasoningTrace(
            question=Question("What is the main concept discussed in this text?", "General", "Medium", 5.0),
            choices=[
                Choice("A", "First concept", "This addresses the main idea", 0.8),
                Choice("B", "Second concept", "This is an alternative view", 0.6),
                Choice("C", "Third concept", "This is a related idea", 0.4),
                Choice("D", "Fourth concept", "This is less likely", 0.2)
            ],
            selected_choice="A",
            reasoning_chain="This is a fallback reasoning chain generated when the main system encountered an error.",
            token_count=50,
            metadata={
                'fallback': True,
                'model_used': self.model_name,
                'interface_used': self.interface,
                'generation_timestamp': datetime.now().isoformat(),
                'source_text_length': len(seed_text)
            }
        )
    
    def process_texts(self, texts: List[str], output_path: Optional[str] = None) -> List[ReasoningTrace]:
        """Process multiple texts and generate reasoning traces"""
        results = []
        
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}/{len(texts)}...")
            print(f"Text length: {len(text)} characters")
            
            try:
                trace = self.generate_reasoning_trace(text)
                results.append(trace)
                
                print(f"✅ Generated reasoning trace ({trace.token_count} tokens)")
                print(f"Question: {trace.question.stem[:100]}...")
                
            except Exception as e:
                print(f"❌ Error processing text {i+1}: {e}")
                continue
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def process_jsonl_file(self, file_path: str, output_path: str, max_samples: Optional[int] = None) -> List[ReasoningTrace]:
        """Process JSONL file and generate reasoning traces"""
        file_path = Path(file_path)
        
        print(f"Processing JSONL file: {file_path}")
        
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
        
        print(f"Loaded {len(texts)} text samples")
        
        return self.process_texts(texts, output_path)
    
    def _save_results(self, results: List[ReasoningTrace], output_path: str):
        """Save reasoning traces to file"""
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_traces_generated": len(results),
                "model_used": self.model_name,
                "interface_used": self.interface,
                "average_token_count": sum(r.token_count for r in results) / len(results) if results else 0
            },
            "reasoning_traces": []
        }
        
        for result in results:
            output_data["reasoning_traces"].append({
                "question": {
                    "stem": result.question.stem,
                    "topic": result.question.topic,
                    "difficulty": result.question.difficulty,
                    "quality_score": result.question.quality_score
                },
                "choices": [
                    {
                        "id": choice.id,
                        "text": choice.text,
                        "reasoning": choice.reasoning,
                        "confidence": choice.confidence
                    }
                    for choice in result.choices
                ],
                "selected_choice": result.selected_choice,
                "reasoning_chain": result.reasoning_chain,
                "token_count": result.token_count,
                "metadata": result.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_path}")
        print(f"Generated {len(results)} reasoning traces")
        print(f"Average token count: {output_data['metadata']['average_token_count']:.1f}")


def main():
    """Example usage of the multi-agent system"""
    
    # Sample scientific text
    sample_text = """
    Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the 
    quantum state of each particle cannot be described independently. This correlation persists even when the 
    particles are separated by large distances, leading Einstein to famously describe it as "spooky action at a 
    distance." The phenomenon has been experimentally verified and forms the basis for quantum computing and 
    quantum cryptography. Bell's theorem shows that no physical theory based on local hidden variables can 
    reproduce all the predictions of quantum mechanics, thus ruling out local realism as a viable explanation 
    for quantum phenomena.
    """
    
    if not SMOLAGENTS_AVAILABLE:
        print("smolagents not available. Install with: pip install smolagents")
        return
    
    try:
        # Initialize the multi-agent system
        reasoner = SmolSyntheticReasoner(model_name="gemini-2.5-flash")
        
        # Generate reasoning trace
        trace = reasoner.generate_reasoning_trace(sample_text)
        
        # Print results
        print("\n=== Generated Reasoning Trace ===")
        print(f"Question: {trace.question.stem}")
        print(f"Topic: {trace.question.topic}")
        print(f"Difficulty: {trace.question.difficulty}")
        print(f"Quality Score: {trace.question.quality_score}")
        print(f"Token Count: {trace.token_count}")
        print(f"Reasoning Chain: {trace.reasoning_chain[:500]}...")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("This might be due to missing smolagents installation or API access.")
        print("Install with: pip install smolagents")


if __name__ == "__main__":
    main() 