#!/usr/bin/env python3
"""
Advanced Synthetic Reasoning Trace Generator v2

Generates realistic reasoning traces from scientific text using LLMs via Argo server.

Usage examples:
    # Use default settings (5 samples for testing)
    python llm_synthetic_reasoner_v2.py --max-traces 5
    
    # Specify different model and input file
    python llm_synthetic_reasoner_v2.py --model gpt4o --input-file my_data.jsonl --max-traces 3
    
    # Enable quality checking for higher quality traces (slower)
    python llm_synthetic_reasoner_v2.py --quality-check --min-quality-score 6.5 --max-traces 5
    
    # Process all samples with different output file
    python llm_synthetic_reasoner_v2.py --output my_traces.json
    
    # Start fresh (ignore existing output file)
    python llm_synthetic_reasoner_v2.py --no-resume --max-traces 5
    
    # Resume from previous run (default behavior)
    python llm_synthetic_reasoner_v2.py --max-traces 10
"""

import asyncio
import random
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from .llm_interface import UnifiedLLMClient
    from .llm_judge_evaluator import LLMJudgeEvaluator, QRSTriplet
except ImportError:
    from llm_interface import UnifiedLLMClient
    from llm_judge_evaluator import LLMJudgeEvaluator, QRSTriplet

class LLMInterface(ABC):
    """Abstract interface for LLM clients"""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        pass

class ArgoLLMAdapter(LLMInterface):
    """Adapter to use UnifiedLLMClient with Argo server as LLMInterface"""
    
    def __init__(self, model_name: str = "claudesonnet4", interface: str = "argo", **kwargs):
        """Initialize with Argo server as default"""
        self.client = UnifiedLLMClient(model_name=model_name, interface=interface, **kwargs)
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using Argo server"""
        messages = [{"role": "user", "content": prompt}]
        return self.client.generate(messages, temperature=temperature)

@dataclass
class ScientificPassage:
    content: str
    section_type: str  # 'methods', 'results', 'discussion', 'mixed', etc.
    source_title: str  # Original paper title if available
    domain: str
    passage_id: str = None  # Unique identifier for this passage

@dataclass
class UserQuestion:
    question: str
    question_type: str  # 'clarification', 'critical', 'interpretation', etc.
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced'
    target_audience: str  # 'student', 'researcher', 'practitioner'

@dataclass
class QuestionContext:
    relevant_sections: List[str]
    reasoning_requirements: List[str]
    potential_misconceptions: List[str]
    background_needed: List[str]

@dataclass
class ReasoningFragment:
    content: str
    type: str
    confidence: float
    uncertainty_markers: List[str]

class ReasoningTraceGenerator:
    def __init__(self, llm: LLMInterface, grade_answers: bool = False):
        self.llm = llm
        self.grade_answers = grade_answers
        self.question_types = [
            'clarification', 'critical', 'interpretation', 
            'comparison', 'application', 'mechanism', 'limitation'
        ]
        self.authenticity_markers = [
            "hmm", "wait", "actually", "let me think", "I wonder",
            "but", "maybe", "oh", "interesting", "that's weird",
            "I'm not sure", "hang on", "let me reconsider"
        ]
        
    async def generate_reasoning_trace(self, passage: ScientificPassage) -> Dict[str, Any]:
        """Generate a complete question-response reasoning trace from a scientific paper."""
        
        # Stage 0: Generate realistic user questions
        user_questions = await self._generate_user_questions(paper)
        selected_question = self._select_question_for_reasoning(user_questions, passage)
        
        # Stage 1: Analyze question context and requirements
        question_context = await self._analyze_question_context(selected_question, passage)
        
        # Stage 2: Generate question-specific reasoning fragments
        fragments = await asyncio.gather(
            self._generate_targeted_hypotheses(selected_question, passage),
            self._evaluate_evidence_for_question(selected_question, passage),
            self._evaluate_methods_for_question(selected_question, passage),
            self._consider_user_perspective(selected_question, passage)
        )
        
        # Stage 3: Assemble into question-responsive trace
        raw_trace = await self._assemble_question_response_chain(
            selected_question, question_context, fragments, paper
        )
        
        # Stage 4: Enhance authenticity and inject realistic errors
        enhanced_trace = await self._enhance_authenticity(raw_trace)
        final_trace = await self._inject_realistic_reasoning_patterns(enhanced_trace)
        
        # Extract final answer from reasoning trace
        final_answer = await self._extract_final_answer(final_trace, selected_question)
        
        # Grade the final answer for RL training (if enabled)
        answer_grades = None
        if self.grade_answers:
            print(f"  Grading final answer...")
            answer_grades = await self._grade_final_answer(
                selected_question, 
                final_answer, 
                passage.content + " " + passage.content, 
                final_trace
            )
        
        return await self.generate_reasoning_trace_for_question(paper, selected_question)
    
    async def generate_reasoning_trace_for_question(
        self, 
        passage: ScientificPassage, 
        question: UserQuestion
    ) -> Dict[str, Any]:
        """Generate a reasoning trace for a specific question using natural randomness."""
        
        # Stage 1: Analyze question context and requirements
        question_context = await self._analyze_question_context(question, passage)
        
        # Stage 2: Generate question-specific reasoning fragments (natural variation at ~0.7 temp)
        fragments = await asyncio.gather(
            self._generate_targeted_hypotheses(question, passage),
            self._evaluate_evidence_for_question(question, passage),
            self._evaluate_methods_for_question(question, passage),
            self._consider_user_perspective(question, passage)
        )
        
        # Stage 3: Assemble into question-responsive trace
        raw_trace = await self._assemble_question_response_chain(
            question, question_context, fragments, passage
        )
        
        # Stage 4: Enhance authenticity and inject realistic errors
        enhanced_trace = await self._enhance_authenticity(raw_trace)
        final_trace = await self._inject_realistic_reasoning_patterns(enhanced_trace)
        
        # Extract final answer from reasoning trace
        final_answer = await self._extract_final_answer(final_trace, question)
        
        # Grade the final answer for RL training (if enabled)
        answer_grades = None
        if self.grade_answers:
            print(f"  Grading final answer...")
            answer_grades = await self._grade_final_answer(
                question, 
                final_answer, 
                passage.content + " " + passage.content, 
                final_trace
            )
        
        return {
            'source_paper': {
                'title': passage.source_title,
                'domain': passage.domain
            },
            'user_interaction': {
                'question': question.question,
                'question_type': question.question_type,
                'difficulty_level': question.difficulty_level
            },
            'reasoning_trace': final_trace,
            'final_answer': final_answer,
            'answer_grades': answer_grades,
            'metadata': {
                'authenticity_score': self._calculate_authenticity_score(final_trace),
                'complexity_level': self._assess_complexity(final_trace),
                'reasoning_patterns': self._identify_patterns(final_trace),
                'question_responsiveness': self._assess_question_responsiveness(final_trace, question)
            }
        }
    
    async def _generate_user_questions(self, passage: ScientificPassage) -> List[UserQuestion]:
        prompt = f"""
Simulate realistic questions that different types of users would ask about this scientific paper:

Paper Title: {passage.source_title}
Abstract: {passage.content}
Methods: {passage.content}
Key Results: {passage.content}

Generate 6-8 questions across these categories:
CLARIFICATION (beginner): Basic understanding questions about concepts or methods
CRITICAL (intermediate): Questions challenging methodology, conclusions, or assumptions  
INTERPRETATION (intermediate): Questions about meaning, implications, or significance
COMPARISON (advanced): Questions relating this to other research or broader context
APPLICATION (intermediate): Questions about practical applications or extensions
MECHANISM (advanced): Questions about underlying processes or how things work
LIMITATION (advanced): Questions about weaknesses, constraints, or scope

IMPORTANT: Make each question SELF-CONTAINED by:
- Including sufficient context within the question itself
- Defining key terms or concepts mentioned
- Not relying on phrases like "this paper", "the authors", "these results"
- Providing enough background so the question makes sense independently

Make each question:
- Natural and conversational (how real people actually ask)
- Specific to this paper's actual content but self-explanatory
- Appropriate for the indicated user level
- Something that would genuinely help understanding
- Complete with necessary context embedded in the question

Format: [TYPE] (Level): "Question text"
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_user_questions(response)
    
    async def _analyze_question_context(self, question: UserQuestion, passage: ScientificPassage) -> QuestionContext:
        prompt = f"""
Analyze what this user question requires in terms of reasoning about the paper:

User Question: {question.question}
Question Type: {question.question_type}
User Level: {question.difficulty_level}

Paper Context:
Title: {passage.source_title}
Abstract: {passage.content}
Methods: {passage.content}

Identify:
1. Which specific parts of the paper are most relevant to this question
2. What types of reasoning are needed (methodological, statistical, conceptual, etc.)
3. What potential misconceptions or confusions the user might have
4. What background knowledge might be missing for this user level
5. What level of technical detail is appropriate
6. What aspects require careful consideration vs. straightforward explanation

Provide specific, actionable analysis for generating a thoughtful response.
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return self._parse_question_context(response)
    
    async def _generate_targeted_hypotheses(self, question: UserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
Think through how to approach this specific user question. Show your reasoning process:

User Question: {question.question}
Paper Context: {passage.content}
Relevant Methods: {passage.content}

Consider:
- What is the user really asking for?
- Multiple ways to interpret or answer their question
- What from the paper directly addresses their concern
- What might be missing or unclear
- How to frame the answer appropriately for their level

Think step by step: "The user seems to be asking about... My first thought is... but they might actually mean... Let me consider what would be most helpful..."

Show your working through of how to best respond to their specific question.
"""
        
        response = await self.llm.generate(prompt, temperature=0.9)
        return ReasoningFragment(
            content=response,
            type='targeted_hypothesis',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _evaluate_evidence_for_question(self, question: UserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
Evaluate the paper's evidence specifically from the perspective of this user's question:

User Question: {question.question}
Paper Results: {passage.content}
Discussion: {passage.content}

Think through:
- Does the evidence actually address what they're asking about?
- How strong is the evidence for answering their specific question?
- What limitations or caveats are relevant to their inquiry?
- Are there alternative interpretations they should consider?
- What would help them understand the strength of the evidence?

Show your reasoning as you work through evaluating the evidence through the lens of their question.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='question_focused_evidence',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _evaluate_methods_for_question(self, question: UserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
Consider the methodology from the perspective of this user's specific question:

User Question: {question.question}
Methods: {passage.content}

Think about:
- Are the methods appropriate for what the user is asking about?
- What methodological details matter for their specific question?
- How do methodological choices affect the answer to their question?
- What would the user need to understand about the methods?
- Are there methodological limitations that affect the response?

Express your reasoning as you evaluate how the methodology relates to their specific concern.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='question_focused_methods',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _consider_user_perspective(self, question: UserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
Think about this question from the user's perspective and what would be most helpful:

User Question: {question.question}
User Level: {question.difficulty_level}
Question Type: {question.question_type}

Consider:
- What might have prompted this question?
- What would be most useful for them to understand?
- What common misconceptions should be addressed?
- How technical should the response be?
- What connections to broader concepts might help?
- What follow-up questions might arise?

Show your thinking about how to craft a response that truly serves this user's needs.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='user_perspective',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _assemble_question_response_chain(
        self, 
        question: UserQuestion,
        context: QuestionContext,
        fragments: List[ReasoningFragment], 
        passage: ScientificPassage
    ) -> str:
        
        prompt = f"""
Create a natural reasoning trace that works through how to respond to this user's question. Show the thinking process of someone carefully considering how to give a thoughtful, accurate answer.

User Question: {question.question}
Question Analysis: {context.reasoning_requirements}
User Level: {question.difficulty_level}

Reasoning to incorporate:
1. Approach considerations: {fragments[0].content}
2. Evidence evaluation: {fragments[1].content}
3. Methods considerations: {fragments[2].content}
4. User perspective: {fragments[3].content}

Create a flowing <thought>...</thought> section that:
- Starts by really understanding what the user is asking
- Works through the relevant aspects of the paper
- Considers how to frame the response appropriately
- Shows reasoning about evidence, methods, and interpretation
- Addresses potential user confusion or misconceptions
- Builds toward a clear, helpful response
- Acknowledges limitations or uncertainties where appropriate

Make it feel like someone genuinely thinking through how to give this user the most helpful and accurate response to their specific question.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    def _select_question_for_reasoning(self, questions: List[UserQuestion], passage: ScientificPassage) -> UserQuestion:
        """Select the most interesting question for generating reasoning trace."""
        print(f"  Generated {len(questions)} questions")
        
        if not questions:
            # Fallback: create a default question if none were generated
            print("  No questions generated, creating fallback question")
            return UserQuestion(
                question="What are the main findings and implications of this research?",
                question_type="interpretation",
                difficulty_level="intermediate",
                target_audience="general"
            )
        
        # Prioritize intermediate/advanced questions as they typically require more reasoning
        advanced_questions = [q for q in questions if q.difficulty_level in ['intermediate', 'advanced']]
        if advanced_questions:
            selected = random.choice(advanced_questions)
            print(f"  Selected advanced question: {selected.question[:50]}...")
            return selected
        
        selected = random.choice(questions)
        print(f"  Selected question: {selected.question[:50]}...")
        return selected
    
    def _parse_user_questions(self, response: str) -> List[UserQuestion]:
        """Parse LLM response into UserQuestion objects."""
        print(f"  Parsing LLM response (length: {len(response)})")
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line and any(qtype in line.upper() for qtype in ['CLARIFICATION', 'CRITICAL', 'INTERPRETATION', 'COMPARISON', 'APPLICATION', 'MECHANISM', 'LIMITATION']):
                try:
                    # Parse format: [TYPE] (Level): "Question text"
                    parts = line.split(':', 1)
                    type_level = parts[0].strip()
                    question_text = parts[1].strip().strip('"')
                    
                    question_type = type_level.split('(')[0].strip('[]').lower()
                    level = type_level.split('(')[1].split(')')[0].lower()
                    
                    questions.append(UserQuestion(
                        question=question_text,
                        question_type=question_type,
                        difficulty_level=level,
                        target_audience='general'
                    ))
                    print(f"  Parsed question: {question_text[:50]}...")
                except Exception as e:
                    print(f"  Failed to parse line: {line[:50]}... Error: {e}")
                    continue  # Skip malformed lines
        
        print(f"  Successfully parsed {len(questions)} questions")
        return questions[:8]  # Limit to 8 questions
    
    def _parse_question_context(self, response: str) -> QuestionContext:
        """Parse question context analysis."""
        # Simple parsing - in practice you'd want more robust parsing
        return QuestionContext(
            relevant_sections=['methods', 'results'],  # Default
            reasoning_requirements=['evidence evaluation'],
            potential_misconceptions=[],
            background_needed=[]
        )
    
    
    def _assess_question_responsiveness(self, trace: str, question: UserQuestion) -> float:
        """Assess how well the trace responds to the specific question."""
        question_words = set(question.question.lower().split())
        trace_words = set(trace.lower().split())
        
        # Simple overlap metric - in practice you'd want semantic similarity
        overlap = len(question_words.intersection(trace_words))
        return min(1.0, overlap / max(len(question_words), 1))
    
    async def _evaluate_evidence(self, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
You're critically examining the evidence presented in this study. Think through it step by step, showing your internal reasoning process:

Results: {passage.content}
Discussion: {passage.content}

Show your thinking as you:
- Examine what the data actually shows vs. what's claimed
- Consider alternative explanations
- Notice potential confounds or limitations
- Question the strength of conclusions

Include moments of:
- Initial acceptance followed by growing skepticism
- "Wait, that doesn't make sense..." realizations
- Uncertainty about interpretation
- Recognition of both strengths and weaknesses

Think aloud naturally, with pauses, corrections, and genuine uncertainty.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='evidence_eval',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _critique_methodology(self, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
You're a researcher evaluating this study's methodology. Show your internal dialogue as you work through potential issues:

Methods: {passage.content}

Think through:
- Are the controls adequate?
- Could confounding factors explain results?
- Are sample sizes appropriate?
- Are measurements valid and reliable?
- What would you do differently?

Show realistic reasoning patterns:
- Initial impressions
- Deeper analysis revealing problems
- Moments of "oh wait, I missed something"
- Balancing criticism with recognition of practical constraints
- Genuine uncertainty about some judgments

Express this as natural internal thought, not a formal critique.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='method_critique',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _explore_connections(self, passage: ScientificPassage) -> ReasoningFragment:
        prompt = f"""
You're thinking about how this research connects to other work you know. Show your thought process as you make connections:

Paper: {passage.source_title}
Key findings: {passage.content[:300]}...

Think about:
- Similar studies you've read
- Contradictory findings in the literature
- Broader theoretical implications
- Potential applications or extensions

Show natural associative thinking:
- "This reminds me of..."
- "But wait, didn't Smith et al. find the opposite?"
- "I wonder if this applies to..."
- "Actually, I'm not sure if these are really comparable..."

Include the messiness of real thinking - false starts, uncertainty, partial connections.
"""
        
        response = await self.llm.generate(prompt, temperature=0.9)
        return ReasoningFragment(
            content=response,
            type='connections',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_uncertainty_markers(response)
        )
    
    async def _assemble_reasoning_chain(
        self, 
        questions: List[str], 
        complexity_points: List[str],
        fragments: List[ReasoningFragment], 
        passage: ScientificPassage
    ) -> str:
        
        # Select the most interesting question and complexity point
        main_question = max(questions, key=len)  # Heuristic: longer questions often more interesting
        main_complexity = random.choice(complexity_points)
        
        prompt = f"""
Weave these reasoning fragments into a natural, flowing internal thought process. The goal is to create an authentic stream of consciousness that feels like someone genuinely working through scientific ideas.

Starting question/focus: {main_question}
Key complexity to address: {main_complexity}

Reasoning fragments to incorporate:
1. Hypotheses: {fragments[0].content}
2. Evidence evaluation: {fragments[1].content}
3. Method critique: {fragments[2].content}
4. Connections: {fragments[3].content}

Create a flowing <thought>...</thought> section that:
- Starts with the question/complexity
- Naturally incorporates insights from each fragment
- Shows realistic thinking patterns (building, questioning, revising)
- Includes natural transitions and connections
- Has moments of clarity, and uncertainty
- Feels like authentic internal dialogue

Don't just concatenate the fragments - weave them into a coherent thought process that flows naturally from one idea to the next.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    async def _extract_final_answer(self, reasoning_trace: str, question: UserQuestion) -> str:
        """Extract a clear, concise final answer from the reasoning trace"""
        
        prompt = f"""
Based on the detailed reasoning trace below, provide a clear, concise final answer to the user's question.

USER QUESTION: {question.question}

REASONING TRACE: {reasoning_trace}

TASK: Extract and synthesize the key points from the reasoning into a clear, direct answer that:
- Directly addresses the user's question
- Summarizes the main conclusions from the reasoning
- Is self-contained and doesn't require reading the reasoning trace
- Maintains appropriate uncertainty where present in the reasoning
- Is written for the user's level ({question.difficulty_level})

Provide just the final answer - no preamble like "Based on the reasoning above" or "The answer is":
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return response.strip()
    
    async def _grade_final_answer(self, question: UserQuestion, final_answer: str, source_text: str, reasoning_trace: str) -> Dict[str, float]:
        """Grade the final answer across multiple dimensions for RL training"""
        
        # Grade each dimension
        grades = await asyncio.gather(
            self._grade_question_alignment(question, final_answer),
            self._grade_scientific_accuracy(final_answer, source_text),
            self._grade_completeness(question, final_answer),
            self._grade_uncertainty_calibration(final_answer, reasoning_trace),
            self._grade_clarity_structure(final_answer),
            self._grade_evidence_usage(final_answer, source_text)
        )
        
        return {
            'question_alignment': grades[0],
            'scientific_accuracy': grades[1], 
            'completeness': grades[2],
            'uncertainty_calibration': grades[3],
            'clarity_structure': grades[4],
            'evidence_usage': grades[5],
            'overall_score': sum(grades) / len(grades)
        }
    
    async def _grade_question_alignment(self, question: UserQuestion, final_answer: str) -> float:
        """Grade how well the answer addresses the specific question asked"""
        
        prompt = f"""
Rate how well this answer addresses the specific question asked (scale 0.0-1.0):

QUESTION: {question.question}
QUESTION TYPE: {question.question_type}
DIFFICULTY LEVEL: {question.difficulty_level}

ANSWER: {final_answer}

Evaluation criteria:
- Does the answer directly address what was asked?
- Are all parts of multi-part questions covered?
- Is the response appropriate for the question type and difficulty level?
- Does it stay focused on the question rather than going off-topic?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5  # Default if parsing fails
    
    async def _grade_scientific_accuracy(self, final_answer: str, source_text: str) -> float:
        """Grade factual accuracy and scientific correctness"""
        
        prompt = f"""
Rate the scientific accuracy of this answer based on the source material (scale 0.0-1.0):

SOURCE TEXT: {source_text[:1000]}...

ANSWER: {final_answer}

Evaluation criteria:
- Are scientific facts stated correctly?
- Are there any clear factual errors?
- Does the answer contradict established scientific knowledge?
- Are technical terms used appropriately?
- Is the answer consistent with the source material?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _grade_completeness(self, question: UserQuestion, final_answer: str) -> float:
        """Grade whether the answer thoroughly addresses the question"""
        
        prompt = f"""
Rate how complete and thorough this answer is (scale 0.0-1.0):

QUESTION: {question.question}

ANSWER: {final_answer}

Evaluation criteria:
- Does the answer cover all aspects of the question?
- Are important points missing?
- Is the depth appropriate for the question complexity?
- For multi-part questions, are all parts addressed?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _grade_uncertainty_calibration(self, final_answer: str, reasoning_trace: str) -> float:
        """Grade appropriate expression of confidence and uncertainty"""
        
        prompt = f"""
Rate how well this answer expresses appropriate uncertainty and confidence (scale 0.0-1.0):

REASONING TRACE (for context): {reasoning_trace[:500]}...

FINAL ANSWER: {final_answer}

Evaluation criteria:
- Does the answer appropriately express uncertainty about uncertain claims?
- Does it avoid overconfident statements about speculative topics?
- Does it appropriately express confidence about well-established facts?
- Is the level of certainty consistent with the reasoning shown?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _grade_clarity_structure(self, final_answer: str) -> float:
        """Grade clarity, organization, and readability"""
        
        prompt = f"""
Rate the clarity and structure of this answer (scale 0.0-1.0):

ANSWER: {final_answer}

Evaluation criteria:
- Is the answer well-organized and logical?
- Is the language clear and appropriate?
- Are ideas presented in a logical order?
- Is it easy to follow and understand?
- Are key points emphasized effectively?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _grade_evidence_usage(self, final_answer: str, source_text: str) -> float:
        """Grade how well the answer uses evidence from source material"""
        
        prompt = f"""
Rate how effectively this answer uses evidence from the source material (scale 0.0-1.0):

SOURCE TEXT: {source_text[:1000]}...

ANSWER: {final_answer}

Evaluation criteria:
- Does the answer draw appropriately from the source material?
- Are claims supported by evidence where appropriate?
- Does it avoid making unsupported claims?
- Is the relationship between evidence and conclusions clear?

Provide only a numerical score (0.0-1.0):
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _enhance_authenticity(self, trace: str) -> str:
        prompt = f"""
Make this reasoning trace more authentic and natural. Add realistic thinking patterns:

Current trace: {trace}

Enhance by adding:
- Natural speech patterns and filler words ("hmm", "let me see", "actually")
- Realistic corrections ("wait, that's not right", "let me reconsider")
- Genuine uncertainty expressions ("I'm not sure", "maybe", "possibly")
- Meta-cognitive awareness ("I'm getting confused", "let me step back")
- Natural pauses and transitions
- Building understanding gradually
- Catching and correcting mistakes

Keep the scientific content accurate but make the thinking process feel genuinely human and authentic.
"""
        
        return await self.llm.generate(prompt, temperature=0.6)
    
    async def _inject_realistic_reasoning_patterns(self, trace: str) -> str:
        prompt = f"""
Add realistic reasoning errors and corrections to make this trace more authentic. Humans don't think perfectly - they make mistakes and correct them.

Current trace: {trace}

Add patterns like:
- Initial wrong assumptions that get corrected ("Oh wait, I misread that")
- Overconfidence that gets tempered ("Actually, I'm less sure about this")
- Logical missteps that get caught ("That doesn't follow")
- Misinterpretations that get clarified ("Let me read this again")
- False starts ("No, that's not the right approach")

Keep the final reasoning sound, but show the messy, error-prone process of getting there. This makes it more realistic and educational.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    def _extract_questions(self, response: str) -> List[str]:
        # Extract questions from LLM response
        import re
        questions = re.findall(r'"([^"]*\?)"', response)
        return questions[:6]  # Limit to 6 questions
    
    def _extract_complexity_points(self, response: str) -> List[str]:
        # Extract complexity points from response
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return [line for line in lines if any(marker in line.lower() 
                for marker in ['complex', 'difficult', 'problem', 'issue', 'challenge'])]
    
    def _estimate_confidence(self, text: str) -> float:
        # Estimate confidence based on language patterns
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'not sure', 'unclear', 'might']
        certainty_words = ['definitely', 'clearly', 'obviously', 'certain', 'sure']
        
        uncertainty_count = sum(text.lower().count(word) for word in uncertainty_words)
        certainty_count = sum(text.lower().count(word) for word in certainty_words)
        
        return max(0.1, min(0.9, 0.5 + 0.1 * (certainty_count - uncertainty_count)))
    
    def _find_uncertainty_markers(self, text: str) -> List[str]:
        markers = []
        for marker in self.authenticity_markers:
            if marker in text.lower():
                markers.append(marker)
        return markers
    
    def _calculate_authenticity_score(self, trace: str) -> float:
        # Calculate authenticity based on presence of natural thinking patterns
        patterns = [
            'hmm', 'wait', 'actually', 'let me', 'I wonder', 'maybe', 'but',
            'oh', 'interesting', "I'm not sure", 'reconsider', 'step back'
        ]
        
        score = sum(1 for pattern in patterns if pattern in trace.lower())
        return min(1.0, score / len(patterns))
    
    def _assess_complexity(self, trace: str) -> str:
        word_count = len(trace.split())
        if word_count < 200:
            return 'low'
        elif word_count < 500:
            return 'medium'
        else:
            return 'high'
    
    def _identify_patterns(self, trace: str) -> List[str]:
        patterns = []
        
        if 'hypothesis' in trace.lower():
            patterns.append('hypothesis_generation')
        if any(word in trace.lower() for word in ['evidence', 'data', 'result']):
            patterns.append('evidence_evaluation')
        if any(word in trace.lower() for word in ['method', 'approach', 'design']):
            patterns.append('methodology_critique')
        if any(word in trace.lower() for word in ['reminds', 'similar', 'connect']):
            patterns.append('connection_making')
        if any(word in trace.lower() for word in ['wait', 'actually', 'reconsider']):
            patterns.append('self_correction')
            
        return patterns

# Example usage and batch processing
class BatchProcessor:
    def __init__(self, generator: ReasoningTraceGenerator, quality_check: bool = False, min_quality_score: float = 6.0):
        self.generator = generator
        self.quality_check = quality_check
        self.min_quality_score = min_quality_score
        self.evaluator = LLMJudgeEvaluator() if quality_check else None
    
    async def _process_paper_with_question_variations(
        self,
        passage: ScientificPassage, 
        traces_per_question: int,
        traces_per_paper: int
    ) -> List[Dict[str, Any]]:
        """Generate one question for a paper, then create multiple reasoning traces for that same question."""
        
        print(f"  Generating {traces_per_question} traces for same question")
        
        # Step 1: Generate a single question for this paper
        max_attempts = 3
        selected_question = None
        
        for attempt in range(max_attempts):
            try:
                # Generate questions for the paper (reuse existing logic)
                questions = await self.generator._generate_user_questions(passage)
                if questions:
                    selected_question = self.generator._select_question_for_reasoning(questions, passage)
                    break
            except Exception as e:
                print(f"    ⚠️ Question generation attempt {attempt + 1} failed: {e}")
                continue
        
        if not selected_question:
            print(f"    ❌ Failed to generate question after {max_attempts} attempts")
            return []
        
        print(f"    📝 Selected question: {selected_question.question[:80]}...")
        
        # Step 2: Generate multiple reasoning traces for this same question using natural randomness
        accepted_traces = []
        
        for trace_idx in range(traces_per_question):
            print(f"    🎯 Generating trace {trace_idx + 1}/{traces_per_question}")
            
            max_trace_attempts = 2
            accepted_trace = None
            
            for attempt in range(max_trace_attempts):
                try:
                    # Generate reasoning trace for this specific question with natural temperature variation
                    trace = await self.generator.generate_reasoning_trace_for_question(
                        paper, selected_question
                    )
                    
                    print(f"      Generated trace (authenticity: {trace['metadata']['authenticity_score']:.2f})")
                    
                    # Quality check if enabled
                    if self.quality_check:
                        quality_score = await self._evaluate_trace_quality(trace)
                        
                        if quality_score >= self.min_quality_score:
                            print(f"      ✅ Quality check passed ({quality_score:.1f} >= {self.min_quality_score})")
                            accepted_trace = trace
                            break
                        else:
                            print(f"      ❌ Quality check failed ({quality_score:.1f} < {self.min_quality_score})")
                            if attempt < max_trace_attempts - 1:
                                print(f"      🔄 Retrying trace generation...")
                            continue
                    else:
                        # No quality check, accept the trace
                        accepted_trace = trace
                        break
                        
                except Exception as e:
                    print(f"      ❌ Error in trace attempt {attempt + 1}: {e}")
                    if attempt < max_trace_attempts - 1:
                        print(f"      🔄 Retrying...")
                    continue
            
            if accepted_trace:
                # Add metadata about the variation
                accepted_trace['metadata']['question_variation_index'] = trace_idx
                accepted_trace['metadata']['same_question_group'] = True
                accepted_traces.append(accepted_trace)
                print(f"      ✅ Accepted trace {trace_idx + 1}")
            else:
                print(f"      ❌ Failed to generate acceptable trace {trace_idx + 1} after {max_trace_attempts} attempts")
        
        print(f"  📊 Successfully generated {len(accepted_traces)}/{traces_per_question} traces for same question")
        return accepted_traces
    
    async def _evaluate_trace_quality(self, trace_data: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated trace using subset of judges"""
        if not self.quality_check or not self.evaluator:
            return 10.0  # Skip evaluation, assume acceptable
        
        # Create QRS triplet from trace data
        triplet = QRSTriplet(
            question=trace_data['user_interaction']['question'],
            reasoning=trace_data['reasoning_trace'],
            solution=trace_data['final_answer'],
            source_file="synthetic_generation",
            sample_index=0
        )
        
        # Use only sonnet 3.7 and gpt-4o for quality check (reuse existing logic)
        subset_judges = ['claude-sonnet-3.7', 'gpt-4o']
        
        try:
            # Reuse the existing evaluate_triplet method
            ratings = await self.evaluator.evaluate_triplet(triplet, subset_judges)
            valid_ratings = [r for r in ratings if r.overall_score > 0]
            
            if not valid_ratings:
                print(f"    ⚠️ Quality check failed - no valid ratings")
                return 0.0
            
            # Reuse existing calculate_statistics method
            stats = self.evaluator.calculate_statistics(valid_ratings)
            avg_score = stats['overall_score']['mean']
            
            print(f"    📊 Quality scores: {[r.overall_score for r in valid_ratings]} → avg: {avg_score:.1f}")
            
            return avg_score
            
        except Exception as e:
            print(f"    ❌ Quality evaluation error: {e}")
            return 0.0  # Assume failed quality check
        
    async def process_paper_corpus(
        self, 
        papers: List[ScientificPassage], 
        traces_per_paper: int = 3,
        traces_per_question: int = 1,
        output_file: str = None,
        save_incrementally: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a corpus of papers to generate reasoning traces with incremental saving.
        
        If traces_per_question > 1, generates one question per paper, then creates
        multiple reasoning traces for that same question using temperature variation.
        """
        
        all_traces = []
        
        # Load existing traces if file exists
        if output_file and save_incrementally:
            existing_traces = self._load_existing_traces(output_file)
            if existing_traces:
                print(f"Loaded {len(existing_traces)} existing traces from {output_file}")
                all_traces.extend(existing_traces)
        
        for i, paper in enumerate(papers):
            print(f"Processing {i+1}/{len(papers)}: {passage.source_title}")
            
            if traces_per_question > 1:
                # New logic: Generate one question, then multiple reasoning traces for it
                accepted_traces = await self._process_paper_with_question_variations(
                    paper, traces_per_question, traces_per_paper
                )
                
                for trace in accepted_traces:
                    trace['sample_index'] = len(all_traces)
                    if self.quality_check:
                        trace['metadata']['quality_checked'] = True
                        trace['metadata']['min_quality_threshold'] = self.min_quality_score
                    
                    all_traces.append(trace)
                    print(f"  ✅ Accepted trace {len(accepted_traces)} for same question")
                    
                    # Save incrementally after each successful generation
                    if output_file and save_incrementally:
                        self._save_traces_incrementally(all_traces, output_file)
                        print(f"  💾 Saved {len(all_traces)} traces to {output_file}")
                        
            else:
                # Original logic: Generate one trace per paper
                accepted_trace = None
                max_attempts = 3  # Maximum attempts per paper
                attempt = 0
                
                while accepted_trace is None and attempt < max_attempts:
                    attempt += 1
                    try:
                        if self.quality_check:
                            print(f"  Attempt {attempt}/{max_attempts} (with quality check)")
                        else:
                            print(f"  Attempt {attempt}/{max_attempts}")
                        
                        # Generate multiple traces per paper for diversity
                        paper_traces = await asyncio.gather(*[
                            self.generator.generate_reasoning_trace(paper) 
                            for _ in range(traces_per_paper)
                        ])
                        
                        # Select best trace based on authenticity score
                        best_trace = max(paper_traces, 
                                       key=lambda x: x['metadata']['authenticity_score'])
                        
                        print(f"    Generated trace (authenticity: {best_trace['metadata']['authenticity_score']:.2f})")
                        
                        # Quality check if enabled
                        if self.quality_check:
                            quality_score = await self._evaluate_trace_quality(best_trace)
                            
                            if quality_score >= self.min_quality_score:
                                print(f"    ✅ Quality check passed ({quality_score:.1f} >= {self.min_quality_score})")
                                accepted_trace = best_trace
                            else:
                                print(f"    ❌ Quality check failed ({quality_score:.1f} < {self.min_quality_score})")
                                if attempt < max_attempts:
                                    print(f"    🔄 Retrying with new generation...")
                                continue
                        else:
                            # No quality check, accept the trace
                            accepted_trace = best_trace
                            
                    except Exception as e:
                        print(f"    ❌ Error in attempt {attempt}: {e}")
                        if attempt < max_attempts:
                            print(f"    🔄 Retrying...")
                        continue
                
                if accepted_trace:
                    accepted_trace['sample_index'] = len(all_traces)
                    accepted_trace['metadata']['generation_attempts'] = attempt
                    if self.quality_check:
                        accepted_trace['metadata']['quality_checked'] = True
                        accepted_trace['metadata']['min_quality_threshold'] = self.min_quality_score
                    
                    all_traces.append(accepted_trace)
                    print(f"  ✅ Accepted trace after {attempt} attempt(s)")
                    
                    # Save incrementally after each successful generation
                    if output_file and save_incrementally:
                        self._save_traces_incrementally(all_traces, output_file)
                        print(f"  💾 Saved {len(all_traces)} traces to {output_file}")
                else:
                    print(f"  ❌ Failed to generate acceptable trace after {max_attempts} attempts")
            
        return all_traces
    
    def _load_existing_traces(self, output_file: str) -> List[Dict[str, Any]]:
        """Load existing traces from output file if it exists"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('questions', [])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return []
    
    def _save_traces_incrementally(self, traces: List[Dict[str, Any]], output_file: str):
        """Save traces incrementally with metadata"""
        from datetime import datetime
        
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_traces_generated": len(traces),
                "last_updated": datetime.now().isoformat(),
                "status": "in_progress"
            },
            "questions": traces
        }
        
        # Write to temporary file first, then rename for atomic operation
        temp_file = output_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            import os
            os.rename(temp_file, output_file)
        except Exception as e:
            print(f"Warning: Could not save incrementally: {e}")
            # Clean up temp file if it exists
            try:
                os.remove(temp_file)
            except:
                pass
    
    def save_training_dataset(self, traces: List[Dict[str, Any]], filename: str):
        """Save traces in final format suitable for training."""
        from datetime import datetime
        
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_traces_generated": len(traces),
                "last_updated": datetime.now().isoformat(),
                "status": "completed"
            },
            "questions": traces
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Final dataset saved with {len(traces)} traces")

def load_data_from_jsonl(file_path: str, max_samples: int = None) -> List[str]:
    """Load text chunks from JSONL file with 'text' field"""
    texts = []
    
    try:
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
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    
    print(f"Loaded {len(texts)} text samples from {file_path}")
    return texts

def convert_texts_to_passages(texts: List[str]) -> List[ScientificPassage]:
    """Convert text chunks to ScientificPassage objects for processing"""
    passages = []
    
    for i, text in enumerate(texts):
        # Detect section type based on content patterns (basic heuristics)
        section_type = _detect_section_type(text)
        
        passages.append(ScientificPassage(
            content=text,
            section_type=section_type,
            source_title=f"Scientific Paper {i+1}",  # Could be extracted from metadata if available
            domain="General Science",  # Could be inferred or provided as metadata
            passage_id=f"passage_{i+1}"
        ))
    
    return passages

def _detect_section_type(text: str) -> str:
    """Basic heuristic to detect what type of section this passage is from"""
    text_lower = text.lower()
    
    # Look for common section indicators
    if any(keyword in text_lower for keyword in ['method', 'procedure', 'experimental', 'protocol', 'approach']):
        return 'methods'
    elif any(keyword in text_lower for keyword in ['result', 'finding', 'observed', 'measured', 'data show']):
        return 'results'
    elif any(keyword in text_lower for keyword in ['discuss', 'interpret', 'implication', 'conclude', 'suggest']):
        return 'discussion'
    elif any(keyword in text_lower for keyword in ['introduction', 'background', 'literature']):
        return 'introduction'
    else:
        return 'mixed'  # Could be combination or unclear

# Example implementation of LLM interface (for reference only)
# Use ArgoLLMAdapter for Argo server integration
class OpenAILLM(LLMInterface):
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        # Initialize OpenAI client here
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Implement actual OpenAI API call
        # This is a placeholder
        return "Generated response based on prompt"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces from scientific text")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="claudesonnet4",
        help="LLM model to use (default: claudesonnet4)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default="../data/heuristic_filtered_cosmo_limited.jsonl",
        help="Input JSONL file with text chunks in 'text' field (default: ../data/heuristic_filtered_cosmo_limited.jsonl)"
    )
    
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to generate for testing (default: process all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reasoning_training_data.json",
        help="Output file for generated traces (default: reasoning_training_data.json)"
    )
    
    parser.add_argument(
        "--traces-per-sample",
        type=int,
        default=1,
        help="Number of reasoning traces to generate per text sample (default: 1)"
    )
    
    parser.add_argument(
        "--traces-per-question",
        type=int,
        default=1,
        help="Number of reasoning traces to generate per question using temperature variation (default: 1)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists (default: True, use --no-resume to start fresh)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing output file"
    )
    
    parser.add_argument(
        "--grade-answers",
        action="store_true",
        help="Grade final answers for RL training (slower but provides reward signals)"
    )
    
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Enable quality checking with LLM judges before accepting traces (slower but higher quality)"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=6.0,
        help="Minimum quality score (1-10) required to accept a trace (default: 6.0)"
    )
    
    return parser.parse_args()

async def main():
    """Main function with command line argument support"""
    args = parse_arguments()
    
    print(f"Synthetic Reasoning Trace Generator")
    print(f"Model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Max traces: {args.max_traces or 'All'}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    # Load data from JSONL file
    texts = load_data_from_jsonl(args.input_file, args.max_traces)
    
    if not texts:
        print("No text data loaded. Exiting.")
        return
    
    # Convert texts to ScientificPassage objects
    papers = convert_texts_to_passages(texts)
    
    # Initialize components with specified model
    try:
        llm = ArgoLLMAdapter(args.model)
        generator = ReasoningTraceGenerator(llm, grade_answers=args.grade_answers)
        processor = BatchProcessor(
            generator, 
            quality_check=args.quality_check,
            min_quality_score=args.min_quality_score
        )
        
        print(f"Initialized LLM with model: {args.model}")
        if args.grade_answers:
            print("Answer grading enabled for RL training")
        if args.quality_check:
            print(f"Quality checking enabled (min score: {args.min_quality_score}) using Claude Sonnet 3.7 & GPT-4o")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return
    
    # Determine resume behavior
    resume_enabled = not args.no_resume  # Default to resume unless --no-resume specified
    
    # Generate traces with incremental saving
    print(f"Generating reasoning traces for {len(papers)} text samples...")
    print(f"Progress will be saved incrementally to: {args.output}")
    if resume_enabled:
        print("(You can interrupt and resume later - progress is saved after each trace)")
        print("(Use --no-resume to start fresh)")
    else:
        print("(Starting fresh - ignoring any existing output file)")
    print("-" * 50)
    
    traces = await processor.process_paper_corpus(
        papers, 
        traces_per_paper=args.traces_per_sample,
        traces_per_question=args.traces_per_question,
        output_file=args.output if resume_enabled else None,
        save_incrementally=resume_enabled
    )
    
    # Final save with completion status
    processor.save_training_dataset(traces, args.output)
    
    print(f"✅ Generated {len(traces)} reasoning traces for training")
    print(f"Final results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())