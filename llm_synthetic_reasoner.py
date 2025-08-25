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
import re
import time
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# MPI support
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    rank = 0
    size = 1
    MPI_AVAILABLE = False

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
        self.logical_rigor_markers = [
            "therefore", "however", "given that", "this leads to", "consequently",
            "alternatively", "nevertheless", "furthermore", "specifically", "precisely",
            "in contrast", "as a result", "checking this against", "verifying that"
        ]
        
    async def generate_reasoning_trace(self, passage: ScientificPassage) -> Dict[str, Any]:
        """Generate a complete question-response reasoning trace from a scientific paper."""
        
        start_time = time.time()
        print(f"  🚀 Starting trace generation for: {passage.source_title[:50]}...")
        
        # Stage 0: Generate realistic user questions
        stage_start = time.time()
        user_questions = await self._generate_user_questions(passage)
        selected_question = self._select_question_for_reasoning(user_questions, passage)
        print(f"     Question generation: {time.time() - stage_start:.1f}s")
        
        # Stage 1: Generate question-specific reasoning fragments (3 parallel calls)
        stage_start = time.time()
        fragments = await asyncio.gather(
            self._generate_targeted_hypotheses(selected_question, passage),
            self._evaluate_evidence_and_methods_combined(selected_question, passage),  # Combined call
            self._consider_user_perspective(selected_question, passage)
        )
        print(f"     Fragment generation: {time.time() - stage_start:.1f}s")
        
        # Stage 2: Analyze context and assemble trace (1 combined call)
        stage_start = time.time()
        raw_trace = await self._analyze_context_and_assemble_trace(
            selected_question, fragments, passage
        )
        print(f"     Context & assembly: {time.time() - stage_start:.1f}s")
        
        # Stage 3: Comprehensive enhancement (1 combined call)
        stage_start = time.time()
        final_trace = await self._enhance_structure_and_inject_corrections(raw_trace)
        print(f"     Enhancement: {time.time() - stage_start:.1f}s")
        
        # Extract final answer from reasoning trace
        stage_start = time.time()
        final_answer = await self._extract_final_answer(final_trace, selected_question)
        print(f"     Answer extraction: {time.time() - stage_start:.1f}s")
        
        # Grade the final answer for RL training (if enabled)
        answer_grades = None
        if self.grade_answers:
            stage_start = time.time()
            print(f"  Grading final answer...")
            answer_grades = await self._grade_final_answer(
                selected_question, 
                final_answer, 
                passage.content + " " + passage.content, 
                final_trace
            )
            print(f"     Answer grading: {time.time() - stage_start:.1f}s")
        
        total_generation_time = time.time() - start_time
        print(f"  📊 Total generation time: {total_generation_time:.1f}s")
        
        return await self.generate_reasoning_trace_for_question(passage, selected_question)
    
    async def generate_reasoning_trace_for_question(
        self, 
        passage: ScientificPassage, 
        question: UserQuestion
    ) -> Dict[str, Any]:
        """Generate a reasoning trace for a specific question using natural randomness."""
        
        trace_start_time = time.time()
        
        # Stage 1: Generate question-specific reasoning fragments (3 parallel calls)
        stage_start = time.time()
        fragments = await asyncio.gather(
            self._generate_targeted_hypotheses(question, passage),
            self._evaluate_evidence_and_methods_combined(question, passage),  # Combined call
            self._consider_user_perspective(question, passage)
        )
        fragment_time = time.time() - stage_start
        
        # Stage 2: Analyze context and assemble trace (1 combined call)
        stage_start = time.time()
        raw_trace = await self._analyze_context_and_assemble_trace(
            question, fragments, passage
        )
        assembly_time = time.time() - stage_start
        
        # Stage 3: Comprehensive enhancement (1 combined call)
        stage_start = time.time()
        # final_trace = await self._enhance_structure_and_inject_corrections(raw_trace)
        final_trace = raw_trace
        enhancement_time = time.time() - stage_start
        
        # Extract final answer from reasoning trace
        stage_start = time.time()
        final_answer = await self._extract_final_answer(final_trace, question)
        extraction_time = time.time() - stage_start
        
        # Grade the final answer for RL training (if enabled)
        answer_grades = None
        grading_time = 0.0
        if self.grade_answers:
            stage_start = time.time()
            print(f"  Grading final answer...")
            answer_grades = await self._grade_final_answer(
                question, 
                final_answer, 
                passage.content + " " + passage.content, 
                final_trace
            )
            grading_time = time.time() - stage_start
        
        total_time = time.time() - trace_start_time
        
        print(f"  ✅ Total trace time: {total_time:.1f}s (fragments: {fragment_time:.1f}s, assembly: {assembly_time:.1f}s, enhancement: {enhancement_time:.1f}s, extraction: {extraction_time:.1f}s{f', grading: {grading_time:.1f}s' if grading_time > 0 else ''})")
        
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
                'logical_rigor_score': self._calculate_logical_rigor_score(final_trace),
                'complexity_level': self._assess_complexity(final_trace),
                'reasoning_patterns': self._identify_patterns(final_trace),
                'question_responsiveness': self._assess_question_responsiveness(final_trace, question),
                'timing': {
                    'total_time': round(total_time, 2),
                    'fragment_generation_time': round(fragment_time, 2),
                    'context_assembly_time': round(assembly_time, 2),
                    'enhancement_time': round(enhancement_time, 2),
                    'answer_extraction_time': round(extraction_time, 2),
                    'grading_time': round(grading_time, 2) if grading_time > 0 else None
                }
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
CLARIFICATION (expert): Understanding questions about nuances in concepts or methods
CRITICAL (intermediate): Questions challenging methodology, conclusions, or assumptions  
INTERPRETATION (intermediate): Questions about meaning, implications, or significance
COMPARISON (advanced): Questions relating this to other research or broader context
APPLICATION (intermediate): Questions about practical applications or extensions
MECHANISM (advanced): Questions about underlying processes or how things work
LIMITATION (advanced): Questions about weaknesses, constraints, or scope

IMPORTANT: Make each question SELF-CONTAINED by:
- Including sufficient context within the question itself
- Defining key terms or concepts mentioned
- Not relying on phrases like "this paper", "the authors", "these results", "the thesis", "the model", etc.
- Providing enough background so the question makes sense independently

Make each question:
- Natural and conversational (how real people actually ask)
- Specific to this paper's actual content but self-contained without the source text
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
            uncertainty_markers=self._find_logical_markers(response)
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
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='question_focused_evidence',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
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
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='question_focused_methods',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
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
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='user_perspective',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _evaluate_evidence_and_methods_combined(self, question: UserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        """Combined evidence and methodological evaluation to maintain quality while reducing calls"""
        prompt = f"""
Evaluate both the evidence and methodology from the perspective of this user's specific question:

User Question: {question.question}
Paper Results: {passage.content}
Methods: {passage.content}

Think through both dimensions systematically:

EVIDENCE EVALUATION:
- Does the evidence actually address what they're asking about?
- How strong is the evidence for answering their specific question?
- What limitations or caveats are relevant to their inquiry?
- Are there alternative interpretations they should consider?

METHODOLOGICAL ANALYSIS:
- Are the methods appropriate for what the user is asking about?
- What methodological details matter for their specific question?
- How do methodological choices affect the answer to their question?
- Are there methodological limitations that affect the response?

INTEGRATED ASSESSMENT:
- How do the methodological strengths/weaknesses affect evidence interpretation?
- What would help them understand both the strength of evidence AND the reliability of methods?
- Are there interactions between methodology and evidence quality that matter for this question?

Show your reasoning as you work through evaluating both evidence and methods through the lens of their question.
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='evidence_and_methods_combined',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _analyze_context_and_assemble_trace(self, question: UserQuestion, fragments: List[ReasoningFragment], passage: ScientificPassage) -> str:
        """Combined context analysis and trace assembly to maintain quality while reducing calls"""
        prompt = f"""
Analyze the question requirements and create a systematic reasoning trace that methodically addresses this question.

User Question: {question.question}
Question Type: {question.question_type}
User Level: {question.difficulty_level}

Available reasoning components:
1. Targeted hypotheses: {fragments[0].content}
2. Evidence & methods analysis: {fragments[1].content}
3. User perspective: {fragments[2].content}

CONTEXT ANALYSIS - First determine:
- Which specific parts of the paper are most relevant to this question
- What types of reasoning are needed (methodological, statistical, conceptual, etc.)
- What potential misconceptions or confusions the user might have
- What background knowledge might be missing for this user level
- What level of technical detail is appropriate

TRACE ASSEMBLY - Then create a structured <thought>...</thought> section that demonstrates:
- Clear problem decomposition and analysis informed by the context requirements
- Systematic integration of the reasoning components above
- Logical connections between concepts and data
- Critical assessment incorporating both evidence and methodological considerations
- Identification and correction of initial assumptions when warranted
- Step-by-step reasoning toward conclusions appropriate for the user level
- Appropriate acknowledgment of uncertainty when evidence is limited
- Coherent synthesis addressing the user's specific needs and potential misconceptions

Weave the reasoning components naturally into a flowing expert thought process while addressing the contextual requirements.
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    async def _enhance_structure_and_inject_corrections(self, trace: str) -> str:
        """Combined logical enhancement and correction injection to maintain quality while reducing calls"""
        prompt = f"""
Enhance this reasoning trace by improving logical flow and adding natural self-correction moments simultaneously.

Current trace: {trace}

DUAL IMPROVEMENT APPROACH:

1. LOGICAL FLOW ENHANCEMENT - Naturally weave in:
   - Clearer logical transitions ("Given that...", "This leads me to think...", "Therefore...")
   - More explicit reasoning steps woven naturally into the flow
   - Natural error checking moments ("Let me double-check this...", "Wait, let me verify...")
   - Organic uncertainty expression ("The evidence seems to support...", "I'm less certain about...")
   - Smoother cause-and-effect relationships
   - Natural evaluation of alternatives ("But I should also consider...", "Another possibility is...")

2. SELF-CORRECTION INTEGRATION - Naturally incorporate moments like:
   - Catching contradictions: "Wait, this contradicts what I said earlier about..."
   - Correcting generalizations: "Actually, let me be more precise - this only applies when..."
   - Recognizing weak evidence: "Hmm, the data doesn't really support such a broad claim..."
   - Refining claims: "Let me be more careful here - the evidence suggests..."
   - Checking assumptions: "I should double-check whether this assumption holds..."
   - Considering alternatives: "But I should also consider that this could mean..."
   - Catching oversights: "Oh, I need to account for the potential bias here..."

**CRITICAL**: Keep the conversational, flowing style of expert thinking. Do NOT add structured headings, bullet points, or artificial phrases. Make these feel like authentic moments of expert self-reflection woven seamlessly into improved logical flow. Maintain the stream-of-consciousness feel while making both the logic clearer and the self-correction more apparent.

Return only the enhanced trace with natural expert language.
"""
        
        return await self.llm.generate(prompt, temperature=0.65)
    
    async def _assemble_question_response_chain(
        self, 
        question: UserQuestion,
        context: QuestionContext,
        fragments: List[ReasoningFragment], 
        passage: ScientificPassage
    ) -> str:
        
        prompt = f"""
Create a systematic reasoning trace that methodically analyzes this question. Focus on logical progression, evidence evaluation, and structured problem-solving.

User Question: {question.question}
Question Analysis: {context.reasoning_requirements}
User Level: {question.difficulty_level}

Reasoning components to integrate:
1. Approach considerations: {fragments[0].content}
2. Evidence evaluation: {fragments[1].content}
3. Methods considerations: {fragments[2].content}
4. User perspective: {fragments[3].content}

Create a structured <thought>...</thought> section that demonstrates:
- Clear problem decomposition and analysis
- Systematic evaluation of available evidence
- Logical connections between concepts and data
- Critical assessment of methodological strengths/limitations
- Identification and correction of initial assumptions when warranted
- Step-by-step reasoning toward conclusions
- Appropriate acknowledgment of uncertainty when evidence is limited
- Coherent synthesis of information to address the question

Emphasize logical rigor, self-correction when errors are detected, and transparent reasoning processes over conversational elements.
Maintain a natural language structure, as if you are an expert working through this thought process.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    def _select_question_for_reasoning(self, questions: List[UserQuestion], passage: ScientificPassage) -> UserQuestion:
        """Select the most interesting question for generating reasoning trace."""
        print(f"  Generated {len(questions)} questions")
        # try to filter out questions referencing the text
        disqualifiers = ['the paper', 'the authors', 'the thesis', 'the project']
        questions = [q for q in questions if not any(disqualifier in q.question.lower() for disqualifier in disqualifiers)]
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
            uncertainty_markers=self._find_logical_markers(response)
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
            uncertainty_markers=self._find_logical_markers(response)
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
            uncertainty_markers=self._find_logical_markers(response)
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
        """Grade the final answer across multiple dimensions for RL training - consolidated into single call"""
        
        prompt = f"""
Evaluate this answer across 6 dimensions and provide scores (scale 0.0-1.0) for each:

QUESTION: {question.question}
QUESTION TYPE: {question.question_type}
DIFFICULTY LEVEL: {question.difficulty_level}

ANSWER: {final_answer}

SOURCE TEXT: {source_text[:1000]}...

REASONING TRACE (for context): {reasoning_trace[:500]}...

Rate the answer on these 6 dimensions (0.0-1.0 scale):

1. QUESTION_ALIGNMENT: How well does the answer directly address the specific question asked?
   - Does it address what was asked?
   - Are all parts of multi-part questions covered?
   - Is response appropriate for question type/difficulty?
   - Does it stay focused rather than going off-topic?

2. SCIENTIFIC_ACCURACY: How factually accurate and scientifically correct is the answer?
   - Are scientific facts stated correctly?
   - Are there clear factual errors?
   - Does it contradict established scientific knowledge?
   - Are technical terms used appropriately?
   - Is it consistent with source material?

3. COMPLETENESS: How thoroughly does the answer address the question?
   - Does it cover all aspects of the question?
   - Are important points missing?
   - Is depth appropriate for question complexity?
   - For multi-part questions, are all parts addressed?

4. UNCERTAINTY_CALIBRATION: How well does it express appropriate uncertainty/confidence?
   - Does it express uncertainty about uncertain claims?
   - Does it avoid overconfidence about speculative topics?
   - Does it express confidence about well-established facts?
   - Is certainty level consistent with reasoning shown?

5. CLARITY_STRUCTURE: How clear, organized, and readable is the answer?
   - Is it well-organized and logical?
   - Is language clear and appropriate?
   - Are ideas in logical order?
   - Is it easy to follow and understand?
   - Are key points emphasized effectively?

6. EVIDENCE_USAGE: How effectively does it use evidence from source material?
   - Does it draw appropriately from source material?
   - Are claims supported by evidence where appropriate?
   - Does it avoid unsupported claims?
   - Is relationship between evidence and conclusions clear?

Provide your scores in this exact format:
QUESTION_ALIGNMENT: X.X
SCIENTIFIC_ACCURACY: X.X
COMPLETENESS: X.X
UNCERTAINTY_CALIBRATION: X.X
CLARITY_STRUCTURE: X.X
EVIDENCE_USAGE: X.X
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        return self._parse_consolidated_grades(response)
    
    def _parse_consolidated_grades(self, response: str) -> Dict[str, float]:
        """Parse the consolidated grading response into individual scores"""
        grades = {
            'question_alignment': 0.5,
            'scientific_accuracy': 0.5,
            'completeness': 0.5,
            'uncertainty_calibration': 0.5,
            'clarity_structure': 0.5,
            'evidence_usage': 0.5
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    try:
                        score = float(value.strip())
                        if key == 'question_alignment':
                            grades['question_alignment'] = score
                        elif key == 'scientific_accuracy':
                            grades['scientific_accuracy'] = score
                        elif key == 'completeness':
                            grades['completeness'] = score
                        elif key == 'uncertainty_calibration':
                            grades['uncertainty_calibration'] = score
                        elif key == 'clarity_structure':
                            grades['clarity_structure'] = score
                        elif key == 'evidence_usage':
                            grades['evidence_usage'] = score
                    except ValueError:
                        continue  # Skip lines that can't be parsed as float
        except Exception as e:
            print(f"Warning: Error parsing consolidated grades: {e}")
        
        # Calculate overall score (exclude the overall_score itself from calculation)
        core_grades = [v for k, v in grades.items() if k != 'overall_score']
        grades['overall_score'] = sum(core_grades) / len(core_grades)
        
        return grades
    
    async def _enhance_logical_structure(self, trace: str) -> str:
        prompt = f"""
Enhance the logical flow and clarity of this reasoning trace while maintaining its natural, conversational style.

Current trace: {trace}

Improve by naturally weaving in:
- Clearer logical transitions ("Given that...", "This leads me to think...", "Therefore...")
- More explicit reasoning steps woven naturally into the flow
- Natural error checking moments ("Let me double-check this...", "Wait, let me verify...")
- Organic uncertainty expression ("The evidence seems to support...", "I'm less certain about...")
- Smoother cause-and-effect relationships
- Natural evaluation of alternatives ("But I should also consider...", "Another possibility is...")
- Authentic validation moments ("This makes sense because...")

**CRITICAL**: Keep the conversational, flowing style of expert thinking. Do NOT add structured headings, bullet points, or artificial phrases. Maintain the stream-of-consciousness feel while making the logic clearer.

Return only the improved trace with natural expert language.
"""
        
        return await self.llm.generate(prompt, temperature=0.6)
    
    async def _inject_systematic_corrections(self, trace: str) -> str:
        prompt = f"""
Add natural self-correction moments to show authentic expert reasoning. Weave these organically into the flow.

Current trace: {trace}

Naturally incorporate moments like:
- Catching contradictions: "Wait, this contradicts what I said earlier about..."
- Correcting generalizations: "Actually, let me be more precise - this only applies when..."
- Recognizing weak evidence: "Hmm, the data doesn't really support such a broad claim..."
- Refining claims: "Let me be more careful here - the evidence suggests..."
- Checking assumptions: "I should double-check whether this assumption holds..."
- Considering alternatives: "But I should also consider that this could mean..."
- Catching oversights: "Oh, I need to account for the potential bias here..."

**CRITICAL**: Make these feel like authentic moments of expert self-reflection, NOT structured analysis. Use natural, conversational language as if catching your own errors in real-time. Maintain the flowing, stream-of-consciousness style.

Return only the enhanced trace with natural self-correction woven in.
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
    
    def _find_logical_markers(self, text: str) -> List[str]:
        markers = []
        for marker in self.logical_rigor_markers:
            if marker in text.lower():
                markers.append(marker)
        return markers
    
    def _calculate_logical_rigor_score(self, trace: str) -> float:
        # Calculate logical rigor based on presence of systematic reasoning patterns
        patterns = [
            'therefore', 'however', 'given that', 'this leads to', 'consequently',
            'specifically', 'checking', 'verifying', 'alternatively', 'precisely',
            'as a result', 'in contrast', 'furthermore', 'systematic'
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
    def __init__(self, generator: ReasoningTraceGenerator, quality_check: bool = False, 
                 min_quality_score: float = 6.0, quality_judges: List[str] = None):
        self.generator = generator
        self.quality_check = quality_check
        self.min_quality_score = min_quality_score
        self.quality_judges = quality_judges or ['claude-sonnet-3.7', 'gpt-4o']
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
                        passage, selected_question
                    )
                    
                    print(f"      Generated trace (logical rigor: {trace['metadata']['logical_rigor_score']:.2f})")
                    
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
        
        # Use configurable judge models for quality check
        try:
            # Reuse the existing evaluate_triplet method
            ratings = await self.evaluator.evaluate_triplet(triplet, self.quality_judges)
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
        processed_titles = set()
        
        # Load existing traces if file exists
        if output_file and save_incrementally:
            existing_traces = self._load_existing_traces(output_file)
            if existing_traces:
                print(f"Loaded {len(existing_traces)} existing traces from {output_file}")
                all_traces.extend(existing_traces)
                processed_titles = self._get_processed_paper_titles(existing_traces)
                print(f"Found {len(processed_titles)} already processed papers")
        
        for i, paper in enumerate(papers):
            print(f"Processing {i+1}/{len(papers)}: {paper.source_title}")
            
            # Skip if paper already processed
            if paper.source_title in processed_titles:
                print(f"  🔄 Skipping {paper.source_title} - already processed")
                continue
            
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
                        
                        # Select best trace based on logical rigor score
                        best_trace = max(paper_traces, 
                                       key=lambda x: x['metadata']['logical_rigor_score'])
                        
                        print(f"    Generated trace (logical rigor: {best_trace['metadata']['logical_rigor_score']:.2f})")
                        
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
    
    def _get_processed_paper_titles(self, traces: List[Dict[str, Any]]) -> set:
        """Extract set of paper titles that have already been processed"""
        processed_titles = set()
        for trace in traces:
            if 'source_paper' in trace and 'title' in trace['source_paper']:
                processed_titles.add(trace['source_paper']['title'])
        return processed_titles
    
    def _save_traces_incrementally(self, traces: List[Dict[str, Any]], output_file: str):
        """Save traces incrementally with metadata"""
        from datetime import datetime
        import os
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
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
        
        # No try-except here - let failures propagate for better debugging
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        os.rename(temp_file, output_file)
    
    def save_training_dataset(self, traces: List[Dict[str, Any]], filename: str):
        """Save traces in final format suitable for training."""
        from datetime import datetime
        import os
        
        # Ensure output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
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

def discover_input_files(pattern: str) -> List[str]:
    """
    Discover input files matching the given pattern (supports wildcards).
    Returns sorted list of matching file paths.
    """
    try:
        files = glob.glob(pattern, recursive=True)
        if not files:
            # Try to handle the case where pattern is a single file without wildcards
            if Path(pattern).exists():
                files = [pattern]
            else:
                print(f"Warning: No files found matching pattern: {pattern}")
                return []
        
        # Sort for consistent ordering across ranks
        files.sort()
        print(f"Discovered {len(files)} files matching pattern: {pattern}")
        return files
    except Exception as e:
        print(f"Error discovering files with pattern '{pattern}': {e}")
        return []

def stream_texts_from_files(file_paths: List[str], max_samples: int = None, 
                          apply_quality_filter: bool = True) -> Iterator[Tuple[str, str, int]]:
    """
    Stream text samples from multiple JSONL files.
    Yields (text, source_file, sample_index) tuples.
    """
    total_yielded = 0
    global_sample_index = 0
    filtered_count = 0
    filter_reasons = {}
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        file_sample_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and total_yielded >= max_samples:
                        return
                    
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data and data['text'].strip():
                            text = data['text'].strip()
                            
                            if apply_quality_filter:
                                is_valid, reason = is_high_quality_scientific_text(text)
                                if is_valid:
                                    yield (text, file_path, global_sample_index)
                                    total_yielded += 1
                                    file_sample_count += 1
                                    global_sample_index += 1
                                else:
                                    filtered_count += 1
                                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                                    global_sample_index += 1
                            else:
                                yield (text, file_path, global_sample_index)
                                total_yielded += 1
                                file_sample_count += 1
                                global_sample_index += 1
                                
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON on line {line_num} in {file_path}")
                        continue
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Warning: Error processing file {file_path}: {e}")
            continue
        
        print(f"  Processed {file_sample_count} valid samples from {file_path}")
    
    print(f"\nTotal samples yielded: {total_yielded}")
    if apply_quality_filter and filtered_count > 0:
        print(f"Filtered out {filtered_count} low-quality samples:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count} samples")

def distribute_files_across_ranks(file_paths: List[str], rank: int, size: int) -> List[str]:
    """
    Distribute files across MPI ranks to ensure no overlap.
    Each rank gets a subset of files to process.
    """
    if size <= 1:
        return file_paths
    
    # Simple round-robin distribution of files
    my_files = [file_paths[i] for i in range(len(file_paths)) if i % size == rank]
    print(f"Rank {rank}: assigned {len(my_files)} files out of {len(file_paths)} total")
    
    return my_files

def is_high_quality_scientific_text(text: str) -> tuple[bool, str]:
    """
    Filter for high-quality scientific text suitable for reasoning trace generation.
    Returns (is_valid, reason_if_invalid)
    """
    import re
    text = text.strip()
    text_lower = text.lower()
    
    # Minimum length filter
    if len(text) < 300:
        return False, f"Too short ({len(text)} chars, min 300)"
    
    # Maximum length filter (avoid extremely long texts)
    if len(text) > 50000:
        return False, f"Too long ({len(text)} chars, max 50000)"
    
    # Check for references/bibliography sections
    reference_indicators = [
        'references', 'bibliography', 'works cited', 'citations',
        'further reading', 'suggested reading'
    ]
    
    # Strong indicators of reference sections
    if any(indicator in text_lower[:100] for indicator in reference_indicators):
        # Check if it's predominantly citations (multiple author-year patterns)
        citation_patterns = [
            r'\b[A-Z][a-z]+,?\s+[A-Z]\.',  # Author initials
            r'\(\d{4}\)',  # Year in parentheses
            r'\bet\s+al\.',  # et al.
            r'\bpp?\.\s*\d+',  # page numbers
            r'\bvol\.\s*\d+',  # volume numbers
            r'\bdoi:',  # DOI indicators
        ]
        
        import re
        citation_matches = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
        # If high density of citation patterns, likely a reference section
        if citation_matches > len(text) / 200:  # Roughly 1 citation pattern per 200 chars
            return False, "Appears to be references/bibliography section"
    
    # Check for acknowledgments sections
    acknowledgment_indicators = [
        'acknowledgment', 'acknowledgement', 'acknowledgments', 'acknowledgements',
        'we thank', 'we are grateful', 'we would like to thank',
        'the authors thank', 'the authors acknowledge',
        'funding', 'supported by', 'grant number'
    ]
    
    if any(indicator in text_lower[:200] for indicator in acknowledgment_indicators):
        # Check if predominantly acknowledgments (gratitude expressions)
        gratitude_patterns = [
            r'\bthank\b', r'\bgrateful\b', r'\backnowledge\b',
            r'\bsupport\b', r'\bfunding\b', r'\bgrant\b'
        ]
        gratitude_matches = sum(len(re.findall(pattern, text_lower)) for pattern in gratitude_patterns)
        if gratitude_matches > 3:  # Multiple gratitude expressions
            return False, "Appears to be acknowledgments section"
    
    # Check for appendix content
    if text_lower.startswith(('appendix', 'supplementary', 'supplemental')):
        return False, "Appears to be appendix/supplementary material"
    
    # Check for abstracts (information-light)
    if 'abstract' in text_lower[:200]:
        return False, "Appears to be abstract/summary (information-light)"
    
    return True, "Valid scientific content"

def load_data_from_jsonl(file_path: str, max_samples: int = None, apply_quality_filter: bool = True) -> List[str]:
    """Load and filter text chunks from JSONL file with 'text' field"""
    texts = []
    filtered_count = 0
    filter_reasons = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and len(texts) >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text'].strip():
                        text = data['text'].strip()
                        
                        if apply_quality_filter:
                            is_valid, reason = is_high_quality_scientific_text(text)
                            if is_valid:
                                texts.append(text)
                            else:
                                filtered_count += 1
                                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                        else:
                            texts.append(text)
                            
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {i+1}")
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    
    print(f"Loaded {len(texts)} high-quality text samples from {file_path}")
    if apply_quality_filter and filtered_count > 0:
        print(f"Filtered out {filtered_count} low-quality samples:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count} samples")
    
    return texts

# Note: convert_texts_to_passages function is now handled inline in streaming processing

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
        "--input-pattern",
        type=str,
        default="../data/heuristic_filtered_cosmo_limited.jsonl",
        help="Input file pattern (supports wildcards) for JSONL files with text chunks in 'text' field (default: ../data/heuristic_filtered_cosmo_limited.jsonl)"
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

async def process_files_streaming(
    processor,
    file_paths: List[str],
    max_samples: int = None,
    traces_per_sample: int = 1,
    traces_per_question: int = 1,
    output_file: str = None,
    save_incrementally: bool = True
) -> List[Dict[str, Any]]:
    """
    Process files using streaming approach to reduce memory usage.
    """
    all_traces = []
    processed_samples = 0
    processed_titles = set()
    
    # Load existing traces if resuming
    if output_file and save_incrementally:
        existing_traces = processor._load_existing_traces(output_file)
        if existing_traces:
            print(f"Loaded {len(existing_traces)} existing traces from {output_file}")
            all_traces.extend(existing_traces)
            processed_titles = processor._get_processed_paper_titles(existing_traces)
            processed_samples = len(existing_traces)
            print(f"Found {len(processed_titles)} already processed papers")
    
    # Stream through files and process samples
    sample_count = 0
    for text, source_file, global_index in stream_texts_from_files(
        file_paths, 
        max_samples=max_samples, 
        apply_quality_filter=True
    ):
        sample_count += 1
        
        # Create a unique title using file and sample info
        file_name = Path(source_file).stem
        sample_title = f"Sample {global_index} from {file_name}"
        
        # Skip if already processed (for resume functionality)
        if sample_title in processed_titles:
            print(f"  🔄 Skipping {sample_title} - already processed")
            continue
        
        # Convert text to ScientificPassage
        passage = ScientificPassage(
            content=text,
            section_type=_detect_section_type(text),
            source_title=sample_title,
            domain="General Science",
            passage_id=f"passage_{global_index}"
        )
        
        print(f"Processing sample {sample_count}: {sample_title}")
        
        try:
            if traces_per_question > 1:
                # Generate multiple traces for same question
                sample_traces = await processor._process_paper_with_question_variations(
                    passage, traces_per_question, traces_per_sample
                )
                
                for trace in sample_traces:
                    trace['sample_index'] = len(all_traces)
                    if processor.quality_check:
                        trace['metadata']['quality_checked'] = True
                        trace['metadata']['min_quality_threshold'] = processor.min_quality_score
                    
                    all_traces.append(trace)
                    
                    # Save incrementally after each trace
                    if output_file and save_incrementally:
                        processor._save_traces_incrementally(all_traces, output_file)
                        print(f"  💾 Saved {len(all_traces)} traces to {output_file}")
            else:
                # Generate single trace per sample
                max_attempts = 3
                accepted_trace = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        print(f"  Attempt {attempt}/{max_attempts}")
                        
                        # Generate traces for this sample
                        paper_traces = await asyncio.gather(*[
                            processor.generator.generate_reasoning_trace(passage) 
                            for _ in range(traces_per_sample)
                        ])
                        
                        # Select best trace
                        best_trace = max(paper_traces, 
                                       key=lambda x: x['metadata']['logical_rigor_score'])
                        
                        print(f"    Generated trace (logical rigor: {best_trace['metadata']['logical_rigor_score']:.2f})")
                        
                        # Quality check if enabled
                        if processor.quality_check:
                            quality_score = await processor._evaluate_trace_quality(best_trace)
                            
                            if quality_score >= processor.min_quality_score:
                                print(f"    ✅ Quality check passed ({quality_score:.1f} >= {processor.min_quality_score})")
                                accepted_trace = best_trace
                                break
                            else:
                                print(f"    ❌ Quality check failed ({quality_score:.1f} < {processor.min_quality_score})")
                                if attempt < max_attempts:
                                    print(f"    🔄 Retrying...")
                                continue
                        else:
                            accepted_trace = best_trace
                            break
                            
                    except Exception as e:
                        print(f"    ❌ Error in attempt {attempt}: {e}")
                        if attempt < max_attempts:
                            print(f"    🔄 Retrying...")
                        continue
                
                if accepted_trace:
                    accepted_trace['sample_index'] = len(all_traces)
                    if processor.quality_check:
                        accepted_trace['metadata']['quality_checked'] = True
                        accepted_trace['metadata']['min_quality_threshold'] = processor.min_quality_score
                    
                    all_traces.append(accepted_trace)
                    print(f"  ✅ Accepted trace")
                    
                    # Save incrementally
                    if output_file and save_incrementally:
                        processor._save_traces_incrementally(all_traces, output_file)
                        print(f"  💾 Saved {len(all_traces)} traces to {output_file}")
                else:
                    print(f"  ❌ Failed to generate acceptable trace after {max_attempts} attempts")
        
        except Exception as e:
            print(f"  ❌ Error processing sample {sample_title}: {e}")
            continue
        
        # Check if we've reached the maximum number of samples
        if max_samples and len(all_traces) >= max_samples:
            print(f"Reached maximum of {max_samples} traces, stopping")
            break
    
    return all_traces

async def main():
    """Main function with command line argument support and MPI distribution"""
    # Configure logging based on MPI rank
    if MPI_AVAILABLE:
        if rank == 0:
            logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format=f'[Rank {rank}] %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger = logging.getLogger(__name__)
    
    args = parse_arguments()
    
    if rank == 0:
        print(f"Synthetic Reasoning Trace Generator")
        print(f"Model: {args.model}")
        print(f"Input pattern: {args.input_pattern}")
        print(f"Max traces: {args.max_traces or 'All'}")
        if MPI_AVAILABLE:
            print(f"MPI processes: {size}")
        print("-" * 50)
    
    # Discover input files using pattern (supports wildcards)
    all_files = discover_input_files(args.input_pattern)
    
    if not all_files:
        if rank == 0:
            print("No input files found. Exiting.")
        return
    
    # Distribute files across MPI processes (each rank gets different files)
    if MPI_AVAILABLE and size > 1:
        my_files = distribute_files_across_ranks(all_files, rank, size)
        
        # Update output filename to include rank
        base_name, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'json')
        my_output = f"{base_name}_rank{rank}.{ext}"
        logger.info(f"Rank {rank} will process {len(my_files)} files")
    else:
        my_files = all_files
        my_output = args.output
        if rank == 0:
            logger.info(f"Processing {len(my_files)} files in single process mode")
    
    if not my_files:
        logger.info(f"Rank {rank} has no files to process")
        return
    
    # Initialize components with specified model
    try:
        llm = ArgoLLMAdapter(args.model)
        generator = ReasoningTraceGenerator(llm, grade_answers=args.grade_answers)
        processor = BatchProcessor(
            generator, 
            quality_check=args.quality_check,
            min_quality_score=args.min_quality_score
        )
        
        logger.info(f"Initialized LLM with model: {args.model}")
        if args.grade_answers and rank == 0:
            print("Answer grading enabled for RL training")
        if args.quality_check and rank == 0:
            print(f"Quality checking enabled (min score: {args.min_quality_score}) using Claude Sonnet 3.7 & GPT-4o")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return
    
    # Determine resume behavior
    resume_enabled = not args.no_resume
    
    # Generate traces with incremental saving using streaming approach
    logger.info(f"Processing files with streaming approach...")
    if rank == 0:
        print(f"Progress will be saved incrementally to: {my_output}")
        if resume_enabled:
            print("(You can interrupt and resume later - progress is saved after each trace)")
            print("(Use --no-resume to start fresh)")
        else:
            print("(Starting fresh - ignoring any existing output file)")
        print("-" * 50)
    
    traces = await process_files_streaming(
        processor=processor,
        file_paths=my_files,
        max_samples=args.max_traces,
        traces_per_sample=args.traces_per_sample,
        traces_per_question=args.traces_per_question,
        output_file=my_output if resume_enabled else None,
        save_incrementally=resume_enabled
    )
    
    # Final save with completion status
    processor.save_training_dataset(traces, my_output)
    
    logger.info(f"Generated {len(traces)} reasoning traces for training")
    logger.info(f"Results saved to: {my_output}")
    
    # Wait for all processes to complete if using MPI
    if MPI_AVAILABLE and size > 1:
        comm.Barrier()
        if rank == 0:
            print(f"✅ All {size} MPI processes completed. Results saved to *_rank*.json files")

if __name__ == "__main__":
    asyncio.run(main())