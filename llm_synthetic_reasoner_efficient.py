#!/usr/bin/env python3
"""
Efficient Synthetic Reasoning Trace Generator v2.1

Two-stage specialized approach for high-quality, efficient reasoning trace generation:
1. Challenging Question Generation (specialized for difficult, logic-oriented questions)
2. Systematic Reasoning Trace Generation (specialized for rigorous analysis)

Performance: 2-3 LLM calls per trace (vs 8-10 in original method)
Quality: Enhanced through specialized prompts for each stage

Usage examples:
    # Use default settings (5 samples for testing)
    python llm_synthetic_reasoner_efficient.py --max-traces 5
    
    # Specify different model and input file
    python llm_synthetic_reasoner_efficient.py --model gpt41 --input-file my_data.jsonl --max-traces 10
    
    # Enable quality checking (slower but higher quality)
    python llm_synthetic_reasoner_efficient.py --quality-check --min-quality-score 6.5 --max-traces 5
    
    # Generate multiple traces per question using temperature variation
    python llm_synthetic_reasoner_efficient.py --traces-per-question 3 --max-traces 5
"""

import asyncio
import random
import argparse
import json
import time
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

class EfficientReasoningTraceGenerator:
    """Efficient version that consolidates multiple LLM calls into comprehensive single calls"""
    
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
        """Generate a complete reasoning trace with 3-4 specialized LLM calls"""
        
        start_time = time.time()
        print(f"  🚀 Starting efficient trace generation for: {passage.source_title[:50]}...")
        
        # CALL 1: Generate challenging, logic-oriented question
        stage_start = time.time()
        question_data = await self._generate_challenging_question(passage)
        question_time = time.time() - stage_start
        print(f"     Question generation: {question_time:.1f}s")
        print(f"  Generated question: {question_data['question'][:80]}...")
        
        # CALL 2: Generate systematic reasoning trace and answer for the question
        stage_start = time.time()
        reasoning_data = await self._generate_reasoning_trace_for_question(passage, question_data)
        reasoning_time = time.time() - stage_start
        print(f"     Reasoning generation: {reasoning_time:.1f}s")
        
        # CALL 3: Fact verification and accuracy refinement
        stage_start = time.time()
        print(f"  Verifying facts and refining accuracy...")
        verified_data = await self._verify_and_refine_accuracy(
            passage, question_data, reasoning_data
        )
        verification_time = time.time() - stage_start
        print(f"     Fact verification: {verification_time:.1f}s")
        
        # CALL 4 (optional): Grade the final answer if enabled
        answer_grades = None
        grading_time = 0.0
        if self.grade_answers:
            stage_start = time.time()
            print(f"  Grading final answer...")
            answer_grades = await self._grade_final_answer_consolidated(
                question_data, 
                verified_data['final_answer'], 
                passage.content, 
                verified_data['reasoning_trace']
            )
            grading_time = time.time() - stage_start
            print(f"     Answer grading: {grading_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"  ✅ Total efficient trace time: {total_time:.1f}s (question: {question_time:.1f}s, reasoning: {reasoning_time:.1f}s, verification: {verification_time:.1f}s{f', grading: {grading_time:.1f}s' if grading_time > 0 else ''})")
        
        return {
            'source_paper': {
                'title': passage.source_title,
                'domain': passage.domain
            },
            'user_interaction': {
                'question': question_data['question'],
                'question_type': question_data['question_type'],
                'difficulty_level': question_data['difficulty_level']
            },
            'reasoning_trace': verified_data['reasoning_trace'],
            'final_answer': verified_data['final_answer'],
            'answer_grades': answer_grades,
            'metadata': {
                'logical_rigor_score': self._calculate_logical_rigor_score(verified_data['reasoning_trace']),
                'complexity_level': self._assess_complexity(verified_data['reasoning_trace']),
                'reasoning_patterns': self._identify_patterns(verified_data['reasoning_trace']),
                'question_responsiveness': self._assess_question_responsiveness(verified_data['reasoning_trace'], question_data),
                'fact_verified': True,
                'verification_improvements': verified_data.get('improvements_made', []),
                'timing': {
                    'total_time': round(total_time, 2),
                    'question_generation_time': round(question_time, 2),
                    'reasoning_generation_time': round(reasoning_time, 2),
                    'fact_verification_time': round(verification_time, 2),
                    'grading_time': round(grading_time, 2) if grading_time > 0 else None
                }
            }
        }
    
    async def _generate_challenging_question(self, passage: ScientificPassage) -> Dict[str, str]:
        """Generate challenging, logic and reasoning oriented questions (CALL 1)"""
        
        prompt = f"""
You are an expert researcher specializing in creating challenging questions that test deep understanding and reasoning abilities. Your task is to generate ONE exceptional question about this scientific content.

SCIENTIFIC PASSAGE:
Title: {passage.source_title}
Content: {passage.content}
Section Type: {passage.section_type}
Domain: {passage.domain}

TASK: Generate ONE challenging question that emphasizes logic, reasoning, and critical thinking.

QUESTION REQUIREMENTS:
1. DIFFICULTY: Should be challenging for intermediate to advanced researchers
2. LOGIC-ORIENTED: Requires systematic reasoning, not just recall
3. SELF-CONTAINED: Includes all necessary context within the question itself
4. REASONING-DEMANDING: Cannot be answered with simple factual lookup
5. INTELLECTUALLY RIGOROUS: Tests understanding of concepts, relationships, causality

PREFERRED QUESTION TYPES (choose the most challenging option):
- CRITICAL ANALYSIS: Challenge methodology, assumptions, or conclusions with specific technical concerns
- MECHANISTIC REASONING: How/why questions requiring step-by-step logical explanation
- LIMITATION ANALYSIS: Identify and explain constraints, boundaries, or failure modes
- COMPARATIVE REASONING: Contrast approaches, compare alternatives, or evaluate trade-offs
- APPLICATION CHALLENGES: Complex scenarios requiring multi-step reasoning to apply findings

QUESTION DESIGN PRINCIPLES:
- Start with sophisticated framing: "How would you reconcile...", "What explains the apparent contradiction between...", "Given the methodological constraints, how reliable is..."
- Require multi-step logical reasoning
- Force consideration of alternative explanations
- Demand evaluation of evidence quality
- Test understanding of causal relationships
- Challenge assumptions or oversimplifications
- Avoid references to the source text, e.g., "this paper", "the authors", "these results", "the thesis", "the model", etc.

FORMAT YOUR RESPONSE AS:
QUESTION_TYPE: [type from the preferred types above]
DIFFICULTY: advanced
QUESTION: [The challenging question text - should be substantial and sophisticated]

Focus on creating a question that would require an expert to engage in systematic, rigorous reasoning to answer properly.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_question_response(response)
    
    async def _generate_reasoning_trace_for_question(self, passage: ScientificPassage, question_data: Dict[str, str]) -> Dict[str, str]:
        """Generate systematic reasoning trace and answer for a given question (CALL 2)"""
        
        prompt = f"""
You are an expert researcher providing a systematic analysis to answer a challenging question. Show rigorous, step-by-step reasoning that demonstrates scientific thinking at its best.

SCIENTIFIC PASSAGE:
Title: {passage.source_title}
Content: {passage.content}
Section Type: {passage.section_type}
Domain: {passage.domain}

QUESTION TO ANSWER:
Type: {question_data['question_type']}
Difficulty: {question_data['difficulty_level']}
Question: {question_data['question']}

TASK: Provide a comprehensive reasoning trace followed by a clear final answer.  In general this  means:
    1. fully decompose the question into parts
    2. enumerate the parts to make sure you have understood the question in full
    3. identify background information for every part of the question
    4. double-check the background for accuracy, consistency, and relevance
    5. in a logical chain, incorporate the background information in ways that lead to the final answer
    6. double-check the final answer for accuracy, consistency, and relevance
    7. if the final answer is not correct, revise the erroneous parts of the reasoning trace rewrite the final answer



FORMAT YOUR RESPONSE AS:
REASONING:
<thought>
[Natural, flowing expert reasoning that systematically works through the question. Use conversational expert language as if you're thinking through a challenging problem out loud.

**CRITICAL STYLE REQUIREMENTS**:
- Write as a continuous stream of expert consciousness, NOT as structured analysis
- Use natural self-correction phrases woven into the flow: "Wait, let me reconsider...", "Actually, I need to correct this...", "On reflection, this assumption is flawed because..."
- NO structured headings, bullet points, or artificial phrases like "**Recognition of inconsistencies:**"
- Maintain logical progression while keeping it conversational and natural
- Show authentic expert thinking with genuine moments of doubt, correction, and insight]
</thought>

FINAL_ANSWER: [Clear, direct answer that addresses all aspects of the question while acknowledging appropriate uncertainties]

Emphasize natural expert thinking, logical rigor, and authentic self-correction within a conversational, flowing style.
"""
        
        response = await self.llm.generate(prompt, temperature=0.7)
        return self._parse_reasoning_response(response)
    
    def _parse_question_response(self, response: str) -> Dict[str, str]:
        """Parse the question generation response"""
        lines = response.strip().split('\n')
        
        # Initialize defaults
        question_data = {
            'question': "What are the main methodological strengths and limitations of this research approach?",
            'question_type': "critical_analysis",
            'difficulty_level': "advanced"
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('QUESTION_TYPE:'):
                question_data['question_type'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('DIFFICULTY:'):
                question_data['difficulty_level'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('QUESTION:'):
                question_data['question'] = line.split(':', 1)[1].strip()
        
        return question_data
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, str]:
        """Parse the reasoning trace response"""
        lines = response.strip().split('\n')
        
        reasoning_trace = ""
        final_answer = ""
        
        current_section = None
        thought_content = []
        collecting_thought = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('REASONING:'):
                current_section = 'reasoning'
            elif line.startswith('FINAL_ANSWER:'):
                current_section = 'final_answer'
                final_answer = line.split(':', 1)[1].strip()
            elif line.startswith('<thought>'):
                collecting_thought = True
                thought_line = line.replace('<thought>', '').strip()
                if thought_line:
                    thought_content.append(thought_line)
            elif line.startswith('</thought>'):
                collecting_thought = False
                reasoning_trace = '\n'.join(thought_content)
            elif collecting_thought:
                thought_content.append(line)
            elif current_section == 'final_answer' and line:
                if final_answer:
                    final_answer += ' ' + line
                else:
                    final_answer = line
        
        # If no reasoning trace was found in <thought> tags, extract from REASONING section
        if not reasoning_trace and current_section == 'reasoning':
            reasoning_start = response.find('REASONING:')
            final_answer_start = response.find('FINAL_ANSWER:')
            if reasoning_start != -1:
                if final_answer_start != -1:
                    reasoning_section = response[reasoning_start:final_answer_start]
                else:
                    reasoning_section = response[reasoning_start:]
                reasoning_trace = reasoning_section.replace('REASONING:', '').strip()
        
        # Fallback if parsing fails
        if not reasoning_trace:
            reasoning_trace = "The analysis requires systematic evaluation of the available evidence and methodological approach to draw appropriate conclusions."
        if not final_answer:
            final_answer = "Based on the systematic analysis, the evidence supports the main conclusions while acknowledging the identified limitations."
        
        return {
            'reasoning_trace': reasoning_trace,
            'final_answer': final_answer
        }
    
    async def _grade_final_answer_consolidated(self, question: Dict[str, Any], final_answer: str, source_text: str, reasoning_trace: str) -> Dict[str, float]:
        """Consolidated answer grading in a single call - reusing the efficient method from original"""
        
        prompt = f"""
Evaluate this answer across 6 dimensions and provide scores (scale 0.0-1.0) for each:

QUESTION: {question['question']}
QUESTION TYPE: {question['question_type']}
DIFFICULTY LEVEL: {question['difficulty_level']}

ANSWER: {final_answer}

SOURCE TEXT: {source_text[:1000]}...

REASONING TRACE (for context): {reasoning_trace[:500]}...

Rate the answer on these 6 dimensions (0.0-1.0 scale):

1. QUESTION_ALIGNMENT: How well does the answer directly address the specific question asked?
2. SCIENTIFIC_ACCURACY: How factually accurate and scientifically correct is the answer?
3. COMPLETENESS: How thoroughly does the answer address the question?
4. UNCERTAINTY_CALIBRATION: How well does it express appropriate uncertainty/confidence?
5. CLARITY_STRUCTURE: How clear, organized, and readable is the answer?
6. EVIDENCE_USAGE: How effectively does it use evidence from source material?

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
                        continue
        except Exception as e:
            print(f"Warning: Error parsing consolidated grades: {e}")
        
        # Calculate overall score (exclude the overall_score itself from calculation)
        core_grades = [v for k, v in grades.items() if k != 'overall_score']
        grades['overall_score'] = sum(core_grades) / len(core_grades)
        
        return grades
    
    def _calculate_logical_rigor_score(self, trace: str) -> float:
        """Calculate logical rigor based on systematic reasoning and self-correction patterns"""
        
        # Core logical reasoning patterns
        logical_patterns = [
            'therefore', 'however', 'given that', 'this leads to', 'consequently',
            'specifically', 'checking', 'verifying', 'alternatively', 'precisely',
            'as a result', 'in contrast', 'furthermore', 'systematic'
        ]
        
        # Self-correction patterns (weighted more heavily for rigor)
        self_correction_patterns = [
            'wait, let me', 'actually', 'on second thought', 'let me reconsider',
            'i need to revise', 'this contradicts', 'let me check', 'i initially missed',
            'on reflection', 'but what if', 'let me double-check', 'i see the flaw'
        ]
        
        # Calculate scores
        logical_score = sum(1 for pattern in logical_patterns if pattern in trace.lower())
        correction_score = sum(1 for pattern in self_correction_patterns if pattern in trace.lower())
        
        # Weight self-correction more heavily (shows rigorous thinking)
        total_score = logical_score + (correction_score * 1.5)
        max_possible = len(logical_patterns) + (len(self_correction_patterns) * 1.5)
        
        return min(1.0, total_score / max_possible)
    
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
        
        if any(word in trace.lower() for word in ['hypothesis', 'hypothesize']):
            patterns.append('hypothesis_generation')
        if any(word in trace.lower() for word in ['evidence', 'data', 'result']):
            patterns.append('evidence_evaluation')
        if any(word in trace.lower() for word in ['method', 'approach', 'design']):
            patterns.append('methodology_critique')
        if any(word in trace.lower() for word in ['connect', 'relate', 'similar']):
            patterns.append('connection_making')
        # Enhanced self-correction detection
        self_correction_indicators = [
            'wait, let me', 'actually', 'on second thought', 'let me reconsider',
            'i need to revise', 'this contradicts', 'let me check', 'i initially missed',
            'on reflection', 'but what if', 'let me double-check', 'i see the flaw',
            'however', 'alternatively', 'correction', 'revise', 'reconsider'
        ]
        if any(indicator in trace.lower() for indicator in self_correction_indicators):
            patterns.append('self_correction')
        if any(word in trace.lower() for word in ['systematic', 'step', 'first', 'next']):
            patterns.append('systematic_analysis')
            
        return patterns
    
    def _assess_question_responsiveness(self, trace: str, question: Dict[str, Any]) -> float:
        """Assess how well the trace responds to the specific question."""
        question_words = set(question['question'].lower().split())
        trace_words = set(trace.lower().split())
        
        # Simple overlap metric - in practice you'd want semantic similarity
        overlap = len(question_words.intersection(trace_words))
        return min(1.0, overlap / max(len(question_words), 1))
    
    async def _verify_and_refine_accuracy(self, passage: ScientificPassage, question_data: Dict[str, str], reasoning_data: Dict[str, str]) -> Dict[str, str]:
        """CALL 3: Fact verification and accuracy refinement to improve scientific accuracy"""
        
        prompt = f"""
You are a scientific fact-checker tasked with verifying and refining the accuracy of reasoning and conclusions. Review the reasoning trace and final answer for factual accuracy, then provide corrections.

ORIGINAL SCIENTIFIC PASSAGE:
{passage.content}

QUESTION: {question_data['question']}

REASONING TRACE TO VERIFY:
{reasoning_data['reasoning_trace']}

FINAL ANSWER TO VERIFY:
{reasoning_data['final_answer']}

YOUR TASK:
1. FACT VERIFICATION: Identify any factual errors, misinterpretations, or unsupported claims in the reasoning and answer
2. ACCURACY REFINEMENT: Correct errors and improve precision of scientific statements
3. EVIDENCE ALIGNMENT: Ensure all claims are properly supported by evidence from the passage
4. COMPLETENESS CHECK: Identify any important factual information that was missed

SPECIFIC FOCUS AREAS:
- Scientific terminology and concepts used correctly
- Numerical values, measurements, and statistical claims accurate
- Methodological details correctly represented
- Causal relationships properly established
- Limitations and uncertainties appropriately acknowledged
- Claims properly scoped to available evidence

FORMAT YOUR RESPONSE AS:
IMPROVEMENTS_MADE: [List specific corrections made, or "No significant errors found"]

VERIFIED_REASONING:
<thought>
[The corrected and refined reasoning trace with improved factual accuracy. 

**CRITICAL**: Maintain the natural flow and conversational style of expert thinking. Do NOT insert structured headings, bullet points, or artificial phrases like "**Recognition of logical inconsistencies:**". Instead, weave corrections seamlessly into the natural thought process using phrases like "Actually, let me reconsider this..." or "Wait, I think I made an error here..." Keep the reasoning as a flowing stream of expert consciousness.]
</thought>

VERIFIED_ANSWER: [The corrected final answer with improved accuracy and precision]

Focus on factual correctness while preserving the natural, flowing style of expert reasoning. Make corrections feel like authentic moments of self-reflection, not structured analysis.
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)  # Lower temperature for accuracy
        return self._parse_verification_response(response, reasoning_data)
    
    def _parse_verification_response(self, response: str, original_data: Dict[str, str]) -> Dict[str, str]:
        """Parse the verification response and extract improvements"""
        lines = response.strip().split('\n')
        
        verified_reasoning = original_data['reasoning_trace']  # Fallback to original
        verified_answer = original_data['final_answer']  # Fallback to original
        improvements_made = []
        
        current_section = None
        thought_content = []
        collecting_thought = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('IMPROVEMENTS_MADE:'):
                improvements_text = line.split(':', 1)[1].strip()
                if improvements_text and improvements_text != "No significant errors found":
                    improvements_made.append(improvements_text)
            elif line.startswith('VERIFIED_REASONING:'):
                current_section = 'reasoning'
            elif line.startswith('VERIFIED_ANSWER:'):
                current_section = 'answer'
                verified_answer = line.split(':', 1)[1].strip()
            elif line.startswith('<thought>'):
                collecting_thought = True
                thought_line = line.replace('<thought>', '').strip()
                if thought_line:
                    thought_content.append(thought_line)
            elif line.startswith('</thought>'):
                collecting_thought = False
                verified_reasoning = '\n'.join(thought_content)
            elif collecting_thought:
                thought_content.append(line)
            elif current_section == 'answer' and line:
                if verified_answer:
                    verified_answer += ' ' + line
                else:
                    verified_answer = line
        
        return {
            'reasoning_trace': verified_reasoning,
            'final_answer': verified_answer,
            'improvements_made': improvements_made
        }

# Reuse the BatchProcessor from the original with minimal modifications
class EfficientBatchProcessor:
    def __init__(self, generator: EfficientReasoningTraceGenerator, quality_check: bool = False, 
                 min_quality_score: float = 6.0, quality_judges: List[str] = None):
        self.generator = generator
        self.quality_check = quality_check
        self.min_quality_score = min_quality_score
        self.quality_judges = quality_judges or ['claude-sonnet-3.7', 'gpt-4o']
        self.evaluator = LLMJudgeEvaluator() if quality_check else None
    
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
            ratings = await self.evaluator.evaluate_triplet(triplet, self.quality_judges)
            valid_ratings = [r for r in ratings if r.overall_score > 0]
            
            if not valid_ratings:
                print(f"    ⚠️ Quality check failed - no valid ratings")
                return 0.0
            
            stats = self.evaluator.calculate_statistics(valid_ratings)
            avg_score = stats['overall_score']['mean']
            
            print(f"    📊 Quality scores: {[r.overall_score for r in valid_ratings]} → avg: {avg_score:.1f}")
            
            return avg_score
            
        except Exception as e:
            print(f"    ❌ Quality evaluation error: {e}")
            return 0.0
    
    async def process_paper_corpus(
        self, 
        papers: List[ScientificPassage], 
        traces_per_paper: int = 3,
        traces_per_question: int = 1,
        output_file: str = None,
        save_incrementally: bool = True
    ) -> List[Dict[str, Any]]:
        """Process corpus with efficient reasoning trace generation"""
        
        all_traces = []
        
        # Load existing traces if file exists
        if output_file and save_incrementally:
            existing_traces = self._load_existing_traces(output_file)
            if existing_traces:
                print(f"Loaded {len(existing_traces)} existing traces from {output_file}")
                all_traces.extend(existing_traces)
        
        for i, paper in enumerate(papers):
            print(f"Processing {i+1}/{len(papers)}: {paper.source_title}")
            
            if traces_per_question > 1:
                # Generate multiple traces using temperature variation
                print(f"  Generating {traces_per_question} traces with temperature variation")
                
                for trace_idx in range(traces_per_question):
                    print(f"    🎯 Generating trace {trace_idx + 1}/{traces_per_question}")
                    
                    max_attempts = 2
                    accepted_trace = None
                    
                    for attempt in range(max_attempts):
                        try:
                            # Vary temperature for diversity
                            temperature_variation = 0.7 + (trace_idx * 0.1)  # 0.7, 0.8, 0.9, etc.
                            
                            trace = await self.generator.generate_reasoning_trace(paper)
                            
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
                                    if attempt < max_attempts - 1:
                                        print(f"      🔄 Retrying...")
                                    continue
                            else:
                                accepted_trace = trace
                                break
                                
                        except Exception as e:
                            print(f"      ❌ Error in attempt {attempt + 1}: {e}")
                            if attempt < max_attempts - 1:
                                print(f"      🔄 Retrying...")
                            continue
                    
                    if accepted_trace:
                        accepted_trace['sample_index'] = len(all_traces)
                        accepted_trace['metadata']['question_variation_index'] = trace_idx
                        accepted_trace['metadata']['same_question_group'] = True
                        
                        if self.quality_check:
                            accepted_trace['metadata']['quality_checked'] = True
                            accepted_trace['metadata']['min_quality_threshold'] = self.min_quality_score
                        
                        all_traces.append(accepted_trace)
                        print(f"      ✅ Accepted trace {trace_idx + 1}")
                        
                        # Save incrementally
                        if output_file and save_incrementally:
                            self._save_traces_incrementally(all_traces, output_file)
                            print(f"      💾 Saved {len(all_traces)} traces to {output_file}")
                    else:
                        print(f"      ❌ Failed to generate acceptable trace {trace_idx + 1}")
            
            else:
                # Generate single trace per paper
                accepted_trace = None
                max_attempts = 3
                attempt = 0
                
                while accepted_trace is None and attempt < max_attempts:
                    attempt += 1
                    try:
                        print(f"  Attempt {attempt}/{max_attempts}")
                        
                        # Generate single trace
                        trace = await self.generator.generate_reasoning_trace(paper)
                        
                        print(f"    Generated trace (logical rigor: {trace['metadata']['logical_rigor_score']:.2f})")
                        
                        # Quality check if enabled
                        if self.quality_check:
                            quality_score = await self._evaluate_trace_quality(trace)
                            
                            if quality_score >= self.min_quality_score:
                                print(f"    ✅ Quality check passed ({quality_score:.1f} >= {self.min_quality_score})")
                                accepted_trace = trace
                            else:
                                print(f"    ❌ Quality check failed ({quality_score:.1f} < {self.min_quality_score})")
                                if attempt < max_attempts:
                                    print(f"    🔄 Retrying...")
                                continue
                        else:
                            accepted_trace = trace
                            
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
                    
                    # Save incrementally
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
                "status": "in_progress",
                "generator_version": "efficient_v2"
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
            try:
                import os
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
                "status": "completed",
                "generator_version": "efficient_v2"
            },
            "questions": traces
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Final dataset saved with {len(traces)} traces")

# Reuse utility functions from original
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
    
    # Check for abstracts (information-light)
    abstract_indicators = ['abstract', 'summary', 'synopsis']
    if any(indicator in text_lower[:50] for indicator in abstract_indicators):
        # Check if it's likely an abstract (brief, high-level summary)
        if len(text) < 1000 and any(phrase in text_lower for phrase in [
            'this study', 'this research', 'we present', 'we investigate',
            'our findings', 'in conclusion', 'these results'
        ]):
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

def convert_texts_to_passages(texts: List[str]) -> List[ScientificPassage]:
    """Convert text chunks to ScientificPassage objects for processing"""
    passages = []
    
    for i, text in enumerate(texts):
        # Detect section type based on content patterns (basic heuristics)
        section_type = _detect_section_type(text)
        
        passages.append(ScientificPassage(
            content=text,
            section_type=section_type,
            source_title=f"Scientific Paper {i+1}",
            domain="General Science",
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
        return 'mixed'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces efficiently from scientific text")
    
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
        help="Input JSONL file with text chunks in 'text' field"
    )
    
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to generate for testing"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reasoning_training_data_efficient.json",
        help="Output file for generated traces"
    )
    
    parser.add_argument(
        "--traces-per-sample",
        type=int,
        default=1,
        help="Number of reasoning traces to generate per text sample"
    )
    
    parser.add_argument(
        "--traces-per-question",
        type=int,
        default=1,
        help="Number of reasoning traces to generate per question using temperature variation"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing output file"
    )
    
    parser.add_argument(
        "--grade-answers",
        action="store_true",
        help="Grade final answers for RL training (adds 1 LLM call per trace)"
    )
    
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Enable quality checking with LLM judges before accepting traces"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=6.0,
        help="Minimum quality score (1-10) required to accept a trace"
    )
    
    parser.add_argument(
        "--quality-judges", 
        nargs="+",
        choices=['claude-sonnet-3.7', 'gemini-flash', 'gpt-4.1', 'gpt-4o', 'gpt-o4-mini'],
        default=['claude-sonnet-3.7', 'gpt-4o'],
        help="Judge models to use for quality checking"
    )
    
    parser.add_argument(
        "--no-input-filter",
        action="store_true",
        help="Disable input text quality filtering (process all text regardless of quality)"
    )
    
    return parser.parse_args()

async def main():
    """Main function with command line argument support"""
    args = parse_arguments()
    
    print(f"Efficient Synthetic Reasoning Trace Generator v2")
    print(f"Model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Max traces: {args.max_traces or 'All'}")
    print(f"Output file: {args.output}")
    print(f"Performance: 2-3 LLM calls per trace (vs 8-10 in original)")
    print("-" * 50)
    
    # Load data from JSONL file with quality filtering
    apply_filter = not args.no_input_filter
    texts = load_data_from_jsonl(args.input_file, args.max_traces, apply_quality_filter=apply_filter)
    
    if apply_filter:
        print("✅ Input quality filtering enabled (use --no-input-filter to disable)")
    else:
        print("⚠️ Input quality filtering disabled - processing all text samples")
    
    if not texts:
        print("No text data loaded. Exiting.")
        return
    
    # Convert texts to ScientificPassage objects
    papers = convert_texts_to_passages(texts)
    
    # Initialize components with specified model
    try:
        llm = ArgoLLMAdapter(args.model)
        generator = EfficientReasoningTraceGenerator(llm, grade_answers=args.grade_answers)
        processor = EfficientBatchProcessor(
            generator, 
            quality_check=args.quality_check,
            min_quality_score=args.min_quality_score,
            quality_judges=args.quality_judges
        )
        
        print(f"Initialized LLM with model: {args.model}")
        if args.grade_answers:
            print("Answer grading enabled (adds 1 LLM call per trace)")
        if args.quality_check:
            print(f"Quality checking enabled (min score: {args.min_quality_score})")
            print(f"Quality judges: {', '.join(args.quality_judges)}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return
    
    # Determine resume behavior
    resume_enabled = not args.no_resume
    
    # Generate traces with incremental saving
    print(f"Generating reasoning traces for {len(papers)} text samples...")
    print(f"Progress will be saved incrementally to: {args.output}")
    if resume_enabled:
        print("(Resumable - progress saved after each trace)")
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
    
    print(f"✅ Generated {len(traces)} reasoning traces efficiently")
    print(f"Performance: ~{2 + (1 if args.grade_answers else 0)} LLM calls per trace")
    print(f"Final results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())