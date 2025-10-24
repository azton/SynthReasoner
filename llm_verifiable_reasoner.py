#!/usr/bin/env python3
"""
Verifiable Response Synthetic Reasoning Trace Generator

Generates realistic reasoning traces with verifiable final answers from scientific text.
Focuses on creating questions where the answer is numerical, code, single unambiguous word, 
equation, or otherwise objectively verifiable.

Usage examples:
    # Use default settings (5 samples for testing)
    python llm_verifiable_reasoner.py --max-traces 5
    
    # Specify different model and input file
    python llm_verifiable_reasoner.py --model gpt4o --input-file my_data.jsonl --max-traces 3
    
    # Enable quality checking for higher quality traces (slower)
    python llm_verifiable_reasoner.py --quality-check --min-quality-score 6.5 --max-traces 5
    
    # Process all samples with different output file
    python llm_verifiable_reasoner.py --output my_verifiable_traces.json
    
    # Start fresh (ignore existing output file)
    python llm_verifiable_reasoner.py --no-resume --max-traces 5
    
    # Resume from previous run (default behavior)
    python llm_verifiable_reasoner.py --max-traces 10
"""

import asyncio
import random
import argparse
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

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
    from .llm_synthetic_reasoner import (
        LLMInterface, ArgoLLMAdapter, ScientificPassage, UserQuestion, 
        QuestionContext, ReasoningFragment, load_data_from_jsonl, 
        convert_texts_to_passages, is_high_quality_scientific_text
    )
except ImportError:
    from llm_interface import UnifiedLLMClient
    from llm_judge_evaluator import LLMJudgeEvaluator, QRSTriplet
    from llm_synthetic_reasoner import (
        LLMInterface, ArgoLLMAdapter, ScientificPassage, UserQuestion,
        QuestionContext, ReasoningFragment, load_data_from_jsonl,
        convert_texts_to_passages, is_high_quality_scientific_text
    )

class VerificationMethod(Enum):
    """Enumeration of different verification methods for answers"""
    NUMERICAL = "numerical"           # Numbers, percentages, counts, measurements
    MATHEMATICAL = "mathematical"     # Equations, formulas, calculations
    CODE = "code"                    # Programming code, algorithms, syntax
    SINGLE_WORD = "single_word"      # Unambiguous single terms, names, classifications
    LIST_ORDERED = "list_ordered"    # Ordered lists with specific sequence
    LIST_UNORDERED = "list_unordered" # Unordered sets of items
    CATEGORICAL = "categorical"      # Specific categories, classifications, types

@dataclass
class VerifiableUserQuestion(UserQuestion):
    """Extended UserQuestion with verification requirements"""
    verification_method: VerificationMethod
    expected_answer_format: str  # Description of expected format
    verification_criteria: str   # How to verify the answer

@dataclass
class VerifiableAnswer:
    """Container for a verifiable answer with metadata"""
    content: str
    verification_method: VerificationMethod
    verification_criteria: str
    confidence_level: float
    reasoning_justification: str

class VerifiableReasoningTraceGenerator:
    """Generator for reasoning traces with verifiable final answers"""
    
    def __init__(self, llm: LLMInterface, grade_answers: bool = False):
        self.llm = llm
        self.grade_answers = grade_answers
        
        # Question types optimized for verifiable answers
        self.verifiable_question_types = [
            'quantitative_analysis',   # How many, what percentage, what value
            'calculation_based',       # Compute, calculate, determine numerically  
            'classification',          # Categorize, classify, identify type
            'identification',          # Name, identify specific terms/concepts
            'comparison_ranking',      # Which is greater, rank in order
            'code_generation',         # Write code, algorithm, pseudocode
            'formula_derivation',      # Derive equation, write formula
        ]
        
        self.logical_rigor_markers = [
            "therefore", "however", "given that", "this leads to", "consequently",
            "alternatively", "nevertheless", "furthermore", "specifically", "precisely",
            "in contrast", "as a result", "checking this against", "verifying that",
            "calculating", "computing", "measuring", "quantifying", "determining"
        ]
    
    async def generate_verifiable_reasoning_trace(self, passage: ScientificPassage) -> Dict[str, Any]:
        """Generate a complete verifiable question-response reasoning trace from scientific text."""
        
        start_time = time.time()
        print(f"  🚀 Starting verifiable trace generation for: {passage.source_title[:50]}...")
        
        # Stage 0: Generate verifiable user questions
        stage_start = time.time()
        verifiable_questions = await self._generate_verifiable_user_questions(passage)
        selected_question = self._select_verifiable_question(verifiable_questions, passage)
        print(f"     Verifiable question generation: {time.time() - stage_start:.1f}s")
        
        return await self.generate_verifiable_trace_for_question(passage, selected_question)
    
    async def generate_verifiable_trace_for_question(
        self, 
        passage: ScientificPassage, 
        question: VerifiableUserQuestion
    ) -> Dict[str, Any]:
        """Generate a verifiable reasoning trace for a specific question."""
        
        trace_start_time = time.time()
        
        # Stage 1: Generate question-specific reasoning fragments (3 parallel calls)
        stage_start = time.time()
        fragments = await asyncio.gather(
            self._generate_verifiable_hypotheses(question, passage),
            self._evaluate_evidence_for_verifiable_answer(question, passage),
            self._consider_verification_requirements(question, passage)
        )
        fragment_time = time.time() - stage_start
        
        # Stage 2: Analyze context and assemble verifiable trace
        stage_start = time.time()
        raw_trace = await self._analyze_context_and_assemble_verifiable_trace(
            question, fragments, passage
        )
        assembly_time = time.time() - stage_start
        
        # Stage 3: Extract and verify final answer
        stage_start = time.time()
        verifiable_answer = await self._extract_and_verify_final_answer(raw_trace, question)
        extraction_time = time.time() - stage_start
        
        # Grade the final answer for RL training (if enabled)
        answer_grades = None
        grading_time = 0.0
        if self.grade_answers:
            stage_start = time.time()
            print(f"  Grading verifiable answer...")
            answer_grades = await self._grade_verifiable_answer(
                question, 
                verifiable_answer, 
                passage.content, 
                raw_trace
            )
            grading_time = time.time() - stage_start
        
        total_time = time.time() - trace_start_time
        
        print(f"  ✅ Total verifiable trace time: {total_time:.1f}s (fragments: {fragment_time:.1f}s, assembly: {assembly_time:.1f}s, extraction: {extraction_time:.1f}s{f', grading: {grading_time:.1f}s' if grading_time > 0 else ''})")
        
        return {
            'source_paper': {
                'title': passage.source_title,
                'domain': passage.domain
            },
            'user_interaction': {
                'question': question.question,
                'question_type': question.question_type,
                'difficulty_level': question.difficulty_level,
                'verification_method': question.verification_method.value,
                'expected_answer_format': question.expected_answer_format,
                'verification_criteria': question.verification_criteria
            },
            'reasoning_trace': raw_trace,
            'verifiable_answer': {
                'content': verifiable_answer.content,
                'verification_method': verifiable_answer.verification_method.value,
                'verification_criteria': verifiable_answer.verification_criteria,
                'confidence_level': verifiable_answer.confidence_level,
                'reasoning_justification': verifiable_answer.reasoning_justification
            },
            'answer_grades': answer_grades,
            'metadata': {
                'logical_rigor_score': self._calculate_logical_rigor_score(raw_trace),
                'complexity_level': self._assess_complexity(raw_trace),
                'reasoning_patterns': self._identify_patterns(raw_trace),
                'question_responsiveness': self._assess_question_responsiveness(raw_trace, question),
                'verifiability_score': self._calculate_verifiability_score(verifiable_answer),
                'timing': {
                    'total_time': round(total_time, 2),
                    'fragment_generation_time': round(fragment_time, 2),
                    'context_assembly_time': round(assembly_time, 2),
                    'answer_extraction_time': round(extraction_time, 2),
                    'grading_time': round(grading_time, 2) if grading_time > 0 else None
                }
            }
        }
    
    async def _generate_verifiable_user_questions(self, passage: ScientificPassage) -> List[VerifiableUserQuestion]:
        """Generate questions designed to have verifiable answers"""
        prompt = f"""
Generate challenging questions that would be appropriate for CAREER RESEARCHERS in this field. Questions should be INSPIRED BY the source material but create original, difficult problems that test deep understanding and analytical skills.

Source Title: {passage.source_title}
Content: {passage.content}

APPROACH: Use the source material as inspiration to create challenging problems that:
1. Build upon concepts, methods, or parameters mentioned in the source
2. Create novel scenarios that extend beyond what's directly stated
3. Require sophisticated reasoning that would challenge domain experts
4. Are completely self-contained with all necessary information embedded

DIFFICULTY LEVEL: Problems should be challenging enough for researchers with PhDs in the relevant field.

Create 6-8 questions across these VERIFIABLE categories (excluding YES/NO questions):

NUMERICAL (answer: calculated values requiring sophisticated multi-step reasoning):
- "Assume a future CMB mission improves constraints by factor 5 over current limits. If current 95% upper limit on interaction parameter is 2.0x10^-9, calculate the projected new limit and percentage improvement"
- "A cosmological survey detects 1847 Type Ia supernovae across redshift range z=0.1-1.2. If systematic uncertainties scale as sigma_sys = 0.02*sqrt(N) and statistical errors follow Poisson statistics, calculate the total uncertainty on the dark energy equation of state parameter w"
- "Given primordial gravitational wave amplitude A_T = 0.01 at pivot scale k_0 = 0.002 Mpc^-1 with spectral index n_T = -0.12, calculate the tensor-to-scalar ratio r and energy scale of inflation assuming slow-roll approximation"
- "A galaxy cluster analysis uses 2500 member galaxies with velocity dispersion sigma_v = 1200±50 km/s. Calculate the virial mass M_200 and estimate systematic uncertainty from triaxiality assuming prolate ellipsoid with axis ratio 2:1"
- "For pulsar timing array detecting nanohertz gravitational waves, calculate required timing precision to achieve SNR=5 detection of stochastic background with characteristic strain h_c = 2x10^-15 at f=1 yr^-1 using 40 pulsars over 15 years"

MATHEMATICAL (answer: equations, formulas requiring advanced derivation and field expertise):
- "For scalar field inflation with potential V(phi) = (1/2)*m^2*phi^2 + lambda*phi^4/4, derive the slow-roll parameters epsilon and eta, then find conditions for graceful exit from inflation and calculate the total number of e-foldings N_e"
- "In modified gravity f(R) cosmology, derive the effective dark energy density and pressure from the field equations, then show that f(R) = R + alpha*R^2 produces accelerated expansion equivalent to LambdaCDM at late times"
- "For weak gravitational lensing, derive the convergence kappa from the deflection angle alpha, then show how the reduced shear g relates to the intrinsic ellipticity distribution and calculate the cosmic shear power spectrum C_l^(gamma-gamma)"
- "In supersymmetric dark matter models, derive the relic abundance Omega*h^2 from the Boltzmann equation for a Majorana fermion chi with s-wave annihilation cross-section <sigma*v> = a + b*<v^2>, including co-annihilation effects"

CLASSIFICATION (answer: specific technical category requiring expert knowledge):
- "Based on the observational signatures described, classify the dark matter candidate as: sterile neutrino, axion, WIMP, or primordial black hole"
- "Given the spectral energy distribution and variability timescales, classify the active galactic nucleus as: Seyfert 1, Seyfert 2, blazar, or radio galaxy"
- "From the stellar population synthesis models and metallicity gradients, classify the galaxy formation scenario as: monolithic collapse, hierarchical assembly, or major merger"

IDENTIFICATION (answer: specific technical term or precise concept):
- "What is the technical term for the phenomenon where quantum fluctuations during inflation become classical density perturbations?"
- "Identify the specific symmetry principle that leads to the conservation of baryon number in the Standard Model"
- "What is the precise name for the statistical measure that quantifies the deviation from Gaussianity in CMB temperature fluctuations?"


CODE/ALGORITHM (answer: sophisticated algorithmic implementation):
- "Write a Python function to implement the Limber approximation for calculating angular power spectra C_ℓ from 3D matter power spectra P(k,z), including proper integration over radial selection functions"
- "Design an algorithm to perform Bayesian parameter estimation for a 6-parameter cosmological model using MCMC sampling with Metropolis-Hastings acceptance, including adaptive step-size tuning"
- "Implement pseudocode for the Voronoi tessellation-based galaxy clustering analysis that accounts for edge effects and survey geometry in the BOSS-like survey footprint"

IMPORTANT Guidelines for RESEARCHER-LEVEL Problems:
- Make questions COMPLETELY SELF-CONTAINED with all necessary information
- CREATE ORIGINAL problems INSPIRED BY the source material, don't just extract data
- Questions should challenge someone with PhD-level expertise in the field
- For NUMERICAL problems: Require sophisticated multi-step reasoning and domain knowledge
- For MATHEMATICAL problems: Require advanced derivations and theoretical understanding
- All required data, constants, and formulas must be embedded in the question
- NEVER reference equations, figures, tables, or sections from the source material
- Problems should synthesize multiple advanced concepts from the field
- Include realistic but challenging parameters that test deep understanding
- Ensure answers can be OBJECTIVELY VERIFIED by domain experts
- Focus on problems that would appear in advanced graduate courses or research papers

AVOID creating questions like:
- "What was the sample size?" (simple lookup)
- "Calculate ν = √(3ξ/2) for ξ = 20" (simple substitution)
- "What is the value of x when y = 5?" (trivial calculation)
- "Derive the expression given in equation (3)" (references source material)
- "Using the data from Table 2" (references source material)

INSTEAD CREATE advanced problems like:
- "For a CMB mission detecting spectral distortions, calculate the required sensitivity to constrain primordial black hole abundance in the mass range 10^15-10^17 g, assuming standard thermal history and accounting for Silk damping effects"
- "In a galaxy survey with photometric redshift uncertainties sigma_z = 0.03(1+z), derive the optimal binning strategy to minimize information loss for weak lensing tomography with 5 redshift bins over z=0.1-2.0"
- "For superfluid dark matter with self-interaction cross-section sigma/m = 0.1 cm^2/g, calculate the core radius and central density profile of a dwarf galaxy halo with M_vir = 10^10 M_sun, including quantum pressure effects"

Format each question using XML tags:

<question>
<type>NUMERICAL|MATHEMATICAL|CLASSIFICATION|IDENTIFICATION|CODE</type>
<level>Expert</level>
<verification>ADVANCED_CALCULATION|DERIVATION|etc</verification>
<text>
Question text goes here (can span multiple lines)
</text>
<format>Expected answer format description</format>
<criteria>How to verify correctness</criteria>
</question>

Example:
<question>
<type>NUMERICAL</type>
<level>Expert</level>
<verification>ADVANCED_CALCULATION</verification>
<text>
A dark matter direct detection experiment observes 127 nuclear recoil events in the energy range 2-50 keV over 365.25 days with 1.2 tonne fiducial mass. Given background expectation of 0.8±0.1 events/keV/tonne/year and detector efficiency epsilon(E) = 0.9*(E/keV)^0.3 for E>2 keV, calculate the 90% CL upper limit on the WIMP-nucleon spin-independent cross-section for m_chi = 100 GeV, including systematic uncertainties.
</text>
<format>Cross-section limit in cm^2 with uncertainties</format>
<criteria>Must include proper Poisson statistics, efficiency corrections, and systematic uncertainty propagation</criteria>
</question>
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_verifiable_user_questions(response)
    
    async def _generate_verifiable_hypotheses(self, question: VerifiableUserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        """Generate targeted hypotheses for answering verifiable questions"""
        prompt = f"""
Think through how to systematically approach this verifiable question. Show your reasoning process:

User Question: {question.question}
Verification Method: {question.verification_method.value}
Expected Answer Format: {question.expected_answer_format}
Verification Criteria: {question.verification_criteria}

Source Context: {passage.content}

Consider systematically:
- What specific information from the source material is needed to answer this?
- What calculation, identification, or analysis steps are required?
- How can I ensure the answer meets the verification criteria?
- What potential sources of error or ambiguity should I watch for?
- What specific evidence would confirm or refute different possible answers?

For {question.verification_method.value} questions specifically:
- What exact format should the final answer take?
- How can I double-check my work?
- What would make this answer objectively verifiable?

Think step by step: "To answer this verifiable question, I need to... The key information appears to be... I should look for... The answer format requires..."

Show your systematic working through of how to reliably produce a verifiable answer.
"""
        
        response = await self.llm.generate(prompt, temperature=0.9)
        return ReasoningFragment(
            content=response,
            type='verifiable_hypothesis',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _evaluate_evidence_for_verifiable_answer(self, question: VerifiableUserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        """Evaluate evidence specifically for generating verifiable answers"""
        prompt = f"""
Systematically evaluate the source evidence to support a verifiable answer to this question:

User Question: {question.question}
Verification Method: {question.verification_method.value}
Expected Format: {question.expected_answer_format}

Source Evidence: {passage.content}

Analyze systematically:

EVIDENCE LOCATION:
- Where exactly in the source material is the relevant information?
- Is the information explicitly stated or must it be calculated/inferred?
- Are there multiple pieces of evidence that need to be combined?

VERIFICATION READINESS:
- Is the evidence sufficient to generate a {question.verification_method.value} answer?
- What specific data points, numbers, or facts are available?
- How clear and unambiguous is the source information?

CALCULATION/DERIVATION NEEDS:
- Does the answer require mathematical computation from the given data?
- What formulas or algorithms would be needed?
- Are all necessary inputs available in the source material?

ANSWER CONFIDENCE:
- How certain can I be about the verifiable answer based on this evidence?
- What assumptions would I need to make, if any?
- Are there alternative interpretations that could affect the answer?

Show your systematic evaluation of whether and how the evidence supports generating a reliable, verifiable answer.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='verifiable_evidence_eval',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _consider_verification_requirements(self, question: VerifiableUserQuestion, passage: ScientificPassage) -> ReasoningFragment:
        """Consider the specific verification requirements for the answer"""
        prompt = f"""
Think about what it will take to make this answer truly verifiable and how to meet those requirements:

User Question: {question.question}
Verification Method: {question.verification_method.value}
Expected Format: {question.expected_answer_format}
Verification Criteria: {question.verification_criteria}

Consider verification requirements:

FORMAT COMPLIANCE:
- What exact format must the final answer take?
- How should numbers be presented (decimals, percentages, units)?
- What level of precision is appropriate?
- Should the answer be a single value or include ranges/confidence intervals?

TRACEABILITY:
- How can someone verify this answer against the source material?
- What specific passages or data points should they check?
- What calculation steps need to be clearly documented?

ERROR PREVENTION:
- What common mistakes could lead to an incorrect answer?
- How can I double-check my reasoning and calculations?
- What would make this answer unambiguous and objective?

VERIFICATION PROCESS:
- What would an independent reviewer need to do to verify this answer?
- What tools or knowledge would they need?
- How could disagreements about the answer be resolved?

Show your thinking about how to ensure the final answer will be genuinely verifiable and meet the specified criteria.
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return ReasoningFragment(
            content=response,
            type='verification_requirements',
            confidence=self._estimate_confidence(response),
            uncertainty_markers=self._find_logical_markers(response)
        )
    
    async def _analyze_context_and_assemble_verifiable_trace(
        self, 
        question: VerifiableUserQuestion, 
        fragments: List[ReasoningFragment], 
        passage: ScientificPassage
    ) -> str:
        """Assemble a reasoning trace focused on producing a verifiable answer"""
        prompt = f"""
Create a systematic reasoning trace that methodically works toward a verifiable answer to this question.

User Question: {question.question}
Verification Method: {question.verification_method.value}
Expected Answer Format: {question.expected_answer_format}
Verification Criteria: {question.verification_criteria}

Reasoning Components:
1. Systematic approach: {fragments[0].content}
2. Evidence evaluation: {fragments[1].content}
3. Verification requirements: {fragments[2].content}

Create a structured <thought>...</thought> section that demonstrates:

SYSTEMATIC ANALYSIS:
- Clear identification of what information is needed from the source material
- Methodical location and extraction of relevant data
- Step-by-step reasoning toward the verifiable answer

CALCULATION/DERIVATION (if needed):
- Show all mathematical steps explicitly
- Explain the logic behind each calculation
- Verify intermediate results along the way

VERIFICATION FOCUS:
- Continuously check that reasoning aligns with verification criteria
- Ensure the approach will produce the required answer format
- Double-check work for potential errors

LOGICAL PROGRESSION:
- Build systematically from evidence to conclusion
- Show how each step contributes to the final verifiable answer
- Include natural error-checking and verification moments

The reasoning should feel like an expert methodically working through a problem where the answer must be objectively correct and verifiable. Focus on precision, systematic thinking, and clear documentation of the reasoning path.

IMPORTANT: Make all reasoning standalone. Avoid phrases like "the paper", "the authors", "the study", "the research", "the manuscript", "the text", or "the document". Instead reference specific data, findings, or information directly.
"""
        
        return await self.llm.generate(prompt, temperature=0.7)
    
    async def _extract_and_verify_final_answer(self, reasoning_trace: str, question: VerifiableUserQuestion) -> VerifiableAnswer:
        """Extract and format the final verifiable answer from the reasoning trace"""
        
        prompt = f"""
Based on the detailed reasoning trace, extract and format the final verifiable answer.

USER QUESTION: {question.question}
VERIFICATION METHOD: {question.verification_method.value}
EXPECTED FORMAT: {question.expected_answer_format}
VERIFICATION CRITERIA: {question.verification_criteria}

REASONING TRACE: {reasoning_trace}

TASK: Extract the final answer and format it according to the verification requirements:

1. FINAL ANSWER: Provide the answer in the exact required format
2. CONFIDENCE LEVEL: Rate confidence 0.0-1.0 based on evidence quality
3. VERIFICATION PATH: Explain how someone could verify this answer
4. REASONING JUSTIFICATION: Summarize the key reasoning steps that led to this answer

Format your response as:
FINAL_ANSWER: [answer in required format]
CONFIDENCE: [0.0-1.0]
VERIFICATION_PATH: [how to verify this answer]
REASONING_JUSTIFICATION: [brief summary of key reasoning - make this standalone without referencing "the paper", "the authors", "the study", etc.]

IMPORTANT: Make all explanations standalone. Avoid phrases like "the paper", "the authors", "the study", "the research", "the manuscript", "the text", or "the document". Instead use specific factual statements that can stand alone.

Be precise and ensure the answer exactly matches the verification criteria and expected format.
"""
        
        response = await self.llm.generate(prompt, temperature=0.4)
        return self._parse_verifiable_answer(response, question)
    
    async def _grade_verifiable_answer(
        self, 
        question: VerifiableUserQuestion, 
        answer: VerifiableAnswer, 
        source_text: str, 
        reasoning_trace: str
    ) -> Dict[str, float]:
        """Grade verifiable answers with additional metrics for verifiability"""
        
        prompt = f"""
Evaluate this verifiable answer across 8 dimensions (scale 0.0-1.0):

QUESTION: {question.question}
VERIFICATION METHOD: {question.verification_method.value}
EXPECTED FORMAT: {question.expected_answer_format}

ANSWER: {answer.content}
CONFIDENCE: {answer.confidence_level}

SOURCE TEXT: {source_text[:1000]}...
REASONING TRACE: {reasoning_trace[:500]}...

Rate the answer on these 8 dimensions:

1. QUESTION_ALIGNMENT: How well does the answer directly address the specific question?
2. SCIENTIFIC_ACCURACY: How factually accurate and scientifically correct is the answer?
3. FORMAT_COMPLIANCE: How well does the answer match the required format specifications?
4. VERIFIABILITY: How easily can this answer be objectively verified against the source?
5. PRECISION: How precise and unambiguous is the answer?
6. CALCULATION_CORRECTNESS: How accurate are any calculations or derivations shown?
7. EVIDENCE_USAGE: How effectively does it use evidence from source material?
8. REASONING_QUALITY: How sound and systematic is the reasoning process?

Provide scores in this format:
QUESTION_ALIGNMENT: X.X
SCIENTIFIC_ACCURACY: X.X
FORMAT_COMPLIANCE: X.X
VERIFIABILITY: X.X
PRECISION: X.X
CALCULATION_CORRECTNESS: X.X
EVIDENCE_USAGE: X.X
REASONING_QUALITY: X.X
"""
        
        response = await self.llm.generate(prompt, temperature=0.3)
        return self._parse_verifiable_grades(response)
    
    def _select_verifiable_question(self, questions: List[VerifiableUserQuestion], passage: ScientificPassage) -> VerifiableUserQuestion:
        """Select the most suitable verifiable question for reasoning trace generation"""
        print(f"  Generated {len(questions)} verifiable questions")
        
        if not questions:
            raise ValueError(f"Failed to generate any verifiable questions for passage: {passage.source_title}. This indicates the LLM failed to create appropriate researcher-level questions inspired by the content.")
        
        # Prioritize questions with clear numerical or mathematical answers
        priority_methods = [
            VerificationMethod.NUMERICAL, 
            VerificationMethod.MATHEMATICAL, 
        ]
        
        priority_questions = [q for q in questions if q.verification_method in priority_methods]
        if priority_questions:
            selected = random.choice(priority_questions)
            print(f"  Selected priority verifiable question: {selected.question[:50]}...")
            return selected
        
        # Fail if no priority questions were generated
        raise ValueError(f"No NUMERICAL or MATHEMATICAL questions generated for passage: {passage.source_title}. Generated {len(questions)} questions but none were of the required types for researcher-level verification.")
    
    def _parse_verifiable_user_questions(self, response: str) -> List[VerifiableUserQuestion]:
        """Parse LLM response with XML tags into VerifiableUserQuestion objects"""
        import re
        
        print(f"  Parsing verifiable questions response (length: {len(response)})")
        questions = []
        
        # Extract all question blocks using regex
        question_pattern = r'<question>(.*?)</question>'
        question_blocks = re.findall(question_pattern, response, re.DOTALL)
        
        print(f"  Found {len(question_blocks)} question blocks")
        
        for i, block in enumerate(question_blocks):
            try:
                # Extract individual fields using regex
                type_match = re.search(r'<type>(.*?)</type>', block, re.DOTALL)
                level_match = re.search(r'<level>(.*?)</level>', block, re.DOTALL)
                verification_match = re.search(r'<verification>(.*?)</verification>', block, re.DOTALL)
                text_match = re.search(r'<text>(.*?)</text>', block, re.DOTALL)
                format_match = re.search(r'<format>(.*?)</format>', block, re.DOTALL)
                criteria_match = re.search(r'<criteria>(.*?)</criteria>', block, re.DOTALL)
                
                # Validate all fields are present
                if not all([type_match, level_match, text_match, format_match, criteria_match]):
                    print(f"  Question {i+1}: Missing required fields, skipping")
                    continue
                    
                # Extract and clean field values
                question_type_str = type_match.group(1).strip().upper()
                level = level_match.group(1).strip().lower()
                verification = verification_match.group(1).strip() if verification_match else ""
                question_text = text_match.group(1).strip()
                expected_format = format_match.group(1).strip()
                criteria = criteria_match.group(1).strip()
                
                # Validate question text is substantial
                if not question_text or len(question_text) < 10:
                    print(f"  Question {i+1}: Question text too short, skipping")
                    continue
                    
                # Map type to verification method and question type
                if question_type_str == 'NUMERICAL':
                    verification_method = VerificationMethod.NUMERICAL
                    question_type = 'quantitative_analysis'
                elif question_type_str == 'MATHEMATICAL':
                    verification_method = VerificationMethod.MATHEMATICAL
                    question_type = 'calculation_based'
                elif question_type_str == 'CLASSIFICATION':
                    verification_method = VerificationMethod.CATEGORICAL
                    question_type = 'classification'
                elif question_type_str == 'IDENTIFICATION':
                    verification_method = VerificationMethod.SINGLE_WORD
                    question_type = 'identification'
                elif question_type_str == 'CODE':
                    verification_method = VerificationMethod.CODE
                    question_type = 'code_generation'
                else:
                    print(f"  Question {i+1}: Unknown type '{question_type_str}', skipping")
                    continue
                
                # Create question object
                questions.append(VerifiableUserQuestion(
                    question=question_text,
                    question_type=question_type,
                    difficulty_level=level,
                    target_audience='researcher',
                    verification_method=verification_method,
                    expected_answer_format=expected_format,
                    verification_criteria=criteria
                ))
                
                print(f"  Question {i+1}: Successfully parsed ({question_type_str}): {question_text[:50]}...")
                
            except Exception as e:
                print(f"  Question {i+1}: Failed to parse - {e}")
                continue
        
        print(f"  Successfully parsed {len(questions)} verifiable questions")
        
        if len(questions) == 0:
            raise ValueError("Failed to parse any valid questions from LLM response. The response format may be incorrect or the questions may not meet quality standards.")
        
        # Filter for numerical/mathematical questions only
        priority_questions = [q for q in questions if q.verification_method in [VerificationMethod.NUMERICAL, VerificationMethod.MATHEMATICAL]]
        if len(priority_questions) == 0:
            raise ValueError(f"No NUMERICAL or MATHEMATICAL questions found among {len(questions)} parsed questions. All generated questions must be of these types for researcher-level verification.")
        
        return questions[:8]  # Limit to 8 questions
    
    def _parse_verifiable_answer(self, response: str, question: VerifiableUserQuestion) -> VerifiableAnswer:
                continue
                
            # Parse main question line: [TYPE] (Level) [VERIFICATION]: "Question text"
            if ':' in line and any(qtype in line.upper() for qtype in ['NUMERICAL', 'MATHEMATICAL', 'CLASSIFICATION', 'IDENTIFICATION', 'CODE']):
                try:
                    parts = line.split(':', 1)
                    type_info = parts[0].strip()
                    question_text = parts[1].strip().strip('"').strip()
                    
                    print(f"  Line {i}: Found potential question line")
                    print(f"    Type info: '{type_info}'")
                    print(f"    Question text: '{question_text[:100]}...'")
                    
                    # If question text is empty or too short, look at next line(s)
                    if not question_text.strip() or len(question_text.strip()) < 10:
                        print(f"  Question text empty/short, looking at subsequent lines...")
                        # Look at next few lines for the actual question content
                        question_lines = []
                        for j in range(i + 1, min(i + 10, len(lines))):
                            next_line = lines[j].strip()
                            if next_line and not next_line.startswith('[') and not next_line.startswith('---'):
                                # Stop if we hit another question type
                                if any(qtype in next_line.upper() for qtype in ['NUMERICAL', 'MATHEMATICAL', 'CLASSIFICATION', 'IDENTIFICATION', 'CODE']):
                                    break
                                # Collect substantial content lines
                                if len(next_line) > 5:
                                    question_lines.append(next_line.strip('"').strip())
                                    
                        if question_lines:
                            # Join all question content lines
                            question_text = ' '.join(question_lines)
                            print(f"    Found multi-line question text: '{question_text[:100]}...'")
                        else:
                            print(f"    No question content found in subsequent lines")
                        
                        # Skip if still no valid question text
                        if not question_text.strip() or len(question_text.strip()) < 10:
                            print(f"  Skipping - no valid question text found")
                            continue
                    
                    # Extract question type and verification method
                    if 'NUMERICAL' in type_info.upper():
                        verification_method = VerificationMethod.NUMERICAL
                        question_type = 'quantitative_analysis'
                    elif 'MATHEMATICAL' in type_info.upper():
                        verification_method = VerificationMethod.MATHEMATICAL
                        question_type = 'calculation_based'
                    elif 'CLASSIFICATION' in type_info.upper():
                        verification_method = VerificationMethod.CATEGORICAL
                        question_type = 'classification'
                    elif 'IDENTIFICATION' in type_info.upper():
                        verification_method = VerificationMethod.SINGLE_WORD
                        question_type = 'identification'
                    elif 'CODE' in type_info.upper():
                        verification_method = VerificationMethod.CODE
                        question_type = 'code_generation'
                    else:
                        verification_method = VerificationMethod.SINGLE_WORD
                        question_type = 'identification'
                    
                    # Extract difficulty level
                    level_match = re.search(r'\((.*?)\)', type_info)
                    level = level_match.group(1).lower() if level_match else 'intermediate'
                    
                    current_question = {
                        'question': question_text,
                        'question_type': question_type,
                        'difficulty_level': level,
                        'verification_method': verification_method
                    }
                    
                except Exception as e:
                    print(f"  Failed to parse question line: {line[:100]}... Error: {e}")
                    continue
            
            # Parse expected format line - handle with or without colon
            elif line.startswith('[Expected format]') and current_question:
                if ':' in line:
                    current_format = line.replace('[Expected format]:', '').strip()
                else:
                    current_format = line.replace('[Expected format]', '').strip()
                # If format is empty, look at next line
                if not current_format and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('['):
                        current_format = next_line
                print(f"  Line {i}: Found format: '{current_format}'")
            
            # Parse criteria line - handle with or without colon  
            elif line.startswith('[Criteria]') and current_question:
                if ':' in line:
                    current_criteria = line.replace('[Criteria]:', '').strip()
                else:
                    current_criteria = line.replace('[Criteria]', '').strip()
                # If criteria is empty, look at next line
                if not current_criteria and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('['):
                        current_criteria = next_line
                print(f"  Line {i}: Found criteria: '{current_criteria}'")
                
                # Complete the question when we have all parts
                if current_question and current_format and current_criteria:
                    # Validate that question is not empty
                    if current_question['question'].strip():
                        questions.append(VerifiableUserQuestion(
                            question=current_question['question'],
                            question_type=current_question['question_type'],
                            difficulty_level=current_question['difficulty_level'],
                            target_audience='researcher',
                            verification_method=current_question['verification_method'],
                            expected_answer_format=current_format,
                            verification_criteria=current_criteria
                        ))
                        print(f"  Successfully parsed verifiable question: {current_question['question'][:50]}...")
                    else:
                        print(f"  Skipped empty question")
                    
                    # Reset for next question
                    current_question = None
                    current_format = None
                    current_criteria = None
            
            # Handle content lines that might be format or criteria without headers
            elif current_question and not current_format and line and not line.startswith('[') and not any(qtype in line.upper() for qtype in ['NUMERICAL', 'MATHEMATICAL', 'CLASSIFICATION', 'IDENTIFICATION', 'CODE']):
                # This might be format content
                current_format = line
                print(f"  Line {i}: Assuming format content: '{current_format}'")
            elif current_question and current_format and not current_criteria and line and not line.startswith('[') and not any(qtype in line.upper() for qtype in ['NUMERICAL', 'MATHEMATICAL', 'CLASSIFICATION', 'IDENTIFICATION', 'CODE']):
                # This might be criteria content  
                current_criteria = line
                print(f"  Line {i}: Assuming criteria content: '{current_criteria}'")
                
                # Try to complete the question
                if current_question and current_format and current_criteria:
                    if current_question['question'].strip():
                        questions.append(VerifiableUserQuestion(
                            question=current_question['question'],
                            question_type=current_question['question_type'],
                            difficulty_level=current_question['difficulty_level'],
                            target_audience='researcher',
                            verification_method=current_question['verification_method'],
                            expected_answer_format=current_format,
                            verification_criteria=current_criteria
                        ))
                        print(f"  Successfully parsed verifiable question: {current_question['question'][:50]}...")
                    
                    # Reset for next question
                    current_question = None
                    current_format = None
                    current_criteria = None
        
        print(f"  Successfully parsed {len(questions)} verifiable questions")
        
        if len(questions) == 0:
            raise ValueError("Failed to parse any valid questions from LLM response. The response format may be incorrect or the questions may not meet quality standards.")
        
        # Filter for numerical/mathematical questions only
        priority_questions = [q for q in questions if q.verification_method in [VerificationMethod.NUMERICAL, VerificationMethod.MATHEMATICAL]]
        if len(priority_questions) == 0:
            raise ValueError(f"No NUMERICAL or MATHEMATICAL questions found among {len(questions)} parsed questions. All generated questions must be of these types for researcher-level verification.")
        
        return questions[:8]  # Limit to 8 questions
    
    def _parse_verifiable_answer(self, response: str, question: VerifiableUserQuestion) -> VerifiableAnswer:
        """Parse the LLM response into a VerifiableAnswer object"""
        
        # Initialize values - will raise error if parsing fails
        content = None
        confidence = 0.5
        verification_path = None
        justification = None
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('FINAL_ANSWER:'):
                    content = line.replace('FINAL_ANSWER:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('VERIFICATION_PATH:'):
                    verification_path = line.replace('VERIFICATION_PATH:', '').strip()
                elif line.startswith('REASONING_JUSTIFICATION:'):
                    justification = line.replace('REASONING_JUSTIFICATION:', '').strip()
        except Exception as e:
            print(f"Warning: Error parsing verifiable answer: {e}")
        
        # Validate that content was successfully extracted
        if not content or not content.strip() or len(content.strip()) < 3:
            raise ValueError(f"Failed to extract valid answer content from LLM response for question: {question.question[:50]}...")
        
        # Validate that justification was successfully extracted
        if not justification or not justification.strip():
            raise ValueError(f"Failed to extract valid reasoning justification from LLM response for question: {question.question[:50]}...")
        
        # Validate that verification path was extracted
        if not verification_path or not verification_path.strip():
            raise ValueError(f"Failed to extract valid verification path from LLM response for question: {question.question[:50]}...")
        
        return VerifiableAnswer(
            content=content,
            verification_method=question.verification_method,
            verification_criteria=verification_path,
            confidence_level=confidence,
            reasoning_justification=justification
        )
    
    def _parse_verifiable_grades(self, response: str) -> Dict[str, float]:
        """Parse the verifiable answer grading response"""
        grades = {
            'question_alignment': 0.5,
            'scientific_accuracy': 0.5,
            'format_compliance': 0.5,
            'verifiability': 0.5,
            'precision': 0.5,
            'calculation_correctness': 0.5,
            'evidence_usage': 0.5,
            'reasoning_quality': 0.5
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    try:
                        score = float(value.strip())
                        if key in grades:
                            grades[key] = score
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Warning: Error parsing verifiable grades: {e}")
        
        # Calculate overall score
        grades['overall_score'] = sum(grades.values()) / len(grades)
        
        return grades
    
    def _calculate_verifiability_score(self, answer: VerifiableAnswer) -> float:
        """Calculate a score for how verifiable the answer is"""
        score = answer.confidence_level
        
        # Bonus for specific verification methods
        if answer.verification_method in [VerificationMethod.NUMERICAL, VerificationMethod.MATHEMATICAL]:
            score += 0.2
        elif answer.verification_method == VerificationMethod.SINGLE_WORD:
            score += 0.1
        
        # Penalty for vague answers
        if 'unable' in answer.content.lower() or 'unclear' in answer.content.lower():
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    # Reuse utility methods from base class
    def _estimate_confidence(self, text: str) -> float:
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'not sure', 'unclear', 'might']
        certainty_words = ['definitely', 'clearly', 'obviously', 'certain', 'sure', 'precisely']
        
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
            'specifically', 'checking', 'verifying', 'alternatively', 'precisely',
            'as a result', 'in contrast', 'furthermore', 'systematic', 'calculating'
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
        if any(word in trace.lower() for word in ['calculate', 'compute', 'formula']):
            patterns.append('calculation_based')
        if any(word in trace.lower() for word in ['verify', 'check', 'confirm']):
            patterns.append('verification_focused')
        if any(word in trace.lower() for word in ['wait', 'actually', 'reconsider']):
            patterns.append('self_correction')
            
        return patterns
    
    def _assess_question_responsiveness(self, trace: str, question: VerifiableUserQuestion) -> float:
        question_words = set(question.question.lower().split())
        trace_words = set(trace.lower().split())
        
        overlap = len(question_words.intersection(trace_words))
        return min(1.0, overlap / max(len(question_words), 1))

# Reuse BatchProcessor from original with modifications for verifiable traces
class VerifiableBatchProcessor:
    def __init__(self, generator: VerifiableReasoningTraceGenerator, quality_check: bool = False, 
                 min_quality_score: float = 6.0, quality_judges: List[str] = None):
        self.generator = generator
        self.quality_check = quality_check
        self.min_quality_score = min_quality_score
        self.quality_judges = quality_judges or ['claude-sonnet-3.7', 'gpt-4o']
        self.evaluator = LLMJudgeEvaluator() if quality_check else None
    
    async def process_paper_corpus(
        self, 
        papers: List[ScientificPassage], 
        traces_per_paper: int = 1,
        output_file: str = None,
        save_incrementally: bool = True
    ) -> List[Dict[str, Any]]:
        """Process papers to generate verifiable reasoning traces"""
        
        all_traces = []
        processed_titles = set()
        
        # Load existing traces if file exists
        if output_file and save_incrementally:
            existing_traces = self._load_existing_traces(output_file)
            if existing_traces:
                print(f"Loaded {len(existing_traces)} existing verifiable traces from {output_file}")
                all_traces.extend(existing_traces)
                processed_titles = self._get_processed_paper_titles(existing_traces)
                print(f"Found {len(processed_titles)} already processed papers")
        
        for i, paper in enumerate(papers):
            print(f"Processing {i+1}/{len(papers)}: {paper.source_title}")
            
            # Skip if paper already processed
            if paper.source_title in processed_titles:
                print(f"  🔄 Skipping {paper.source_title} - already processed")
                continue
            
            accepted_trace = None
            max_attempts = 3
            attempt = 0
            
            while accepted_trace is None and attempt < max_attempts:
                attempt += 1
                try:
                    print(f"  Attempt {attempt}/{max_attempts} (verifiable trace)")
                    
                    # Generate verifiable trace
                    trace = await self.generator.generate_verifiable_reasoning_trace(paper)
                    
                    verifiability_score = trace['metadata']['verifiability_score']
                    print(f"    Generated verifiable trace (verifiability: {verifiability_score:.2f})")
                    
                    # Quality check if enabled
                    if self.quality_check:
                        quality_score = await self._evaluate_trace_quality(trace)
                        
                        if quality_score >= self.min_quality_score:
                            print(f"    ✅ Quality check passed ({quality_score:.1f} >= {self.min_quality_score})")
                            accepted_trace = trace
                        else:
                            print(f"    ❌ Quality check failed ({quality_score:.1f} < {self.min_quality_score})")
                            if attempt < max_attempts:
                                print(f"    🔄 Retrying with new generation...")
                            continue
                    else:
                        # No quality check, accept if verifiability is reasonable
                        if verifiability_score >= 0.6:
                            accepted_trace = trace
                        else:
                            print(f"    ❌ Low verifiability score ({verifiability_score:.2f} < 0.6)")
                            if attempt < max_attempts:
                                print(f"    🔄 Retrying...")
                            continue
                        
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
                print(f"  ✅ Accepted verifiable trace after {attempt} attempt(s)")
                
                # Save incrementally
                if output_file and save_incrementally:
                    self._save_traces_incrementally(all_traces, output_file)
                    print(f"  💾 Saved {len(all_traces)} verifiable traces to {output_file}")
            else:
                print(f"  ❌ Failed to generate acceptable verifiable trace after {max_attempts} attempts")
        
        return all_traces
    
    async def _evaluate_trace_quality(self, trace_data: Dict[str, Any]) -> float:
        """Evaluate the quality of a verifiable trace"""
        if not self.quality_check or not self.evaluator:
            return 10.0
        
        try:
            triplet = QRSTriplet(
                question=trace_data['user_interaction']['question'],
                reasoning=trace_data['reasoning_trace'],
                solution=trace_data['verifiable_answer']['content'],
                source_file="verifiable_generation",
                sample_index=0
            )
            
            ratings = await self.evaluator.evaluate_triplet(triplet, self.quality_judges)
            valid_ratings = [r for r in ratings if r.overall_score > 0]
            
            if not valid_ratings:
                return 0.0
            
            stats = self.evaluator.calculate_statistics(valid_ratings)
            avg_score = stats['overall_score']['mean']
            
            # Bonus for high verifiability
            verifiability_bonus = trace_data['metadata']['verifiability_score'] * 2.0
            
            return min(10.0, avg_score + verifiability_bonus)
            
        except Exception as e:
            print(f"    ❌ Quality evaluation error: {e}")
            return 0.0
    
    def _load_existing_traces(self, output_file: str) -> List[Dict[str, Any]]:
        """Load existing verifiable traces from output file"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('verifiable_traces', [])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return []
    
    def _get_processed_paper_titles(self, traces: List[Dict[str, Any]]) -> set:
        """Extract processed paper titles"""
        processed_titles = set()
        for trace in traces:
            if 'source_paper' in trace and 'title' in trace['source_paper']:
                processed_titles.add(trace['source_paper']['title'])
        return processed_titles
    
    def _save_traces_incrementally(self, traces: List[Dict[str, Any]], output_file: str):
        """Save verifiable traces incrementally"""
        from datetime import datetime
        
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_verifiable_traces_generated": len(traces),
                "last_updated": datetime.now().isoformat(),
                "status": "in_progress",
                "trace_type": "verifiable_reasoning"
            },
            "verifiable_traces": traces
        }
        
        temp_file = output_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            import os
            os.rename(temp_file, output_file)
        except Exception as e:
            print(f"Warning: Could not save incrementally: {e}")
            try:
                os.remove(temp_file)
            except:
                pass
    
    def save_training_dataset(self, traces: List[Dict[str, Any]], filename: str):
        """Save verifiable traces in final format"""
        from datetime import datetime
        
        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_verifiable_traces_generated": len(traces),
                "last_updated": datetime.now().isoformat(),
                "status": "completed",
                "trace_type": "verifiable_reasoning"
            },
            "verifiable_traces": traces
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Final verifiable dataset saved with {len(traces)} traces")

def parse_arguments():
    """Parse command line arguments for verifiable reasoner"""
    parser = argparse.ArgumentParser(description="Generate verifiable synthetic reasoning traces")
    
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
        help="Input JSONL file with text chunks (default: ../data/heuristic_filtered_cosmo_limited.jsonl)"
    )
    
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of verifiable traces to generate (default: process all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="verifiable_reasoning_training_data.json",
        help="Output file for verifiable traces (default: verifiable_reasoning_training_data.json)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (default: True, use --no-resume to start fresh)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring existing output file"
    )
    
    parser.add_argument(
        "--grade-answers",
        action="store_true",
        help="Grade verifiable answers for RL training (slower)"
    )
    
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Enable quality checking with LLM judges (slower but higher quality)"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=6.0,
        help="Minimum quality score required to accept a trace (default: 6.0)"
    )
    
    return parser.parse_args()

async def main():
    """Main function for verifiable reasoning trace generation"""
    # Configure logging
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
        print(f"Verifiable Reasoning Trace Generator")
        print(f"Model: {args.model}")
        print(f"Input file: {args.input_file}")
        print(f"Max traces: {args.max_traces or 'All'}")
        if MPI_AVAILABLE:
            print(f"MPI processes: {size}")
        print("-" * 50)
    
    # Load data
    texts = load_data_from_jsonl(args.input_file, args.max_traces, apply_quality_filter=(rank == 0))
    
    if not texts:
        if rank == 0:
            print("No text data loaded. Exiting.")
        return
    
    # Distribute texts across MPI processes
    if MPI_AVAILABLE and size > 1:
        my_text_indices = [i for i in range(len(texts)) if i % size == rank]
        my_texts = [texts[i] for i in my_text_indices]
        start_index = my_text_indices[0] if my_text_indices else 0
        logger.info(f"Assigned {len(my_texts)} texts to process")
        
        base_name, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'json')
        my_output = f"{base_name}_rank{rank}.{ext}"
    else:
        my_texts = texts
        start_index = 0
        my_output = args.output
        if rank == 0:
            logger.info(f"Processing {len(my_texts)} texts in single process mode")
    
    # Convert to passages
    papers = convert_texts_to_passages(my_texts, start_index)
    
    # Initialize components
    try:
        llm = ArgoLLMAdapter(args.model)
        generator = VerifiableReasoningTraceGenerator(llm, grade_answers=args.grade_answers)
        processor = VerifiableBatchProcessor(
            generator,
            quality_check=args.quality_check,
            min_quality_score=args.min_quality_score
        )
        
        logger.info(f"Initialized verifiable LLM with model: {args.model}")
        if args.grade_answers and rank == 0:
            print("Verifiable answer grading enabled")
        if args.quality_check and rank == 0:
            print(f"Quality checking enabled (min score: {args.min_quality_score})")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return
    
    # Determine resume behavior
    resume_enabled = not args.no_resume
    
    # Generate verifiable traces
    logger.info(f"Generating verifiable reasoning traces for {len(papers)} text samples...")
    if rank == 0:
        print(f"Progress will be saved incrementally to: {my_output}")
        if resume_enabled:
            print("(Resumable - progress saved after each verifiable trace)")
        else:
            print("(Starting fresh)")
        print("-" * 50)
    
    traces = await processor.process_paper_corpus(
        papers,
        traces_per_paper=1,
        output_file=my_output if resume_enabled else None,
        save_incrementally=resume_enabled
    )
    
    # Final save
    processor.save_training_dataset(traces, my_output)
    
    logger.info(f"Generated {len(traces)} verifiable reasoning traces")
    logger.info(f"Results saved to: {my_output}")
    
    # Wait for all processes
    if MPI_AVAILABLE and size > 1:
        comm.Barrier()
        if rank == 0:
            print(f"✅ All {size} MPI processes completed verifiable trace generation")

if __name__ == "__main__":
    asyncio.run(main())