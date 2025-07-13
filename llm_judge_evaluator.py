#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation System for QRS Triplets

Evaluates Question-Reasoning-Solution triplets from trace files using multiple LLM judges.
Supports Sonnet 4, Gemini Flash, GPT-4.1, and GPT-4o as judges.
Provides statistical analysis including mean and standard deviation of ratings.
"""

import json
import asyncio
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import argparse
import random
from argo import ArgoModel, ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QRSTriplet:
    """Represents a Question-Reasoning-Solution triplet"""
    question: str
    reasoning: str
    solution: str
    source_file: str
    sample_index: int
    metadata: Optional[Dict] = None

@dataclass
class JudgeRating:
    """Represents a judge's rating of a QRS triplet"""
    judge_model: str
    overall_score: float  # 1-10 scale
    reasoning_quality: float  # 1-10 scale
    solution_accuracy: float  # 1-10 scale
    coherence: float  # 1-10 scale
    explanation: str

class LLMJudgeEvaluator:
    """Multi-model LLM judge evaluation system using Argo interface"""
    
    def __init__(self):
        # Initialize Argo models
        self.models = {
            'claude-sonnet-3.7': ArgoModel('claudesonnet37'),
            'gemini-flash': ArgoModel('gemini25flash'),
            'gpt-4.1': ArgoModel('gpt41'),
            'gpt-4o': ArgoModel('gpt4o'),
            'gpt-o4-mini': ArgoModel('gpto4mini')
        }
        
        self.judge_models = {
            'claude-sonnet-3.7': self._judge_with_claude_sonnet37,
            'gemini-flash': self._judge_with_gemini_flash,
            'gpt-4.1': self._judge_with_gpt41,
            'gpt-4o': self._judge_with_gpt4o,
            'gpt-o4-mini': self._judge_with_gpt_o4_mini
        }
        
        self.evaluation_prompt = """
You are an expert AI judge evaluating the quality of reasoning and problem-solving in Question-Reasoning-Solution (QRS) triplets.

**Question:** {question}

**Reasoning:** {reasoning}

**Solution:** {solution}

Rate each aspect on a 1-10 scale using these STRICT criteria. Be discriminating - most responses should score 4-7, with 8+ reserved for truly exceptional work:

## 1. **Reasoning Quality** (1-10)
Evaluate the internal thought process for logical structure, depth, and authenticity. **BE HARSH:**
- **9-10**: EXCEPTIONAL - Demonstrates PhD-level reasoning with sophisticated analysis, elegant logical flow, meaningful self-correction, deep domain expertise, and insights that go beyond the obvious. Should be extremely rare.
- **7-8**: STRONG - Clear logical progression with good depth, some sophisticated analysis, appropriate self-correction, well-organized thoughts. Still has minor flaws or could be deeper.
- **5-6**: ADEQUATE - Basic logical structure present but lacks depth, may be verbose/meandering, shallow analysis, obvious reasoning without insight. Most "good enough" responses fall here.
- **3-4**: POOR - Significant logical gaps, disorganized thoughts, superficial analysis, confused reasoning, lots of uncertainty without resolution.
- **1-2**: TERRIBLE - Major logical errors, incoherent structure, minimal effort, fundamentally flawed thinking.

## 2. **Solution Accuracy** (1-10)
Evaluate factual correctness, completeness, and appropriateness. **DEMAND EXCELLENCE:**
- **9-10**: EXCEPTIONAL - Highly accurate with precise technical details, comprehensive coverage, perfect scope, demonstrates deep expertise. Textbook-quality answer.
- **7-8**: STRONG - Mostly accurate with good detail, covers main points well, appropriate scope, minor gaps only.
- **5-6**: ADEQUATE - Generally accurate but lacks important details, somewhat generic, missing key specifics, over-generalized. "Wikipedia-level" answers.
- **3-4**: POOR - Notable omissions, some factual errors, doesn't fully address question, incomplete coverage.
- **1-2**: TERRIBLE - Major factual errors, significant omissions, completely misses the point.

## 3. **Coherence** (1-10)
Evaluate how well reasoning connects to solution. **NO TOLERANCE FOR DISCONNECTS:**
- **9-10**: EXCEPTIONAL - Perfect seamless flow from reasoning to solution, every thought contributes, reasoning fully justifies and explains the final answer.
- **7-8**: STRONG - Good alignment, reasoning generally supports solution, clear connection with minor gaps.
- **5-6**: ADEQUATE - Reasonable connection but some reasoning doesn't contribute to solution, some disconnect between depth of reasoning and simplicity of answer.
- **3-4**: POOR - Significant gaps, reasoning explores topics not reflected in solution, weak connection.
- **1-2**: TERRIBLE - Reasoning and solution seem completely disconnected or contradictory.

## 4. **Overall Score** (1-10)
Your holistic assessment. **RESERVE HIGH SCORES FOR TRULY IMPRESSIVE WORK:**
- **9-10**: EXCEPTIONAL - Demonstrates mastery across all dimensions, could be used as a training example, genuinely impressive reasoning and solution.
- **7-8**: STRONG - Solid performance with clear strengths, good but not exceptional, has some notable positive qualities.
- **5-6**: ADEQUATE - Decent work that gets the job done but nothing special, meets basic expectations, "fine but forgettable."
- **3-4**: POOR - Below expectations, notable weaknesses outweigh strengths, problematic in multiple ways.
- **1-2**: TERRIBLE - Fundamentally flawed, should not be used as an example of good reasoning.

**GRADING GUIDANCE - BE TOUGH:**
- **Most responses should score 4-7**. Scores of 8+ should be genuinely impressive and rare.
- **Common flaws to penalize heavily**: Verbose rambling, obvious insights, lack of technical depth, generic solutions, poor organization, excessive uncertainty
- **Reasoning Quality**: Penalize stream-of-consciousness style, repetition, lack of clear progression, shallow analysis
- **Solution Accuracy**: Demand specific details, technical terms, comprehensive coverage. Generic answers should score 5-6 maximum
- **Coherence**: If reasoning is 3x longer than solution but doesn't justify that depth, penalize heavily

**Examples of what deserves high scores:**
- Reasoning with genuine insights, sophisticated analysis, clear logical progression, meaningful self-correction
- Solutions with precise technical details, specific mechanisms, appropriate scope and depth
- Perfect flow where extensive reasoning clearly justifies and builds to a comprehensive solution

**Examples of what should score low-mid range (4-6):**
- "Wikipedia-level" generic answers without deep insight
- Verbose reasoning that doesn't lead to proportionally sophisticated solutions  
- Obvious analysis without novel perspectives or deep understanding
- Solutions that are "correct but basic"

Provide your response in this exact JSON format:
{{
    "reasoning_quality": <score>,
    "solution_accuracy": <score>, 
    "coherence": <score>,
    "overall_score": <score>,
    "explanation": "<detailed explanation of your ratings with specific examples from the text>"
}}
"""

    async def _get_judge_response_with_retry(self, model_key: str, messages: list, max_tokens: int = 3000, temperature: float = 0.1, max_retries: int = 1) -> JudgeRating:
        """Get judge response with retry on JSON parsing failure"""
        judge_model_name = model_key.replace('_', '-')
        
        for attempt in range(max_retries + 1):
            try:
                # Get response from model
                if temperature == 0.1 and model_key in ['gpt-4.1', 'gpt-4o', 'gpt-o4-mini']:
                    response = self.models[model_key](messages, max_tokens=max_tokens, temperature=temperature)
                else:
                    response = self.models[model_key](messages, max_tokens=max_tokens)
                
                # Extract and parse content
                content = response.content
                
                # Check for server errors first
                if "Server Error:" in content or "Error:" in content:
                    raise Exception(f"Server error: {content}")
                
                # First, extract from <code> tags if present
                if '<code>' in content and '</code>' in content:
                    start = content.find('<code>') + 6
                    end = content.find('</code>')
                    content = content[start:end].strip('"""').strip()
                
                # Then, handle markdown code blocks (which might be nested inside <code> tags)
                if '```json' in content:
                    start = content.find('```json') + 7
                    end = content.find('```', start)
                    if end != -1:
                        content = content[start:end].strip()
                elif '```' in content:
                    # Handle cases where there are ``` without json specifier
                    start = content.find('```') + 3
                    end = content.find('```', start)
                    if end != -1:
                        content = content[start:end].strip()
                
                # Clean up any remaining quotes or formatting artifacts
                content = content.strip('"""').strip("'''").strip()
                
                # Check if content is empty after extraction
                if not content or content.isspace():
                    raise Exception("Empty response content after extraction")
                
                # Parse JSON
                result = json.loads(content)
                
                # Handle different coherence field names
                coherence_score = result.get('coherence', result.get('solution_coherence', 0.0))
                
                return JudgeRating(
                    judge_model=judge_model_name,
                    overall_score=float(result['overall_score']),
                    reasoning_quality=float(result['reasoning_quality']),
                    solution_accuracy=float(result['solution_accuracy']),
                    coherence=float(coherence_score),
                    explanation=result['explanation']
                )
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"{judge_model_name} JSON parsing failed on attempt {attempt + 1}, retrying...")
                    logger.warning(f"Raw content: {content[:200]}...")
                    continue
                else:
                    logger.error(f"{judge_model_name} JSON parsing failed after {max_retries + 1} attempts: {e}")
                    logger.error(f"Final raw content: {content[:500]}...")
                    return self._create_error_rating(judge_model_name, f"JSON parsing failed after retries: {str(e)}")
                    
            except Exception as e:
                logger.error(f"{judge_model_name} evaluation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    logger.warning(f"Retrying {judge_model_name} evaluation...")
                    continue
                else:
                    logger.error(f"Raw response content: {response.content[:500]}...")
                    return self._create_error_rating(judge_model_name, str(e))
        
        # Should never reach here, but just in case
        return self._create_error_rating(judge_model_name, "Unexpected error in retry logic")

    def load_qrs_triplets(self, trace_files: List[str]) -> List[QRSTriplet]:
        """Load QRS triplets from trace JSON files"""
        triplets = []
        
        for file_path in trace_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i, question_data in enumerate(data.get('questions', [])):
                    triplet = QRSTriplet(
                        question=question_data['user_interaction']['question'],
                        reasoning=question_data['reasoning_trace'],
                        solution=question_data['final_answer'],
                        source_file=file_path,
                        sample_index=i,
                        metadata=question_data.get('metadata', {})
                    )
                    triplets.append(triplet)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return triplets

    async def _judge_with_claude_sonnet37(self, triplet: QRSTriplet) -> JudgeRating:
        """Judge using Claude Sonnet 3.7 via Argo"""
        prompt = self.evaluation_prompt.format(
            question=triplet.question,
            reasoning=triplet.reasoning,
            solution=triplet.solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        return await self._get_judge_response_with_retry('claude-sonnet-3.7', messages)

    async def _judge_with_gemini_flash(self, triplet: QRSTriplet) -> JudgeRating:
        """Judge using Gemini Flash via Argo"""
        prompt = self.evaluation_prompt.format(
            question=triplet.question,
            reasoning=triplet.reasoning,
            solution=triplet.solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        return await self._get_judge_response_with_retry('gemini-flash', messages)

    async def _judge_with_gpt41(self, triplet: QRSTriplet) -> JudgeRating:
        """Judge using GPT-4.1 via Argo"""
        prompt = self.evaluation_prompt.format(
            question=triplet.question,
            reasoning=triplet.reasoning,
            solution=triplet.solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        return await self._get_judge_response_with_retry('gpt-4.1', messages, temperature=0.1)

    async def _judge_with_gpt4o(self, triplet: QRSTriplet) -> JudgeRating:
        """Judge using GPT-4o via Argo"""
        prompt = self.evaluation_prompt.format(
            question=triplet.question,
            reasoning=triplet.reasoning,
            solution=triplet.solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        return await self._get_judge_response_with_retry('gpt-4o', messages, temperature=0.1)

    async def _judge_with_gpt_o4_mini(self, triplet: QRSTriplet) -> JudgeRating:
        """Judge using GPT-o4-mini via Argo"""
        prompt = self.evaluation_prompt.format(
            question=triplet.question,
            reasoning=triplet.reasoning,
            solution=triplet.solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        return await self._get_judge_response_with_retry('gpt-o4-mini', messages, temperature=0.1)

    def _create_error_rating(self, judge_model: str, error_msg: str) -> JudgeRating:
        """Create an error rating when judge fails"""
        return JudgeRating(
            judge_model=judge_model,
            overall_score=0.0,
            reasoning_quality=0.0,
            solution_accuracy=0.0,
            coherence=0.0,
            explanation=f"Evaluation failed: {error_msg}"
        )

    async def evaluate_triplet(self, triplet: QRSTriplet, judge_models: List[str]) -> List[JudgeRating]:
        """Evaluate a single QRS triplet with multiple judges"""
        tasks = []
        
        for model in judge_models:
            if model in self.judge_models:
                task = self.judge_models[model](triplet)
                tasks.append(task)
            else:
                logger.warning(f"Unknown judge model: {model}")
        
        ratings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid ratings
        valid_ratings = [r for r in ratings if isinstance(r, JudgeRating)]
        return valid_ratings

    def calculate_statistics(self, ratings: List[JudgeRating]) -> Dict:
        """Calculate mean and standard deviation for ratings"""
        if not ratings:
            return {}
        
        metrics = ['overall_score', 'reasoning_quality', 'solution_accuracy', 'coherence']
        stats = {}
        
        for metric in metrics:
            values = [getattr(r, metric) for r in ratings if getattr(r, metric) > 0]
            if values:
                stats[metric] = {
                    'mean': statistics.mean(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'count': len(values)
                }
            else:
                stats[metric] = {'mean': 0.0, 'std_dev': 0.0, 'count': 0}
        
        return stats

    async def evaluate_all_triplets(self, 
                                  trace_files: List[str], 
                                  judge_models: List[str] = None,
                                  output_file: str = "evaluation_results.json") -> Dict:
        """Evaluate all QRS triplets and generate comprehensive results"""
        
        if judge_models is None:
            judge_models = list(self.judge_models.keys())
        
        # Load triplets
        triplets = self.load_qrs_triplets(trace_files)
        logger.info(f"Loaded {len(triplets)} QRS triplets from {len(trace_files)} files")
        
        # Evaluate each triplet
        all_results = []
        
        for i, triplet in enumerate(triplets):
            logger.info(f"Evaluating triplet {i+1}/{len(triplets)}")
            
            ratings = await self.evaluate_triplet(triplet, judge_models)
            
            triplet_result = {
                'triplet_info': {
                    'source_file': triplet.source_file,
                    'sample_index': triplet.sample_index,
                    'question': triplet.question[:200] + "..." if len(triplet.question) > 200 else triplet.question
                },
                'ratings': [
                    {
                        'judge_model': r.judge_model,
                        'overall_score': r.overall_score,
                        'reasoning_quality': r.reasoning_quality,
                        'solution_accuracy': r.solution_accuracy,
                        'coherence': r.coherence,
                        'explanation': r.explanation
                    } for r in ratings
                ],
                'statistics': self.calculate_statistics(ratings)
            }
            
            all_results.append(triplet_result)
        
        # Calculate overall statistics across all triplets
        all_ratings = []
        for result in all_results:
            all_ratings.extend([JudgeRating(**rating) for rating in result['ratings']])
        
        overall_stats = self.calculate_statistics(all_ratings)
        
        # Create final results
        final_results = {
            'evaluation_metadata': {
                'total_triplets': len(triplets),
                'judge_models': judge_models,
                'trace_files': trace_files
            },
            'overall_statistics': overall_stats,
            'individual_results': all_results
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation complete. Results saved to {output_file}")
        return final_results

async def run_comparative_trace_evaluation():
    """Run comparative evaluation across all trace files to determine best reasoning model"""
    print("🔍 LLM Judge Evaluation System - Comparative Analysis")
    print("=" * 60)
    
    # Find all trace files
    trace_files = list(Path('.').glob('*_traces.json'))
    if not trace_files:
        print("❌ No trace files found. Please ensure *_traces.json files are present.")
        return
    
    print(f"📁 Found {len(trace_files)} trace files:")
    for f in trace_files:
        print(f"   - {f}")
    
    # Use real evaluator
    evaluator = LLMJudgeEvaluator()
    judge_models = ['claude-sonnet-3.7', 'gemini-flash', 'gpt-4.1', 'gpt-4o', 'gpt-o4-mini']
    
    # Results by model
    model_results = {}
    all_results = []
    
    for trace_file in trace_files:
        print(f"\n🔎 Evaluating {trace_file}")
        
        # Extract model name from filename
        model_name = str(trace_file).replace('_traces.json', '').replace('.json', '')
        
        # Load triplets from this file
        triplets = evaluator.load_qrs_triplets([str(trace_file)])
        
        if not triplets:
            print(f"   ❌ No triplets found in {trace_file}")
            continue
            
        # Evaluate first triplet from each file
        triplet = triplets[0]
        print(f"   📊 Evaluating triplet: {triplet.question[:100]}...")
        
        ratings = await evaluator.evaluate_triplet(triplet, judge_models)
        valid_ratings = [r for r in ratings if r.overall_score > 0]  # Filter out failed evaluations
        
        print(f"   📝 Received {len(valid_ratings)}/{len(ratings)} successful judge ratings")
        
        if valid_ratings:
            stats = evaluator.calculate_statistics(valid_ratings)
            
            # Store results for this model (convert JudgeRating objects to dicts)
            model_results[model_name] = {
                'file': str(trace_file),
                'ratings': [
                    {
                        'judge_model': r.judge_model,
                        'overall_score': r.overall_score,
                        'reasoning_quality': r.reasoning_quality,
                        'solution_accuracy': r.solution_accuracy,
                        'coherence': r.coherence,
                        'explanation': r.explanation
                    } for r in valid_ratings
                ],
                'statistics': stats,
                'judge_count': len(valid_ratings)
            }
            
            print(f"   📈 Average scores: Overall={stats['overall_score']['mean']:.2f}, "
                  f"Reasoning={stats['reasoning_quality']['mean']:.2f}, "
                  f"Accuracy={stats['solution_accuracy']['mean']:.2f}, "
                  f"Coherence={stats['coherence']['mean']:.2f}")
        else:
            print(f"   ❌ No successful evaluations for {trace_file}")
    
    # Comparative analysis
    print(f"\n📊 COMPARATIVE ANALYSIS ACROSS ALL MODELS:")
    print("=" * 70)
    
    # Sort models by overall score
    ranked_models = sorted(model_results.items(), 
                          key=lambda x: x[1]['statistics']['overall_score']['mean'], 
                          reverse=True)
    
    print(f"{'Rank':<4} {'Model':<20} {'Overall':<10} {'Reasoning':<12} {'Accuracy':<10} {'Coherence':<10} {'Judges':<7}")
    print("-" * 70)
    
    for rank, (model_name, results) in enumerate(ranked_models, 1):
        stats = results['statistics']
        overall_str = f"{stats['overall_score']['mean']:.2f}±{stats['overall_score']['std_dev']:.2f}"
        reasoning_str = f"{stats['reasoning_quality']['mean']:.2f}±{stats['reasoning_quality']['std_dev']:.2f}"
        accuracy_str = f"{stats['solution_accuracy']['mean']:.2f}±{stats['solution_accuracy']['std_dev']:.2f}"
        coherence_str = f"{stats['coherence']['mean']:.2f}±{stats['coherence']['std_dev']:.2f}"
        
        print(f"{rank:<4} {model_name:<20} "
              f"{overall_str:<10} "
              f"{reasoning_str:<12} "
              f"{accuracy_str:<10} "
              f"{coherence_str:<10} "
              f"{results['judge_count']:<7}")
    
    # Identify best performers
    print(f"\n🏆 BEST PERFORMERS:")
    print("-" * 30)
    
    metrics = ['overall_score', 'reasoning_quality', 'solution_accuracy', 'coherence']
    for metric in metrics:
        best_model = max(model_results.items(), 
                        key=lambda x: x[1]['statistics'][metric]['mean'])
        print(f"{metric.replace('_', ' ').title():<20}: {best_model[0]} "
              f"({best_model[1]['statistics'][metric]['mean']:.2f})")
    
    # Save comprehensive results
    final_results = {
        'evaluation_metadata': {
            'total_trace_files': len(trace_files),
            'successful_evaluations': len(model_results),
            'judge_models': judge_models,
            'evaluation_type': 'comparative_trace_analysis'
        },
        'model_rankings': [
            {
                'rank': rank,
                'model_name': model_name,
                'trace_file': results['file'],
                'statistics': results['statistics'],
                'judge_count': results['judge_count']
            }
            for rank, (model_name, results) in enumerate(ranked_models, 1)
        ],
        'best_performers': {
            metric: {
                'model': max(model_results.items(), 
                           key=lambda x: x[1]['statistics'][metric]['mean'])[0],
                'score': max(model_results.items(), 
                           key=lambda x: x[1]['statistics'][metric]['mean'])[1]['statistics'][metric]['mean']
            }
            for metric in metrics
        },
        'detailed_results': model_results
    }
    
    output_file = "comparative_trace_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    print("\n✅ Comparative evaluation completed!")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation for QRS Triplets")
    parser.add_argument("--trace-files", nargs="+", help="Trace JSON files to evaluate")
    parser.add_argument("--judges", nargs="+", 
                        choices=['claude-sonnet-3.7', 'gemini-flash', 'gpt-4.1', 'gpt-4o', 'gpt-o4-mini'],
                        default=['claude-sonnet-3.7', 'gemini-flash', 'gpt-4.1', 'gpt-4o', 'gpt-o4-mini'],
                        help="Judge models to use")
    parser.add_argument("--output", default="evaluation_results.json", 
                        help="Output file for results")
    
    args = parser.parse_args()
    
    if not args.trace_files:
        # Auto-discover trace files
        trace_files = list(Path('.').glob('*_traces.json'))
        args.trace_files = [str(f) for f in trace_files]
        print(f"Auto-discovered trace files: {args.trace_files}")
    
    evaluator = LLMJudgeEvaluator()
    
    # Run evaluation
    asyncio.run(evaluator.evaluate_all_triplets(
        trace_files=args.trace_files,
        judge_models=args.judges,
        output_file=args.output
    ))

if __name__ == "__main__":
    asyncio.run(run_comparative_trace_evaluation())