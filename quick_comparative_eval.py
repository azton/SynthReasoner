#!/usr/bin/env python3
"""
Quick comparative evaluation across trace files using working judge models
"""

import asyncio
import json
from pathlib import Path
from llm_judge_evaluator import LLMJudgeEvaluator

async def quick_comparative_evaluation():
    """Quick evaluation using only reliable judge models"""
    print("🔍 Quick Comparative Analysis - Reliable Judges Only")
    print("=" * 55)
    
    # Find all trace files
    trace_files = list(Path('.').glob('*_traces.json'))
    if not trace_files:
        print("❌ No trace files found.")
        return
    
    print(f"📁 Found {len(trace_files)} trace files")
    
    # Use only the most reliable judge models
    evaluator = LLMJudgeEvaluator()
    reliable_judges = ['claude-sonnet-4', 'gpt-4.1', 'gpt-o4-mini']  # Skip problematic ones
    
    model_results = {}
    
    for trace_file in trace_files:
        print(f"\n🔎 Evaluating {trace_file}")
        
        model_name = str(trace_file).replace('_traces.json', '')
        triplets = evaluator.load_qrs_triplets([str(trace_file)])
        
        if not triplets:
            continue
            
        triplet = triplets[0]
        print(f"   📊 Question: {triplet.question[:80]}...")
        
        ratings = await evaluator.evaluate_triplet(triplet, reliable_judges)
        valid_ratings = [r for r in ratings if r.overall_score > 0]
        
        if valid_ratings:
            stats = evaluator.calculate_statistics(valid_ratings)
            model_results[model_name] = stats
            
            print(f"   📈 Scores: Overall={stats['overall_score']['mean']:.2f}, "
                  f"Reasoning={stats['reasoning_quality']['mean']:.2f}, "
                  f"Accuracy={stats['solution_accuracy']['mean']:.2f}, "
                  f"Coherence={stats['coherence']['mean']:.2f} "
                  f"(n={stats['overall_score']['count']})")
    
    # Rankings
    print(f"\n📊 FINAL RANKINGS:")
    print("=" * 60)
    
    ranked_models = sorted(model_results.items(), 
                          key=lambda x: x[1]['overall_score']['mean'], 
                          reverse=True)
    
    print(f"{'Rank':<4} {'Model':<20} {'Overall':<10} {'Reasoning':<10} {'Accuracy':<10} {'Coherence':<10}")
    print("-" * 60)
    
    for rank, (model_name, stats) in enumerate(ranked_models, 1):
        print(f"{rank:<4} {model_name:<20} "
              f"{stats['overall_score']['mean']:.2f}±{stats['overall_score']['std_dev']:.1f:<6} "
              f"{stats['reasoning_quality']['mean']:.2f}±{stats['reasoning_quality']['std_dev']:.1f:<6} "
              f"{stats['solution_accuracy']['mean']:.2f}±{stats['solution_accuracy']['std_dev']:.1f:<6} "
              f"{stats['coherence']['mean']:.2f}±{stats['coherence']['std_dev']:.1f:<6}")
    
    # Best performers
    print(f"\n🏆 CHAMPION BY CATEGORY:")
    print("-" * 35)
    
    metrics = ['overall_score', 'reasoning_quality', 'solution_accuracy', 'coherence']
    for metric in metrics:
        if model_results:
            best = max(model_results.items(), key=lambda x: x[1][metric]['mean'])
            print(f"{metric.replace('_', ' ').title():<20}: {best[0]} ({best[1][metric]['mean']:.2f})")
    
    # Save results
    results = {
        'rankings': [
            {'rank': rank, 'model': name, 'stats': stats}
            for rank, (name, stats) in enumerate(ranked_models, 1)
        ],
        'champions': {
            metric: max(model_results.items(), key=lambda x: x[1][metric]['mean'])[0]
            for metric in metrics if model_results
        }
    }
    
    with open('quick_comparative_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: quick_comparative_results.json")
    print("✅ Quick evaluation completed!")

if __name__ == "__main__":
    asyncio.run(quick_comparative_evaluation())