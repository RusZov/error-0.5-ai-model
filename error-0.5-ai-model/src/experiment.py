"""
Experiment runner for Error 0.5 AI Model.

Orchestrates the full experimental pipeline:
1. Load prompts and configuration
2. Run LLM inference with N repetitions
3. Normalize and categorize responses
4. Calculate metrics
5. Save and compare results
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .llm_client import LLMClient
from .metrics import MetricsCalculator
from .utils import (
    normalize_response, 
    filter_noise, 
    save_results,
    calculate_category_distribution,
)


class ExperimentRunner:
    """
    Run experiments to analyze LLM response consistency using Error 0.5 theory.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration dictionary (or load from config.yaml)
        """
        self.config = config or {}
        
        # Extract configuration sections
        llm_config = self.config.get('llm', {})
        gen_config = self.config.get('generation', {})
        exp_config = self.config.get('experiment', {})
        metrics_config = self.config.get('metrics', {})
        output_config = self.config.get('output', {})
        
        # Initialize LLM client
        provider = llm_config.get('provider', 'openai')
        if provider == 'openai':
            client_config = llm_config.get('openai', {})
        else:
            client_config = llm_config.get('local', {})
        
        self.llm_client = LLMClient.create(provider=provider, **client_config)
        
        # Store generation parameters
        self.default_temperature = gen_config.get('temperature', 0.7)
        self.default_top_p = gen_config.get('top_p', 0.9)
        self.default_top_k = gen_config.get('top_k', 50)
        self.default_system_prompt = gen_config.get('system_prompt', '')
        
        # Store experiment parameters
        self.default_repetitions = exp_config.get('default_repetitions', 100)
        self.noise_filter_enabled = exp_config.get('noise_filter_enabled', True)
        self.noise_keywords = exp_config.get('noise_filter_keywords', [])
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            quality_threshold=metrics_config.get('quality_threshold', 0.8),
            p_gain=metrics_config.get('p_gain', 1.0),
            b_value=metrics_config.get('b_value', 1.0),
            p_loss=metrics_config.get('p_loss', 0.5),
            l_value=metrics_config.get('l_value', 0.5),
        )
        
        # Output settings
        self.results_dir = output_config.get('results_dir', 'results')
        self.save_csv = output_config.get('save_csv', True)
        self.save_json = output_config.get('save_json', True)
    
    def run_single_prompt(self, prompt: str, n_repetitions: int,
                          temperature: float = None, top_p: float = None,
                          top_k: int = None, system_prompt: str = None,
                          context: str = None, normalize_method: str = "keyword",
                          filter_noise_flag: bool = None) -> Dict[str, Any]:
        """
        Run experiment for a single prompt.
        
        Args:
            prompt: The input prompt
            n_repetitions: Number of generations
            temperature: Sampling temperature (override default)
            top_p: Nucleus sampling parameter (override default)
            top_k: Top-k sampling (override default)
            system_prompt: System instruction (override default)
            context: Additional context/RAG content
            normalize_method: Response normalization method
            filter_noise_flag: Whether to filter noise (override default)
            
        Returns:
            Dictionary with responses and calculated metrics
        """
        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        top_k = top_k if top_k is not None else self.default_top_k
        system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        filter_noise_flag = filter_noise_flag if filter_noise_flag is not None else self.noise_filter_enabled
        
        # Add context to prompt if provided (simple RAG)
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt
        
        print(f"Running {n_repetitions} generations for prompt: {prompt[:50]}...")
        
        # Generate responses
        responses = []
        for i in range(n_repetitions):
            response = self.llm_client.generate(
                full_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            responses.append(response)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_repetitions}")
        
        # Filter noise if enabled
        if filter_noise_flag and self.noise_keywords:
            original_count = len(responses)
            responses = filter_noise(responses, self.noise_keywords)
            filtered_count = original_count - len(responses)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} noisy responses")
        
        # Normalize responses to categories
        categories = [normalize_response(r, method=normalize_method) for r in responses]
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(categories)
        
        # Build result dictionary
        result = {
            'prompt': prompt,
            'full_prompt': full_prompt,
            'n_repetitions': n_repetitions,
            'actual_responses': len(responses),
            'parameters': {
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'system_prompt': system_prompt,
                'context': context is not None,
            },
            'responses': responses,
            'categories': categories,
            'metrics': metrics,
        }
        
        return result
    
    def run_experiment(self, prompts: List[str], 
                       n_repetitions: int = None,
                       temperature: float = None,
                       top_p: float = None,
                       top_k: int = None,
                       system_prompt: str = None,
                       context: str = None,
                       normalize_method: str = "keyword",
                       save_results_flag: bool = True) -> Dict[str, Any]:
        """
        Run full experiment across multiple prompts.
        
        Args:
            prompts: List of input prompts
            n_repetitions: Number of generations per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling
            system_prompt: System instruction
            context: Additional context (applied to all prompts)
            normalize_method: Response normalization method
            save_results_flag: Whether to save results to files
            
        Returns:
            Complete experiment results
        """
        n_repetitions = n_repetitions if n_repetitions is not None else self.default_repetitions
        
        print("=" * 60)
        print("Error 0.5 AI Model Experiment")
        print("=" * 60)
        print(f"Prompts: {len(prompts)}")
        print(f"Repetitions per prompt: {n_repetitions}")
        print(f"Temperature: {temperature if temperature is not None else self.default_temperature}")
        print("=" * 60)
        
        # Run for each prompt
        prompt_results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}] Processing prompt...")
            result = self.run_single_prompt(
                prompt=prompt,
                n_repetitions=n_repetitions,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                system_prompt=system_prompt,
                context=context,
                normalize_method=normalize_method,
            )
            result['prompt_id'] = i
            prompt_results.append(result)
        
        # Build summary
        summary = []
        for result in prompt_results:
            summary.append({
                'prompt_id': result['prompt_id'],
                'prompt': result['prompt'][:100],  # Truncate for summary
                'n_responses': result['actual_responses'],
                'k_rep': result['metrics']['k_rep'],
                'k_div': result['metrics']['k_div'],
                'entropy': result['metrics']['entropy'],
                'd_lack': result['metrics']['d_lack'],
                'omega_05': result['metrics']['omega_05'],
                'n_categories': result['metrics']['n_categories'],
            })
        
        # Compile full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'configuration': {
                'n_repetitions': n_repetitions,
                'temperature': temperature if temperature is not None else self.default_temperature,
                'top_p': top_p if top_p is not None else self.default_top_p,
                'top_k': top_k if top_k is not None else self.default_top_k,
                'system_prompt': system_prompt if system_prompt is not None else self.default_system_prompt,
                'normalize_method': normalize_method,
            },
            'summary': summary,
            'prompts': prompt_results,
        }
        
        # Save results
        if save_results_flag:
            csv_path, json_path = save_results(results, self.results_dir)
            results['output_files'] = {
                'csv': str(csv_path) if csv_path else None,
                'json': str(json_path),
            }
            print(f"\nResults saved to:")
            if csv_path:
                print(f"  CSV: {csv_path}")
            print(f"  JSON: {json_path}")
        
        return results
    
    def compare_experiments(self, prompts: List[str],
                            params_a: Dict[str, Any],
                            params_b: Dict[str, Any],
                            n_repetitions: int = None) -> Dict[str, Any]:
        """
        Run comparison experiment with two different parameter sets.
        
        Args:
            prompts: List of input prompts
            params_a: Parameters for experiment A
            params_b: Parameters for experiment B
            n_repetitions: Number of generations per prompt
            
        Returns:
            Comparison results
        """
        print("\n" + "=" * 60)
        print("COMPARISON EXPERIMENT")
        print("=" * 60)
        
        # Run experiment A
        print("\n--- Running Experiment A ---")
        results_a = self.run_experiment(
            prompts=prompts,
            n_repetitions=n_repetitions,
            temperature=params_a.get('temperature'),
            top_p=params_a.get('top_p'),
            top_k=params_a.get('top_k'),
            system_prompt=params_a.get('system_prompt'),
            context=params_a.get('context'),
            save_results_flag=False,  # Don't save intermediate results
        )
        
        # Run experiment B
        print("\n--- Running Experiment B ---")
        results_b = self.run_experiment(
            prompts=prompts,
            n_repetitions=n_repetitions,
            temperature=params_b.get('temperature'),
            top_p=params_b.get('top_p'),
            top_k=params_b.get('top_k'),
            system_prompt=params_b.get('system_prompt'),
            context=params_b.get('context'),
            save_results_flag=False,
        )
        
        # Compare distributions
        comparisons = []
        for i, (result_a, result_b) in enumerate(zip(results_a['prompts'], results_b['prompts'])):
            dist_a = result_a['metrics']['frequencies']
            dist_b = result_b['metrics']['frequencies']
            
            comparison = self.metrics_calculator.compare_distributions(dist_a, dist_b)
            comparison['prompt_id'] = i
            comparison['prompt'] = prompts[i][:100]
            comparison['params_a'] = params_a
            comparison['params_b'] = params_b
            
            comparisons.append(comparison)
        
        # Build comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_results = {
            'timestamp': timestamp,
            'experiment_a_params': params_a,
            'experiment_b_params': params_b,
            'n_repetitions': n_repetitions,
            'comparisons': comparisons,
            'summary_a': results_a['summary'],
            'summary_b': results_b['summary'],
        }
        
        # Save comparison results
        csv_path, json_path = save_results(
            comparison_results, 
            self.results_dir,
            prefix="comparison"
        )
        
        print(f"\nComparison results saved to:")
        if csv_path:
            print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        for comp in comparisons:
            print(f"\nPrompt {comp['prompt_id']}: {comp['prompt']}...")
            print(f"  K_rep change: {comp['k_rep_change']:+.4f}")
            print(f"  Entropy change: {comp['entropy_change']:+.4f} bits")
            print(f"  KL divergence: {comp['kl_divergence']:.4f}")
        
        return comparison_results


def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Error 0.5 AI Model Experiment")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--prompts', type=str, nargs='+',
                        default=['What is 2+2?', 'Explain quantum computing briefly'],
                        help='List of prompts to test')
    parser.add_argument('--n-repetitions', type=int, default=None,
                        help='Number of repetitions per prompt')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison mode')
    parser.add_argument('--temp-b', type=float, default=1.0,
                        help='Temperature for comparison experiment B')
    
    args = parser.parse_args()
    
    # Load configuration
    from .utils import load_config
    config = load_config(args.config)
    
    # Create runner
    runner = ExperimentRunner(config)
    
    if args.compare:
        # Run comparison
        results = runner.compare_experiments(
            prompts=args.prompts,
            params_a={'temperature': args.temperature},
            params_b={'temperature': args.temp_b},
            n_repetitions=args.n_repetitions,
        )
    else:
        # Run single experiment
        results = runner.run_experiment(
            prompts=args.prompts,
            n_repetitions=args.n_repetitions,
            temperature=args.temperature,
        )
    
    print("\nExperiment completed successfully!")
    return results


if __name__ == "__main__":
    main()
