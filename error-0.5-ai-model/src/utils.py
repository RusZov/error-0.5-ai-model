"""
Utility functions for the Error 0.5 AI Model Experiment.
"""

import os
import json
import yaml
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = _expand_env_vars(config)
    
    return config


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in configuration."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def normalize_response(response: str, method: str = "keyword") -> str:
    """
    Normalize LLM response into categories.
    
    Args:
        response: Raw response text from LLM
        method: Normalization method ("keyword", "first_word", "regex")
        
    Returns:
        Normalized category string
    """
    response = response.strip()
    
    if not response:
        return "empty"
    
    if method == "keyword":
        # Check for key phrases
        response_lower = response.lower()
        
        if any(phrase in response_lower for phrase in ["i don't know", "i do not know", "unknown", "cannot answer", "not sure"]):
            return "uncertain"
        elif any(phrase in response_lower for phrase in ["yes", "correct", "true"]):
            return "affirmative"
        elif any(phrase in response_lower for phrase in ["no", "incorrect", "false"]):
            return "negative"
        elif any(phrase in response_lower for phrase in ["maybe", "possibly", "perhaps"]):
            return "ambiguous"
        else:
            return "informative"
    
    elif method == "first_word":
        # Categorize by first word
        first_word = response.split()[0].lower() if response.split() else ""
        
        if first_word in ["yes", "yeah", "yep", "correct"]:
            return "affirmative"
        elif first_word in ["no", "nope", "incorrect"]:
            return "negative"
        elif first_word in ["maybe", "possibly", "perhaps", "unsure"]:
            return "ambiguous"
        elif first_word in ["i", "the", "a", "an", "it", "this", "that"]:
            return "informative"
        else:
            return "other"
    
    elif method == "regex":
        # Use regex patterns for categorization
        patterns = {
            "uncertain": r"(?i)(don't know|do not know|unknown|uncertain|not sure)",
            "affirmative": r"(?i)^(yes|yeah|yep|correct|true)",
            "negative": r"(?i)^(no|nope|incorrect|false)",
            "ambiguous": r"(?i)(maybe|possibly|perhaps|might)",
        }
        
        for category, pattern in patterns.items():
            if re.search(pattern, response):
                return category
        
        return "informative"
    
    return "other"


def filter_noise(responses: List[str], keywords: List[str]) -> List[str]:
    """
    Filter out noisy responses that contain specified keywords.
    
    Args:
        responses: List of response strings
        keywords: List of keywords indicating noise
        
    Returns:
        Filtered list of responses
    """
    filtered = []
    for response in responses:
        response_lower = response.lower()
        if not any(keyword.lower() in response_lower for keyword in keywords):
            filtered.append(response)
    
    return filtered


def save_results(results: Dict[str, Any], output_dir: str = "results", 
                 prefix: str = "experiment") -> tuple:
    """
    Save experiment results to CSV and JSON files.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save results
        prefix: Filename prefix
        
    Returns:
        Tuple of (csv_path, json_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = Path(results.get('timestamp', 'output')).stem if 'timestamp' in results else 'output'
    base_name = f"{prefix}_{timestamp}" if timestamp != 'output' else prefix
    
    csv_path = output_path / f"{base_name}.csv"
    json_path = output_path / f"{base_name}.json"
    
    # Save as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save summary as CSV if pandas is available
    try:
        import pandas as pd
        
        if 'summary' in results:
            df = pd.DataFrame(results['summary'])
            df.to_csv(csv_path, index=False)
        elif 'prompts' in results:
            # Convert prompt results to DataFrame
            summary_data = []
            for prompt_result in results['prompts']:
                summary_data.append({
                    'prompt_id': prompt_result.get('prompt_id', 0),
                    'prompt': prompt_result.get('prompt', '')[:100],
                    'n_responses': prompt_result.get('n_responses', 0),
                    'k_rep': prompt_result.get('metrics', {}).get('k_rep', 0),
                    'k_div': prompt_result.get('metrics', {}).get('k_div', 0),
                    'entropy': prompt_result.get('metrics', {}).get('entropy', 0),
                    'd_lack': prompt_result.get('metrics', {}).get('d_lack', 0),
                    'omega_05': prompt_result.get('metrics', {}).get('omega_05', 0),
                })
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
        csv_path = None
    
    return csv_path, json_path


def calculate_category_distribution(responses: List[str], 
                                    method: str = "keyword") -> Dict[str, int]:
    """
    Calculate distribution of response categories.
    
    Args:
        responses: List of response strings
        method: Normalization method
        
    Returns:
        Dictionary mapping categories to counts
    """
    distribution = {}
    
    for response in responses:
        category = normalize_response(response, method)
        distribution[category] = distribution.get(category, 0) + 1
    
    return distribution
