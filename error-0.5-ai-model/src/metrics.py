"""
Metrics calculator for Error 0.5 AI Model Experiment.

Implements the scientific theory of "Error 0.5" including:
- K_rep (Repetition Coefficient)
- K_div (Diversity Coefficient)
- Entropy (H)
- D_lack (Data Lack Index)
- Ω_0.5 (Useful Error Index)
"""

import math
from typing import Dict, List, Any, Optional
from collections import Counter


class MetricsCalculator:
    """
    Calculate metrics for analyzing LLM response consistency and uncertainty.
    """
    
    def __init__(self, quality_threshold: float = 0.8,
                 p_gain: float = 1.0, b_value: float = 1.0,
                 p_loss: float = 0.5, l_value: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            quality_threshold: Threshold for quality assessment (Q_C)
            p_gain: Probability weight for gain in Ω_0.5
            b_value: Benefit multiplier in Ω_0.5
            p_loss: Probability weight for loss in Ω_0.5
            l_value: Loss multiplier in Ω_0.5
        """
        self.quality_threshold = quality_threshold
        self.p_gain = p_gain
        self.b_value = b_value
        self.p_loss = p_loss
        self.l_value = l_value
    
    def calculate_frequencies(self, categories: List[str]) -> Dict[str, float]:
        """
        Calculate frequency distribution p_i = n_i / N.
        
        Args:
            categories: List of category labels
            
        Returns:
            Dictionary mapping categories to frequencies
        """
        n_total = len(categories)
        if n_total == 0:
            return {}
        
        counts = Counter(categories)
        frequencies = {cat: count / n_total for cat, count in counts.items()}
        
        return frequencies
    
    def calculate_k_rep(self, frequencies: Dict[str, float]) -> float:
        """
        Calculate Repetition Coefficient: K_rep = max(p_i).
        
        Measures the dominance of the most frequent category.
        Higher values indicate more consistent/repetitive responses.
        
        Args:
            frequencies: Category frequency distribution
            
        Returns:
            K_rep value in range [0, 1]
        """
        if not frequencies:
            return 0.0
        
        return max(frequencies.values())
    
    def calculate_k_div(self, k_rep: float) -> float:
        """
        Calculate Diversity Coefficient: K_div = 1 - K_rep.
        
        Measures the diversity of responses.
        Higher values indicate more varied responses.
        
        Args:
            k_rep: Repetition coefficient
            
        Returns:
            K_div value in range [0, 1]
        """
        return 1.0 - k_rep
    
    def calculate_entropy(self, frequencies: Dict[str, float]) -> float:
        """
        Calculate Shannon Entropy: H = -Σ p_i * log2(p_i).
        
        Measures the uncertainty/information content in responses.
        Higher entropy indicates more unpredictable responses.
        
        Args:
            frequencies: Category frequency distribution
            
        Returns:
            Entropy value in bits
        """
        if not frequencies:
            return 0.0
        
        entropy = 0.0
        for p in frequencies.values():
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_d_lack(self, categories: List[str], 
                         quality_threshold: Optional[float] = None) -> float:
        """
        Calculate Data Lack Index: D_lack = 1 - Q_C.
        
        Q_C is the quality/confidence score, calculated as:
        - User-provided threshold, or
        - 1 - (proportion of "uncertain" or contradictory responses)
        
        Args:
            categories: List of category labels
            quality_threshold: Optional user-provided Q_C value
            
        Returns:
            D_lack value in range [0, 1]
        """
        if quality_threshold is not None:
            return 1.0 - quality_threshold
        
        if not categories:
            return 1.0
        
        # Count uncertain/contradictory responses
        uncertain_categories = {'uncertain', 'ambiguous', 'empty', 'other'}
        n_uncertain = sum(1 for cat in categories if cat.lower() in uncertain_categories)
        
        proportion_uncertain = n_uncertain / len(categories)
        q_c = 1.0 - proportion_uncertain
        
        return 1.0 - q_c
    
    def calculate_omega_05(self, frequencies: Dict[str, float],
                           categories: List[str],
                           p_gain: Optional[float] = None,
                           b_value: Optional[float] = None,
                           p_loss: Optional[float] = None,
                           l_value: Optional[float] = None) -> float:
        """
        Calculate Useful Error Index: Ω_0.5 = P_gain * B - P_loss * L.
        
        This metric evaluates whether unexpected/diverse responses provide
        value (gain) versus confusion/cost (loss).
        
        P_gain: Probability of beneficial unexpected responses
        B: Benefit multiplier for unexpected insights
        P_loss: Probability of harmful errors/contradictions
        L: Loss multiplier for errors
        
        Args:
            frequencies: Category frequency distribution
            categories: List of category labels
            p_gain: Override for P_gain parameter
            b_value: Override for B parameter
            p_loss: Override for P_loss parameter
            l_value: Override for L parameter
            
        Returns:
            Ω_0.5 value (can be negative)
        """
        # Use instance defaults if not overridden
        p_gain = p_gain if p_gain is not None else self.p_gain
        b_value = b_value if b_value is not None else self.b_value
        p_loss = p_loss if p_loss is not None else self.p_loss
        l_value = l_value if l_value is not None else self.l_value
        
        if not frequencies or not categories:
            return 0.0
        
        # Estimate P_gain: probability of diverse but informative responses
        # Consider "informative" and "ambiguous" as potentially beneficial
        beneficial_categories = {'informative', 'ambiguous'}
        p_gain_est = sum(frequencies.get(cat, 0) for cat in beneficial_categories)
        
        # Estimate P_loss: probability of uncertain/erroneous responses
        loss_categories = {'uncertain', 'empty', 'other'}
        p_loss_est = sum(frequencies.get(cat, 0) for cat in loss_categories)
        
        # Calculate Ω_0.5
        omega = (p_gain * p_gain_est * b_value) - (p_loss * p_loss_est * l_value)
        
        return omega
    
    def calculate_all_metrics(self, categories: List[str],
                              quality_threshold: Optional[float] = None,
                              p_gain: Optional[float] = None,
                              b_value: Optional[float] = None,
                              p_loss: Optional[float] = None,
                              l_value: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate all metrics for a set of categorized responses.
        
        Args:
            categories: List of category labels
            quality_threshold: Optional Q_C override
            p_gain, b_value, p_loss, l_value: Ω_0.5 parameter overrides
            
        Returns:
            Dictionary with all calculated metrics
        """
        frequencies = self.calculate_frequencies(categories)
        k_rep = self.calculate_k_rep(frequencies)
        k_div = self.calculate_k_div(k_rep)
        entropy = self.calculate_entropy(frequencies)
        d_lack = self.calculate_d_lack(categories, quality_threshold)
        omega_05 = self.calculate_omega_05(
            frequencies, categories, p_gain, b_value, p_loss, l_value
        )
        
        return {
            'frequencies': frequencies,
            'k_rep': k_rep,
            'k_div': k_div,
            'entropy': entropy,
            'd_lack': d_lack,
            'omega_05': omega_05,
            'n_responses': len(categories),
            'n_categories': len(frequencies),
        }
    
    def compare_distributions(self, dist1: Dict[str, float], 
                              dist2: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare two frequency distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Comparison results including KL divergence approximation
        """
        # Get all categories
        all_categories = set(dist1.keys()) | set(dist2.keys())
        
        # Calculate differences
        changes = {}
        for cat in all_categories:
            p1 = dist1.get(cat, 0)
            p2 = dist2.get(cat, 0)
            changes[cat] = {
                'before': p1,
                'after': p2,
                'delta': p2 - p1,
                'relative_change': (p2 - p1) / p1 if p1 > 0 else float('inf') if p2 > 0 else 0
            }
        
        # Calculate approximate KL divergence
        kl_div = 0.0
        for cat in all_categories:
            p1 = dist1.get(cat, 1e-10)  # Small epsilon to avoid log(0)
            p2 = dist2.get(cat, 1e-10)
            if p1 > 0 and p2 > 0:
                kl_div += p2 * math.log2(p2 / p1)
        
        # Compare summary metrics
        metrics1 = {
            'k_rep': self.calculate_k_rep(dist1),
            'entropy': self.calculate_entropy(dist1),
        }
        metrics2 = {
            'k_rep': self.calculate_k_rep(dist2),
            'entropy': self.calculate_entropy(dist2),
        }
        
        return {
            'category_changes': changes,
            'kl_divergence': kl_div,
            'metrics_before': metrics1,
            'metrics_after': metrics2,
            'k_rep_change': metrics2['k_rep'] - metrics1['k_rep'],
            'entropy_change': metrics2['entropy'] - metrics1['entropy'],
        }
