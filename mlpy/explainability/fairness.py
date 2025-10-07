"""
Fairness and Bias Detection
===========================

Analyze models for fairness and detect potential biases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings
import logging
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    
    demographic_parity: float
    equal_opportunity: float
    equalized_odds: float
    disparate_impact: float
    statistical_parity: float
    groups: Dict[str, Dict[str, float]]  # Group-specific metrics
    
    def is_fair(self, threshold: float = 0.8) -> bool:
        """Check if model meets fairness threshold."""
        return (self.disparate_impact >= threshold and 
                abs(self.demographic_parity) <= (1 - threshold))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of fairness metrics."""
        return {
            'demographic_parity': self.demographic_parity,
            'equal_opportunity': self.equal_opportunity,
            'equalized_odds': self.equalized_odds,
            'disparate_impact': self.disparate_impact,
            'statistical_parity': self.statistical_parity,
            'is_fair': self.is_fair(),
            'num_groups': len(self.groups)
        }


class FairnessAnalyzer:
    """Analyze model fairness across different groups."""
    
    def __init__(
        self,
        model: Any,
        sensitive_features: List[str],
        favorable_outcome: Union[int, float] = 1,
        reference_group: Optional[str] = None
    ):
        """
        Initialize fairness analyzer.
        
        Args:
            model: Trained model to analyze
            sensitive_features: List of sensitive feature names
            favorable_outcome: Value indicating favorable outcome
            reference_group: Reference group for comparisons
        """
        self.model = model
        self.sensitive_features = sensitive_features
        self.favorable_outcome = favorable_outcome
        self.reference_group = reference_group
    
    def analyze(
        self,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: str
    ) -> FairnessMetrics:
        """
        Analyze fairness for a sensitive feature.
        
        Args:
            X: Feature data
            y_true: True labels
            sensitive_feature: Name of sensitive feature to analyze
            
        Returns:
            FairnessMetrics object
        """
        if sensitive_feature not in X.columns:
            raise ValueError(f"Sensitive feature {sensitive_feature} not in data")
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Get unique groups
        groups = X[sensitive_feature].unique()
        
        # Calculate metrics for each group
        group_metrics = {}
        
        for group in groups:
            group_mask = X[sensitive_feature] == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate group-specific metrics
            group_metrics[str(group)] = self._calculate_group_metrics(
                group_y_true, group_y_pred
            )
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(group_metrics)
        
        return FairnessMetrics(
            demographic_parity=fairness_metrics['demographic_parity'],
            equal_opportunity=fairness_metrics['equal_opportunity'],
            equalized_odds=fairness_metrics['equalized_odds'],
            disparate_impact=fairness_metrics['disparate_impact'],
            statistical_parity=fairness_metrics['statistical_parity'],
            groups=group_metrics
        )
    
    def _calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a specific group."""
        # Positive rate
        positive_rate = np.mean(y_pred == self.favorable_outcome)
        
        # True positive rate (sensitivity)
        true_positives = np.sum((y_true == self.favorable_outcome) & 
                               (y_pred == self.favorable_outcome))
        actual_positives = np.sum(y_true == self.favorable_outcome)
        tpr = true_positives / actual_positives if actual_positives > 0 else 0
        
        # False positive rate
        false_positives = np.sum((y_true != self.favorable_outcome) & 
                                (y_pred == self.favorable_outcome))
        actual_negatives = np.sum(y_true != self.favorable_outcome)
        fpr = false_positives / actual_negatives if actual_negatives > 0 else 0
        
        # Accuracy
        accuracy = np.mean(y_true == y_pred)
        
        return {
            'size': len(y_true),
            'positive_rate': positive_rate,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'accuracy': accuracy,
            'favorable_count': np.sum(y_pred == self.favorable_outcome),
            'favorable_ratio': positive_rate
        }
    
    def _calculate_fairness_metrics(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate fairness metrics across groups."""
        # Get reference group
        if self.reference_group and self.reference_group in group_metrics:
            ref_group = self.reference_group
        else:
            # Use group with highest positive rate as reference
            ref_group = max(group_metrics.keys(), 
                          key=lambda g: group_metrics[g]['positive_rate'])
        
        ref_metrics = group_metrics[ref_group]
        
        # Demographic parity: difference in positive rates
        dp_diffs = []
        for group, metrics in group_metrics.items():
            if group != ref_group:
                dp_diffs.append(metrics['positive_rate'] - ref_metrics['positive_rate'])
        
        demographic_parity = max(abs(d) for d in dp_diffs) if dp_diffs else 0
        
        # Equal opportunity: difference in TPR
        eo_diffs = []
        for group, metrics in group_metrics.items():
            if group != ref_group:
                eo_diffs.append(metrics['true_positive_rate'] - 
                              ref_metrics['true_positive_rate'])
        
        equal_opportunity = max(abs(d) for d in eo_diffs) if eo_diffs else 0
        
        # Equalized odds: max difference in TPR and FPR
        odds_diffs = []
        for group, metrics in group_metrics.items():
            if group != ref_group:
                tpr_diff = abs(metrics['true_positive_rate'] - 
                             ref_metrics['true_positive_rate'])
                fpr_diff = abs(metrics['false_positive_rate'] - 
                             ref_metrics['false_positive_rate'])
                odds_diffs.append(max(tpr_diff, fpr_diff))
        
        equalized_odds = max(odds_diffs) if odds_diffs else 0
        
        # Disparate impact: ratio of positive rates
        di_ratios = []
        for group, metrics in group_metrics.items():
            if group != ref_group and ref_metrics['positive_rate'] > 0:
                ratio = metrics['positive_rate'] / ref_metrics['positive_rate']
                di_ratios.append(ratio)
        
        disparate_impact = min(di_ratios) if di_ratios else 1.0
        
        # Statistical parity
        positive_rates = [m['positive_rate'] for m in group_metrics.values()]
        statistical_parity = np.std(positive_rates)
        
        return {
            'demographic_parity': demographic_parity,
            'equal_opportunity': equal_opportunity,
            'equalized_odds': equalized_odds,
            'disparate_impact': disparate_impact,
            'statistical_parity': statistical_parity
        }
    
    def plot_fairness(
        self,
        fairness_metrics: FairnessMetrics,
        metric_names: Optional[List[str]] = None
    ):
        """Plot fairness metrics visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if metric_names is None:
            metric_names = ['demographic_parity', 'equal_opportunity', 
                          'equalized_odds', 'disparate_impact']
        
        # Prepare data
        metrics = []
        values = []
        
        for name in metric_names:
            if hasattr(fairness_metrics, name):
                metrics.append(name.replace('_', ' ').title())
                values.append(getattr(fairness_metrics, name))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if v >= 0.8 and v <= 1.25 else 'red' for v in values]
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add threshold lines
        ax.axhline(y=0.8, color='orange', linestyle='--', label='Fairness threshold')
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Fairness Metrics Analysis')
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class BiasDetector:
    """Detect various types of bias in model and data."""
    
    def __init__(self, model: Any = None):
        """
        Initialize bias detector.
        
        Args:
            model: Optional model to analyze
        """
        self.model = model
    
    def detect_data_bias(
        self,
        data: pd.DataFrame,
        sensitive_features: List[str],
        target: str
    ) -> Dict[str, Any]:
        """
        Detect bias in training data.
        
        Args:
            data: Training data
            sensitive_features: Sensitive feature columns
            target: Target column
            
        Returns:
            Dictionary of bias indicators
        """
        biases = {}
        
        for feature in sensitive_features:
            if feature not in data.columns:
                continue
            
            # Check class imbalance per group
            group_targets = data.groupby(feature)[target].value_counts(normalize=True)
            
            # Calculate imbalance
            imbalance_scores = []
            for group in data[feature].unique():
                if group in group_targets.index:
                    group_dist = group_targets[group]
                    # Compare to overall distribution
                    overall_dist = data[target].value_counts(normalize=True)
                    
                    # KL divergence as imbalance measure
                    kl_div = 0
                    for class_val in overall_dist.index:
                        if class_val in group_dist:
                            p = overall_dist[class_val]
                            q = group_dist[class_val]
                            if p > 0 and q > 0:
                                kl_div += p * np.log(p / q)
                    
                    imbalance_scores.append(kl_div)
            
            biases[feature] = {
                'group_sizes': data[feature].value_counts().to_dict(),
                'max_imbalance': max(imbalance_scores) if imbalance_scores else 0,
                'mean_imbalance': np.mean(imbalance_scores) if imbalance_scores else 0
            }
        
        return biases
    
    def detect_prediction_bias(
        self,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        sensitive_features: List[str]
    ) -> Dict[str, Any]:
        """
        Detect bias in model predictions.
        
        Args:
            X: Feature data
            y_pred: Model predictions
            sensitive_features: Sensitive feature columns
            
        Returns:
            Dictionary of bias indicators
        """
        if self.model is None:
            raise ValueError("Model required for prediction bias detection")
        
        biases = {}
        
        for feature in sensitive_features:
            if feature not in X.columns:
                continue
            
            # Analyze prediction distribution per group
            groups = X[feature].unique()
            group_predictions = {}
            
            for group in groups:
                group_mask = X[feature] == group
                group_preds = y_pred[group_mask]
                
                group_predictions[str(group)] = {
                    'mean': np.mean(group_preds),
                    'std': np.std(group_preds),
                    'positive_rate': np.mean(group_preds == 1) if len(np.unique(group_preds)) == 2 else None
                }
            
            # Calculate bias metrics
            pred_means = [p['mean'] for p in group_predictions.values()]
            
            biases[feature] = {
                'groups': group_predictions,
                'mean_difference': max(pred_means) - min(pred_means),
                'std_between_groups': np.std(pred_means),
                'coefficient_of_variation': np.std(pred_means) / (np.mean(pred_means) + 1e-10)
            }
        
        return biases
    
    def detect_representation_bias(
        self,
        data: pd.DataFrame,
        sensitive_features: List[str],
        expected_proportions: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Detect representation bias in data.
        
        Args:
            data: Dataset to analyze
            sensitive_features: Sensitive feature columns
            expected_proportions: Expected proportions for each group
            
        Returns:
            Dictionary of representation bias indicators
        """
        biases = {}
        
        for feature in sensitive_features:
            if feature not in data.columns:
                continue
            
            actual_proportions = data[feature].value_counts(normalize=True).to_dict()
            
            bias_info = {
                'actual_proportions': actual_proportions,
                'total_groups': len(actual_proportions),
                'smallest_group': min(actual_proportions.values()),
                'largest_group': max(actual_proportions.values()),
                'imbalance_ratio': max(actual_proportions.values()) / (min(actual_proportions.values()) + 1e-10)
            }
            
            # Compare to expected if provided
            if expected_proportions and feature in expected_proportions:
                expected = expected_proportions[feature]
                differences = {}
                
                for group, expected_prop in expected.items():
                    actual_prop = actual_proportions.get(group, 0)
                    differences[group] = abs(actual_prop - expected_prop)
                
                bias_info['expected_proportions'] = expected
                bias_info['max_deviation'] = max(differences.values())
                bias_info['mean_deviation'] = np.mean(list(differences.values()))
            
            biases[feature] = bias_info
        
        return biases