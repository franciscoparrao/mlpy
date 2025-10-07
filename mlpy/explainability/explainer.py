"""
Unified Explainer Interface
===========================

Single interface for all explainability methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
import logging

from .shap_explainer import SHAPExplainer, SHAPResults
from .lime_explainer import LIMEExplainer, LIMEExplanation
from .importance import FeatureImportance, PermutationImportance, ImportanceResults
from .counterfactual import CounterfactualExplainer, Counterfactual
from .fairness import FairnessAnalyzer, BiasDetector, FairnessMetrics
from .model_cards import ModelCardGenerator, ModelCard

logger = logging.getLogger(__name__)


class Explainer:
    """
    Unified interface for model explainability.
    
    Provides a single entry point for all explanation methods.
    """
    
    def __init__(
        self,
        model: Any,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        task_type: str = "classification",
        categorical_features: Optional[List[str]] = None,
        sensitive_features: Optional[List[str]] = None
    ):
        """
        Initialize unified explainer.
        
        Args:
            model: Trained model to explain
            data: Reference/training data
            feature_names: Names of features
            class_names: Names of classes (for classification)
            task_type: "classification" or "regression"
            categorical_features: List of categorical feature names
            sensitive_features: List of sensitive feature names for fairness
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names or self._infer_feature_names(data)
        self.class_names = class_names
        self.task_type = task_type
        self.categorical_features = categorical_features or []
        self.sensitive_features = sensitive_features or []
        
        # Initialize sub-explainers lazily
        self._shap_explainer = None
        self._lime_explainer = None
        self._importance_calculator = None
        self._counterfactual_explainer = None
        self._fairness_analyzer = None
        self._bias_detector = None
        self._card_generator = None
    
    def _infer_feature_names(self, data: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Infer feature names from data."""
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        elif isinstance(data, np.ndarray) and data is not None:
            return [f"feature_{i}" for i in range(data.shape[1])]
        else:
            return []
    
    # ============= SHAP Methods =============
    
    def shap_explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        explainer_type: str = "auto"
    ) -> SHAPResults:
        """
        Generate SHAP explanations.
        
        Args:
            X: Data to explain
            explainer_type: Type of SHAP explainer
            
        Returns:
            SHAPResults object
        """
        if self._shap_explainer is None:
            self._shap_explainer = SHAPExplainer(
                self.model,
                self.data,
                self.feature_names,
                self.task_type,
                explainer_type
            )
        
        return self._shap_explainer.explain(X)
    
    def plot_shap_summary(
        self,
        shap_results: Optional[SHAPResults] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        plot_type: str = "dot"
    ):
        """Plot SHAP summary."""
        if shap_results is None and X is not None:
            shap_results = self.shap_explain(X)
        
        if shap_results and self._shap_explainer:
            self._shap_explainer.plot_summary(shap_results, plot_type)
    
    def plot_shap_waterfall(
        self,
        instance_idx: int = 0,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ):
        """Plot SHAP waterfall for single instance."""
        if X is not None:
            shap_results = self.shap_explain(X)
            if self._shap_explainer:
                self._shap_explainer.plot_waterfall(shap_results, instance_idx)
    
    # ============= LIME Methods =============
    
    def lime_explain(
        self,
        instance: Union[pd.Series, np.ndarray],
        num_features: int = 10
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for single instance.
        
        Args:
            instance: Instance to explain
            num_features: Number of features in explanation
            
        Returns:
            LIMEExplanation object
        """
        if self._lime_explainer is None:
            if self.data is None:
                raise ValueError("Reference data required for LIME")
            
            self._lime_explainer = LIMEExplainer(
                self.model,
                self.data,
                self.feature_names,
                self.class_names,
                mode="tabular"
            )
        
        return self._lime_explainer.explain_instance(instance, num_features)
    
    def plot_lime_explanation(
        self,
        explanation: Optional[LIMEExplanation] = None,
        instance: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        """Plot LIME explanation."""
        if explanation is None and instance is not None:
            explanation = self.lime_explain(instance)
        
        if explanation and self._lime_explainer:
            self._lime_explainer.plot_explanation(explanation)
    
    # ============= Feature Importance =============
    
    def global_importance(
        self,
        method: str = "auto",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> ImportanceResults:
        """
        Calculate global feature importance.
        
        Args:
            method: Importance method
            X: Feature data (for some methods)
            y: Target data (for some methods)
            
        Returns:
            ImportanceResults object
        """
        if self._importance_calculator is None:
            self._importance_calculator = FeatureImportance(
                self.model,
                self.feature_names,
                self.task_type
            )
        
        return self._importance_calculator.calculate(method, X, y)
    
    def plot_importance_ranking(
        self,
        importance_results: Optional[ImportanceResults] = None,
        top_n: int = 20
    ):
        """Plot feature importance ranking."""
        if importance_results is None:
            importance_results = self.global_importance()
        
        if importance_results:
            importance_results.plot(top_n)
    
    # ============= Counterfactuals =============
    
    def counterfactual(
        self,
        instance: Union[pd.Series, np.ndarray],
        desired_outcome: Optional[Union[int, float]] = None,
        max_features_changed: Optional[int] = None,
        method: str = "optimization"
    ) -> Counterfactual:
        """
        Generate counterfactual explanation.
        
        Args:
            instance: Instance to explain
            desired_outcome: Desired prediction
            max_features_changed: Max features to change
            method: Generation method
            
        Returns:
            Counterfactual object
        """
        if self._counterfactual_explainer is None:
            self._counterfactual_explainer = CounterfactualExplainer(
                self.model,
                self.feature_names,
                categorical_features=self.categorical_features
            )
        
        return self._counterfactual_explainer.generate(
            instance,
            desired_outcome,
            max_features_changed=max_features_changed,
            method=method
        )
    
    # ============= Fairness Analysis =============
    
    def analyze_fairness(
        self,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: str
    ) -> FairnessMetrics:
        """
        Analyze model fairness.
        
        Args:
            X: Feature data
            y_true: True labels
            sensitive_feature: Sensitive feature to analyze
            
        Returns:
            FairnessMetrics object
        """
        if self._fairness_analyzer is None:
            self._fairness_analyzer = FairnessAnalyzer(
                self.model,
                self.sensitive_features
            )
        
        return self._fairness_analyzer.analyze(X, y_true, sensitive_feature)
    
    def detect_bias(
        self,
        data: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """
        Detect bias in data.
        
        Args:
            data: Dataset to analyze
            target: Target column name
            
        Returns:
            Dictionary of bias indicators
        """
        if self._bias_detector is None:
            self._bias_detector = BiasDetector(self.model)
        
        return self._bias_detector.detect_data_bias(
            data,
            self.sensitive_features,
            target
        )
    
    # ============= Model Cards =============
    
    def generate_model_card(
        self,
        model_name: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        intended_uses: Optional[List[str]] = None,
        ethical_considerations: Optional[List[str]] = None,
        owner: str = "Unknown",
        contact: str = "Unknown"
    ) -> ModelCard:
        """
        Generate model card.
        
        Args:
            model_name: Name of the model
            performance_metrics: Performance metrics
            intended_uses: List of intended uses
            ethical_considerations: Ethical considerations
            owner: Model owner
            contact: Contact info
            
        Returns:
            ModelCard object
        """
        if self._card_generator is None:
            self._card_generator = ModelCardGenerator(
                self.model,
                model_name or "Model"
            )
        
        return self._card_generator.generate(
            performance_metrics=performance_metrics,
            intended_uses=intended_uses,
            ethical_considerations=ethical_considerations,
            owner=owner,
            contact=contact
        )
    
    # ============= Comprehensive Report =============
    
    def generate_full_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_instance: Optional[Union[pd.Series, np.ndarray]] = None,
        output_dir: str = "./explainability_report"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report.
        
        Args:
            X: Data to analyze
            y: True labels (optional)
            sample_instance: Sample instance for local explanations
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with all explanation results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'model_type': self.model.__class__.__name__,
            'task_type': self.task_type,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        # 1. Global Feature Importance
        try:
            logger.info("Calculating global feature importance...")
            importance = self.global_importance(X=X, y=y)
            report['feature_importance'] = importance.to_dataframe().to_dict()
            
            # Save plot
            import matplotlib.pyplot as plt
            plt.figure()
            importance.plot()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
        
        # 2. SHAP Analysis
        try:
            logger.info("Running SHAP analysis...")
            shap_results = self.shap_explain(X[:100] if len(X) > 100 else X)
            report['shap_importance'] = shap_results.get_feature_importance().to_dict()
        except Exception as e:
            logger.warning(f"Could not run SHAP analysis: {e}")
        
        # 3. LIME for sample instance
        if sample_instance is not None:
            try:
                logger.info("Generating LIME explanation...")
                lime_exp = self.lime_explain(sample_instance)
                report['lime_explanation'] = lime_exp.as_dict()
            except Exception as e:
                logger.warning(f"Could not generate LIME explanation: {e}")
        
        # 4. Counterfactual
        if sample_instance is not None:
            try:
                logger.info("Generating counterfactual...")
                cf = self.counterfactual(sample_instance)
                report['counterfactual'] = {
                    'changes': cf.feature_changes,
                    'distance': cf.distance,
                    'sparsity': cf.sparsity,
                    'valid': cf.validity
                }
            except Exception as e:
                logger.warning(f"Could not generate counterfactual: {e}")
        
        # 5. Fairness (if sensitive features defined)
        if self.sensitive_features and isinstance(X, pd.DataFrame) and y is not None:
            try:
                logger.info("Analyzing fairness...")
                fairness_results = {}
                for feature in self.sensitive_features:
                    if feature in X.columns:
                        metrics = self.analyze_fairness(X, y, feature)
                        fairness_results[feature] = metrics.get_summary()
                report['fairness'] = fairness_results
            except Exception as e:
                logger.warning(f"Could not analyze fairness: {e}")
        
        # 6. Model Card
        try:
            logger.info("Generating model card...")
            card = self.generate_model_card(
                performance_metrics=report.get('performance_metrics', {})
            )
            
            # Save in multiple formats
            card.to_json(os.path.join(output_dir, 'model_card.json'))
            card.to_html(os.path.join(output_dir, 'model_card.html'))
            card.to_markdown(os.path.join(output_dir, 'model_card.md'))
            
            report['model_card_generated'] = True
        except Exception as e:
            logger.warning(f"Could not generate model card: {e}")
        
        # Save full report
        import json
        with open(os.path.join(output_dir, 'full_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Explainability report saved to {output_dir}")
        
        return report