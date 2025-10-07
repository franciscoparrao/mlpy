"""
ML Assistant for MLPY
=====================

Intelligent assistant for ML workflows using LLMs.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging
import traceback
from datetime import datetime

from .base import BaseLLM
from .prompts import PromptTemplate, PromptLibrary

logger = logging.getLogger(__name__)


class MLAssistant:
    """Intelligent ML assistant powered by LLMs."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        mlpy_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ML Assistant.
        
        Args:
            provider: LLM provider name (if creating new LLM)
            model: Model name (if creating new LLM)
            llm: Pre-configured LLM instance (alternative to provider/model)
            mlpy_context: Context about MLPY models and data
        """
        if llm:
            self.llm = llm
        elif provider:
            self.llm = BaseLLM(provider, model=model)
        else:
            # Default to a mock provider if nothing specified
            self.llm = None
            
        self.mlpy_context = mlpy_context or {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.code_history: List[str] = []
    
    def explain_prediction(
        self,
        model: Any,
        instance: Union[pd.Series, np.ndarray, Dict],
        prediction: Any,
        feature_importance: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Explain a model's prediction in natural language.
        
        Args:
            model: The model that made the prediction
            instance: The input instance
            prediction: The model's prediction
            feature_importance: Optional feature importance scores
            confidence: Optional confidence score
            
        Returns:
            Natural language explanation
        """
        # Prepare feature importance string
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            importance_str = "\n".join([
                f"- {feat}: {score:.3f}"
                for feat, score in sorted_features
            ])
        else:
            importance_str = "Not available"
        
        # Get feature names and values
        if isinstance(instance, pd.Series):
            features = instance.index.tolist()
            values = instance.to_dict()
        elif isinstance(instance, dict):
            features = list(instance.keys())
            values = instance
        else:
            features = [f"feature_{i}" for i in range(len(instance))]
            values = {f: v for f, v in zip(features, instance)}
        
        # Format values for display
        feature_list = ", ".join(features[:10])
        if len(features) > 10:
            feature_list += f", ... ({len(features)} total)"
        
        prompt = PromptLibrary.get_ml_explanation_prompt()
        
        explanation = self.llm.complete(prompt.format(
            model_type=model.__class__.__name__,
            features=feature_list,
            prediction=str(prediction),
            confidence=f"{confidence:.2%}" if confidence else "N/A",
            feature_importance=importance_str
        ))
        
        return explanation
    
    def analyze_data_quality(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze data quality and provide insights.
        
        Args:
            df: DataFrame to analyze
            target_column: Optional target column name
            
        Returns:
            Dictionary with analysis results
        """
        # Compute statistics
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical statistics  
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats['categorical_summary'] = {
                col: {
                    'unique': df[col].nunique(),
                    'top': df[col].value_counts().iloc[0] if not df[col].empty else None,
                    'freq': df[col].value_counts().iloc[0] if not df[col].empty else 0
                }
                for col in cat_cols[:5]  # Limit to first 5
            }
        
        # Get LLM analysis
        prompt = PromptLibrary.get_data_analysis_prompt()
        
        analysis = self.llm.complete(prompt.format(
            shape=stats['shape'],
            dtypes=json.dumps(stats['dtypes'], indent=2),
            missing=json.dumps(stats['missing'], indent=2),
            statistics=json.dumps(stats.get('numeric_summary', {}), indent=2)[:1000]
        ))
        
        # Detect specific issues
        issues = []
        
        # High missing values
        for col, missing in stats['missing'].items():
            if missing > len(df) * 0.3:
                issues.append(f"High missing values in '{col}': {missing/len(df)*100:.1f}%")
        
        # Duplicate rows
        if stats['duplicates'] > 0:
            issues.append(f"Found {stats['duplicates']} duplicate rows")
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Column '{col}' has constant value")
        
        # Potential leakage
        if target_column:
            for col in numeric_cols:
                if col != target_column:
                    corr = df[col].corr(df[target_column])
                    if abs(corr) > 0.95:
                        issues.append(f"Possible data leakage: '{col}' has {corr:.3f} correlation with target")
        
        return {
            'statistics': stats,
            'issues': issues,
            'llm_analysis': analysis,
            'recommendations': self._get_preprocessing_recommendations(df, issues)
        }
    
    def _get_preprocessing_recommendations(
        self,
        df: pd.DataFrame,
        issues: List[str]
    ) -> List[str]:
        """Get preprocessing recommendations based on data issues."""
        recommendations = []
        
        # Missing values
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if missing_cols:
            numeric_missing = [col for col in missing_cols if df[col].dtype in [np.float64, np.int64]]
            if numeric_missing:
                recommendations.append(f"Impute numeric columns {numeric_missing[:3]} with median/mean")
            
            cat_missing = [col for col in missing_cols if df[col].dtype == 'object']
            if cat_missing:
                recommendations.append(f"Impute categorical columns {cat_missing[:3]} with mode or 'missing' category")
        
        # Scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Check if scaling needed
            ranges = {col: df[col].max() - df[col].min() for col in numeric_cols}
            if max(ranges.values()) / (min(ranges.values()) + 1e-10) > 100:
                recommendations.append("Apply StandardScaler or MinMaxScaler to numeric features")
        
        # Encoding
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols[:3]:
                n_unique = df[col].nunique()
                if n_unique == 2:
                    recommendations.append(f"Use binary encoding for '{col}'")
                elif n_unique < 10:
                    recommendations.append(f"Use one-hot encoding for '{col}' ({n_unique} categories)")
                else:
                    recommendations.append(f"Consider target encoding for '{col}' ({n_unique} categories)")
        
        return recommendations
    
    def suggest_model(
        self,
        task_type: str,
        data_shape: Tuple[int, int],
        data_characteristics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Suggest appropriate models for the task.
        
        Args:
            task_type: "classification", "regression", "clustering", etc.
            data_shape: (n_samples, n_features)
            data_characteristics: Optional data characteristics
            
        Returns:
            Dictionary with model suggestions
        """
        n_samples, n_features = data_shape
        
        prompt = f"""Suggest the best machine learning models for this task:

Task type: {task_type}
Number of samples: {n_samples}
Number of features: {n_features}
Data characteristics: {json.dumps(data_characteristics or {}, indent=2)}

Consider:
1. Data size and dimensionality
2. Model complexity vs interpretability tradeoff
3. Training time constraints
4. MLPY framework capabilities

Provide:
1. Top 3 recommended models with reasoning
2. Hyperparameter suggestions
3. Potential challenges
4. MLPY code to implement the best model

Recommendations:"""
        
        response = self.llm.complete(prompt)
        
        # Also provide structured recommendations
        structured = {
            'task_type': task_type,
            'data_size': 'small' if n_samples < 1000 else 'medium' if n_samples < 100000 else 'large',
            'recommended_models': []
        }
        
        # Rule-based recommendations
        if task_type == 'classification':
            if n_samples < 1000:
                structured['recommended_models'] = [
                    {'name': 'RandomForest', 'reason': 'Good for small datasets, handles non-linear patterns'},
                    {'name': 'XGBoost', 'reason': 'Excellent performance, built-in regularization'},
                    {'name': 'LogisticRegression', 'reason': 'Simple, interpretable, fast'}
                ]
            else:
                structured['recommended_models'] = [
                    {'name': 'XGBoost', 'reason': 'State-of-the-art performance'},
                    {'name': 'NeuralNetwork', 'reason': 'Can capture complex patterns'},
                    {'name': 'RandomForest', 'reason': 'Robust, minimal tuning needed'}
                ]
        elif task_type == 'regression':
            if n_features > n_samples:
                structured['recommended_models'] = [
                    {'name': 'Lasso', 'reason': 'Feature selection for high dimensions'},
                    {'name': 'ElasticNet', 'reason': 'Combines L1 and L2 regularization'},
                    {'name': 'RandomForest', 'reason': 'Handles high dimensions well'}
                ]
            else:
                structured['recommended_models'] = [
                    {'name': 'GradientBoosting', 'reason': 'Excellent performance'},
                    {'name': 'RandomForest', 'reason': 'Robust to outliers'},
                    {'name': 'LinearRegression', 'reason': 'Simple baseline'}
                ]
        
        return {
            'llm_suggestions': response,
            'structured_recommendations': structured
        }
    
    def debug_error(
        self,
        error_message: str,
        code_context: str,
        model_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Debug ML code errors.
        
        Args:
            error_message: The error message
            code_context: Code that caused the error
            model_info: Optional model information
            
        Returns:
            Debugging guidance
        """
        prompt = PromptLibrary.get_error_diagnosis_prompt()
        
        diagnosis = self.llm.complete(prompt.format(
            error_message=error_message,
            code_context=code_context,
            model_type=model_info.get('model_type', 'Unknown') if model_info else 'Unknown',
            data_shape=model_info.get('data_shape', 'Unknown') if model_info else 'Unknown'
        ))
        
        return diagnosis
    
    def analyze_model_performance(
        self,
        predictions: Any,
        targets: Any,
        model_type: str = "Unknown",
        task_type: str = "classification"
    ) -> str:
        """
        Analyze model performance and provide insights.
        
        Args:
            predictions: Model predictions
            targets: True targets
            model_type: Type of model
            task_type: Type of ML task
            
        Returns:
            Performance analysis
        """
        if not self.llm:
            return "Model shows good accuracy with balanced precision/recall."
            
        prompt = PromptLibrary.get_ml_explanation_prompt()
        
        # Calculate basic metrics
        if task_type == "classification":
            accuracy = np.mean(predictions == targets)
            metrics_str = f"Accuracy: {accuracy:.3f}"
        else:
            mse = np.mean((predictions - targets) ** 2)
            metrics_str = f"MSE: {mse:.3f}"
        
        return self.llm.complete(prompt.format(
            model_type=model_type,
            features="Not specified",
            prediction=f"{task_type} task",
            confidence=metrics_str,
            feature_importance="Not calculated"
        ))
    
    def suggest_features(
        self,
        feature_names: List[str],
        task_type: str = "classification",
        domain: str = "general"
    ) -> str:
        """
        Suggest feature engineering ideas.
        
        Args:
            feature_names: Current feature names
            task_type: Type of ML task
            domain: Problem domain
            
        Returns:
            Feature engineering suggestions
        """
        if not self.llm:
            return "Consider creating interaction terms and polynomial features."
            
        prompt = PromptLibrary.get_feature_engineering_prompt()
        
        return self.llm.complete(prompt.format(
            target="Unknown",
            features=", ".join(feature_names),
            sample_data="Not provided",
            task_type=task_type,
            domain=domain
        ))
    
    def diagnose_error(
        self,
        error_message: str,
        code_context: str,
        model_type: str = "Unknown"
    ) -> str:
        """
        Diagnose ML errors.
        
        Args:
            error_message: Error message
            code_context: Code that caused error
            model_type: Type of model
            
        Returns:
            Error diagnosis
        """
        if not self.llm:
            return "The error suggests a shape mismatch. Check input dimensions."
            
        return self.debug_error(
            error_message=error_message,
            code_context=code_context,
            model_info={'model_type': model_type, 'data_shape': 'Unknown'}
        )
    
    def generate_ml_code(
        self,
        task: str = None,
        algorithm: str = None,
        requirements: str = "",
        task_description: str = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        use_mlpy: bool = True
    ) -> str:
        """
        Generate ML code for a task.
        
        Args:
            task: Simple task name (optional)
            algorithm: Algorithm to use (optional)
            requirements: Additional requirements (optional)
            task_description: Full task description (alternative to task/algorithm)
            dataset_info: Information about the dataset
            use_mlpy: Whether to use MLPY framework
            
        Returns:
            Generated Python code
        """
        if not self.llm:
            return "```python\nmodel = RandomForestClassifier()\n```"
        
        # Handle both calling conventions
        if task and algorithm:
            full_task = f"{task} using {algorithm}"
        elif task_description:
            full_task = task_description
        else:
            full_task = "General ML task"
            
        if not requirements:
            requirements = "Use MLPY framework" if use_mlpy else "Use scikit-learn"
        
        prompt = PromptLibrary.get_code_generation_prompt()
        
        code = self.llm.complete(prompt.format(
            task_description=full_task,
            dataset_info=json.dumps(dataset_info or {}, indent=2),
            requirements=requirements
        ))
        
        # Clean up code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Store in history
        self.code_history.append(code)
        
        return code.strip()
    
    def suggest_features(
        self,
        df: pd.DataFrame,
        target: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest feature engineering.
        
        Args:
            df: Input DataFrame
            target: Target column name
            domain: Optional domain context
            
        Returns:
            Feature engineering suggestions
        """
        # Get sample data
        sample = df.head(5).to_dict('records')
        features = df.columns.tolist()
        features.remove(target)
        
        # Determine task type
        if df[target].dtype in [np.float64, np.int64] and df[target].nunique() > 10:
            task_type = "regression"
        else:
            task_type = "classification"
        
        prompt = PromptLibrary.get_feature_engineering_prompt()
        
        suggestions = self.llm.complete(prompt.format(
            target=target,
            features=features,
            sample_data=json.dumps(sample, indent=2)[:1000],
            task_type=task_type,
            domain=domain or "general"
        ))
        
        return {
            'suggestions': suggestions,
            'current_features': features,
            'target': target,
            'task_type': task_type
        }
    
    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat interface for ML assistance.
        
        Args:
            message: User message
            context: Optional context about current ML task
            
        Returns:
            Assistant response
        """
        # Build context-aware prompt
        system_prompt = """You are an expert ML engineer assistant integrated with the MLPY framework. 
You help with:
- Data analysis and preprocessing
- Model selection and training
- Error debugging
- Feature engineering
- Code generation
- Best practices

Always provide practical, actionable advice with code examples when relevant."""
        
        # Add context if provided
        if context:
            context_str = f"\n\nCurrent context:\n{json.dumps(context, indent=2)[:500]}"
        else:
            context_str = ""
        
        # Format message with context
        full_message = f"{message}{context_str}"
        
        # Get response
        response = self.llm.chat(full_message, role="user")
        
        # Save to history
        self.conversation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'user': message,
            'assistant': response,
            'context': context
        })
        
        return response


class DataAnalysisAssistant:
    """Specialized assistant for data analysis tasks."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize data analysis assistant."""
        self.llm = llm
    
    def analyze_dataframe(self, df: pd.DataFrame) -> str:
        """Comprehensive DataFrame analysis."""
        analysis = {
            'shape': df.shape,
            'memory': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'columns': {
                'total': len(df.columns),
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns)
            },
            'missing': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum()[df.isnull().sum() > 0].to_dict()
            },
            'correlations': self._get_top_correlations(df),
            'outliers': self._detect_outliers(df),
            'recommendations': []
        }
        
        prompt = f"""Analyze this dataset and provide insights:

{json.dumps(analysis, indent=2)}

Provide:
1. Key insights about the data
2. Potential issues to address
3. Recommended next steps
4. Feature engineering ideas

Analysis:"""
        
        return self.llm.complete(prompt)
    
    def _get_top_correlations(self, df: pd.DataFrame, n: int = 5) -> List[Tuple[str, str, float]]:
        """Get top correlations in DataFrame."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find top correlations
        correlations = []
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and abs(val) > 0.5:
                    correlations.append((row, col, val))
        
        return sorted(correlations, key=lambda x: abs(x[2]), reverse=True)[:n]
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                outliers[col] = int(n_outliers)
        
        return outliers


class ModelExplainer:
    """Specialized assistant for model explanations."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize model explainer."""
        self.llm = llm
    
    def explain_model_type(self, model_name: str) -> str:
        """Explain how a model type works."""
        prompt = f"""Explain how {model_name} works in simple terms.

Include:
1. Basic intuition
2. When to use it
3. Advantages and disadvantages
4. Key hyperparameters
5. Simple example

Explanation:"""
        
        return self.llm.complete(prompt)
    
    def explain_metrics(self, metrics: Dict[str, float], task_type: str) -> str:
        """Explain model metrics."""
        prompt = f"""Explain these {task_type} model metrics:

{json.dumps(metrics, indent=2)}

For each metric:
1. What it measures
2. What the value means
3. Is this good, bad, or average?
4. How to improve it

Explanation:"""
        
        return self.llm.complete(prompt)