"""
Feature Importance Analysis
===========================

Global and local feature importance using various methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings
import logging
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class ImportanceResults:
    """Container for feature importance results."""
    
    feature_names: List[str]
    importances: np.ndarray
    std: Optional[np.ndarray] = None
    method: str = "unknown"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importances
        })
        
        if self.std is not None:
            df['std'] = self.std
            df['ci_lower'] = df['importance'] - 1.96 * df['std']
            df['ci_upper'] = df['importance'] + 1.96 * df['std']
        
        return df.sort_values('importance', ascending=False)
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top n most important features."""
        df = self.to_dataframe()
        return df.head(n)['feature'].tolist()
    
    def plot(self, top_n: int = 20, show_std: bool = True):
        """Plot feature importances."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        df = self.to_dataframe().head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(df))
        
        if self.std is not None and show_std:
            ax.barh(y_pos, df['importance'], xerr=df['std'], 
                   color='skyblue', alpha=0.7, capsize=3)
        else:
            ax.barh(y_pos, df['importance'], color='skyblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance ({self.method})')
        
        plt.tight_layout()
        plt.show()


class FeatureImportance:
    """Calculate feature importance using various methods."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification"
    ):
        """
        Initialize feature importance calculator.
        
        Args:
            model: Trained model
            feature_names: Names of features
            task_type: "classification" or "regression"
        """
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
    
    def calculate(
        self,
        method: str = "auto",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> ImportanceResults:
        """
        Calculate feature importance.
        
        Args:
            method: "native", "permutation", "drop_column", or "auto"
            X: Feature data (required for some methods)
            y: Target data (required for some methods)
            
        Returns:
            ImportanceResults object
        """
        if method == "auto":
            method = self._detect_method()
        
        if method == "native":
            return self._native_importance()
        elif method == "permutation":
            if X is None or y is None:
                raise ValueError("X and y required for permutation importance")
            return self._permutation_importance(X, y)
        elif method == "drop_column":
            if X is None or y is None:
                raise ValueError("X and y required for drop column importance")
            return self._drop_column_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_method(self) -> str:
        """Detect appropriate method based on model type."""
        # Check for native feature importance
        if hasattr(self.model, 'feature_importances_'):
            return "native"
        elif hasattr(self.model, 'coef_'):
            return "native"
        else:
            return "permutation"
    
    def _native_importance(self) -> ImportanceResults:
        """Get native feature importance from model."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            
            # Get std if available (RandomForest)
            if hasattr(self.model, 'estimators_'):
                std = np.std([
                    tree.feature_importances_ 
                    for tree in self.model.estimators_
                ], axis=0)
            else:
                std = None
                
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if len(coef.shape) > 1:
                # Multi-class: average across classes
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
            std = None
        else:
            raise ValueError("Model has no native feature importance")
        
        # Normalize
        importances = importances / importances.sum()
        
        return ImportanceResults(
            feature_names=self.feature_names or [f"f{i}" for i in range(len(importances))],
            importances=importances,
            std=std,
            method="native"
        )
    
    def _permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        scoring: Optional[str] = None
    ) -> ImportanceResults:
        """Calculate permutation importance."""
        if scoring is None:
            scoring = "accuracy" if self.task_type == "classification" else "neg_mean_squared_error"
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=42
        )
        
        return ImportanceResults(
            feature_names=self.feature_names or [f"f{i}" for i in range(X.shape[1])],
            importances=result.importances_mean,
            std=result.importances_std,
            method="permutation"
        )
    
    def _drop_column_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> ImportanceResults:
        """Calculate importance by dropping columns."""
        from sklearn.model_selection import KFold
        
        # Baseline score with all features
        baseline_scores = cross_val_score(
            self.model, X, y, cv=cv_folds,
            scoring="accuracy" if self.task_type == "classification" else "neg_mean_squared_error"
        )
        baseline_score = baseline_scores.mean()
        
        importances = []
        stds = []
        
        # Drop each feature and measure impact
        for i in range(X.shape[1]):
            X_dropped = np.delete(X, i, axis=1)
            
            scores = cross_val_score(
                self.model.__class__(**self.model.get_params()),
                X_dropped, y, cv=cv_folds,
                scoring="accuracy" if self.task_type == "classification" else "neg_mean_squared_error"
            )
            
            # Importance is the drop in performance
            importance = baseline_score - scores.mean()
            importances.append(importance)
            stds.append(scores.std())
        
        importances = np.array(importances)
        stds = np.array(stds)
        
        # Normalize to positive values
        importances = np.abs(importances)
        importances = importances / importances.sum()
        
        return ImportanceResults(
            feature_names=self.feature_names or [f"f{i}" for i in range(X.shape[1])],
            importances=importances,
            std=stds,
            method="drop_column"
        )
    
    def calculate_all_methods(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, ImportanceResults]:
        """Calculate importance using all applicable methods."""
        results = {}
        
        # Try native importance
        try:
            results['native'] = self._native_importance()
        except:
            pass
        
        # Try permutation importance
        if X is not None and y is not None:
            try:
                results['permutation'] = self._permutation_importance(X, y)
            except:
                pass
            
            # Try drop column importance
            try:
                results['drop_column'] = self._drop_column_importance(X, y)
            except:
                pass
        
        return results


class PermutationImportance:
    """Advanced permutation importance with confidence intervals."""
    
    def __init__(
        self,
        model: Any,
        scoring: Union[str, Callable],
        n_repeats: int = 10,
        random_state: int = 42
    ):
        """
        Initialize permutation importance calculator.
        
        Args:
            model: Trained model
            scoring: Scoring function or string
            n_repeats: Number of permutation repeats
            random_state: Random seed
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def calculate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> ImportanceResults:
        """
        Calculate permutation importance.
        
        Args:
            X: Feature data
            y: Target data
            sample_weight: Sample weights
            
        Returns:
            ImportanceResults object
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Calculate baseline score
        baseline_score = self._score(X_array, y_array, sample_weight)
        
        # Calculate importance for each feature
        importances = []
        importance_scores = {i: [] for i in range(X_array.shape[1])}
        
        np.random.seed(self.random_state)
        
        for _ in range(self.n_repeats):
            for col_idx in range(X_array.shape[1]):
                # Create copy and permute column
                X_permuted = X_array.copy()
                X_permuted[:, col_idx] = np.random.permutation(X_permuted[:, col_idx])
                
                # Calculate score with permuted feature
                permuted_score = self._score(X_permuted, y_array, sample_weight)
                
                # Importance is the decrease in score
                importance = baseline_score - permuted_score
                importance_scores[col_idx].append(importance)
        
        # Calculate mean and std
        mean_importances = []
        std_importances = []
        
        for col_idx in range(X_array.shape[1]):
            scores = importance_scores[col_idx]
            mean_importances.append(np.mean(scores))
            std_importances.append(np.std(scores))
        
        return ImportanceResults(
            feature_names=feature_names,
            importances=np.array(mean_importances),
            std=np.array(std_importances),
            method="permutation"
        )
    
    def _score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray]
    ) -> float:
        """Calculate model score."""
        if isinstance(self.scoring, str):
            # Use sklearn scoring
            from sklearn.metrics import get_scorer
            scorer = get_scorer(self.scoring)
            return scorer(self.model, X, y, sample_weight=sample_weight)
        else:
            # Use custom scoring function
            y_pred = self.model.predict(X)
            return self.scoring(y, y_pred, sample_weight=sample_weight)
    
    def plot_with_confidence(
        self,
        results: ImportanceResults,
        top_n: int = 20,
        confidence_level: float = 0.95
    ):
        """Plot importance with confidence intervals."""
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            logger.warning("Matplotlib/scipy not available for plotting")
            return
        
        df = results.to_dataframe().head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(df))
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_width = z_score * df['std']
        
        # Plot bars with error bars
        ax.barh(y_pos, df['importance'], xerr=ci_width,
               color='lightcoral', alpha=0.7, capsize=5,
               error_kw={'linewidth': 1.5})
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Permutation Feature Importance ({confidence_level*100:.0f}% CI)')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()