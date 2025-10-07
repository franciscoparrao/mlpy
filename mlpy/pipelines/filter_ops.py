"""
Filter pipeline operators for MLPY.

Integrates the comprehensive filter system with pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

from .base import PipeOp, PipeOpInput, PipeOpOutput
from ..tasks.base import Task
from ..filters import create_filter, Filter


class PipeOpFilter(PipeOp):
    """Filter features using any available filter method.
    
    This operator integrates the comprehensive filter system
    into MLPY pipelines.
    
    Parameters
    ----------
    id : str, default="filter"
        Unique identifier.
    method : str, default="auto"
        Filter method to use. See mlpy.filters.list_filters().
    k : int, optional
        Number of features to select.
    threshold : float, optional
        Score threshold for selection.
    percentile : float, optional
        Percentile of features to select.
    filter_params : dict, optional
        Additional parameters for the filter.
    cache_scores : bool, default=True
        Whether to cache filter scores for faster prediction.
    **kwargs
        Additional parameters for PipeOp.
        
    Examples
    --------
    >>> from mlpy.pipelines import PipeOpFilter
    >>> 
    >>> # Select top 10 features using ANOVA
    >>> filter_anova = PipeOpFilter(method='anova', k=10)
    >>> 
    >>> # Select top 25% features using mutual information
    >>> filter_mi = PipeOpFilter(method='mutual_info', percentile=25)
    >>> 
    >>> # Use ensemble of filters
    >>> filter_ensemble = PipeOpFilter(
    ...     method='ensemble',
    ...     filter_params={'filters': ['anova', 'mutual_info', 'correlation']},
    ...     k=20
    ... )
    """
    
    def __init__(
        self,
        id: str = "filter",
        method: str = "auto",
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        filter_params: Optional[Dict] = None,
        cache_scores: bool = True,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        
        self.method = method
        self.k = k
        self.threshold = threshold
        self.percentile = percentile
        self.filter_params = filter_params or {}
        self.cache_scores = cache_scores
        
        # Runtime state
        self._filter = None
        self._selected_features = None
        self._filter_result = None
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a Task."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns a Task with selected features."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate feature scores and select features."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        
        # Create filter
        self._filter = create_filter(self.method, **self.filter_params)
        
        # Calculate scores
        self._filter_result = self._filter.calculate(task)
        
        # Select features
        if self.k is not None:
            self._selected_features = self._filter_result.select_top_k(self.k)
        elif self.threshold is not None:
            self._selected_features = self._filter_result.select_threshold(self.threshold)
        elif self.percentile is not None:
            self._selected_features = self._filter_result.select_percentile(self.percentile)
        else:
            # Default: select top 50%
            n_select = max(1, len(task.feature_names) // 2)
            self._selected_features = self._filter_result.select_top_k(n_select)
            
        # Create filtered task
        filtered_task = task.select_cols(self._selected_features + task.target_names)
        
        # Store state
        self.state.is_trained = True
        self.state["selected_features"] = self._selected_features
        self.state["filter_method"] = self.method
        if self.cache_scores:
            self.state["filter_scores"] = self._filter_result.scores.to_dict()
            
        return {"output": filtered_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature selection to new data."""
        if not self.is_trained:
            raise RuntimeError("PipeOpFilter must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        
        # Apply same feature selection
        filtered_task = task.select_cols(self._selected_features + task.target_names)
        
        return {"output": filtered_task}
        
    def get_importance(self) -> pd.Series:
        """Get feature importance scores.
        
        Returns
        -------
        pd.Series
            Feature scores from the filter.
        """
        if self._filter_result is None:
            raise RuntimeError("Filter must be trained first")
        return self._filter_result.scores
        
    def plot_importance(self, top_n: int = 20, figsize=(10, 6)):
        """Plot feature importance scores.
        
        Parameters
        ----------
        top_n : int, default=20
            Number of top features to plot.
        figsize : tuple, default=(10, 6)
            Figure size.
        """
        import matplotlib.pyplot as plt
        
        scores = self.get_importance()
        top_features = scores.nlargest(top_n)
        
        plt.figure(figsize=figsize)
        top_features.plot(kind='barh')
        plt.xlabel('Score')
        plt.title(f'Top {top_n} Features ({self.method} filter)')
        plt.tight_layout()
        plt.show()


class PipeOpFilterRank(PipeOp):
    """Rank features and optionally filter by rank.
    
    This operator ranks all features but doesn't necessarily
    remove any. Useful for downstream operators that can use
    feature rankings.
    
    Parameters
    ----------
    id : str, default="rank"
        Unique identifier.
    method : str, default="auto"
        Filter method for ranking.
    min_rank : int, optional
        Minimum rank to keep (1 = best).
    max_rank : int, optional
        Maximum rank to keep.
    add_rank_column : bool, default=False
        Whether to add rank as a feature.
    **kwargs
        Additional parameters.
    """
    
    def __init__(
        self,
        id: str = "rank",
        method: str = "auto",
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        add_rank_column: bool = False,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        
        self.method = method
        self.min_rank = min_rank or 1
        self.max_rank = max_rank
        self.add_rank_column = add_rank_column
        
        self._filter = None
        self._ranks = None
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate feature ranks."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        
        # Create filter and calculate scores
        self._filter = create_filter(self.method)
        result = self._filter.calculate(task)
        
        # Calculate ranks (1 = best)
        self._ranks = pd.Series(
            range(1, len(result.features) + 1),
            index=result.features
        )
        
        # Select features by rank if specified
        if self.max_rank is not None:
            selected = self._ranks[self._ranks <= self.max_rank].index.tolist()
            filtered_task = task.select_cols(selected + task.target_names)
        else:
            filtered_task = task.clone()
            
        # Add rank column if requested
        if self.add_rank_column:
            data = filtered_task.data(data_format='dataframe')
            for feat in filtered_task.feature_names:
                if feat in self._ranks:
                    data[f"{feat}_rank"] = self._ranks[feat]
            
            # Create new task with rank columns
            task_class = type(filtered_task)
            filtered_task = task_class(
                data=data,
                target=filtered_task.target_names[0]
            )
            
        # Store state
        self.state.is_trained = True
        self.state["ranks"] = self._ranks.to_dict()
        
        return {"output": filtered_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ranking-based filtering."""
        if not self.is_trained:
            raise RuntimeError("PipeOpFilterRank must be trained before predict")
            
        task = inputs["input"]
        
        # Apply same filtering
        if self.max_rank is not None:
            selected = [f for f in task.feature_names if f in self._ranks and self._ranks[f] <= self.max_rank]
            filtered_task = task.select_cols(selected + task.target_names)
        else:
            filtered_task = task.clone()
            
        return {"output": filtered_task}


class PipeOpFilterCorr(PipeOp):
    """Remove highly correlated features.
    
    Removes features that are highly correlated with other features,
    keeping the one with higher importance score.
    
    Parameters
    ----------
    id : str, default="filter_corr"
        Unique identifier.
    threshold : float, default=0.95
        Correlation threshold above which to remove features.
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'.
    importance_method : str, default='variance'
        Method to determine feature importance for tie-breaking.
    **kwargs
        Additional parameters.
    """
    
    def __init__(
        self,
        id: str = "filter_corr",
        threshold: float = 0.95,
        method: str = 'pearson',
        importance_method: str = 'variance',
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        
        self.threshold = threshold
        self.method = method
        self.importance_method = importance_method
        
        self._selected_features = None
        self._correlation_matrix = None
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove correlated features."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        
        # Get numeric features only
        data = task.data(cols=task.feature_names, data_format='dataframe')
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            # No numeric features, return as is
            self._selected_features = task.feature_names
            return {"output": task}
            
        # Calculate correlation matrix
        numeric_data = data[numeric_features]
        self._correlation_matrix = numeric_data.corr(method=self.method).abs()
        
        # Get feature importance for tie-breaking
        if self.importance_method != 'none':
            filter_obj = create_filter(self.importance_method)
            importance_result = filter_obj.calculate(task, numeric_features)
            importance = importance_result.scores
        else:
            importance = pd.Series(1.0, index=numeric_features)
            
        # Find features to remove
        to_remove = set()
        
        for i, feat1 in enumerate(numeric_features):
            if feat1 in to_remove:
                continue
                
            for j, feat2 in enumerate(numeric_features[i+1:], i+1):
                if feat2 in to_remove:
                    continue
                    
                if self._correlation_matrix.loc[feat1, feat2] > self.threshold:
                    # Remove the less important feature
                    if importance[feat1] >= importance[feat2]:
                        to_remove.add(feat2)
                    else:
                        to_remove.add(feat1)
                        break
                        
        # Select features
        self._selected_features = [f for f in task.feature_names if f not in to_remove]
        
        # Create filtered task
        filtered_task = task.select_cols(self._selected_features + task.target_names)
        
        # Store state
        self.state.is_trained = True
        self.state["selected_features"] = self._selected_features
        self.state["removed_features"] = list(to_remove)
        
        return {"output": filtered_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply correlation filtering."""
        if not self.is_trained:
            raise RuntimeError("PipeOpFilterCorr must be trained before predict")
            
        task = inputs["input"]
        filtered_task = task.select_cols(self._selected_features + task.target_names)
        
        return {"output": filtered_task}