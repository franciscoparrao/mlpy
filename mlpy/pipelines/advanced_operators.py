"""Advanced pipeline operators for MLPY.

This module provides additional pipeline operations for
dimensionality reduction, outlier detection, feature engineering,
and other advanced preprocessing tasks.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .base import PipeOp, PipeOpInput, PipeOpOutput, mlpy_pipeops
from ..tasks import Task, TaskClassif, TaskRegr


class PipeOpPCA(PipeOp):
    """Principal Component Analysis for dimensionality reduction.
    
    Parameters
    ----------
    id : str, default="pca"
        Unique identifier.
    n_components : int, float, or 'mle', default=None
        Number of components to keep.
        - If int, select n_components
        - If float between 0 and 1, select enough components to explain that fraction of variance
        - If 'mle', use Minka's MLE to guess the dimension
        - If None, keep all components
    whiten : bool, default=False
        When True, components are multiplied by sqrt(n_samples) and 
        divided by singular values to ensure uncorrelated outputs.
    svd_solver : str, default='auto'
        Algorithm to use: 'auto', 'full', 'arpack', 'randomized'.
    """
    
    def __init__(
        self,
        id: str = "pca",
        n_components: Union[int, float, str, None] = None,
        whiten: bool = False,
        svd_solver: str = 'auto',
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self._pca = None
        self._feature_cols = None
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit PCA and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get numeric feature columns
        self._feature_cols = [col for col in task.feature_names 
                             if col in data.columns and 
                             pd.api.types.is_numeric_dtype(data[col])]
        
        if not self._feature_cols:
            # No numeric features
            self.state.is_trained = True
            return {"output": task}
            
        # Create and fit PCA
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver
        )
        
        X = data[self._feature_cols].values
        X_transformed = self._pca.fit_transform(X)
        
        # Create new data with PCA components
        n_components = X_transformed.shape[1]
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        
        # Create new dataframe
        new_data = pd.DataFrame(X_transformed, columns=pca_cols, index=data.index)
        
        # Add target and non-feature columns
        for col in data.columns:
            if col not in self._feature_cols:
                new_data[col] = data[col]
                
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update column roles - PCA columns are features
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = pca_cols
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["pca"] = self._pca
        self.state["feature_cols"] = self._feature_cols
        self.state["n_components"] = n_components
        self.state["explained_variance_ratio"] = self._pca.explained_variance_ratio_.tolist()
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted PCA."""
        if not self.is_trained:
            raise RuntimeError("PipeOpPCA must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._feature_cols:
            return {"output": task}
            
        # Transform data
        X = data[self._feature_cols].values
        X_transformed = self._pca.transform(X)
        
        # Create new data
        n_components = self.state["n_components"]
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        
        new_data = pd.DataFrame(X_transformed, columns=pca_cols, index=data.index)
        
        # Add other columns
        for col in data.columns:
            if col not in self._feature_cols:
                new_data[col] = data[col]
                
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = pca_cols
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


class PipeOpTargetEncode(PipeOp):
    """Target encoding for categorical features.
    
    Replaces categorical values with the mean of the target for that category.
    Useful for high-cardinality categorical features.
    
    Parameters
    ----------
    id : str, default="target_encode"
        Unique identifier.
    columns : list of str, optional
        Specific columns to encode. If None, encodes all categorical features.
    smoothing : float, default=1.0
        Smoothing parameter for target encoding to prevent overfitting.
        Higher values mean more smoothing.
    min_samples_leaf : int, default=1
        Minimum samples required in a category to use its mean.
        Categories with fewer samples use the global mean.
    """
    
    def __init__(
        self,
        id: str = "target_encode",
        columns: Optional[List[str]] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.columns = columns
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self._encodings = {}
        self._global_mean = None
        self._target_col = None
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit target encoder and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        
        # Only works for supervised tasks
        if not hasattr(task, 'target_names') or not task.target_names:
            raise ValueError("Target encoding requires a supervised task with target")
            
        # Only works for regression or binary classification
        if isinstance(task, TaskClassif) and len(np.unique(task.truth())) > 2:
            raise ValueError("Target encoding only supports regression or binary classification")
            
        data = task.data()
        self._target_col = task.target_names[0]
        
        # Get categorical columns
        if self.columns:
            cat_cols = self.columns
        else:
            cat_cols = [col for col in task.feature_names
                       if col in data.columns and 
                       pd.api.types.is_object_dtype(data[col])]
            
        if not cat_cols:
            # No categorical columns
            self.state.is_trained = True
            return {"output": task}
            
        # Get target values
        y = data[self._target_col]
        
        # For classification, convert to 0/1
        if isinstance(task, TaskClassif):
            # Assume positive class is second class
            classes = np.unique(y)
            y = (y == classes[1]).astype(float)
            
        # Calculate global mean
        self._global_mean = y.mean()
        
        # Calculate encodings for each categorical column
        new_data = data.copy()
        
        for col in cat_cols:
            # Calculate category statistics
            category_stats = data.groupby(col)[self._target_col].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_means = {}
            for cat, row in category_stats.iterrows():
                cat_mean = row['mean']
                cat_count = row['count']
                
                if cat_count >= self.min_samples_leaf:
                    # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
                    smooth_mean = (cat_count * cat_mean + self.smoothing * self._global_mean) / (cat_count + self.smoothing)
                else:
                    smooth_mean = self._global_mean
                    
                smoothed_means[cat] = smooth_mean
                
            self._encodings[col] = smoothed_means
            
            # Apply encoding
            new_data[col] = data[col].map(smoothed_means).fillna(self._global_mean)
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=self._target_col,
            id=task.id,
            label=task.label
        )
        
        new_task.set_col_roles({role: list(cols) for role, cols in task.col_roles.items()})
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["encodings"] = self._encodings
        self.state["global_mean"] = self._global_mean
        self.state["encoded_columns"] = cat_cols
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted encoder."""
        if not self.is_trained:
            raise RuntimeError("PipeOpTargetEncode must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._encodings:
            return {"output": task}
            
        # Apply encodings
        new_data = data.copy()
        
        for col, encoding in self._encodings.items():
            if col in data.columns:
                # Map known categories, use global mean for unknown
                new_data[col] = data[col].map(encoding).fillna(self._global_mean)
                
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        new_task.set_col_roles({role: list(cols) for role, cols in task.col_roles.items()})
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


class PipeOpOutlierDetect(PipeOp):
    """Detect and optionally remove outliers.
    
    Parameters
    ----------
    id : str, default="outlier_detect"
        Unique identifier.
    method : str, default="isolation"
        Detection method: "isolation", "elliptic", "lof".
    contamination : float, default='auto'
        Expected proportion of outliers in the dataset.
    action : str, default="flag"
        What to do with outliers: "flag", "remove", "impute".
    flag_column : str, default="is_outlier"
        Name of the flag column when action="flag".
    """
    
    def __init__(
        self,
        id: str = "outlier_detect",
        method: str = "isolation",
        contamination: Union[float, str] = 'auto',
        action: str = "flag",
        flag_column: str = "is_outlier",
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.method = method
        self.contamination = contamination
        self.action = action
        self.flag_column = flag_column
        self._detector = None
        self._feature_cols = None
        self._outlier_indices = None
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit outlier detector and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get numeric features
        self._feature_cols = [col for col in task.feature_names 
                             if col in data.columns and 
                             pd.api.types.is_numeric_dtype(data[col])]
        
        if not self._feature_cols:
            # No numeric features
            self.state.is_trained = True
            return {"output": task}
            
        X = data[self._feature_cols].values
        
        # Create detector
        if self.method == "isolation":
            self._detector = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == "elliptic":
            self._detector = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == "lof":
            self._detector = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True  # Allow predict on new data
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
            
        # Fit and predict outliers (-1 for outliers, 1 for inliers)
        outlier_labels = self._detector.fit_predict(X)
        self._outlier_indices = np.where(outlier_labels == -1)[0]
        
        # Apply action
        if self.action == "flag":
            # Add outlier flag column
            new_data = data.copy()
            new_data[self.flag_column] = (outlier_labels == -1).astype(int)
            
        elif self.action == "remove":
            # Remove outlier rows
            inlier_mask = outlier_labels == 1
            new_data = data[inlier_mask].copy()
            
        elif self.action == "impute":
            # Replace outlier values with median
            new_data = data.copy()
            for col in self._feature_cols:
                col_median = data[col].median()
                outlier_mask = outlier_labels == -1
                new_data.loc[outlier_mask, col] = col_median
        else:
            raise ValueError(f"Unknown action: {self.action}")
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update column roles
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        if self.action == "flag":
            # Add flag column as feature
            col_roles["feature"].append(self.flag_column)
            
        new_task.set_col_roles(col_roles)
        
        # Update row roles if rows were removed
        if self.action == "remove":
            # Reset row roles as indices changed
            new_task.set_row_roles({})
        else:
            new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
            
        self.state.is_trained = True
        self.state["detector"] = self._detector
        self.state["feature_cols"] = self._feature_cols
        self.state["n_outliers"] = len(self._outlier_indices)
        self.state["outlier_fraction"] = len(self._outlier_indices) / len(data)
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply outlier detection to new data."""
        if not self.is_trained:
            raise RuntimeError("PipeOpOutlierDetect must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._feature_cols:
            return {"output": task}
            
        X = data[self._feature_cols].values
        
        # Predict outliers
        if self.method == "lof":
            outlier_labels = self._detector.predict(X)
        else:
            outlier_labels = self._detector.predict(X)
            
        # Apply same action as in training
        if self.action == "flag":
            new_data = data.copy()
            new_data[self.flag_column] = (outlier_labels == -1).astype(int)
            
        elif self.action == "remove":
            inlier_mask = outlier_labels == 1
            new_data = data[inlier_mask].copy()
            
        elif self.action == "impute":
            new_data = data.copy()
            # Use training medians stored in state
            if "feature_medians" not in self.state:
                # Calculate medians from current data as fallback
                for col in self._feature_cols:
                    col_median = data[col].median()
                    outlier_mask = outlier_labels == -1
                    new_data.loc[outlier_mask, col] = col_median
            else:
                for col in self._feature_cols:
                    col_median = self.state["feature_medians"][col]
                    outlier_mask = outlier_labels == -1
                    new_data.loc[outlier_mask, col] = col_median
                    
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        if self.action == "flag" and self.flag_column not in col_roles.get("feature", []):
            col_roles.setdefault("feature", []).append(self.flag_column)
            
        new_task.set_col_roles(col_roles)
        
        if self.action == "remove":
            new_task.set_row_roles({})
        else:
            new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
            
        return {"output": new_task}


class PipeOpBin(PipeOp):
    """Bin continuous features into discrete intervals.
    
    Parameters
    ----------
    id : str, default="bin"
        Unique identifier.
    n_bins : int, default=5
        Number of bins.
    columns : list of str, optional
        Specific columns to bin. If None, bins all numeric features.
    strategy : str, default='quantile'
        Strategy to define bins: 'uniform', 'quantile', 'kmeans'.
    encode : str, default='ordinal'
        How to encode bins: 'ordinal', 'onehot'.
    """
    
    def __init__(
        self,
        id: str = "bin",
        n_bins: int = 5,
        columns: Optional[List[str]] = None,
        strategy: str = 'quantile',
        encode: str = 'ordinal',
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.n_bins = n_bins
        self.columns = columns
        self.strategy = strategy
        self.encode = encode
        self._binners = {}
        self._binned_columns = []
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit binning and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get columns to bin
        if self.columns:
            bin_cols = self.columns
        else:
            bin_cols = [col for col in task.feature_names 
                       if col in data.columns and 
                       pd.api.types.is_numeric_dtype(data[col])]
            
        if not bin_cols:
            self.state.is_trained = True
            return {"output": task}
            
        # Fit binners
        new_data = data.copy()
        new_feature_cols = []
        
        for col in bin_cols:
            binner = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy
            )
            
            X_col = data[[col]].values
            X_binned = binner.fit_transform(X_col)
            
            self._binners[col] = binner
            
            if self.encode == 'ordinal':
                # Replace original column
                new_data[col] = X_binned.ravel()
                new_feature_cols.append(col)
            else:  # onehot
                # Create new columns
                n_bins_actual = X_binned.shape[1]
                bin_cols_names = [f"{col}_bin_{i}" for i in range(n_bins_actual)]
                
                for i, bin_col in enumerate(bin_cols_names):
                    new_data[bin_col] = X_binned[:, i]
                    new_feature_cols.append(bin_col)
                    
                # Drop original column
                new_data = new_data.drop(columns=[col])
                
            self._binned_columns.append(col)
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update features
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        
        if self.encode == 'onehot':
            # Remove original columns, add new ones
            col_roles["feature"] = [f for f in col_roles["feature"] if f not in self._binned_columns]
            col_roles["feature"].extend(new_feature_cols)
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["binners"] = self._binners
        self.state["binned_columns"] = self._binned_columns
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply binning to new data."""
        if not self.is_trained:
            raise RuntimeError("PipeOpBin must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._binners:
            return {"output": task}
            
        new_data = data.copy()
        new_feature_cols = []
        
        for col, binner in self._binners.items():
            if col not in data.columns:
                continue
                
            X_col = data[[col]].values
            X_binned = binner.transform(X_col)
            
            if self.encode == 'ordinal':
                new_data[col] = X_binned.ravel()
                new_feature_cols.append(col)
            else:  # onehot
                n_bins_actual = X_binned.shape[1]
                bin_cols_names = [f"{col}_bin_{i}" for i in range(n_bins_actual)]
                
                for i, bin_col in enumerate(bin_cols_names):
                    new_data[bin_col] = X_binned[:, i]
                    new_feature_cols.append(bin_col)
                    
                new_data = new_data.drop(columns=[col])
                
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        
        if self.encode == 'onehot':
            col_roles["feature"] = [f for f in col_roles["feature"] if f not in self._binned_columns]
            col_roles["feature"].extend(new_feature_cols)
            
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


class PipeOpTextVectorize(PipeOp):
    """Convert text features to numeric vectors.
    
    Parameters
    ----------
    id : str, default="text_vectorize"
        Unique identifier.
    columns : list of str
        Text columns to vectorize.
    method : str, default="tfidf"
        Vectorization method: "tfidf", "count".
    max_features : int, optional
        Maximum number of features (vocabulary size).
    ngram_range : tuple, default=(1, 1)
        Range of n-grams to extract.
    min_df : int or float, default=1
        Minimum document frequency for terms.
    max_df : int or float, default=1.0
        Maximum document frequency for terms.
    """
    
    def __init__(
        self,
        columns: List[str],
        id: str = "text_vectorize",
        method: str = "tfidf",
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.columns = columns
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self._vectorizers = {}
        self._feature_names = {}
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit vectorizers and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Check columns exist
        missing_cols = [col for col in self.columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Text columns not found: {missing_cols}")
            
        # Create vectorizers
        new_data = data.drop(columns=self.columns).copy()
        all_new_features = []
        
        for col in self.columns:
            # Create vectorizer
            if self.method == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df
                )
            else:  # count
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df
                )
                
            # Fit and transform
            texts = data[col].fillna('')  # Handle missing as empty string
            X_vec = vectorizer.fit_transform(texts)
            
            # Get feature names
            feature_names = [f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()]
            self._feature_names[col] = feature_names
            
            # Add to dataframe
            vec_df = pd.DataFrame(
                X_vec.toarray(),
                columns=feature_names,
                index=data.index
            )
            new_data = pd.concat([new_data, vec_df], axis=1)
            all_new_features.extend(feature_names)
            
            self._vectorizers[col] = vectorizer
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update features
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        # Remove original text columns from features, add new ones
        col_roles["feature"] = [f for f in col_roles["feature"] if f not in self.columns]
        col_roles["feature"].extend(all_new_features)
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["vectorizers"] = self._vectorizers
        self.state["feature_names"] = self._feature_names
        self.state["n_features_total"] = len(all_new_features)
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply vectorization to new data."""
        if not self.is_trained:
            raise RuntimeError("PipeOpTextVectorize must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._vectorizers:
            return {"output": task}
            
        # Apply vectorizers
        new_data = data.copy()
        all_new_features = []
        
        for col, vectorizer in self._vectorizers.items():
            if col not in data.columns:
                warnings.warn(f"Text column '{col}' not found in prediction data")
                continue
                
            texts = data[col].fillna('')
            X_vec = vectorizer.transform(texts)
            
            feature_names = self._feature_names[col]
            vec_df = pd.DataFrame(
                X_vec.toarray(),
                columns=feature_names,
                index=data.index
            )
            
            # Drop original column and add vectorized
            new_data = new_data.drop(columns=[col])
            new_data = pd.concat([new_data, vec_df], axis=1)
            all_new_features.extend(feature_names)
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = [f for f in col_roles["feature"] if f not in self.columns]
        col_roles["feature"].extend(all_new_features)
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


class PipeOpPolynomial(PipeOp):
    """Generate polynomial and interaction features.
    
    Parameters
    ----------
    id : str, default="polynomial"
        Unique identifier.
    degree : int, default=2
        Maximum degree of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=False
        If True, include a bias column.
    columns : list of str, optional
        Specific columns to use. If None, uses all numeric features.
    """
    
    def __init__(
        self,
        id: str = "polynomial",
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.columns = columns
        self._poly = None
        self._feature_cols = None
        self._poly_feature_names = None
        
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
        """Returns a modified Task."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Task,
                predict=Task
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fit polynomial features and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get columns
        if self.columns:
            self._feature_cols = self.columns
        else:
            self._feature_cols = [col for col in task.feature_names 
                                 if col in data.columns and 
                                 pd.api.types.is_numeric_dtype(data[col])]
            
        if not self._feature_cols:
            self.state.is_trained = True
            return {"output": task}
            
        # Create polynomial features
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        X = data[self._feature_cols].values
        X_poly = self._poly.fit_transform(X)
        
        # Get feature names
        poly_names = self._poly.get_feature_names_out(self._feature_cols)
        
        # Create new dataframe
        poly_df = pd.DataFrame(X_poly, columns=poly_names, index=data.index)
        
        # Combine with non-polynomial columns
        other_cols = [col for col in data.columns if col not in self._feature_cols]
        new_data = pd.concat([poly_df, data[other_cols]], axis=1)
        
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update features
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = list(poly_names)
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["poly"] = self._poly
        self.state["feature_cols"] = self._feature_cols
        self.state["poly_feature_names"] = list(poly_names)
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply polynomial transformation to new data."""
        if not self.is_trained:
            raise RuntimeError("PipeOpPolynomial must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._feature_cols:
            return {"output": task}
            
        # Apply transformation
        X = data[self._feature_cols].values
        X_poly = self._poly.transform(X)
        
        poly_names = self.state["poly_feature_names"]
        poly_df = pd.DataFrame(X_poly, columns=poly_names, index=data.index)
        
        other_cols = [col for col in data.columns if col not in self._feature_cols]
        new_data = pd.concat([poly_df, data[other_cols]], axis=1)
        
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = poly_names
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


# Register all advanced operators
mlpy_pipeops.register("pca", PipeOpPCA)
mlpy_pipeops.register("target_encode", PipeOpTargetEncode)
mlpy_pipeops.register("outlier_detect", PipeOpOutlierDetect)
mlpy_pipeops.register("bin", PipeOpBin)
mlpy_pipeops.register("text_vectorize", PipeOpTextVectorize)
mlpy_pipeops.register("polynomial", PipeOpPolynomial)


__all__ = [
    "PipeOpPCA",
    "PipeOpTargetEncode",
    "PipeOpOutlierDetect",
    "PipeOpBin",
    "PipeOpTextVectorize",
    "PipeOpPolynomial"
]