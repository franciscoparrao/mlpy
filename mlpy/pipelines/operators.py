"""Common pipeline operators for MLPY.

This module provides pre-built pipeline operations for
common preprocessing and transformation tasks.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from .base import PipeOp, PipeOpInput, PipeOpOutput, mlpy_pipeops
from ..tasks import Task, TaskClassif, TaskRegr


class PipeOpScale(PipeOp):
    """Scale numeric features.
    
    Parameters
    ----------
    id : str, default="scale"
        Unique identifier.
    method : str, default="standard"
        Scaling method: "standard", "minmax", "robust".
    with_mean : bool, default=True
        Whether to center the data (for standard scaling).
    with_std : bool, default=True
        Whether to scale to unit variance (for standard scaling).
    """
    
    def __init__(
        self,
        id: str = "scale",
        method: str = "standard",
        with_mean: bool = True,
        with_std: bool = True,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.method = method
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = None
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
        """Fit scaler and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get numeric feature columns
        self._feature_cols = [col for col in task.feature_names 
                             if col in data.columns and 
                             pd.api.types.is_numeric_dtype(data[col])]
        
        if not self._feature_cols:
            # No numeric features to scale
            self.state.is_trained = True
            return {"output": task}
            
        # Create and fit scaler
        if self.method == "standard":
            self._scaler = StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std
            )
        elif self.method == "minmax":
            self._scaler = MinMaxScaler()
        elif self.method == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
            
        # Fit on training data
        X = data[self._feature_cols].values
        self._scaler.fit(X)
        
        # Transform data
        X_scaled = self._scaler.transform(X)
        
        # Create new task with scaled data
        new_data = data.copy()
        new_data[self._feature_cols] = X_scaled
        
        # Create new task preserving all properties
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Preserve row/column roles
        new_task.set_col_roles({role: list(cols) for role, cols in task.col_roles.items()})
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["scaler"] = self._scaler
        self.state["feature_cols"] = self._feature_cols
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted scaler."""
        if not self.is_trained:
            raise RuntimeError("PipeOpScale must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        
        if not self._feature_cols:
            # No features to scale
            return {"output": task}
            
        data = task.data()
        
        # Check that expected columns exist
        missing = set(self._feature_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
            
        # Transform data
        X = data[self._feature_cols].values
        X_scaled = self._scaler.transform(X)
        
        # Create new task
        new_data = data.copy()
        new_data[self._feature_cols] = X_scaled
        
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


class PipeOpImpute(PipeOp):
    """Impute missing values.
    
    Parameters
    ----------
    id : str, default="impute"
        Unique identifier.
    strategy : str, default="mean"
        Imputation strategy: "mean", "median", "most_frequent", "constant".
    fill_value : Any, default=None
        Value to use for constant strategy.
    add_indicator : bool, default=False
        Whether to add a missing indicator column.
    """
    
    def __init__(
        self,
        id: str = "impute",
        strategy: str = "mean",
        fill_value: Any = None,
        add_indicator: bool = False,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self._imputers = {}
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
        """Fit imputers and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get feature columns
        self._feature_cols = task.feature_names
        
        # Check which columns have missing values
        cols_with_missing = [col for col in self._feature_cols 
                            if data[col].isnull().any()]
        
        if not cols_with_missing:
            # No missing values
            self.state.is_trained = True
            return {"output": task}
            
        # Create new data
        new_data = data.copy()
        
        # Fit imputers for each column type
        for col in cols_with_missing:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Numeric imputation
                if self.strategy in ["mean", "median"]:
                    imputer = SimpleImputer(
                        strategy=self.strategy,
                        add_indicator=self.add_indicator
                    )
                else:
                    imputer = SimpleImputer(
                        strategy=self.strategy,
                        fill_value=self.fill_value,
                        add_indicator=self.add_indicator
                    )
            else:
                # Categorical imputation
                imputer = SimpleImputer(
                    strategy="most_frequent" if self.strategy != "constant" else "constant",
                    fill_value=self.fill_value if self.strategy == "constant" else None,
                    add_indicator=self.add_indicator
                )
                
            # Fit and transform
            values = data[[col]].values
            
            # Handle None values in object dtype columns
            if pd.api.types.is_object_dtype(data[col]) and values.dtype == object:
                # Replace None with np.nan for sklearn compatibility
                values = np.where(values == None, np.nan, values)
                
            transformed = imputer.fit_transform(values)
            
            if self.add_indicator:
                # Split imputed values and indicator
                new_data[col] = transformed[:, 0]
                new_data[f"{col}_was_missing"] = transformed[:, 1]
            else:
                new_data[col] = transformed[:, 0]
                
            self._imputers[col] = imputer
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update column roles if indicators were added
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        if self.add_indicator:
            for col in cols_with_missing:
                col_roles["feature"].append(f"{col}_was_missing")
        
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["imputers"] = self._imputers
        self.state["feature_cols"] = self._feature_cols
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted imputers."""
        if not self.is_trained:
            raise RuntimeError("PipeOpImpute must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._imputers:
            # No imputation needed
            return {"output": task}
            
        # Apply imputers
        new_data = data.copy()
        
        for col, imputer in self._imputers.items():
            if col in data.columns:
                values = data[[col]].values
                
                # Handle None values in object dtype columns
                if pd.api.types.is_object_dtype(data[col]) and values.dtype == object:
                    values = np.where(values == None, np.nan, values)
                    
                transformed = imputer.transform(values)
                
                if self.add_indicator:
                    new_data[col] = transformed[:, 0]
                    new_data[f"{col}_was_missing"] = transformed[:, 1]
                else:
                    new_data[col] = transformed[:, 0]
                    
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


class PipeOpSelect(PipeOp):
    """Select features based on statistical tests.
    
    Parameters
    ----------
    id : str, default="select"
        Unique identifier.
    k : int, default=10
        Number of features to select.
    score_func : str, default="auto"
        Score function: "auto", "f_classif", "f_regression".
    """
    
    def __init__(
        self,
        id: str = "select",
        k: int = 10,
        score_func: str = "auto",
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.k = k
        self.score_func = score_func
        self._selector = None
        self._selected_features = None
        
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
        """Fit selector and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get features and target
        X = data[task.feature_names]
        
        # Get target values
        if hasattr(task, 'truth'):
            y = task.truth()
        else:
            # For unsupervised tasks, we can't do feature selection
            raise ValueError("Feature selection requires a supervised task with a target variable")
        
        # Determine score function
        if self.score_func == "auto":
            if isinstance(task, TaskClassif):
                score_func = f_classif
            else:
                score_func = f_regression
        elif self.score_func == "f_classif":
            score_func = f_classif
        elif self.score_func == "f_regression":
            score_func = f_regression
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")
            
        # Fit selector
        self._selector = SelectKBest(score_func=score_func, k=min(self.k, len(task.feature_names)))
        X_selected = self._selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = self._selector.get_support()
        self._selected_features = [f for f, selected in zip(task.feature_names, mask) if selected]
        
        # Create new data with selected features only
        new_data = pd.DataFrame(X_selected, columns=self._selected_features, index=data.index)
        
        # Add target column(s)
        for target in task.target_names:
            new_data[target] = data[target]
            
        # Add any other non-feature columns
        other_cols = [col for col in data.columns 
                     if col not in task.feature_names and col not in task.target_names]
        for col in other_cols:
            new_data[col] = data[col]
            
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
        col_roles["feature"] = self._selected_features
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["selector"] = self._selector
        self.state["selected_features"] = self._selected_features
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted selector."""
        if not self.is_trained:
            raise RuntimeError("PipeOpSelect must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        # Check we have all expected features
        missing = set(task.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
            
        # Transform
        X = data[task.feature_names]
        X_selected = self._selector.transform(X)
        
        # Create new data
        new_data = pd.DataFrame(X_selected, columns=self._selected_features, index=data.index)
        
        # Add other columns
        for col in data.columns:
            if col not in task.feature_names:
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
        col_roles["feature"] = self._selected_features
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


class PipeOpEncode(PipeOp):
    """Encode categorical features.
    
    Parameters
    ----------
    id : str, default="encode"
        Unique identifier.
    method : str, default="onehot"
        Encoding method: "onehot", "label".
    drop : str, optional
        For onehot encoding, whether to drop first category.
    """
    
    def __init__(
        self,
        id: str = "encode",
        method: str = "onehot",
        drop: Optional[str] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.method = method
        self.drop = drop
        self._encoders = {}
        self._encoded_cols = {}
        
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
        """Fit encoders and transform task."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Find categorical features
        cat_features = [col for col in task.feature_names
                       if col in data.columns and 
                       pd.api.types.is_object_dtype(data[col])]
        
        if not cat_features:
            # No categorical features
            self.state.is_trained = True
            return {"output": task}
            
        # Encode each categorical column
        encoded_dfs = []
        remaining_cols = []
        
        for col in data.columns:
            if col in cat_features:
                if self.method == "onehot":
                    encoder = OneHotEncoder(
                        drop=self.drop,
                        sparse_output=False,
                        handle_unknown='ignore'
                    )
                    encoded = encoder.fit_transform(data[[col]])
                    
                    # Create column names
                    if hasattr(encoder, 'get_feature_names_out'):
                        col_names = encoder.get_feature_names_out([col])
                    else:
                        col_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                        
                    encoded_df = pd.DataFrame(encoded, columns=col_names, index=data.index)
                    encoded_dfs.append(encoded_df)
                    
                    self._encoders[col] = encoder
                    self._encoded_cols[col] = list(col_names)
                    
                elif self.method == "label":
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(data[col])
                    remaining_cols.append(pd.Series(encoded, name=col, index=data.index))
                    
                    self._encoders[col] = encoder
                    self._encoded_cols[col] = [col]
                    
            else:
                remaining_cols.append(data[col])
                
        # Combine all columns
        if encoded_dfs:
            new_data = pd.concat(
                [pd.DataFrame(remaining_cols).T] + encoded_dfs,
                axis=1
            )
        else:
            new_data = pd.DataFrame(remaining_cols).T
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update feature list
        new_features = []
        for col in task.feature_names:
            if col in self._encoded_cols:
                new_features.extend(self._encoded_cols[col])
            else:
                new_features.append(col)
                
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = new_features
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["encoders"] = self._encoders
        self.state["encoded_cols"] = self._encoded_cols
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform task with fitted encoders."""
        if not self.is_trained:
            raise RuntimeError("PipeOpEncode must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        
        if not self._encoders:
            return {"output": task}
            
        # Apply encoders
        encoded_dfs = []
        remaining_cols = []
        
        for col in data.columns:
            if col in self._encoders:
                encoder = self._encoders[col]
                
                if self.method == "onehot":
                    encoded = encoder.transform(data[[col]])
                    col_names = self._encoded_cols[col]
                    encoded_df = pd.DataFrame(encoded, columns=col_names, index=data.index)
                    encoded_dfs.append(encoded_df)
                    
                elif self.method == "label":
                    # Handle unknown categories
                    values = data[col].copy()
                    known_classes = set(encoder.classes_)
                    mask = values.isin(known_classes)
                    
                    encoded = np.zeros(len(values), dtype=int)
                    encoded[mask] = encoder.transform(values[mask])
                    encoded[~mask] = -1  # Unknown category
                    
                    remaining_cols.append(pd.Series(encoded, name=col, index=data.index))
                    
            else:
                remaining_cols.append(data[col])
                
        # Combine
        if encoded_dfs:
            new_data = pd.concat(
                [pd.DataFrame(remaining_cols).T] + encoded_dfs,
                axis=1
            )
        else:
            new_data = pd.DataFrame(remaining_cols).T
            
        # Create new task
        task_class = type(task)
        new_task = task_class(
            data=new_data,
            target=task.target_names[0] if task.target_names else None,
            id=task.id,
            label=task.label
        )
        
        # Update features
        new_features = []
        for col in task.feature_names:
            if col in self._encoded_cols:
                new_features.extend(self._encoded_cols[col])
            else:
                new_features.append(col)
                
        col_roles = {role: list(cols) for role, cols in task.col_roles.items()}
        col_roles["feature"] = new_features
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        return {"output": new_task}


# Register operators
mlpy_pipeops.register("scale", PipeOpScale)
mlpy_pipeops.register("impute", PipeOpImpute)
mlpy_pipeops.register("select", PipeOpSelect)
mlpy_pipeops.register("encode", PipeOpEncode)


__all__ = [
    "PipeOpScale",
    "PipeOpImpute", 
    "PipeOpSelect",
    "PipeOpEncode"
]