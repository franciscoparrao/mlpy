"""Automated feature engineering for MLPY.

Provides pipeline operators that automatically create new features:
- Numeric transformations
- Categorical encoding variations
- Feature interactions
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from itertools import combinations

from ..pipelines import PipeOp, PipeOpInput, PipeOpOutput
from ..tasks import Task, TaskClassif, TaskRegr


class AutoFeaturesNumeric(PipeOp):
    """Automatically generate numeric feature transformations.
    
    Creates new features using:
    - Log transformations
    - Square root
    - Squared values
    - Reciprocals
    - Binning
    
    Parameters
    ----------
    id : str
        Operator ID.
    transforms : List[str]
        Which transformations to apply.
        Options: "log", "sqrt", "square", "reciprocal", "bins"
    n_bins : int
        Number of bins for binning.
    """
    
    def __init__(
        self,
        id: str = "auto_features_numeric",
        transforms: List[str] = ["log", "sqrt", "square"],
        n_bins: int = 5,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.transforms = transforms
        self.n_bins = n_bins
        self._feature_info = {}
        
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
        """Generate and add numeric features."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get numeric features
        numeric_features = [col for col in task.feature_names
                          if pd.api.types.is_numeric_dtype(data[col])]
        
        if not numeric_features:
            self.state.is_trained = True
            return {"output": task}
            
        new_data = data.copy()
        new_features = []
        
        for col in numeric_features:
            values = data[col].values
            
            # Skip if too many missing values
            if pd.isna(values).sum() > len(values) * 0.5:
                continue
                
            # Log transform (only for positive values)
            if "log" in self.transforms:
                min_val = np.nanmin(values)
                if min_val > 0:
                    new_col = f"{col}_log"
                    new_data[new_col] = np.log(values)
                    new_features.append(new_col)
                    self._feature_info[new_col] = {"type": "log", "source": col}
                elif min_val > -1:  # Can use log1p
                    new_col = f"{col}_log1p"
                    new_data[new_col] = np.log1p(values)
                    new_features.append(new_col)
                    self._feature_info[new_col] = {"type": "log1p", "source": col}
                    
            # Square root (only for non-negative)
            if "sqrt" in self.transforms:
                min_val = np.nanmin(values)
                if min_val >= 0:
                    new_col = f"{col}_sqrt"
                    new_data[new_col] = np.sqrt(values)
                    new_features.append(new_col)
                    self._feature_info[new_col] = {"type": "sqrt", "source": col}
                    
            # Square
            if "square" in self.transforms:
                new_col = f"{col}_sq"
                new_data[new_col] = values ** 2
                new_features.append(new_col)
                self._feature_info[new_col] = {"type": "square", "source": col}
                
            # Reciprocal (avoid division by zero)
            if "reciprocal" in self.transforms:
                mask = values != 0
                if mask.sum() > len(values) * 0.5:  # At least 50% non-zero
                    new_col = f"{col}_inv"
                    new_data[new_col] = np.where(mask, 1.0 / values, np.nan)
                    new_features.append(new_col)
                    self._feature_info[new_col] = {"type": "reciprocal", "source": col}
                    
            # Binning
            if "bins" in self.transforms:
                # Calculate quantiles on non-missing values
                valid_values = values[~pd.isna(values)]
                if len(valid_values) > self.n_bins:
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bin_edges = np.percentile(valid_values, quantiles * 100)
                    bin_edges[0] = -np.inf
                    bin_edges[-1] = np.inf
                    
                    new_col = f"{col}_bin"
                    new_data[new_col] = pd.cut(values, bins=bin_edges, labels=False)
                    new_features.append(new_col)
                    self._feature_info[new_col] = {
                        "type": "bins",
                        "source": col,
                        "edges": bin_edges
                    }
                    
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
        col_roles["feature"].extend(new_features)
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["feature_info"] = self._feature_info
        self.state["new_features"] = new_features
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply same transformations to new data."""
        if not self.is_trained:
            raise RuntimeError("AutoFeaturesNumeric must be trained first")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        new_data = data.copy()
        
        # Apply transformations based on stored info
        for new_col, info in self._feature_info.items():
            source_col = info["source"]
            if source_col not in data.columns:
                continue
                
            values = data[source_col].values
            
            if info["type"] == "log":
                new_data[new_col] = np.log(values)
            elif info["type"] == "log1p":
                new_data[new_col] = np.log1p(values)
            elif info["type"] == "sqrt":
                new_data[new_col] = np.sqrt(values)
            elif info["type"] == "square":
                new_data[new_col] = values ** 2
            elif info["type"] == "reciprocal":
                mask = values != 0
                new_data[new_col] = np.where(mask, 1.0 / values, np.nan)
            elif info["type"] == "bins":
                new_data[new_col] = pd.cut(values, bins=info["edges"], labels=False)
                
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


class AutoFeaturesCategorical(PipeOp):
    """Automatically generate categorical feature transformations.
    
    Creates new features using:
    - Count encoding
    - Target encoding (for supervised tasks)
    - Frequency encoding
    - Rare category grouping
    
    Parameters
    ----------
    id : str
        Operator ID.
    methods : List[str]
        Encoding methods to use.
    min_frequency : float
        Minimum frequency to keep category (else group as "rare").
    """
    
    def __init__(
        self,
        id: str = "auto_features_categorical",
        methods: List[str] = ["count", "frequency"],
        min_frequency: float = 0.01,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.methods = methods
        self.min_frequency = min_frequency
        self._encodings = {}
        
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
        """Generate categorical features."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Get categorical features
        cat_features = [col for col in task.feature_names
                       if pd.api.types.is_object_dtype(data[col])]
        
        if not cat_features:
            self.state.is_trained = True
            return {"output": task}
            
        new_data = data.copy()
        new_features = []
        
        for col in cat_features:
            values = data[col]
            
            # Count encoding
            if "count" in self.methods:
                counts = values.value_counts()
                count_map = counts.to_dict()
                new_col = f"{col}_count"
                new_data[new_col] = values.map(count_map).fillna(0)
                new_features.append(new_col)
                self._encodings[new_col] = {"type": "count", "map": count_map}
                
            # Frequency encoding
            if "frequency" in self.methods:
                freq = values.value_counts(normalize=True)
                freq_map = freq.to_dict()
                new_col = f"{col}_freq"
                new_data[new_col] = values.map(freq_map).fillna(0)
                new_features.append(new_col)
                self._encodings[new_col] = {"type": "frequency", "map": freq_map}
                
            # Target encoding (only for supervised tasks)
            if "target" in self.methods and hasattr(task, 'truth'):
                target = task.truth()
                if isinstance(task, TaskRegr):
                    # Mean target per category
                    target_means = data.groupby(col)[task.target_names[0]].mean()
                    target_map = target_means.to_dict()
                    new_col = f"{col}_target_mean"
                    new_data[new_col] = values.map(target_map).fillna(target.mean())
                    new_features.append(new_col)
                    self._encodings[new_col] = {
                        "type": "target_mean",
                        "map": target_map,
                        "default": target.mean()
                    }
                    
            # Rare category grouping
            if "rare" in self.methods:
                freq = values.value_counts(normalize=True)
                rare_cats = freq[freq < self.min_frequency].index
                if len(rare_cats) > 0:
                    new_col = f"{col}_grouped"
                    new_data[new_col] = values.replace(rare_cats, "__rare__")
                    new_features.append(new_col)
                    self._encodings[new_col] = {
                        "type": "rare",
                        "rare_cats": list(rare_cats)
                    }
                    
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
        col_roles["feature"].extend(new_features)
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["encodings"] = self._encodings
        self.state["new_features"] = new_features
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply encodings to new data."""
        if not self.is_trained:
            raise RuntimeError("AutoFeaturesCategorical must be trained first")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        new_data = data.copy()
        
        # Apply encodings
        for new_col, info in self._encodings.items():
            # Extract source column from new column name
            if info["type"] in ["count", "frequency", "target_mean"]:
                # Remove suffix to get source column
                suffix = f"_{info['type'].split('_')[0]}"
                if info["type"] == "target_mean":
                    suffix = "_target_mean"
                source_col = new_col[:-len(suffix)]
            else:  # rare
                source_col = new_col[:-len("_grouped")]
                
            if source_col not in data.columns:
                continue
                
            values = data[source_col]
            
            if info["type"] in ["count", "frequency", "target_mean"]:
                default = info.get("default", 0)
                new_data[new_col] = values.map(info["map"]).fillna(default)
            elif info["type"] == "rare":
                new_data[new_col] = values.replace(info["rare_cats"], "__rare__")
                
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


class AutoFeaturesInteraction(PipeOp):
    """Generate feature interactions.
    
    Creates:
    - Numeric × Numeric: multiplication, division
    - Categorical × Categorical: concatenation
    - Numeric × Categorical: group statistics
    
    Parameters
    ----------
    id : str
        Operator ID.
    max_interactions : int
        Maximum number of interactions to create.
    numeric_ops : List[str]
        Operations for numeric interactions.
    """
    
    def __init__(
        self,
        id: str = "auto_features_interaction",
        max_interactions: int = 10,
        numeric_ops: List[str] = ["multiply", "divide"],
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.max_interactions = max_interactions
        self.numeric_ops = numeric_ops
        self._interactions = []
        
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
        """Generate interaction features."""
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        data = task.data()
        
        # Separate numeric and categorical features
        numeric_features = [col for col in task.feature_names
                          if pd.api.types.is_numeric_dtype(data[col])]
        cat_features = [col for col in task.feature_names
                       if pd.api.types.is_object_dtype(data[col])]
        
        new_data = data.copy()
        new_features = []
        interaction_count = 0
        
        # Numeric × Numeric interactions
        if len(numeric_features) >= 2:
            for col1, col2 in combinations(numeric_features, 2):
                if interaction_count >= self.max_interactions:
                    break
                    
                values1 = data[col1].values
                values2 = data[col2].values
                
                # Multiplication
                if "multiply" in self.numeric_ops:
                    new_col = f"{col1}_x_{col2}"
                    new_data[new_col] = values1 * values2
                    new_features.append(new_col)
                    self._interactions.append({
                        "type": "multiply",
                        "cols": [col1, col2],
                        "name": new_col
                    })
                    interaction_count += 1
                    
                # Division (avoid division by zero)
                if "divide" in self.numeric_ops and interaction_count < self.max_interactions:
                    mask = values2 != 0
                    if mask.sum() > len(values2) * 0.5:
                        new_col = f"{col1}_div_{col2}"
                        new_data[new_col] = np.where(mask, values1 / values2, np.nan)
                        new_features.append(new_col)
                        self._interactions.append({
                            "type": "divide",
                            "cols": [col1, col2],
                            "name": new_col
                        })
                        interaction_count += 1
                        
        # Categorical × Categorical interactions
        if len(cat_features) >= 2:
            for col1, col2 in combinations(cat_features, 2):
                if interaction_count >= self.max_interactions:
                    break
                    
                new_col = f"{col1}_AND_{col2}"
                new_data[new_col] = data[col1].astype(str) + "_" + data[col2].astype(str)
                new_features.append(new_col)
                self._interactions.append({
                    "type": "concat",
                    "cols": [col1, col2],
                    "name": new_col
                })
                interaction_count += 1
                
        # Numeric × Categorical: group statistics
        if numeric_features and cat_features:
            for num_col in numeric_features[:3]:  # Limit to avoid explosion
                for cat_col in cat_features[:3]:
                    if interaction_count >= self.max_interactions:
                        break
                        
                    # Mean of numeric per category
                    group_means = data.groupby(cat_col)[num_col].mean()
                    mean_map = group_means.to_dict()
                    
                    new_col = f"{num_col}_by_{cat_col}_mean"
                    new_data[new_col] = data[cat_col].map(mean_map).fillna(data[num_col].mean())
                    new_features.append(new_col)
                    self._interactions.append({
                        "type": "group_mean",
                        "cols": [num_col, cat_col],
                        "name": new_col,
                        "map": mean_map,
                        "default": data[num_col].mean()
                    })
                    interaction_count += 1
                    
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
        col_roles["feature"].extend(new_features)
        new_task.set_col_roles(col_roles)
        new_task.set_row_roles({role: list(rows) for role, rows in task.row_roles.items()})
        
        self.state.is_trained = True
        self.state["interactions"] = self._interactions
        self.state["new_features"] = new_features
        
        return {"output": new_task}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interactions to new data."""
        if not self.is_trained:
            raise RuntimeError("AutoFeaturesInteraction must be trained first")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        data = task.data()
        new_data = data.copy()
        
        # Apply interactions
        for interaction in self._interactions:
            cols = interaction["cols"]
            new_col = interaction["name"]
            
            # Check columns exist
            if not all(col in data.columns for col in cols):
                continue
                
            if interaction["type"] == "multiply":
                new_data[new_col] = data[cols[0]] * data[cols[1]]
            elif interaction["type"] == "divide":
                values2 = data[cols[1]].values
                mask = values2 != 0
                new_data[new_col] = np.where(
                    mask,
                    data[cols[0]].values / values2,
                    np.nan
                )
            elif interaction["type"] == "concat":
                new_data[new_col] = data[cols[0]].astype(str) + "_" + data[cols[1]].astype(str)
            elif interaction["type"] == "group_mean":
                new_data[new_col] = data[cols[1]].map(interaction["map"]).fillna(interaction["default"])
                
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


__all__ = [
    "AutoFeaturesNumeric",
    "AutoFeaturesCategorical",
    "AutoFeaturesInteraction"
]