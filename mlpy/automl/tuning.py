"""Hyperparameter tuning functionality for MLPY.

Provides tools for automated hyperparameter optimization including:
- Parameter spaces definition
- Grid search
- Random search
"""

from typing import Dict, List, Any, Union, Optional, Tuple, Type
from abc import ABC, abstractmethod
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy

from ..base import MLPYObject
from ..learners import Learner
from ..tasks import Task
from ..resamplings import Resampling
from ..measures import Measure
from ..resample import resample
from ..callbacks import CallbackSet, Callback


def _set_nested_param(obj, param_path: str, value: Any) -> None:
    """Set a nested parameter on an object.
    
    Handles special cases like GraphLearner where we need to
    find the right PipeOp.
    """
    from ..pipelines import GraphLearner, PipeOpLearner
    
    parts = param_path.split('.')
    
    # Special handling for GraphLearner
    if isinstance(obj, GraphLearner) and parts[0] == "learner":
        # Find the PipeOpLearner in the graph
        learner_op = None
        for op in obj.graph.pipeops.values():
            if isinstance(op, PipeOpLearner):
                learner_op = op
                break
                
        if learner_op is None:
            raise ValueError("No PipeOpLearner found in GraphLearner")
            
        # Now set the parameter on the actual learner
        remaining_path = '.'.join(parts[1:])
        if remaining_path:
            _set_nested_param(learner_op.learner, remaining_path, value)
        else:
            # This shouldn't happen
            raise ValueError("Cannot set 'learner' directly")
            
    else:
        # Standard nested parameter setting
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)


class Param(MLPYObject, ABC):
    """Abstract base class for hyperparameter definitions.
    
    Parameters
    ----------
    id : str
        Parameter name.
    """
    
    def __init__(self, id: str):
        super().__init__(id=id)
        
    @abstractmethod
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Sample values from parameter space.
        
        Parameters
        ----------
        n : int
            Number of values to sample.
        seed : int, optional
            Random seed.
            
        Returns
        -------
        List[Any]
            Sampled values.
        """
        pass
        
    @abstractmethod
    def grid(self, resolution: int = 10) -> List[Any]:
        """Generate grid of values.
        
        Parameters
        ----------
        resolution : int
            Number of grid points.
            
        Returns
        -------
        List[Any]
            Grid values.
        """
        pass


class ParamInt(Param):
    """Integer hyperparameter.
    
    Parameters
    ----------
    id : str
        Parameter name.
    lower : int
        Lower bound (inclusive).
    upper : int
        Upper bound (inclusive).
    log_scale : bool
        Whether to sample on log scale.
    """
    
    def __init__(self, id: str, lower: int, upper: int, log_scale: bool = False):
        super().__init__(id=id)
        self.lower = lower
        self.upper = upper
        self.log_scale = log_scale
        
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[int]:
        """Sample integer values."""
        if seed is not None:
            # Ensure seed is within valid range
            seed = abs(seed) % (2**32)
            np.random.seed(seed)
            
        if self.log_scale:
            # Sample on log scale
            log_lower = np.log(max(1, self.lower))
            log_upper = np.log(self.upper)
            values = np.exp(np.random.uniform(log_lower, log_upper, n))
            return [int(np.round(v)) for v in values]
        else:
            return [int(x) for x in np.random.randint(self.lower, self.upper + 1, n)]
            
    def grid(self, resolution: int = 10) -> List[int]:
        """Generate integer grid."""
        if self.log_scale:
            log_lower = np.log(max(1, self.lower))
            log_upper = np.log(self.upper)
            values = np.exp(np.linspace(log_lower, log_upper, resolution))
            return sorted(list(set([int(np.round(v)) for v in values])))
        else:
            step = max(1, (self.upper - self.lower) // (resolution - 1))
            return list(range(self.lower, self.upper + 1, step))


class ParamFloat(Param):
    """Float hyperparameter.
    
    Parameters
    ----------
    id : str
        Parameter name.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    log_scale : bool
        Whether to sample on log scale.
    """
    
    def __init__(self, id: str, lower: float, upper: float, log_scale: bool = False):
        super().__init__(id=id)
        self.lower = lower
        self.upper = upper
        self.log_scale = log_scale
        
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[float]:
        """Sample float values."""
        if seed is not None:
            # Ensure seed is within valid range
            seed = abs(seed) % (2**32)
            np.random.seed(seed)
            
        if self.log_scale:
            log_lower = np.log(self.lower)
            log_upper = np.log(self.upper)
            values = np.exp(np.random.uniform(log_lower, log_upper, n))
            return list(values)
        else:
            return list(np.random.uniform(self.lower, self.upper, n))
            
    def grid(self, resolution: int = 10) -> List[float]:
        """Generate float grid."""
        if self.log_scale:
            log_lower = np.log(self.lower)
            log_upper = np.log(self.upper)
            values = np.exp(np.linspace(log_lower, log_upper, resolution))
            return list(values)
        else:
            return list(np.linspace(self.lower, self.upper, resolution))


class ParamCategorical(Param):
    """Categorical hyperparameter.
    
    Parameters
    ----------
    id : str
        Parameter name.
    values : List[Any]
        Possible values.
    """
    
    def __init__(self, id: str, values: List[Any]):
        super().__init__(id=id)
        self.values = values
        
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Sample categorical values."""
        if seed is not None:
            # Ensure seed is within valid range
            seed = abs(seed) % (2**32)
            np.random.seed(seed)
        return list(np.random.choice(self.values, n))
        
    def grid(self, resolution: int = 10) -> List[Any]:
        """Return all categorical values."""
        return self.values


class ParamSet:
    """Set of hyperparameters.
    
    Parameters
    ----------
    params : List[Param]
        List of parameter definitions.
    """
    
    def __init__(self, params: List[Param]):
        self.params = {p.id: p for p in params}
        
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample parameter combinations.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of parameter dictionaries.
        """
        if seed is not None:
            np.random.seed(seed)
            
        configs = []
        for i in range(n):
            config = {}
            for name, param in self.params.items():
                # Use different seed for each param
                param_seed = None if seed is None else seed + i + hash(name)
                config[name] = param.sample(1, seed=param_seed)[0]
            configs.append(config)
            
        return configs
        
    def grid(self, resolution: int = 10) -> List[Dict[str, Any]]:
        """Generate grid of parameter combinations."""
        # Get grid for each parameter
        param_grids = {}
        for name, param in self.params.items():
            param_grids[name] = param.grid(resolution)
            
        # Generate all combinations
        keys = list(param_grids.keys())
        values = [param_grids[k] for k in keys]
        
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
            
        return configs


class TuneResult(MLPYObject):
    """Result of hyperparameter tuning.
    
    Stores results from tuning including:
    - Best configuration
    - All evaluated configurations
    - Performance scores
    """
    
    def __init__(
        self,
        learner: Learner,
        param_set: ParamSet,
        configs: List[Dict[str, Any]],
        scores: List[float],
        best_config: Dict[str, Any],
        best_score: float,
        measure: Measure,
        runtime: float
    ):
        super().__init__(id="tune_result")
        self.learner = learner
        self.param_set = param_set
        self.configs = configs
        self.scores = scores
        self.best_config = best_config
        self.best_score = best_score
        self.measure = measure
        self.runtime = runtime
        
    def as_data_frame(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        df = pd.DataFrame(self.configs)
        df[f"{self.measure.id}_score"] = self.scores
        df["is_best"] = False
        
        # Mark best configuration
        best_idx = np.argmin(self.scores) if self.measure.minimize else np.argmax(self.scores)
        df.loc[best_idx, "is_best"] = True
        
        return df
        
    def plot(self, param: str, ax=None):
        """Plot scores vs parameter values.
        
        Parameters
        ----------
        param : str
            Parameter to plot.
        ax : matplotlib axis, optional
            Axis to plot on.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
            
        if ax is None:
            fig, ax = plt.subplots()
            
        df = self.as_data_frame()
        
        # Group by parameter value and aggregate
        grouped = df.groupby(param)[f"{self.measure.id}_score"].agg(['mean', 'std'])
        
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                   marker='o', capsize=5)
        ax.set_xlabel(param)
        ax.set_ylabel(f"{self.measure.id} score")
        ax.set_title(f"Tuning results for {param}")
        
        # Mark best value
        best_val = self.best_config[param]
        ax.axvline(best_val, color='red', linestyle='--', alpha=0.5, 
                  label=f"Best: {best_val}")
        ax.legend()
        
        return ax
        
    def __repr__(self) -> str:
        return (
            f"TuneResult(n_configs={len(self.configs)}, "
            f"best_score={self.best_score:.4f}, "
            f"runtime={self.runtime:.1f}s)"
        )


class Tuner(MLPYObject, ABC):
    """Abstract base class for hyperparameter tuners.
    
    Parameters
    ----------
    id : str
        Tuner identifier.
    """
    
    def __init__(self, id: str = None):
        super().__init__(id=id)
        
    @abstractmethod
    def tune(
        self,
        learner: Learner,
        task: Task,
        resampling: Resampling,
        measure: Measure,
        param_set: ParamSet,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        **kwargs
    ) -> TuneResult:
        """Run hyperparameter tuning.
        
        Parameters
        ----------
        learner : Learner
            Learner to tune.
        task : Task
            Task to evaluate on.
        resampling : Resampling
            Resampling strategy.
        measure : Measure
            Performance measure.
        param_set : ParamSet
            Parameter space.
            
        Returns
        -------
        TuneResult
            Tuning results.
        """
        pass


class TunerGrid(Tuner):
    """Grid search tuner.
    
    Exhaustively evaluates all parameter combinations.
    
    Parameters
    ----------
    resolution : int
        Grid resolution for numeric parameters.
    """
    
    def __init__(self, resolution: int = 10):
        super().__init__(id="grid_search")
        self.resolution = resolution
        
    def tune(
        self,
        learner: Learner,
        task: Task,
        resampling: Resampling,
        measure: Measure,
        param_set: ParamSet,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        **kwargs
    ) -> TuneResult:
        """Run grid search."""
        import time
        start_time = time.time()
        
        # Setup callbacks
        if callbacks is None:
            callback_set = CallbackSet()
        elif isinstance(callbacks, Callback):
            callback_set = CallbackSet([callbacks])
        else:
            callback_set = CallbackSet(callbacks)
        
        # Generate grid
        configs = param_set.grid(self.resolution)
        scores = []
        
        # Notify callbacks of tuning start
        callback_set.on_tune_begin(learner, param_set, len(configs))
        
        # Evaluate each configuration
        for i, config in enumerate(configs):
            try:
                # Notify callbacks of config start
                callback_set.on_config_begin(i, config)
                
                # Clone learner and set parameters
                learner_clone = learner.clone()
                for param, value in config.items():
                    _set_nested_param(learner_clone, param, value)
                    
                # Evaluate
                result = resample(
                    task=task,
                    learner=learner_clone,
                    resampling=resampling,
                    measures=measure
                )
                
                score = result.score(measure.id)
                scores.append(score)
                
                print(f"Config {i+1}/{len(configs)}: {config} -> {score:.4f}")
                
                # Notify callbacks of config end
                callback_set.on_config_end(i, score)
                
            except Exception as e:
                # Notify callbacks of error
                callback_set.on_error(e, f"config {i}")
                scores.append(float('nan'))
                print(f"Config {i+1}/{len(configs)}: FAILED - {e}")
            
        # Find best
        valid_scores = [(idx, s) for idx, s in enumerate(scores) if not np.isnan(s)]
        if valid_scores:
            if measure.minimize:
                best_idx = min(valid_scores, key=lambda x: x[1])[0]
            else:
                best_idx = max(valid_scores, key=lambda x: x[1])[0]
        else:
            best_idx = 0  # Default to first if all failed
            
        runtime = time.time() - start_time
        
        result = TuneResult(
            learner=learner,
            param_set=param_set,
            configs=configs,
            scores=scores,
            best_config=configs[best_idx],
            best_score=scores[best_idx],
            measure=measure,
            runtime=runtime
        )
        
        # Notify callbacks of tuning end
        callback_set.on_tune_end(result)
        
        return result


class TunerRandom(Tuner):
    """Random search tuner.
    
    Randomly samples parameter combinations.
    
    Parameters
    ----------
    n_evals : int
        Number of configurations to evaluate.
    seed : int, optional
        Random seed.
    """
    
    def __init__(self, n_evals: int = 50, seed: Optional[int] = None):
        super().__init__(id="random_search")
        self.n_evals = n_evals
        self.seed = seed
        
    def tune(
        self,
        learner: Learner,
        task: Task,
        resampling: Resampling,
        measure: Measure,
        param_set: ParamSet,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        **kwargs
    ) -> TuneResult:
        """Run random search."""
        import time
        start_time = time.time()
        
        # Setup callbacks
        if callbacks is None:
            callback_set = CallbackSet()
        elif isinstance(callbacks, Callback):
            callback_set = CallbackSet([callbacks])
        else:
            callback_set = CallbackSet(callbacks)
        
        # Sample configurations
        configs = param_set.sample(self.n_evals, seed=self.seed)
        scores = []
        
        # Notify callbacks of tuning start
        callback_set.on_tune_begin(learner, param_set, len(configs))
        
        # Evaluate each
        for i, config in enumerate(configs):
            try:
                # Notify callbacks of config start
                callback_set.on_config_begin(i, config)
                
                # Clone and configure
                learner_clone = learner.clone()
                for param, value in config.items():
                    _set_nested_param(learner_clone, param, value)
                    
                # Evaluate
                result = resample(
                    task=task,
                    learner=learner_clone,
                    resampling=resampling,
                    measures=measure
                )
                
                score = result.score(measure.id)
                scores.append(score)
                
                print(f"Config {i+1}/{len(configs)}: score={score:.4f}")
                
                # Notify callbacks of config end
                callback_set.on_config_end(i, score)
                
            except Exception as e:
                # Notify callbacks of error
                callback_set.on_error(e, f"config {i}")
                scores.append(float('nan'))
                print(f"Config {i+1}/{len(configs)}: FAILED - {e}")
            
        # Find best
        valid_scores = [(idx, s) for idx, s in enumerate(scores) if not np.isnan(s)]
        if valid_scores:
            if measure.minimize:
                best_idx = min(valid_scores, key=lambda x: x[1])[0]
            else:
                best_idx = max(valid_scores, key=lambda x: x[1])[0]
        else:
            best_idx = 0  # Default to first if all failed
            
        runtime = time.time() - start_time
        
        result = TuneResult(
            learner=learner,
            param_set=param_set,
            configs=configs,
            scores=scores,
            best_config=configs[best_idx],
            best_score=scores[best_idx],
            measure=measure,
            runtime=runtime
        )
        
        # Notify callbacks of tuning end
        callback_set.on_tune_end(result)
        
        return result


__all__ = [
    "Param",
    "ParamInt",
    "ParamFloat",
    "ParamCategorical",
    "ParamSet",
    "TuneResult",
    "Tuner",
    "TunerGrid",
    "TunerRandom"
]