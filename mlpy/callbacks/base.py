"""Base callback classes for MLPY.

This module provides the abstract base class for callbacks
and a container for managing multiple callbacks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time

from ..base import MLPYObject
from ..tasks import Task
from ..learners import Learner
from ..predictions import Prediction
from ..measures import Measure


class Callback(MLPYObject, ABC):
    """Abstract base class for callbacks.
    
    Callbacks allow monitoring and controlling the execution
    of ML experiments at various stages.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier for the callback.
    """
    
    def __init__(self, id: Optional[str] = None):
        super().__init__(id=id or self.__class__.__name__)
        
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Called when resample starts.
        
        Parameters
        ----------
        task : Task
            The task being resampled.
        learner : Learner
            The learner being evaluated.
        resampling : Resampling
            The resampling strategy.
        """
        pass
        
    def on_resample_end(self, result: Any) -> None:
        """Called when resample ends.
        
        Parameters
        ----------
        result : ResampleResult
            The resample result.
        """
        pass
        
    def on_iteration_begin(self, iteration: int, train_set: List[int], test_set: List[int]) -> None:
        """Called before each resampling iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        train_set : list of int
            Training row indices.
        test_set : list of int
            Test row indices.
        """
        pass
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Called after each resampling iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        scores : dict
            Scores for each measure.
        train_time : float
            Time taken to train.
        predict_time : float
            Time taken to predict.
        """
        pass
        
    # Training callbacks
    def on_train_begin(self, task: Task, learner: Learner) -> None:
        """Called before training starts.
        
        Parameters
        ----------
        task : Task
            The training task.
        learner : Learner
            The learner being trained.
        """
        pass
        
    def on_train_end(self, learner: Learner) -> None:
        """Called after training ends.
        
        Parameters
        ----------
        learner : Learner
            The trained learner.
        """
        pass
        
    # Prediction callbacks
    def on_predict_begin(self, task: Task, learner: Learner) -> None:
        """Called before prediction starts.
        
        Parameters
        ----------
        task : Task
            The task to predict on.
        learner : Learner
            The learner making predictions.
        """
        pass
        
    def on_predict_end(self, prediction: Prediction) -> None:
        """Called after prediction ends.
        
        Parameters
        ----------
        prediction : Prediction
            The predictions made.
        """
        pass
        
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Called when benchmark starts.
        
        Parameters
        ----------
        tasks : list of Task
            Tasks in the benchmark.
        learners : list of Learner
            Learners in the benchmark.
        """
        pass
        
    def on_benchmark_end(self, result: Any) -> None:
        """Called when benchmark ends.
        
        Parameters
        ----------
        result : BenchmarkResult
            The benchmark result.
        """
        pass
        
    def on_experiment_begin(self, task: Task, learner: Learner, experiment_num: int) -> None:
        """Called before each benchmark experiment.
        
        Parameters
        ----------
        task : Task
            Current task.
        learner : Learner
            Current learner.
        experiment_num : int
            Experiment number.
        """
        pass
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Called after each benchmark experiment.
        
        Parameters
        ----------
        task : Task
            Current task.
        learner : Learner
            Current learner.
        result : ResampleResult, optional
            Result if successful.
        error : Exception, optional
            Error if failed.
        """
        pass
        
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Called when tuning starts.
        
        Parameters
        ----------
        learner : Learner
            Learner being tuned.
        param_set : ParamSet
            Parameter space.
        n_configs : int
            Number of configurations to evaluate.
        """
        pass
        
    def on_tune_end(self, result: Any) -> None:
        """Called when tuning ends.
        
        Parameters
        ----------
        result : TuneResult
            The tuning result.
        """
        pass
        
    def on_config_begin(self, config_num: int, config: Dict[str, Any]) -> None:
        """Called before evaluating a configuration.
        
        Parameters
        ----------
        config_num : int
            Configuration number.
        config : dict
            Parameter configuration.
        """
        pass
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Called after evaluating a configuration.
        
        Parameters
        ----------
        config_num : int
            Configuration number.
        score : float
            Configuration score.
        """
        pass
        
    # Error handling
    def on_error(self, error: Exception, context: str) -> None:
        """Called when an error occurs.
        
        Parameters
        ----------
        error : Exception
            The error that occurred.
        context : str
            Context where error occurred.
        """
        pass


class CallbackSet:
    """Container for managing multiple callbacks.
    
    Parameters
    ----------
    callbacks : list of Callback
        List of callbacks to manage.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        
    def add(self, callback: Callback) -> None:
        """Add a callback to the set."""
        self.callbacks.append(callback)
        
    def remove(self, callback: Callback) -> None:
        """Remove a callback from the set."""
        self.callbacks.remove(callback)
        
    def clear(self) -> None:
        """Remove all callbacks."""
        self.callbacks.clear()
        
    def _call_method(self, method_name: str, *args, **kwargs) -> None:
        """Call a method on all callbacks.
        
        Parameters
        ----------
        method_name : str
            Name of the method to call.
        *args
            Positional arguments for the method.
        **kwargs
            Keyword arguments for the method.
        """
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method is not None:
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    # Log error but continue with other callbacks
                    import warnings
                    warnings.warn(
                        f"Callback {callback.id} failed in {method_name}: {e}"
                    )
                    
    # Delegate all callback methods
    def on_resample_begin(self, *args, **kwargs):
        self._call_method("on_resample_begin", *args, **kwargs)
        
    def on_resample_end(self, *args, **kwargs):
        self._call_method("on_resample_end", *args, **kwargs)
        
    def on_iteration_begin(self, *args, **kwargs):
        self._call_method("on_iteration_begin", *args, **kwargs)
        
    def on_iteration_end(self, *args, **kwargs):
        self._call_method("on_iteration_end", *args, **kwargs)
        
    def on_train_begin(self, *args, **kwargs):
        self._call_method("on_train_begin", *args, **kwargs)
        
    def on_train_end(self, *args, **kwargs):
        self._call_method("on_train_end", *args, **kwargs)
        
    def on_predict_begin(self, *args, **kwargs):
        self._call_method("on_predict_begin", *args, **kwargs)
        
    def on_predict_end(self, *args, **kwargs):
        self._call_method("on_predict_end", *args, **kwargs)
        
    def on_benchmark_begin(self, *args, **kwargs):
        self._call_method("on_benchmark_begin", *args, **kwargs)
        
    def on_benchmark_end(self, *args, **kwargs):
        self._call_method("on_benchmark_end", *args, **kwargs)
        
    def on_experiment_begin(self, *args, **kwargs):
        self._call_method("on_experiment_begin", *args, **kwargs)
        
    def on_experiment_end(self, *args, **kwargs):
        self._call_method("on_experiment_end", *args, **kwargs)
        
    def on_tune_begin(self, *args, **kwargs):
        self._call_method("on_tune_begin", *args, **kwargs)
        
    def on_tune_end(self, *args, **kwargs):
        self._call_method("on_tune_end", *args, **kwargs)
        
    def on_config_begin(self, *args, **kwargs):
        self._call_method("on_config_begin", *args, **kwargs)
        
    def on_config_end(self, *args, **kwargs):
        self._call_method("on_config_end", *args, **kwargs)
        
    def on_error(self, *args, **kwargs):
        self._call_method("on_error", *args, **kwargs)
        
    def __repr__(self) -> str:
        return f"CallbackSet({len(self.callbacks)} callbacks)"


__all__ = ["Callback", "CallbackSet"]