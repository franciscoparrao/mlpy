"""
Optuna integration for hyperparameter tuning in MLPY.

This module provides a tuner that uses Optuna for efficient
hyperparameter optimization with MLPY learners.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import warnings
from copy import deepcopy

try:
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None
    Trial = None

from ..learners.base import Learner
from ..tasks.base import Task
from ..measures.base import Measure
from ..resamplings.base import Resampling
from ..resamplings.cv import ResamplingCV
from ..param_set import ParamSet, Param
from .. import resample


class OptunaTuner:
    """Hyperparameter tuner using Optuna.
    
    This tuner integrates Optuna's Bayesian optimization
    with MLPY's learner and resampling system.
    
    Parameters
    ----------
    learner : Learner
        The learner to tune.
    search_space : dict or callable
        Either a dictionary defining the search space or
        a function that takes a Trial and returns params.
    resampling : Resampling
        Resampling strategy for evaluation.
    measure : Measure
        Performance measure to optimize.
    n_trials : int, default=100
        Number of optimization trials.
    direction : str, default='auto'
        'minimize' or 'maximize'. Auto-detected from measure.
    sampler : optuna.samplers.BaseSampler, optional
        Optuna sampler. Defaults to TPESampler.
    pruner : optuna.pruners.BasePruner, optional
        Optuna pruner for early stopping.
    study_name : str, optional
        Name for the Optuna study.
    storage : str, optional
        Database URL for distributed optimization.
    load_if_exists : bool, default=False
        Whether to load existing study.
    n_jobs : int, default=1
        Number of parallel jobs. -1 for all cores.
    timeout : float, optional
        Time limit for optimization in seconds.
    show_progress_bar : bool, default=True
        Whether to show optimization progress.
    **kwargs
        Additional arguments passed to study.optimize().
    """
    
    def __init__(
        self,
        learner: Learner,
        search_space: Union[Dict[str, Any], Callable],
        resampling: Optional[Resampling] = None,
        measure: Optional[Measure] = None,
        n_trials: int = 100,
        direction: str = 'auto',
        sampler: Optional['optuna.samplers.BaseSampler'] = None,
        pruner: Optional['optuna.pruners.BasePruner'] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        show_progress_bar: bool = True,
        **kwargs
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install it with: pip install optuna"
            )
            
        self.learner = learner
        self.search_space = search_space
        self.resampling = resampling or ResamplingCV(folds=5)
        self.measure = measure
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs
        
        # Determine optimization direction
        if direction == 'auto':
            if measure and hasattr(measure, 'minimize'):
                self.direction = 'minimize' if measure.minimize else 'maximize'
            else:
                self.direction = 'minimize'
        else:
            self.direction = direction
            
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            sampler=sampler or TPESampler(seed=42),
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists
        )
        
        # Results storage
        self.best_params_ = None
        self.best_learner_ = None
        self.best_score_ = None
        self.results_ = []
        
    def _create_params(self, trial: Trial) -> Dict[str, Any]:
        """Create parameter dictionary from trial.
        
        Parameters
        ----------
        trial : Trial
            Optuna trial object.
            
        Returns
        -------
        dict
            Parameter values for this trial.
        """
        if callable(self.search_space):
            return self.search_space(trial)
            
        params = {}
        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, dict):
                param_type = param_spec.get('type', 'float')
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec['choices']
                    )
            elif isinstance(param_spec, (list, tuple)):
                # Simple categorical
                params[param_name] = trial.suggest_categorical(
                    param_name, param_spec
                )
            elif isinstance(param_spec, range):
                # Integer range
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec.start,
                    param_spec.stop - 1
                )
            else:
                raise ValueError(
                    f"Unknown parameter specification for {param_name}: {param_spec}"
                )
                
        return params
        
    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna.
        
        Parameters
        ----------
        trial : Trial
            Optuna trial object.
            
        Returns
        -------
        float
            Objective value (score).
        """
        # Get parameters for this trial
        params = self._create_params(trial)
        
        # Create learner with these parameters
        learner_class = type(self.learner)
        if hasattr(self.learner, 'estimator'):
            # For wrapped learners (sklearn, xgboost, etc.)
            base_params = self.learner.estimator.get_params()
            base_params.update(params)
            
            if hasattr(self.learner, '__class__'):
                # Create new estimator
                estimator_class = type(self.learner.estimator)
                new_estimator = estimator_class(**base_params)
                trial_learner = learner_class(new_estimator)
            else:
                trial_learner = deepcopy(self.learner)
                trial_learner.estimator.set_params(**params)
        else:
            # For native MLPY learners
            trial_learner = learner_class(**params)
            
        # Evaluate with resampling
        try:
            result = resample(
                task=self._task,
                learner=trial_learner,
                resampling=self.resampling,
                measures=self.measure
            )
            
            # Get score
            scores = result.score(measures=self.measure)
            score = scores.mean()
            
            # Store result
            self.results_.append({
                'trial': trial.number,
                'params': params,
                'score': score,
                'scores_cv': scores.values
            })
            
            # Report intermediate value for pruning
            if hasattr(trial, 'report'):
                trial.report(score, step=0)
                
            return score
            
        except Exception as e:
            # Handle failed trials
            warnings.warn(f"Trial {trial.number} failed: {e}")
            return float('inf') if self.direction == 'minimize' else float('-inf')
            
    def tune(self, task: Task) -> 'OptunaTuner':
        """Run hyperparameter optimization.
        
        Parameters
        ----------
        task : Task
            The task to optimize on.
            
        Returns
        -------
        self : OptunaTuner
            Fitted tuner with best parameters.
        """
        self._task = task
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
            **self.kwargs
        )
        
        # Store best results
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value
        
        # Create best learner
        learner_class = type(self.learner)
        if hasattr(self.learner, 'estimator'):
            base_params = self.learner.estimator.get_params()
            base_params.update(self.best_params_)
            estimator_class = type(self.learner.estimator)
            new_estimator = estimator_class(**base_params)
            self.best_learner_ = learner_class(new_estimator)
        else:
            self.best_learner_ = learner_class(**self.best_params_)
            
        return self
        
    def get_best_learner(self) -> Learner:
        """Get the best learner found.
        
        Returns
        -------
        Learner
            Learner with best hyperparameters.
        """
        if self.best_learner_ is None:
            raise RuntimeError("Must call tune() first")
        return self.best_learner_
        
    def get_results_df(self) -> pd.DataFrame:
        """Get optimization results as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Results with trials, parameters, and scores.
        """
        if not self.results_:
            raise RuntimeError("Must call tune() first")
            
        df = pd.DataFrame(self.results_)
        return df.sort_values('score', 
                            ascending=(self.direction == 'minimize'))
        
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self.study)
        except ImportError:
            warnings.warn("Plotly not installed. Cannot create visualization.")
            
    def plot_param_importances(self):
        """Plot parameter importances."""
        try:
            import optuna.visualization as vis
            return vis.plot_param_importances(self.study)
        except ImportError:
            warnings.warn("Plotly not installed. Cannot create visualization.")


def tune_learner(
    learner: Learner,
    task: Task,
    search_space: Union[Dict[str, Any], Callable],
    resampling: Optional[Resampling] = None,
    measure: Optional[Measure] = None,
    n_trials: int = 100,
    **kwargs
) -> Tuple[Learner, Dict[str, Any], float]:
    """Convenience function for hyperparameter tuning.
    
    Parameters
    ----------
    learner : Learner
        The learner to tune.
    task : Task
        The task to optimize on.
    search_space : dict or callable
        Search space definition.
    resampling : Resampling, optional
        Resampling strategy.
    measure : Measure, optional
        Performance measure.
    n_trials : int, default=100
        Number of trials.
    **kwargs
        Additional arguments for OptunaTuner.
        
    Returns
    -------
    best_learner : Learner
        Learner with best parameters.
    best_params : dict
        Best parameter values.
    best_score : float
        Best score achieved.
        
    Examples
    --------
    >>> from mlpy.tuning import tune_learner
    >>> from mlpy.learners import learner_sklearn
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> # Define search space
    >>> search_space = {
    ...     'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
    ...     'max_depth': {'type': 'int', 'low': 2, 'high': 20},
    ...     'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
    ... }
    >>> 
    >>> # Tune
    >>> learner = learner_sklearn(RandomForestClassifier())
    >>> best_learner, best_params, score = tune_learner(
    ...     learner, task, search_space, n_trials=50
    ... )
    """
    tuner = OptunaTuner(
        learner=learner,
        search_space=search_space,
        resampling=resampling,
        measure=measure,
        n_trials=n_trials,
        **kwargs
    )
    
    tuner.tune(task)
    
    return (
        tuner.get_best_learner(),
        tuner.best_params_,
        tuner.best_score_
    )