"""Benchmark function for comparing multiple learners.

This module provides the benchmark() function for evaluating and comparing
multiple learners on multiple tasks using resampling strategies.
"""

import time
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from itertools import product

from .tasks import Task
from .learners import Learner
from .resamplings import Resampling
from .measures import Measure
from .resample import resample, ResampleResult
from .utils.logging import get_logger
from .parallel import Backend, get_backend
from .callbacks import CallbackSet, Callback

logger = get_logger(__name__)


class BenchmarkResult:
    """Result of a benchmark comparison.
    
    Stores results from evaluating multiple learners on multiple tasks
    using resampling strategies.
    
    Parameters
    ----------
    tasks : List[Task]
        The tasks that were evaluated.
    learners : List[Learner]
        The learners that were evaluated.
    resampling : Resampling
        The resampling strategy used.
    measures : List[Measure]
        The measures used for evaluation.
    """
    
    def __init__(
        self,
        tasks: List[Task],
        learners: List[Learner],
        resampling: Resampling,
        measures: List[Measure]
    ):
        self.tasks = tasks
        self.learners = learners
        self.resampling = resampling
        self.measures = measures
        
        # Storage for results - indexed by (task_id, learner_id)
        self.results: Dict[Tuple[str, str], ResampleResult] = {}
        self.errors: Dict[Tuple[str, str], Exception] = {}
        
    def add_result(
        self,
        task_id: str,
        learner_id: str,
        result: Optional[ResampleResult],
        error: Optional[Exception] = None
    ) -> None:
        """Add result from one task-learner combination.
        
        Parameters
        ----------
        task_id : str
            The task ID.
        learner_id : str
            The learner ID.
        result : ResampleResult, optional
            The resample result. None if evaluation failed.
        error : Exception, optional
            Exception if evaluation failed.
        """
        key = (task_id, learner_id)
        if result is not None:
            self.results[key] = result
        if error is not None:
            self.errors[key] = error
            
    def get_result(self, task_id: str, learner_id: str) -> Optional[ResampleResult]:
        """Get result for a specific task-learner combination.
        
        Parameters
        ----------
        task_id : str
            The task ID.
        learner_id : str
            The learner ID.
            
        Returns
        -------
        ResampleResult or None
            The result if available, None otherwise.
        """
        return self.results.get((task_id, learner_id))
        
    def get_error(self, task_id: str, learner_id: str) -> Optional[Exception]:
        """Get error for a specific task-learner combination.
        
        Parameters
        ----------
        task_id : str
            The task ID.
        learner_id : str
            The learner ID.
            
        Returns
        -------
        Exception or None
            The error if evaluation failed, None otherwise.
        """
        return self.errors.get((task_id, learner_id))
        
    def aggregate(self, measure_id: str, aggr: str = "mean") -> pd.DataFrame:
        """Aggregate scores across resampling iterations.
        
        Parameters
        ----------
        measure_id : str
            The measure to aggregate.
        aggr : str, default="mean"
            Aggregation method ('mean', 'std', 'min', 'max', 'median').
            
        Returns
        -------
        pd.DataFrame
            DataFrame with tasks as rows and learners as columns.
        """
        # Build matrix of aggregated scores
        task_ids = [t.id for t in self.tasks]
        learner_ids = [l.id for l in self.learners]
        
        data = np.full((len(task_ids), len(learner_ids)), np.nan)
        
        for i, task_id in enumerate(task_ids):
            for j, learner_id in enumerate(learner_ids):
                result = self.get_result(task_id, learner_id)
                if result is not None:
                    scores = result.scores.get(measure_id, [])
                    if scores:
                        valid_scores = [s for s in scores if not np.isnan(s)]
                        if valid_scores:
                            if aggr == "mean":
                                data[i, j] = np.mean(valid_scores)
                            elif aggr == "std":
                                data[i, j] = np.std(valid_scores)
                            elif aggr == "min":
                                data[i, j] = np.min(valid_scores)
                            elif aggr == "max":
                                data[i, j] = np.max(valid_scores)
                            elif aggr == "median":
                                data[i, j] = np.median(valid_scores)
                                
        return pd.DataFrame(data, index=task_ids, columns=learner_ids)
        
    def score_table(self, measure_id: Optional[str] = None) -> pd.DataFrame:
        """Get a table of mean scores.
        
        Parameters
        ----------
        measure_id : str, optional
            The measure to show. If None, uses first measure.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with tasks as rows and learners as columns.
        """
        if measure_id is None:
            measure_id = self.measures[0].id
        return self.aggregate(measure_id, "mean")
        
    def rank_learners(self, measure_id: Optional[str] = None) -> pd.DataFrame:
        """Rank learners by average performance across tasks.
        
        Parameters
        ----------
        measure_id : str, optional
            The measure to rank by. If None, uses first measure.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with learner rankings.
        """
        scores = self.score_table(measure_id)
        
        # Calculate mean score across tasks for each learner
        mean_scores = scores.mean(axis=0)
        
        # Determine if measure should be minimized or maximized
        measure = next((m for m in self.measures if m.id == (measure_id or self.measures[0].id)), None)
        minimize = measure.minimize if measure else True
        
        # Sort learners
        if minimize:
            ranking = mean_scores.sort_values()
        else:
            ranking = mean_scores.sort_values(ascending=False)
            
        # Create ranking DataFrame
        rank_df = pd.DataFrame({
            'learner': ranking.index,
            'mean_score': ranking.values,
            'rank': range(1, len(ranking) + 1)
        })
        
        return rank_df
        
    def to_long_format(self) -> pd.DataFrame:
        """Convert results to long format DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with one row per task-learner-measure-iteration.
        """
        rows = []
        
        for (task_id, learner_id), result in self.results.items():
            for iteration in result.iterations:
                row = {
                    'task_id': task_id,
                    'learner_id': learner_id,
                    'iteration': iteration,
                    'train_time': result.train_times[iteration],
                    'predict_time': result.predict_times[iteration]
                }
                
                # Add scores for each measure
                for measure_id, scores in result.scores.items():
                    if iteration < len(scores):
                        row[f'score_{measure_id}'] = scores[iteration]
                        
                rows.append(row)
                
        return pd.DataFrame(rows)
        
    @property
    def n_experiments(self) -> int:
        """Total number of experiments (task-learner combinations)."""
        return len(self.tasks) * len(self.learners)
        
    @property
    def n_successful(self) -> int:
        """Number of successful experiments."""
        return len(self.results)
        
    @property
    def n_errors(self) -> int:
        """Number of failed experiments."""
        return len(self.errors)
        
    def __repr__(self) -> str:
        """String representation."""
        status_parts = []
        
        if self.n_successful > 0:
            status_parts.append(f"{self.n_successful} successful")
        if self.n_errors > 0:
            status_parts.append(f"{self.n_errors} errors")
            
        status = ", ".join(status_parts) if status_parts else "no results"
        
        return (
            f"<BenchmarkResult: {len(self.tasks)} tasks × {len(self.learners)} learners, "
            f"{status}>"
        )


def benchmark(
    tasks: Union[Task, List[Task]],
    learners: Union[Learner, List[Learner]],
    resampling: Resampling,
    measures: Union[Measure, List[Measure]],
    encapsulate: bool = True,
    store_models: bool = False,
    store_backends: bool = False,
    backend: Optional[Backend] = None,
    callbacks: Optional[Union[Callback, List[Callback]]] = None
) -> BenchmarkResult:
    """Benchmark multiple learners on multiple tasks.
    
    This function evaluates multiple learners on multiple tasks using a
    resampling strategy, providing a comprehensive comparison.
    
    Parameters
    ----------
    tasks : Task or List[Task]
        Task(s) to evaluate on.
    learners : Learner or List[Learner]
        Learner(s) to evaluate.
    resampling : Resampling
        Resampling strategy to use.
    measures : Measure or List[Measure]
        Performance measure(s) to calculate.
    encapsulate : bool, default=True
        Whether to encapsulate learners (clone before training).
    store_models : bool, default=False
        Whether to store trained models.
    store_backends : bool, default=False
        Whether to store data backends.
    backend : Backend, optional
        Parallel backend to use. If None, uses sequential execution.
        Can parallelize across task-learner combinations.
    callbacks : Callback or list of Callback, optional
        Callbacks to use during benchmarking.
        
    Returns
    -------
    BenchmarkResult
        Object containing all benchmark results.
        
    Examples
    --------
    >>> from mlpy import benchmark
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners import learner_sklearn
    >>> from mlpy.resamplings import ResamplingCV
    >>> from mlpy.measures import MeasureClassifAccuracy
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> # Multiple learners
    >>> learners = [
    ...     learner_sklearn(DecisionTreeClassifier()),
    ...     learner_sklearn(RandomForestClassifier())
    ... ]
    >>> 
    >>> # Benchmark
    >>> result = benchmark(
    ...     tasks=task,
    ...     learners=learners,
    ...     resampling=ResamplingCV(folds=5),
    ...     measures=MeasureClassifAccuracy()
    ... )
    >>> 
    >>> # Compare results
    >>> print(result.score_table())
    >>> print(result.rank_learners())
    """
    # Ensure inputs are lists
    if not isinstance(tasks, list):
        tasks = [tasks]
    if not isinstance(learners, list):
        learners = [learners]
    if not isinstance(measures, list):
        measures = [measures]
        
    # Validate inputs
    if not tasks:
        raise ValueError("At least one task must be provided")
    if not learners:
        raise ValueError("At least one learner must be provided")
    if not measures:
        raise ValueError("At least one measure must be provided")
        
    # Create result object
    result = BenchmarkResult(tasks, learners, resampling, measures)
    
    # Setup callbacks
    if callbacks is None:
        callback_set = CallbackSet()
    elif isinstance(callbacks, Callback):
        callback_set = CallbackSet([callbacks])
    else:
        callback_set = CallbackSet(callbacks)
        
    # Notify callbacks of benchmark start
    callback_set.on_benchmark_begin(tasks, learners)
    
    # Log benchmark start
    logger.info(
        f"Starting benchmark: {len(tasks)} tasks × {len(learners)} learners × "
        f"{resampling.__class__.__name__}"
    )
    
    # Create all task-learner combinations
    experiments = list(product(tasks, learners))
    total_experiments = len(experiments)
    
    # Check if we should use parallel execution
    if backend is not None and total_experiments > 1:
        # Parallel execution across task-learner combinations
        logger.info(f"Using parallel backend: {backend.id}")
        
        # Define worker function
        def _benchmark_worker(args: Tuple[Task, Learner]) -> Dict[str, Any]:
            task, learner = args
            
            try:
                # Run resample for this combination
                resample_result = resample(
                    task=task,
                    learner=learner,
                    resampling=resampling,
                    measures=measures,
                    encapsulate=encapsulate,
                    store_models=store_models,
                    store_backends=store_backends
                    # Note: Don't pass backend to resample to avoid nested parallelism
                )
                
                return {
                    'task_id': task.id,
                    'learner_id': learner.id,
                    'result': resample_result,
                    'error': None
                }
                
            except Exception as e:
                return {
                    'task_id': task.id,
                    'learner_id': learner.id,
                    'result': None,
                    'error': e
                }
        
        # Execute in parallel
        results = backend.map(_benchmark_worker, experiments)
        
        # Process results
        for i, res in enumerate(results):
            experiment_num = i + 1
            
            # Get task and learner objects for callbacks
            task = next(t for t in tasks if t.id == res['task_id'])
            learner = next(l for l in learners if l.id == res['learner_id'])
            
            logger.info(
                f"Experiment {experiment_num}/{total_experiments}: "
                f"task={res['task_id']}, learner={res['learner_id']}"
            )
            
            # Notify callbacks of experiment (even though it already ran)
            callback_set.on_experiment_begin(task, learner, experiment_num)
            
            if res['result'] is not None:
                result.add_result(res['task_id'], res['learner_id'], res['result'])
                
                # Log summary
                scores_summary = []
                for measure in measures:
                    mean_score = res['result'].score(measure.id)
                    scores_summary.append(f"{measure.id}={mean_score:.4f}")
                logger.info(f"  Results: {', '.join(scores_summary)}")
                
                # Notify callbacks of success
                callback_set.on_experiment_end(task, learner, res['result'], None)
            else:
                result.add_result(res['task_id'], res['learner_id'], None, error=res['error'])
                logger.error(
                    f"  Failed: {type(res['error']).__name__}: {res['error']}"
                )
                
                # Notify callbacks of error
                callback_set.on_error(res['error'], f"experiment {res['task_id']} × {res['learner_id']}")
                callback_set.on_experiment_end(task, learner, None, res['error'])
                
    else:
        # Sequential execution
        experiment_num = 0
        
        for task in tasks:
            for learner in learners:
                experiment_num += 1
                logger.info(
                    f"Experiment {experiment_num}/{total_experiments}: "
                    f"task={task.id}, learner={learner.id}"
                )
                
                # Notify callbacks of experiment start
                callback_set.on_experiment_begin(task, learner, experiment_num)
                
                try:
                    # Run resample for this combination
                    resample_result = resample(
                        task=task,
                        learner=learner,
                        resampling=resampling,
                        measures=measures,
                        encapsulate=encapsulate,
                        store_models=store_models,
                        store_backends=store_backends,
                        backend=backend  # Pass backend to resample for within-resample parallelism
                    )
                    
                    # Store result
                    result.add_result(task.id, learner.id, resample_result)
                    
                    # Log summary
                    scores_summary = []
                    for measure in measures:
                        mean_score = resample_result.score(measure.id)
                        scores_summary.append(f"{measure.id}={mean_score:.4f}")
                    logger.info(f"  Results: {', '.join(scores_summary)}")
                    
                    # Notify callbacks of experiment end
                    callback_set.on_experiment_end(task, learner, resample_result, None)
                    
                except Exception as e:
                    # Store error
                    result.add_result(task.id, learner.id, None, error=e)
                    logger.error(
                        f"  Failed: {type(e).__name__}: {e}"
                    )
                    
                    # Notify callbacks of error
                    callback_set.on_error(e, f"experiment {task.id} × {learner.id}")
                    callback_set.on_experiment_end(task, learner, None, e)
                
    # Log benchmark summary
    logger.info(
        f"Benchmark complete: {result.n_successful} successful, "
        f"{result.n_errors} errors"
    )
    
    # Notify callbacks of benchmark end
    callback_set.on_benchmark_end(result)
    
    return result


__all__ = ["benchmark", "BenchmarkResult"]