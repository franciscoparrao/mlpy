"""Resample function for model evaluation.

This module provides the main resample() function for evaluating
learners using resampling strategies.
"""

import time
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .tasks import Task
from .learners import Learner
from .resamplings import Resampling
from .measures import Measure
from .predictions import Prediction
from .utils.logging import get_logger
from .parallel import Backend, get_backend
from .callbacks import CallbackSet, Callback

logger = get_logger(__name__)


class ResampleResult:
    """Result of a resampling evaluation.
    
    Stores predictions and performance scores from each iteration
    of a resampling strategy.
    
    Parameters
    ----------
    task : Task
        The task that was resampled.
    learner : Learner
        The learner that was evaluated.
    resampling : Resampling
        The resampling strategy used.
    measures : List[Measure]
        The measures used for evaluation.
    """
    
    def __init__(
        self,
        task: Task,
        learner: Learner,
        resampling: Resampling,
        measures: List[Measure]
    ):
        self.task = task
        self.learner = learner
        self.resampling = resampling
        self.measures = measures
        
        # Storage for results
        self.iterations: List[int] = []
        self.predictions: List[Prediction] = []
        self.scores: Dict[str, List[float]] = {m.id: [] for m in measures}
        self.train_times: List[float] = []
        self.predict_times: List[float] = []
        self.errors: List[Optional[Exception]] = []
        
    def add_iteration(
        self,
        iteration: int,
        prediction: Optional[Prediction],
        scores: Optional[Dict[str, float]],
        train_time: float,
        predict_time: float,
        error: Optional[Exception] = None
    ) -> None:
        """Add results from one resampling iteration.
        
        Parameters
        ----------
        iteration : int
            The iteration number.
        prediction : Prediction, optional
            The prediction object. None if iteration failed.
        scores : dict, optional
            Scores for each measure. None if iteration failed.
        train_time : float
            Time taken to train the model.
        predict_time : float
            Time taken to make predictions.
        error : Exception, optional
            Exception if iteration failed.
        """
        self.iterations.append(iteration)
        self.predictions.append(prediction)
        self.train_times.append(train_time)
        self.predict_times.append(predict_time)
        self.errors.append(error)
        
        # Add scores
        if scores is not None:
            for measure_id, score in scores.items():
                self.scores[measure_id].append(score)
        else:
            # Add NaN for failed iterations
            for measure_id in self.scores:
                self.scores[measure_id].append(float('nan'))
                
    @property
    def n_iters(self) -> int:
        """Number of completed iterations."""
        return len(self.iterations)
        
    @property
    def n_errors(self) -> int:
        """Number of failed iterations."""
        return sum(1 for e in self.errors if e is not None)
        
    def aggregate(self, measure_id: Optional[str] = None) -> pd.DataFrame:
        """Aggregate scores across iterations.
        
        Parameters
        ----------
        measure_id : str, optional
            Specific measure to aggregate. If None, aggregate all.
            
        Returns
        -------
        pd.DataFrame
            Aggregated scores with columns for each aggregation method.
        """
        if measure_id is not None:
            measures = [m for m in self.measures if m.id == measure_id]
            if not measures:
                raise ValueError(f"Measure '{measure_id}' not found in results")
        else:
            measures = self.measures
            
        results = []
        for measure in measures:
            scores = self.scores[measure.id]
            
            # Default aggregation methods
            agg_values = {}
            
            # Calculate common aggregations
            valid_scores = [s for s in scores if not np.isnan(s)]
            
            if valid_scores:
                agg_values['mean'] = np.mean(valid_scores)
                agg_values['std'] = np.std(valid_scores)
                agg_values['min'] = np.min(valid_scores)
                agg_values['max'] = np.max(valid_scores)
                agg_values['median'] = np.median(valid_scores)
            else:
                # All scores are NaN
                agg_values['mean'] = float('nan')
                agg_values['std'] = float('nan')
                agg_values['min'] = float('nan')
                agg_values['max'] = float('nan')
                agg_values['median'] = float('nan')
                    
            results.append({
                'measure': measure.id,
                **agg_values
            })
            
        return pd.DataFrame(results)
        
    def score(self, measure_id: Optional[str] = None, average: str = "mean") -> float:
        """Get aggregated score.
        
        Parameters
        ----------
        measure_id : str, optional
            Measure to get score for. If None, use first measure.
        average : str, default="mean"
            Aggregation method to use ('mean', 'std', 'min', 'max', 'median').
            
        Returns
        -------
        float
            The aggregated score.
        """
        if measure_id is None:
            measure_id = self.measures[0].id
            
        if measure_id not in self.scores:
            raise ValueError(f"Measure '{measure_id}' not found")
            
        scores = self.scores[measure_id]
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if not valid_scores:
            return float('nan')
            
        # Calculate requested aggregation
        if average == 'mean':
            return np.mean(valid_scores)
        elif average == 'std':
            return np.std(valid_scores)
        elif average == 'min':
            return np.min(valid_scores)
        elif average == 'max':
            return np.max(valid_scores)
        elif average == 'median':
            return np.median(valid_scores)
        else:
            raise ValueError(
                f"Aggregation '{average}' not available. "
                f"Available: mean, std, min, max, median"
            )
        
    def __repr__(self) -> str:
        """String representation."""
        status = f"{self.n_iters} iterations"
        if self.n_errors > 0:
            status += f" ({self.n_errors} errors)"
            
        # Get primary score
        if self.measures and self.scores[self.measures[0].id]:
            primary_measure = self.measures[0]
            primary_score = self.score(primary_measure.id, "mean")
            status += f"\n{primary_measure.id}: {primary_score:.4f}"
            
        return f"<ResampleResult> {status}"


def resample(
    task: Task,
    learner: Learner,
    resampling: Resampling,
    measures: Union[Measure, List[Measure]],
    store_models: bool = False,
    store_backends: bool = False,
    encapsulate: bool = True,
    backend: Optional[Backend] = None,
    callbacks: Optional[Union[Callback, List[Callback]]] = None
) -> ResampleResult:
    """Evaluate a learner using resampling.
    
    Trains and evaluates the learner on each train/test split
    generated by the resampling strategy.
    
    Parameters
    ----------
    task : Task
        The task to evaluate on.
    learner : Learner
        The learner to evaluate.
    resampling : Resampling
        The resampling strategy to use.
    measures : Measure or list of Measure
        Performance measures to compute.
    store_models : bool, default=False
        Whether to store trained models in the result.
    store_backends : bool, default=False
        Whether to store task backends in the result.
    encapsulate : bool, default=True
        Whether to encapsulate learners to prevent modification.
    backend : Backend, optional
        Parallel backend to use. If None, uses sequential execution.
    callbacks : Callback or list of Callback, optional
        Callbacks to use during resampling.
        
    Returns
    -------
    ResampleResult
        Object containing predictions and performance scores.
        
    Examples
    --------
    >>> from mlpy import resample
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners import LearnerClassifSklearn
    >>> from mlpy.resamplings import ResamplingCV
    >>> from mlpy.measures import MeasureClassifAccuracy
    >>> 
    >>> # Create task
    >>> task = TaskClassif(data=df, target='species')
    >>> 
    >>> # Create learner
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> learner = LearnerClassifSklearn(
    ...     model=DecisionTreeClassifier(),
    ...     id='decision_tree'
    ... )
    >>> 
    >>> # Evaluate with 5-fold CV
    >>> result = resample(
    ...     task=task,
    ...     learner=learner,
    ...     resampling=ResamplingCV(folds=5),
    ...     measures=MeasureClassifAccuracy()
    ... )
    >>> 
    >>> # Get mean accuracy
    >>> print(result.score())
    """
    # Ensure measures is a list
    if isinstance(measures, Measure):
        measures = [measures]
        
    # Validate inputs
    if not measures:
        raise ValueError("At least one measure must be provided")
        
    # Check measure compatibility with task
    for measure in measures:
        if not measure.is_applicable(task):
            raise ValueError(
                f"Measure '{measure.id}' is not applicable to task type '{task.task_type}'"
            )
            
    # Encapsulate learner if requested
    if encapsulate:
        learner = learner.clone()
        
    # Instantiate resampling if needed
    if not resampling.is_instantiated:
        resampling = resampling.clone()
        resampling.instantiate(task)
        
    # Create result object
    result = ResampleResult(
        task=task,
        learner=learner,
        resampling=resampling,
        measures=measures
    )
    
    # Setup callbacks
    if callbacks is None:
        callback_set = CallbackSet()
    elif isinstance(callbacks, Callback):
        callback_set = CallbackSet([callbacks])
    else:
        callback_set = CallbackSet(callbacks)
        
    # Notify callbacks of resample start
    callback_set.on_resample_begin(task, learner, resampling)
    
    # Log start
    logger.info(
        f"Starting resampling: {learner.id} on {task.id} "
        f"using {resampling.id} ({resampling.iters} iterations)"
    )
    
    # Check if we should use parallel execution
    if backend is not None and resampling.iters > 1:
        # Parallel execution
        logger.info(f"Using parallel backend: {backend.id}")
        
        # Prepare iteration data
        iteration_data = list(enumerate(resampling))
        
        # Define worker function
        def _resample_worker(args: Tuple[int, Tuple[List[int], List[int]]]) -> Dict[str, Any]:
            i, (train_set, test_set) = args
            
            # Clone learner for this iteration
            iter_learner = learner.clone()
            
            try:
                # Time training
                start_time = time.time()
                iter_learner.train(task, row_ids=train_set)
                train_time = time.time() - start_time
                
                # Time prediction
                start_time = time.time()
                prediction = iter_learner.predict(task, row_ids=test_set)
                predict_time = time.time() - start_time
                
                # Score prediction
                scores = {}
                for measure in measures:
                    score = measure.score(prediction, task)
                    scores[measure.id] = score
                    
                return {
                    'iteration': i,
                    'prediction': prediction,
                    'scores': scores,
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'error': None
                }
                
            except Exception as e:
                return {
                    'iteration': i,
                    'prediction': None,
                    'scores': None,
                    'train_time': 0.0,
                    'predict_time': 0.0,
                    'error': e
                }
        
        # Execute in parallel
        results = backend.map(_resample_worker, iteration_data)
        
        # Add results to ResampleResult
        for res in results:
            result.add_iteration(**res)
            
    else:
        # Sequential execution
        for i, (train_set, test_set) in enumerate(resampling):
            logger.debug(f"Iteration {i+1}/{resampling.iters}")
            
            # Notify callbacks of iteration start
            callback_set.on_iteration_begin(i, train_set, test_set)
            
            try:
                # Notify callbacks of training start
                callback_set.on_train_begin(task, learner)
                
                # Time training
                start_time = time.time()
                learner.train(task, row_ids=train_set)
                train_time = time.time() - start_time
                
                # Notify callbacks of training end
                callback_set.on_train_end(learner)
                
                # Notify callbacks of prediction start
                callback_set.on_predict_begin(task, learner)
                
                # Time prediction
                start_time = time.time()
                prediction = learner.predict(task, row_ids=test_set)
                predict_time = time.time() - start_time
                
                # Notify callbacks of prediction end
                callback_set.on_predict_end(prediction)
                
                # Score prediction
                scores = {}
                for measure in measures:
                    score = measure.score(prediction, task)
                    scores[measure.id] = score
                    
                # Notify callbacks of iteration end
                callback_set.on_iteration_end(i, scores, train_time, predict_time)
                    
                # Add to results
                result.add_iteration(
                    iteration=i,
                    prediction=prediction,
                    scores=scores,
                    train_time=train_time,
                    predict_time=predict_time
                )
                
            except Exception as e:
                logger.error(f"Iteration {i} failed: {e}")
                
                # Notify callbacks of error
                callback_set.on_error(e, f"iteration {i}")
                
                # Still notify iteration end with empty scores
                callback_set.on_iteration_end(i, {}, 0.0, 0.0)
                
                result.add_iteration(
                    iteration=i,
                    prediction=None,
                    scores=None,
                    train_time=0.0,
                    predict_time=0.0,
                    error=e
                )
            
    # Log completion
    logger.info(
        f"Resampling complete: {result.n_iters} iterations, "
        f"{result.n_errors} errors"
    )
    
    # Notify callbacks of resample end
    callback_set.on_resample_end(result)
    
    # Optionally store models and backends
    if store_models:
        result.models = [learner.clone() for _ in range(result.n_iters)]
        # Note: Actual model storage would be implemented when we have learners
        
    if store_backends:
        result.train_backends = []
        result.test_backends = []
        # Note: Backend subset storage would be implemented here
        
    return result


__all__ = ["resample", "ResampleResult"]