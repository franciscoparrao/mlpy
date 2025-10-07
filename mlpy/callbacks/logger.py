"""Logger callback for MLPY.

This callback logs events to the Python logging system.
"""

import logging
from typing import Dict, Any, List, Optional

from .base import Callback
from ..tasks import Task
from ..learners import Learner
from ..predictions import Prediction


class CallbackLogger(Callback):
    """Callback that logs events using Python's logging system.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, creates a new logger.
    level : int, default=logging.INFO
        Logging level for regular events.
    error_level : int, default=logging.ERROR
        Logging level for errors.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        error_level: int = logging.ERROR
    ):
        super().__init__(id="logger")
        self.logger = logger or logging.getLogger("mlpy.callbacks")
        self.level = level
        self.error_level = error_level
        
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Log resample start."""
        self.logger.log(
            self.level,
            f"Starting resample: learner={learner.id}, task={task.id}, "
            f"resampling={resampling.id}, iterations={resampling.iters}"
        )
        
    def on_resample_end(self, result: Any) -> None:
        """Log resample end."""
        self.logger.log(
            self.level,
            f"Resample complete: {result.n_iters} iterations, "
            f"{result.n_errors} errors"
        )
        
    def on_iteration_begin(self, iteration: int, train_set: List[int], test_set: List[int]) -> None:
        """Log iteration start."""
        self.logger.debug(
            f"Iteration {iteration}: train_size={len(train_set)}, "
            f"test_size={len(test_set)}"
        )
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Log iteration end."""
        scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
        self.logger.log(
            self.level,
            f"Iteration {iteration} complete: {scores_str} "
            f"(train={train_time:.2f}s, predict={predict_time:.2f}s)"
        )
        
    # Training callbacks
    def on_train_begin(self, task: Task, learner: Learner) -> None:
        """Log training start."""
        self.logger.debug(
            f"Training {learner.id} on {task.id} "
            f"({task.nrow} rows, {task.ncol} features)"
        )
        
    def on_train_end(self, learner: Learner) -> None:
        """Log training end."""
        self.logger.debug(f"Training complete for {learner.id}")
        
    # Prediction callbacks
    def on_predict_begin(self, task: Task, learner: Learner) -> None:
        """Log prediction start."""
        self.logger.debug(
            f"Predicting with {learner.id} on {task.id} ({task.nrow} rows)"
        )
        
    def on_predict_end(self, prediction: Prediction) -> None:
        """Log prediction end."""
        self.logger.debug(
            f"Prediction complete: {len(prediction.response)} predictions"
        )
        
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Log benchmark start."""
        self.logger.log(
            self.level,
            f"Starting benchmark: {len(tasks)} tasks × {len(learners)} learners"
        )
        
    def on_benchmark_end(self, result: Any) -> None:
        """Log benchmark end."""
        self.logger.log(
            self.level,
            f"Benchmark complete: {result.n_successful} successful, "
            f"{result.n_errors} errors"
        )
        
    def on_experiment_begin(self, task: Task, learner: Learner, experiment_num: int) -> None:
        """Log experiment start."""
        self.logger.log(
            self.level,
            f"Experiment {experiment_num}: task={task.id}, learner={learner.id}"
        )
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Log experiment end."""
        if error is not None:
            self.logger.log(
                self.error_level,
                f"Experiment failed: {type(error).__name__}: {error}"
            )
        else:
            # Log primary score if available
            if result and hasattr(result, 'measures') and result.measures:
                measure = result.measures[0]
                score = result.score(measure.id)
                self.logger.log(
                    self.level,
                    f"Experiment complete: {measure.id}={score:.4f}"
                )
            else:
                self.logger.log(self.level, "Experiment complete")
                
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Log tuning start."""
        self.logger.log(
            self.level,
            f"Starting tuning: learner={learner.id}, "
            f"configs={n_configs}, params={len(param_set.params)}"
        )
        
    def on_tune_end(self, result: Any) -> None:
        """Log tuning end."""
        self.logger.log(
            self.level,
            f"Tuning complete: best_score={result.best_score:.4f}, "
            f"runtime={result.runtime:.1f}s"
        )
        self.logger.log(
            self.level,
            f"Best config: {result.best_config}"
        )
        
    def on_config_begin(self, config_num: int, config: Dict[str, Any]) -> None:
        """Log config evaluation start."""
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        self.logger.debug(f"Config {config_num}: {config_str}")
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Log config evaluation end."""
        self.logger.log(
            self.level,
            f"Config {config_num} score: {score:.4f}"
        )
        
    # Error handling
    def on_error(self, error: Exception, context: str) -> None:
        """Log error."""
        self.logger.log(
            self.error_level,
            f"Error in {context}: {type(error).__name__}: {error}"
        )


__all__ = ["CallbackLogger"]