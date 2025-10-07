"""History callback for recording experiment details.

This callback records detailed information about the experiment
execution for later analysis.
"""

from typing import Dict, Any, List, Optional
import time
from dataclasses import dataclass, field

from .base import Callback
from ..tasks import Task
from ..learners import Learner
from ..predictions import Prediction


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    iteration: int
    train_set_size: int
    test_set_size: int
    scores: Dict[str, float]
    train_time: float
    predict_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRecord:
    """Record of a benchmark experiment."""
    task_id: str
    learner_id: str
    experiment_num: int
    result: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


@dataclass
class ConfigRecord:
    """Record of a tuning configuration."""
    config_num: int
    config: Dict[str, Any]
    score: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class CallbackHistory(Callback):
    """Callback that records detailed history of experiments.
    
    This callback keeps track of all events during experiment
    execution for later analysis and debugging.
    
    Parameters
    ----------
    record_predictions : bool, default=False
        Whether to store prediction objects.
    """
    
    def __init__(self, record_predictions: bool = False):
        super().__init__(id="history")
        self.record_predictions = record_predictions
        self.clear()
        
    def clear(self) -> None:
        """Clear all recorded history."""
        # Resample history
        self.resample_task = None
        self.resample_learner = None
        self.resample_result = None
        self.iterations: List[IterationRecord] = []
        
        # Training history
        self.train_tasks: List[Task] = []
        self.train_learners: List[Learner] = []
        self.train_times: List[float] = []
        
        # Prediction history
        self.predict_tasks: List[Task] = []
        self.predictions: List[Prediction] = []
        self.predict_times: List[float] = []
        
        # Benchmark history
        self.benchmark_tasks: List[Task] = []
        self.benchmark_learners: List[Learner] = []
        self.benchmark_result = None
        self.experiments: List[ExperimentRecord] = []
        
        # Tuning history
        self.tune_learner = None
        self.tune_param_set = None
        self.tune_result = None
        self.configs: List[ConfigRecord] = []
        
        # Error history
        self.errors: List[Dict[str, Any]] = []
        
        # Timing
        self._start_times: Dict[str, float] = {}
        
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Record resample start."""
        self.resample_task = task
        self.resample_learner = learner
        self._start_times['resample'] = time.time()
        
    def on_resample_end(self, result: Any) -> None:
        """Record resample end."""
        self.resample_result = result
        
    def on_iteration_begin(self, iteration: int, train_set: List[int], test_set: List[int]) -> None:
        """Record iteration start."""
        self._current_iteration = IterationRecord(
            iteration=iteration,
            train_set_size=len(train_set),
            test_set_size=len(test_set),
            scores={},
            train_time=0.0,
            predict_time=0.0
        )
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Record iteration end."""
        if hasattr(self, '_current_iteration'):
            self._current_iteration.scores = scores
            self._current_iteration.train_time = train_time
            self._current_iteration.predict_time = predict_time
            self.iterations.append(self._current_iteration)
            
    # Training callbacks
    def on_train_begin(self, task: Task, learner: Learner) -> None:
        """Record training start."""
        self._start_times['train'] = time.time()
        
    def on_train_end(self, learner: Learner) -> None:
        """Record training end."""
        if 'train' in self._start_times:
            train_time = time.time() - self._start_times['train']
            self.train_times.append(train_time)
            self.train_learners.append(learner)
            
    # Prediction callbacks
    def on_predict_begin(self, task: Task, learner: Learner) -> None:
        """Record prediction start."""
        self._start_times['predict'] = time.time()
        self.predict_tasks.append(task)
        
    def on_predict_end(self, prediction: Prediction) -> None:
        """Record prediction end."""
        if 'predict' in self._start_times:
            predict_time = time.time() - self._start_times['predict']
            self.predict_times.append(predict_time)
            
        if self.record_predictions:
            self.predictions.append(prediction)
            
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Record benchmark start."""
        self.benchmark_tasks = tasks
        self.benchmark_learners = learners
        self._start_times['benchmark'] = time.time()
        
    def on_benchmark_end(self, result: Any) -> None:
        """Record benchmark end."""
        self.benchmark_result = result
        
    def on_experiment_begin(self, task: Task, learner: Learner, experiment_num: int) -> None:
        """Record experiment start."""
        self._current_experiment = ExperimentRecord(
            task_id=task.id,
            learner_id=learner.id,
            experiment_num=experiment_num
        )
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Record experiment end."""
        if hasattr(self, '_current_experiment'):
            self._current_experiment.result = result
            self._current_experiment.error = error
            self._current_experiment.end_time = time.time()
            self.experiments.append(self._current_experiment)
            
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Record tuning start."""
        self.tune_learner = learner
        self.tune_param_set = param_set
        self._start_times['tune'] = time.time()
        
    def on_tune_end(self, result: Any) -> None:
        """Record tuning end."""
        self.tune_result = result
        
    def on_config_begin(self, config_num: int, config: Dict[str, Any]) -> None:
        """Record config evaluation start."""
        self._current_config = ConfigRecord(
            config_num=config_num,
            config=config.copy()
        )
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Record config evaluation end."""
        if hasattr(self, '_current_config'):
            self._current_config.score = score
            self._current_config.end_time = time.time()
            self.configs.append(self._current_config)
            
    # Error handling
    def on_error(self, error: Exception, context: str) -> None:
        """Record error."""
        self.errors.append({
            'error': error,
            'context': context,
            'timestamp': time.time()
        })
        
    # Analysis methods
    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of iteration history."""
        if not self.iterations:
            return {}
            
        scores_by_measure = {}
        for iteration in self.iterations:
            for measure, score in iteration.scores.items():
                if measure not in scores_by_measure:
                    scores_by_measure[measure] = []
                scores_by_measure[measure].append(score)
                
        return {
            'n_iterations': len(self.iterations),
            'total_train_time': sum(it.train_time for it in self.iterations),
            'total_predict_time': sum(it.predict_time for it in self.iterations),
            'avg_train_time': sum(it.train_time for it in self.iterations) / len(self.iterations),
            'avg_predict_time': sum(it.predict_time for it in self.iterations) / len(self.iterations),
            'scores_summary': {
                measure: {
                    'mean': sum(scores) / len(scores),
                    'std': self._std(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
                for measure, scores in scores_by_measure.items()
            }
        }
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark experiments."""
        if not self.experiments:
            return {}
            
        successful = [exp for exp in self.experiments if exp.error is None]
        failed = [exp for exp in self.experiments if exp.error is not None]
        
        return {
            'n_experiments': len(self.experiments),
            'n_successful': len(successful),
            'n_failed': len(failed),
            'failed_experiments': [
                {'task': exp.task_id, 'learner': exp.learner_id, 'error': str(exp.error)}
                for exp in failed
            ]
        }
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of tuning configurations."""
        if not self.configs:
            return {}
            
        valid_scores = [c.score for c in self.configs if c.score is not None]
        
        if not valid_scores:
            return {'n_configs': len(self.configs)}
            
        best_idx = valid_scores.index(max(valid_scores))
        
        return {
            'n_configs': len(self.configs),
            'best_config': self.configs[best_idx].config,
            'best_score': valid_scores[best_idx],
            'score_range': (min(valid_scores), max(valid_scores))
        }
        
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
        
    def __repr__(self) -> str:
        parts = [f"CallbackHistory("]
        if self.iterations:
            parts.append(f"iterations={len(self.iterations)}")
        if self.experiments:
            parts.append(f"experiments={len(self.experiments)}")
        if self.configs:
            parts.append(f"configs={len(self.configs)}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        parts.append(")")
        return "".join(parts) if len(parts) == 2 else ", ".join(parts[1:])


__all__ = ["CallbackHistory", "IterationRecord", "ExperimentRecord", "ConfigRecord"]