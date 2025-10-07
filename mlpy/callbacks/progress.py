"""Progress callback for displaying experiment progress.

This callback displays progress bars and status information.
"""

from typing import Dict, Any, List, Optional
import sys
import time

from .base import Callback
from ..tasks import Task
from ..learners import Learner


class CallbackProgress(Callback):
    """Callback that displays progress information.
    
    Parameters
    ----------
    show_time : bool, default=True
        Whether to show timing information.
    width : int, default=50
        Width of progress bars.
    """
    
    def __init__(self, show_time: bool = True, width: int = 50):
        super().__init__(id="progress")
        self.show_time = show_time
        self.width = width
        self._start_times = {}
        
    def _print_progress(self, current: int, total: int, prefix: str = "", suffix: str = "") -> None:
        """Print a progress bar."""
        percent = current / total if total > 0 else 1.0
        filled = int(self.width * percent)
        bar = "█" * filled + "-" * (self.width - filled)
        
        # Calculate time remaining
        time_info = ""
        if self.show_time and prefix in self._start_times:
            elapsed = time.time() - self._start_times[prefix]
            if current > 0:
                rate = elapsed / current
                remaining = rate * (total - current)
                time_info = f" [{self._format_time(elapsed)}<{self._format_time(remaining)}]"
                
        print(f"\r{prefix} |{bar}| {current}/{total} {suffix}{time_info}", end="")
        
        if current == total:
            print()  # New line when complete
            
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
            
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Show resample start."""
        print(f"\nResampling {learner.id} on {task.id} ({resampling.iters} iterations)")
        self._start_times['resample'] = time.time()
        self._total_iterations = resampling.iters
        self._current_iteration = 0
        
    def on_resample_end(self, result: Any) -> None:
        """Show resample end."""
        if self.show_time and 'resample' in self._start_times:
            elapsed = time.time() - self._start_times['resample']
            print(f"Resample complete in {self._format_time(elapsed)}")
            
    def on_iteration_begin(self, iteration: int, train_set: List[int], test_set: List[int]) -> None:
        """Update progress for iteration."""
        self._current_iteration = iteration + 1
        self._print_progress(
            self._current_iteration - 1,
            self._total_iterations,
            "Iterations",
            f"[{len(train_set)}/{len(test_set)} split]"
        )
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Update progress after iteration."""
        # Show primary score
        score_str = ""
        if scores:
            first_measure = list(scores.keys())[0]
            score_str = f" {first_measure}={scores[first_measure]:.4f}"
            
        self._print_progress(
            self._current_iteration,
            self._total_iterations,
            "Iterations",
            score_str
        )
        
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Show benchmark start."""
        n_experiments = len(tasks) * len(learners)
        print(f"\nBenchmarking {len(learners)} learners on {len(tasks)} tasks ({n_experiments} experiments)")
        self._start_times['benchmark'] = time.time()
        self._total_experiments = n_experiments
        self._current_experiment = 0
        
    def on_benchmark_end(self, result: Any) -> None:
        """Show benchmark end."""
        if self.show_time and 'benchmark' in self._start_times:
            elapsed = time.time() - self._start_times['benchmark']
            print(f"\nBenchmark complete in {self._format_time(elapsed)}")
            print(f"Successful: {result.n_successful}, Errors: {result.n_errors}")
            
    def on_experiment_begin(self, task: Task, learner: Learner, experiment_num: int) -> None:
        """Update progress for experiment."""
        self._current_experiment = experiment_num
        self._print_progress(
            experiment_num - 1,
            self._total_experiments,
            "Experiments",
            f" [{task.id} × {learner.id}]"
        )
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Update progress after experiment."""
        status = "✓" if error is None else "✗"
        self._print_progress(
            self._current_experiment,
            self._total_experiments,
            "Experiments",
            f" {status}"
        )
        
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Show tuning start."""
        print(f"\nTuning {learner.id} with {n_configs} configurations")
        self._start_times['tune'] = time.time()
        self._total_configs = n_configs
        self._current_config = 0
        self._start_times['configs'] = time.time()
        
    def on_tune_end(self, result: Any) -> None:
        """Show tuning end."""
        if self.show_time and 'tune' in self._start_times:
            elapsed = time.time() - self._start_times['tune']
            print(f"\nTuning complete in {self._format_time(elapsed)}")
            print(f"Best score: {result.best_score:.4f}")
            
    def on_config_begin(self, config_num: int, config: Dict[str, Any]) -> None:
        """Update progress for config."""
        self._current_config = config_num
        self._print_progress(
            config_num - 1,
            self._total_configs,
            "Configs"
        )
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Update progress after config."""
        self._print_progress(
            self._current_config,
            self._total_configs,
            "Configs",
            f" score={score:.4f}"
        )


__all__ = ["CallbackProgress"]