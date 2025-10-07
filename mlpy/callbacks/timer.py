"""Timer callback for tracking execution times.

This callback tracks and reports timing information for
various stages of experiment execution.
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .base import Callback
from ..tasks import Task
from ..learners import Learner


class CallbackTimer(Callback):
    """Callback that tracks execution times.
    
    Parameters
    ----------
    verbose : bool, default=True
        Whether to print timing information.
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(id="timer")
        self.verbose = verbose
        self.clear()
        
    def clear(self) -> None:
        """Clear all timing data."""
        self._start_times: Dict[str, float] = {}
        self.timings: Dict[str, List[float]] = defaultdict(list)
        
    def _start(self, key: str) -> None:
        """Start timing for a key."""
        self._start_times[key] = time.time()
        
    def _stop(self, key: str) -> float:
        """Stop timing for a key and record elapsed time."""
        if key in self._start_times:
            elapsed = time.time() - self._start_times[key]
            self.timings[key].append(elapsed)
            del self._start_times[key]
            return elapsed
        return 0.0
        
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Start timing resample."""
        self._start('resample')
        
    def on_resample_end(self, result: Any) -> None:
        """Stop timing resample."""
        elapsed = self._stop('resample')
        if self.verbose:
            print(f"Resample completed in {elapsed:.2f}s")
            
    def on_iteration_begin(self, iteration: int, train_set: List[int], test_set: List[int]) -> None:
        """Start timing iteration."""
        self._start(f'iteration_{iteration}')
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Stop timing iteration."""
        self._stop(f'iteration_{iteration}')
        self.timings['train'].append(train_time)
        self.timings['predict'].append(predict_time)
        
    # Training callbacks
    def on_train_begin(self, task: Task, learner: Learner) -> None:
        """Start timing training."""
        self._start('train_single')
        
    def on_train_end(self, learner: Learner) -> None:
        """Stop timing training."""
        self._stop('train_single')
        
    # Prediction callbacks
    def on_predict_begin(self, task: Task, learner: Learner) -> None:
        """Start timing prediction."""
        self._start('predict_single')
        
    def on_predict_end(self, prediction: Any) -> None:
        """Stop timing prediction."""
        self._stop('predict_single')
        
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Start timing benchmark."""
        self._start('benchmark')
        
    def on_benchmark_end(self, result: Any) -> None:
        """Stop timing benchmark."""
        elapsed = self._stop('benchmark')
        if self.verbose:
            print(f"\nBenchmark completed in {elapsed:.2f}s")
            self._print_summary()
            
    def on_experiment_begin(self, task: Task, learner: Learner, experiment_num: int) -> None:
        """Start timing experiment."""
        self._start(f'experiment_{experiment_num}')
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Stop timing experiment."""
        self._stop(f'experiment_{experiment_num}')
        
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Start timing tuning."""
        self._start('tune')
        
    def on_tune_end(self, result: Any) -> None:
        """Stop timing tuning."""
        elapsed = self._stop('tune')
        if self.verbose:
            print(f"\nTuning completed in {elapsed:.2f}s")
            avg_per_config = elapsed / len(result.configs) if result.configs else 0
            print(f"Average time per config: {avg_per_config:.2f}s")
            
    def on_config_begin(self, config_num: int, config: Dict[str, Any]) -> None:
        """Start timing config."""
        self._start(f'config_{config_num}')
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Stop timing config."""
        self._stop(f'config_{config_num}')
        
    # Analysis methods
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        
        for key, times in self.timings.items():
            if times:
                summary[key] = {
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
                
        return summary
        
    def get_experiment_times(self) -> Dict[str, Dict[str, Any]]:
        """Get experiment timing data organized by experiment type.
        
        Returns
        -------
        dict
            Timing data organized by experiment type (resample, benchmark, tune).
        """
        result = {}
        
        # Resample timings
        if 'resample' in self.timings:
            result['resample'] = {
                'total': self.timings['resample'][0] if self.timings['resample'] else 0.0,
                'iterations': []
            }
            for i in range(100):  # Check up to 100 iterations
                key = f'iteration_{i}'
                if key in self.timings and self.timings[key]:
                    result['resample']['iterations'].append(self.timings[key][0])
                    
        # Benchmark timings
        if 'benchmark' in self.timings:
            result['benchmark'] = {
                'total': self.timings['benchmark'][0] if self.timings['benchmark'] else 0.0,
                'experiments': []
            }
            for i in range(100):  # Check up to 100 experiments
                key = f'experiment_{i}'
                if key in self.timings and self.timings[key]:
                    result['benchmark']['experiments'].append(self.timings[key][0])
                    
        # Tuning timings
        if 'tune' in self.timings:
            result['tuning'] = {
                'total': self.timings['tune'][0] if self.timings['tune'] else 0.0,
                'configs': []
            }
            for i in range(1000):  # Check up to 1000 configs
                key = f'config_{i}'
                if key in self.timings and self.timings[key]:
                    result['tuning']['configs'].append(self.timings[key][0])
                    
        return result
        
    def _print_summary(self) -> None:
        """Print timing summary."""
        summary = self.get_summary()
        
        if not summary:
            return
            
        print("\nTiming Summary:")
        print("-" * 50)
        
        for key, stats in sorted(summary.items()):
            print(f"{key:20} Total: {stats['total']:8.2f}s  "
                  f"Mean: {stats['mean']:6.2f}s  "
                  f"Count: {stats['count']:4d}")
                  
    def __repr__(self) -> str:
        n_timings = sum(len(times) for times in self.timings.values())
        return f"CallbackTimer({n_timings} timings recorded)"


__all__ = ["CallbackTimer"]