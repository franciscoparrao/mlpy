"""Checkpoint callback for saving experiment state.

This callback saves intermediate results and models during
long-running experiments.
"""

import os
import pickle
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import Callback
from ..tasks import Task
from ..learners import Learner


class CallbackCheckpoint(Callback):
    """Callback for saving checkpoints during experiments.
    
    Parameters
    ----------
    checkpoint_dir : str, default="checkpoints"
        Directory to save checkpoints.
    save_freq : int, default=10
        Save checkpoint every N iterations/configs.
    save_best : bool, default=True
        Whether to save the best model separately.
    measure : str, optional
        Measure to monitor for best model. If None, uses first measure.
    minimize : bool, optional
        Whether to minimize the measure. If None, inferred from measure.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_freq: int = 10,
        save_best: bool = True,
        measure: Optional[str] = None,
        minimize: Optional[bool] = None
    ):
        super().__init__(id="checkpoint")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.measure = measure
        self.minimize = minimize
        
        # State
        self.best_score = None
        self._current_experiment = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_checkpoint_path(self, name: str) -> Path:
        """Get path for a checkpoint file."""
        return self.checkpoint_dir / f"{name}.pkl"
        
    def _save_checkpoint(self, data: Dict[str, Any], name: str) -> None:
        """Save checkpoint data."""
        path = self._get_checkpoint_path(name)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to save checkpoint {name}: {e}")
            
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.minimize:
            return current < best
        else:
            return current > best
            
    # Resample callbacks
    def on_resample_begin(self, task: Task, learner: Learner, resampling: Any) -> None:
        """Initialize checkpoint for resample."""
        self._current_experiment = f"resample_{task.id}_{learner.id}"
        self.best_score = None
        
        # Save initial state
        self._save_checkpoint({
            'type': 'resample',
            'task_id': task.id,
            'learner_id': learner.id,
            'resampling_id': resampling.id,
            'status': 'started'
        }, f"{self._current_experiment}_init")
        
    def on_iteration_end(
        self,
        iteration: int,
        scores: Dict[str, float],
        train_time: float,
        predict_time: float
    ) -> None:
        """Save checkpoint after iteration."""
        # Regular checkpoint
        if (iteration + 1) % self.save_freq == 0:
            self._save_checkpoint({
                'type': 'resample_iteration',
                'iteration': iteration,
                'scores': scores,
                'train_time': train_time,
                'predict_time': predict_time
            }, f"{self._current_experiment}_iter_{iteration}")
            
        # Best model checkpoint
        if self.save_best and scores:
            # Get score to monitor
            if self.measure and self.measure in scores:
                current_score = scores[self.measure]
            else:
                current_score = list(scores.values())[0]
                
            # Initialize minimize if needed
            if self.minimize is None and self.measure:
                self.minimize = any(
                    term in self.measure.lower()
                    for term in ['error', 'loss', 'mse', 'mae', 'rmse']
                )
                
            # Check if best
            if self.best_score is None or self._is_better(current_score, self.best_score):
                self.best_score = current_score
                self._save_checkpoint({
                    'type': 'best_model',
                    'iteration': iteration,
                    'score': current_score,
                    'scores': scores
                }, f"{self._current_experiment}_best")
                
    def on_resample_end(self, result: Any) -> None:
        """Save final checkpoint for resample."""
        self._save_checkpoint({
            'type': 'resample_final',
            'n_iterations': result.n_iters,
            'n_errors': result.n_errors,
            'status': 'completed'
        }, f"{self._current_experiment}_final")
        
    # Benchmark callbacks
    def on_benchmark_begin(self, tasks: List[Task], learners: List[Learner]) -> None:
        """Initialize checkpoint for benchmark."""
        self._current_experiment = "benchmark"
        
        self._save_checkpoint({
            'type': 'benchmark',
            'task_ids': [t.id for t in tasks],
            'learner_ids': [l.id for l in learners],
            'status': 'started'
        }, f"{self._current_experiment}_init")
        
    def on_experiment_end(
        self,
        task: Task,
        learner: Learner,
        result: Optional[Any],
        error: Optional[Exception]
    ) -> None:
        """Save checkpoint after experiment."""
        exp_name = f"{task.id}_{learner.id}"
        
        checkpoint_data = {
            'type': 'benchmark_experiment',
            'task_id': task.id,
            'learner_id': learner.id,
            'success': error is None
        }
        
        if error is not None:
            checkpoint_data['error'] = str(error)
        elif result is not None:
            # Save summary scores
            checkpoint_data['scores'] = {
                m.id: result.score(m.id) for m in result.measures
            }
            
        self._save_checkpoint(checkpoint_data, f"benchmark_{exp_name}")
        
    # Tuning callbacks
    def on_tune_begin(self, learner: Learner, param_set: Any, n_configs: int) -> None:
        """Initialize checkpoint for tuning."""
        self._current_experiment = f"tune_{learner.id}"
        self.best_score = None
        
        self._save_checkpoint({
            'type': 'tuning',
            'learner_id': learner.id,
            'n_configs': n_configs,
            'param_names': list(param_set.params.keys()),
            'status': 'started'
        }, f"{self._current_experiment}_init")
        
    def on_config_end(self, config_num: int, score: float) -> None:
        """Save checkpoint after config evaluation."""
        # Regular checkpoint
        if config_num % self.save_freq == 0:
            self._save_checkpoint({
                'type': 'tuning_config',
                'config_num': config_num,
                'score': score
            }, f"{self._current_experiment}_config_{config_num}")
            
        # Best config checkpoint
        if self.save_best:
            if self.best_score is None or self._is_better(score, self.best_score):
                self.best_score = score
                self._save_checkpoint({
                    'type': 'best_config',
                    'config_num': config_num,
                    'score': score
                }, f"{self._current_experiment}_best")
                
    def list_checkpoints(self) -> List[str]:
        """List all saved checkpoints.
        
        Returns
        -------
        list of str
            Names of saved checkpoints.
        """
        return [p.stem for p in self.checkpoint_dir.glob("*.pkl")]
        
    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Parameters
        ----------
        name : str
            Name of the checkpoint to load.
            
        Returns
        -------
        dict
            Checkpoint data.
        """
        path = self._get_checkpoint_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {name} not found")
            
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def clear_checkpoints(self) -> None:
        """Remove all checkpoints."""
        for path in self.checkpoint_dir.glob("*.pkl"):
            path.unlink()


__all__ = ["CallbackCheckpoint"]