"""
A/B Testing and Experiment Tracking
====================================

Manage model experiments and A/B testing in production.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from scipy import stats
import hashlib
import json
import logging
from pathlib import Path
import random
from enum import Enum

from ..learners.base import Learner

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AllocationStrategy(Enum):
    """Traffic allocation strategy."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class Variant:
    """Represents a model variant in an experiment."""
    
    variant_id: str
    model_id: str
    model_version: str
    description: str
    weight: float = 0.5
    is_control: bool = False
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric observation."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "count": len(values)
        }


@dataclass
class Experiment:
    """Represents an A/B test experiment."""
    
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    status: ExperimentStatus
    allocation_strategy: AllocationStrategy
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    target_metric: str = "accuracy"
    minimum_sample_size: int = 100
    confidence_level: float = 0.95
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": [asdict(v) for v in self.variants],
            "status": self.status.value,
            "allocation_strategy": self.allocation_strategy.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "target_metric": self.target_metric,
            "minimum_sample_size": self.minimum_sample_size,
            "confidence_level": self.confidence_level,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "tags": self.tags
        }


class ABTester:
    """Manages A/B testing for models."""
    
    def __init__(self, storage_path: str = "./experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
        self.load_experiments()
    
    def load_experiments(self):
        """Load experiments from storage."""
        experiments_file = self.storage_path / "experiments.json"
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                data = json.load(f)
                for exp_id, exp_data in data.items():
                    # Reconstruct experiment
                    variants = [Variant(**v) for v in exp_data.pop("variants")]
                    exp_data["status"] = ExperimentStatus(exp_data["status"])
                    exp_data["allocation_strategy"] = AllocationStrategy(exp_data["allocation_strategy"])
                    self.experiments[exp_id] = Experiment(variants=variants, **exp_data)
                    
                    if self.experiments[exp_id].status == ExperimentStatus.RUNNING:
                        self.active_experiments.append(exp_id)
    
    def save_experiments(self):
        """Save experiments to storage."""
        experiments_file = self.storage_path / "experiments.json"
        data = {
            exp_id: exp.to_dict()
            for exp_id, exp in self.experiments.items()
        }
        with open(experiments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        control_model: Tuple[str, str],  # (model_id, version)
        treatment_models: List[Tuple[str, str]],  # [(model_id, version), ...]
        description: str = "",
        allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM,
        weights: Optional[List[float]] = None,
        target_metric: str = "accuracy",
        minimum_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> Experiment:
        """Create a new A/B test experiment."""
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id(name)
        
        # Create variants
        variants = []
        
        # Control variant
        control = Variant(
            variant_id=f"{experiment_id}_control",
            model_id=control_model[0],
            model_version=control_model[1],
            description="Control variant",
            weight=weights[0] if weights else 1.0 / (len(treatment_models) + 1),
            is_control=True
        )
        variants.append(control)
        
        # Treatment variants
        for i, (model_id, version) in enumerate(treatment_models):
            variant = Variant(
                variant_id=f"{experiment_id}_treatment_{i+1}",
                model_id=model_id,
                model_version=version,
                description=f"Treatment variant {i+1}",
                weight=weights[i+1] if weights else 1.0 / (len(treatment_models) + 1),
                is_control=False
            )
            variants.append(variant)
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description or f"A/B test: {name}",
            variants=variants,
            status=ExperimentStatus.DRAFT,
            allocation_strategy=allocation_strategy,
            target_metric=target_metric,
            minimum_sample_size=minimum_sample_size,
            confidence_level=confidence_level
        )
        
        self.experiments[experiment_id] = experiment
        self.save_experiments()
        
        logger.info(f"Created experiment {experiment_id}")
        return experiment
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment must be in DRAFT status to start")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow().isoformat()
        self.active_experiments.append(experiment_id)
        
        self.save_experiments()
        logger.info(f"Started experiment {experiment_id}")
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment is not running")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.utcnow().isoformat()
        
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)
        
        self.save_experiments()
        logger.info(f"Stopped experiment {experiment_id}")
    
    def select_variant(self, experiment_id: str, context: Optional[Dict] = None) -> Variant:
        """Select a variant for a request based on allocation strategy."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            # Return control if experiment not running
            return next(v for v in experiment.variants if v.is_control)
        
        strategy = experiment.allocation_strategy
        
        if strategy == AllocationStrategy.RANDOM:
            return self._random_allocation(experiment)
        elif strategy == AllocationStrategy.WEIGHTED:
            return self._weighted_allocation(experiment)
        elif strategy == AllocationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_allocation(experiment)
        elif strategy == AllocationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_allocation(experiment)
        else:
            return self._random_allocation(experiment)
    
    def _random_allocation(self, experiment: Experiment) -> Variant:
        """Random variant selection."""
        return random.choice(experiment.variants)
    
    def _weighted_allocation(self, experiment: Experiment) -> Variant:
        """Weighted random variant selection."""
        weights = [v.weight for v in experiment.variants]
        return random.choices(experiment.variants, weights=weights)[0]
    
    def _epsilon_greedy_allocation(self, experiment: Experiment, epsilon: float = 0.1) -> Variant:
        """Epsilon-greedy variant selection."""
        if random.random() < epsilon:
            # Explore: random selection
            return random.choice(experiment.variants)
        else:
            # Exploit: select best performing
            metric = experiment.target_metric
            best_variant = None
            best_score = -float('inf')
            
            for variant in experiment.variants:
                if metric in variant.metrics and variant.metrics[metric]:
                    score = np.mean(variant.metrics[metric])
                    if score > best_score:
                        best_score = score
                        best_variant = variant
            
            return best_variant or experiment.variants[0]
    
    def _thompson_sampling_allocation(self, experiment: Experiment) -> Variant:
        """Thompson sampling variant selection."""
        # Use Beta distribution for binary metrics
        samples = []
        
        for variant in experiment.variants:
            # Use success/failure counts
            alpha = variant.success_count + 1
            beta = max(1, variant.request_count - variant.success_count) + 1
            sample = np.random.beta(alpha, beta)
            samples.append(sample)
        
        # Select variant with highest sample
        best_idx = np.argmax(samples)
        return experiment.variants[best_idx]
    
    def record_outcome(
        self,
        experiment_id: str,
        variant_id: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Record the outcome of a variant selection."""
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        variant = next((v for v in experiment.variants if v.variant_id == variant_id), None)
        
        if not variant:
            return
        
        # Update counts
        variant.request_count += 1
        if success:
            variant.success_count += 1
        else:
            variant.error_count += 1
        
        # Record metrics
        if metrics:
            for metric_name, value in metrics.items():
                variant.add_metric(metric_name, value)
        
        # Check if experiment should end
        if self._should_end_experiment(experiment):
            self.stop_experiment(experiment_id)
        
        self.save_experiments()
    
    def _should_end_experiment(self, experiment: Experiment) -> bool:
        """Check if experiment has reached significance."""
        # Check minimum sample size
        for variant in experiment.variants:
            if variant.request_count < experiment.minimum_sample_size:
                return False
        
        # Perform statistical test
        control = next(v for v in experiment.variants if v.is_control)
        
        for variant in experiment.variants:
            if variant.is_control:
                continue
            
            # Check if we have enough data
            metric = experiment.target_metric
            if metric not in control.metrics or metric not in variant.metrics:
                continue
            
            control_values = control.metrics[metric]
            variant_values = variant.metrics[metric]
            
            if len(control_values) < 30 or len(variant_values) < 30:
                continue
            
            # Perform t-test
            _, p_value = stats.ttest_ind(control_values, variant_values)
            
            # Check significance
            alpha = 1 - experiment.confidence_level
            if p_value < alpha:
                logger.info(f"Experiment {experiment.experiment_id} reached significance (p={p_value:.4f})")
                return True
        
        return False
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive results for an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        control = next(v for v in experiment.variants if v.is_control)
        
        results = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "duration": self._calculate_duration(experiment),
            "total_requests": sum(v.request_count for v in experiment.variants),
            "variants": []
        }
        
        for variant in experiment.variants:
            variant_result = {
                "variant_id": variant.variant_id,
                "model_id": variant.model_id,
                "is_control": variant.is_control,
                "request_count": variant.request_count,
                "success_rate": variant.success_count / max(1, variant.request_count),
                "metrics": {}
            }
            
            # Calculate metric statistics
            for metric_name in variant.metrics:
                stats = variant.get_metric_stats(metric_name)
                variant_result["metrics"][metric_name] = stats
                
                # Calculate lift vs control
                if not variant.is_control and metric_name in control.metrics:
                    control_mean = np.mean(control.metrics[metric_name])
                    variant_mean = stats["mean"]
                    lift = ((variant_mean - control_mean) / control_mean) * 100
                    variant_result["metrics"][metric_name]["lift_vs_control"] = lift
                    
                    # Statistical significance
                    if len(control.metrics[metric_name]) >= 30 and stats["count"] >= 30:
                        from scipy import stats as scipy_stats
                        _, p_value = scipy_stats.ttest_ind(
                            control.metrics[metric_name],
                            variant.metrics[metric_name]
                        )
                        variant_result["metrics"][metric_name]["p_value"] = p_value
                        variant_result["metrics"][metric_name]["significant"] = p_value < (1 - experiment.confidence_level)
            
            results["variants"].append(variant_result)
        
        # Winner determination
        if experiment.status == ExperimentStatus.COMPLETED:
            results["winner"] = self._determine_winner(experiment)
        
        return results
    
    def _calculate_duration(self, experiment: Experiment) -> Optional[float]:
        """Calculate experiment duration in hours."""
        if not experiment.start_time:
            return None
        
        start = datetime.fromisoformat(experiment.start_time)
        end = datetime.fromisoformat(experiment.end_time) if experiment.end_time else datetime.utcnow()
        
        return (end - start).total_seconds() / 3600
    
    def _determine_winner(self, experiment: Experiment) -> Optional[str]:
        """Determine the winning variant."""
        metric = experiment.target_metric
        best_variant = None
        best_score = -float('inf')
        
        for variant in experiment.variants:
            if metric in variant.metrics and variant.metrics[metric]:
                score = np.mean(variant.metrics[metric])
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        return best_variant.variant_id if best_variant else None
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"exp_{timestamp}_{name_hash}"


class ExperimentTracker:
    """Track and manage ML experiments."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or "./mlruns"
        self.experiments: Dict[str, Dict] = {}
        self.active_run = None
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a new experiment run."""
        run_id = hashlib.md5(f"{experiment_name}_{datetime.utcnow()}".encode()).hexdigest()[:8]
        
        self.active_run = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "run_name": run_name or run_id,
            "start_time": datetime.utcnow().isoformat(),
            "parameters": {},
            "metrics": {},
            "artifacts": []
        }
        
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = {}
        
        logger.info(f"Started run {run_id} for experiment {experiment_name}")
        return run_id
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.active_run:
            self.active_run["parameters"][key] = value
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self.active_run:
            if key not in self.active_run["metrics"]:
                self.active_run["metrics"][key] = []
            self.active_run["metrics"][key].append({
                "value": value,
                "step": step or len(self.active_run["metrics"][key]),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log an artifact."""
        if self.active_run:
            self.active_run["artifacts"].append({
                "path": file_path,
                "type": artifact_type,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def end_run(self):
        """End the current run."""
        if self.active_run:
            self.active_run["end_time"] = datetime.utcnow().isoformat()
            
            # Save run
            exp_name = self.active_run["experiment_name"]
            run_id = self.active_run["run_id"]
            self.experiments[exp_name][run_id] = self.active_run
            
            logger.info(f"Ended run {run_id}")
            self.active_run = None
    
    def get_experiment_runs(self, experiment_name: str) -> List[Dict]:
        """Get all runs for an experiment."""
        if experiment_name not in self.experiments:
            return []
        return list(self.experiments[experiment_name].values())