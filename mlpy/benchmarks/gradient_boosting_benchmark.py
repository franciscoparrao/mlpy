"""Gradient Boosting Benchmark Suite for MLPY.

This module provides comprehensive benchmarking for gradient boosting models,
comparing XGBoost, LightGBM, and CatBoost across various scenarios.
"""

import numpy as np
import pandas as pd
import time
import psutil
import GPUtil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import warnings
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from ..tasks import TaskClassif, TaskRegr
from ..data import Data
from ..measures import MeasureClassifAccuracy, MeasureClassifAUC, MeasureRegrRMSE, MeasureRegrMAE
from ..resampling import ResamplingCV
from ..learners import learner_gradient_boosting
from ..learners.gradient_boosting import GBOptimizationProfile

# Try to import individual backends
try:
    from ..learners.xgboost_wrapper import learner_xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from ..learners.lightgbm_wrapper import learner_lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from ..learners.catboost_wrapper import learner_catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    backend: str
    task_name: str
    task_type: str
    n_samples: int
    n_features: int
    n_categorical: int
    has_missing: bool
    training_time: float
    prediction_time: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    accuracy_score: float
    additional_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    hardware_info: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class GradientBoostingBenchmark:
    """Comprehensive benchmark suite for gradient boosting models.
    
    This class provides tools to compare XGBoost, LightGBM, and CatBoost
    across various datasets and scenarios.
    
    Parameters
    ----------
    backends : List[str], optional
        List of backends to benchmark. If None, uses all available.
    n_estimators : int, default=100
        Number of boosting rounds.
    early_stopping : bool, default=True
        Whether to use early stopping.
    use_gpu : bool, default=False
        Whether to test GPU performance.
    cv_folds : int, default=5
        Number of cross-validation folds.
    save_results : bool, default=True
        Whether to save results to file.
    output_dir : Path, optional
        Directory to save results.
    verbose : bool, default=True
        Verbosity mode.
    """
    
    def __init__(
        self,
        backends: Optional[List[str]] = None,
        n_estimators: int = 100,
        early_stopping: bool = True,
        use_gpu: bool = False,
        cv_folds: int = 5,
        save_results: bool = True,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        self.backends = backends or self._get_available_backends()
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.use_gpu = use_gpu
        self.cv_folds = cv_folds
        self.save_results = save_results
        self.output_dir = Path(output_dir or "./benchmark_results")
        self.verbose = verbose
        
        self.results = []
        self.hardware_info = self._get_hardware_info()
        
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        available = []
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        if CATBOOST_AVAILABLE:
            available.append('catboost')
        return available
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory_mb': []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_available'] = True
                info['gpu_count'] = len(gpus)
                info['gpu_names'] = [gpu.name for gpu in gpus]
                info['gpu_memory_mb'] = [gpu.memoryTotal for gpu in gpus]
        except:
            pass
            
        return info
        
    def _measure_memory(self) -> float:
        """Measure current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)  # MB
        
    def _measure_gpu_memory(self) -> Optional[float]:
        """Measure GPU memory usage."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return sum(gpu.memoryUsed for gpu in gpus)
        except:
            pass
        return None
        
    def create_synthetic_dataset(
        self,
        n_samples: int = 10000,
        n_features: int = 50,
        n_categorical: int = 10,
        n_informative: int = 40,
        missing_ratio: float = 0.1,
        task_type: str = 'classification',
        n_classes: int = 2,
        random_state: int = 42
    ) -> Tuple[Task, str]:
        """Create synthetic dataset for benchmarking.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_features : int
            Total number of features.
        n_categorical : int
            Number of categorical features.
        n_informative : int
            Number of informative features.
        missing_ratio : float
            Ratio of missing values.
        task_type : str
            'classification' or 'regression'.
        n_classes : int
            Number of classes for classification.
        random_state : int
            Random seed.
            
        Returns
        -------
        Task, str
            The task and dataset name.
        """
        from sklearn.datasets import make_classification, make_regression
        
        np.random.seed(random_state)
        
        # Generate base dataset
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features - n_categorical,
                n_informative=n_informative - n_categorical,
                n_classes=n_classes,
                random_state=random_state
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features - n_categorical,
                n_informative=n_informative - n_categorical,
                random_state=random_state
            )
            
        # Add categorical features
        if n_categorical > 0:
            cat_features = np.random.randint(
                0, min(10, n_classes + 3),
                size=(n_samples, n_categorical)
            )
            X = np.hstack([X, cat_features])
            
        # Add missing values
        if missing_ratio > 0:
            n_missing = int(X.size * missing_ratio)
            missing_indices = np.random.choice(
                X.size, n_missing, replace=False
            )
            X.flat[missing_indices] = np.nan
            
        # Create DataFrame
        feature_names = [f"num_{i}" for i in range(n_features - n_categorical)]
        feature_names += [f"cat_{i}" for i in range(n_categorical)]
        
        df = pd.DataFrame(X, columns=feature_names)
        
        # Set categorical dtypes
        for i in range(n_categorical):
            col = f"cat_{i}"
            df[col] = df[col].fillna(-1).astype(int).astype('category')
            
        df['target'] = y
        
        # Create Data and Task
        data = Data(
            id=f"synthetic_{task_type}_{n_samples}_{n_features}",
            backend=df,
            target_names=['target']
        )
        
        if task_type == 'classification':
            task = TaskClassif(
                id=f"task_synthetic_{task_type}",
                backend=data,
                target='target'
            )
        else:
            task = TaskRegr(
                id=f"task_synthetic_{task_type}",
                backend=data,
                target='target'
            )
            
        dataset_name = f"Synthetic_{task_type}_{n_samples}x{n_features}"
        if n_categorical > 0:
            dataset_name += f"_cat{n_categorical}"
        if missing_ratio > 0:
            dataset_name += f"_miss{int(missing_ratio*100)}"
            
        return task, dataset_name
        
    def benchmark_single(
        self,
        task: Task,
        backend: str,
        task_name: str = "Unknown",
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Benchmark a single backend on a task.
        
        Parameters
        ----------
        task : Task
            The task to benchmark.
        backend : str
            Backend to use.
        task_name : str
            Name of the task/dataset.
        hyperparameters : Dict, optional
            Custom hyperparameters.
            
        Returns
        -------
        BenchmarkResult
            Benchmark results.
        """
        if self.verbose:
            print(f"\nBenchmarking {backend} on {task_name}...")
            
        # Prepare hyperparameters
        params = {
            'n_estimators': self.n_estimators,
            'backend': backend,
            'verbose': False
        }
        
        if self.early_stopping:
            params['early_stopping_rounds'] = 10
            
        if hyperparameters:
            params.update(hyperparameters)
            
        # Configure GPU if requested
        if self.use_gpu and self.hardware_info['gpu_available']:
            profile = GBOptimizationProfile(use_gpu=True, handle_categorical=True)
            params['optimization_profile'] = profile
            
        # Create learner
        learner = learner_gradient_boosting(**params)
        
        # Measure training
        mem_before = self._measure_memory()
        gpu_mem_before = self._measure_gpu_memory() if self.use_gpu else None
        
        start_time = time.time()
        try:
            learner.train(task)
            training_time = time.time() - start_time
            error = None
        except Exception as e:
            training_time = time.time() - start_time
            error = str(e)
            logger.error(f"Training failed for {backend}: {error}")
            
        mem_after = self._measure_memory()
        gpu_mem_after = self._measure_gpu_memory() if self.use_gpu else None
        
        memory_usage = mem_after - mem_before
        gpu_memory = (gpu_mem_after - gpu_mem_before) if gpu_mem_after else None
        
        # Measure prediction
        start_time = time.time()
        if error is None:
            try:
                predictions = learner.predict(task)
                prediction_time = time.time() - start_time
            except Exception as e:
                prediction_time = 0
                error = str(e)
        else:
            prediction_time = 0
            
        # Calculate metrics
        metrics = {}
        accuracy_score = 0.0
        
        if error is None:
            if isinstance(task, TaskClassif):
                # Classification metrics
                measure_acc = MeasureClassifAccuracy()
                accuracy_score = predictions.score(measure_acc)
                metrics['accuracy'] = accuracy_score
                
                try:
                    measure_auc = MeasureClassifAUC()
                    metrics['auc'] = predictions.score(measure_auc)
                except:
                    pass
            else:
                # Regression metrics
                measure_rmse = MeasureRegrRMSE()
                accuracy_score = -predictions.score(measure_rmse)  # Negative for consistency
                metrics['rmse'] = -accuracy_score
                
                measure_mae = MeasureRegrMAE()
                metrics['mae'] = predictions.score(measure_mae)
                
        # Get data characteristics
        X_sample = task.data(cols=task.feature_names, data_format='dataframe')
        n_categorical = len(X_sample.select_dtypes(include=['category', 'object']).columns)
        has_missing = X_sample.isnull().any().any()
        
        # Create result
        result = BenchmarkResult(
            backend=backend,
            task_name=task_name,
            task_type='classification' if isinstance(task, TaskClassif) else 'regression',
            n_samples=len(task.row_roles['use']),
            n_features=len(task.feature_names),
            n_categorical=n_categorical,
            has_missing=has_missing,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory,
            accuracy_score=accuracy_score,
            additional_metrics=metrics,
            hyperparameters=params,
            hardware_info=self.hardware_info,
            timestamp=datetime.now().isoformat(),
            error=error
        )
        
        return result
        
    def benchmark_task(
        self,
        task: Task,
        task_name: str = "Unknown",
        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark all backends on a single task.
        
        Parameters
        ----------
        task : Task
            The task to benchmark.
        task_name : str
            Name of the task.
        hyperparameters : Dict[str, Dict], optional
            Backend-specific hyperparameters.
            
        Returns
        -------
        List[BenchmarkResult]
            Results for all backends.
        """
        results = []
        
        for backend in self.backends:
            backend_params = None
            if hyperparameters and backend in hyperparameters:
                backend_params = hyperparameters[backend]
                
            result = self.benchmark_single(
                task, backend, task_name, backend_params
            )
            results.append(result)
            self.results.append(result)
            
        return results
        
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple scenarios.
        
        Returns
        -------
        pd.DataFrame
            Benchmark results summary.
        """
        scenarios = [
            # Small dataset, no categorical
            {
                'n_samples': 1000,
                'n_features': 20,
                'n_categorical': 0,
                'missing_ratio': 0.0,
                'name': 'Small_Numeric'
            },
            # Medium dataset with categorical
            {
                'n_samples': 10000,
                'n_features': 50,
                'n_categorical': 10,
                'missing_ratio': 0.0,
                'name': 'Medium_Mixed'
            },
            # Large dataset
            {
                'n_samples': 100000,
                'n_features': 100,
                'n_categorical': 20,
                'missing_ratio': 0.0,
                'name': 'Large_Mixed'
            },
            # Dataset with missing values
            {
                'n_samples': 10000,
                'n_features': 50,
                'n_categorical': 10,
                'missing_ratio': 0.2,
                'name': 'Medium_Missing'
            },
            # Wide dataset
            {
                'n_samples': 5000,
                'n_features': 500,
                'n_categorical': 50,
                'missing_ratio': 0.0,
                'name': 'Wide_Mixed'
            },
            # Heavily categorical
            {
                'n_samples': 10000,
                'n_features': 50,
                'n_categorical': 40,
                'missing_ratio': 0.0,
                'name': 'Heavy_Categorical'
            }
        ]
        
        print("=" * 80)
        print("GRADIENT BOOSTING COMPREHENSIVE BENCHMARK")
        print("=" * 80)
        print(f"Backends: {', '.join(self.backends)}")
        print(f"Hardware: {self.hardware_info['cpu_count']} CPUs, "
              f"{self.hardware_info['total_memory_gb']:.1f}GB RAM")
        if self.hardware_info['gpu_available']:
            print(f"GPU: {self.hardware_info['gpu_names'][0]}")
        print("=" * 80)
        
        # Run benchmarks for each scenario
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            print("-" * 40)
            
            # Classification task
            task_clf, name_clf = self.create_synthetic_dataset(
                task_type='classification',
                **{k: v for k, v in scenario.items() if k != 'name'}
            )
            self.benchmark_task(task_clf, f"{scenario['name']}_Classification")
            
            # Regression task
            task_reg, name_reg = self.create_synthetic_dataset(
                task_type='regression',
                **{k: v for k, v in scenario.items() if k != 'name'}
            )
            self.benchmark_task(task_reg, f"{scenario['name']}_Regression")
            
        # Create summary DataFrame
        df = self._create_summary_dataframe()
        
        # Save results
        if self.save_results:
            self._save_results(df)
            
        return df
        
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame from results.
        
        Returns
        -------
        pd.DataFrame
            Summary of benchmark results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Add derived metrics
        df['throughput_samples_per_sec'] = df['n_samples'] / df['training_time']
        df['memory_per_sample'] = df['memory_usage_mb'] / df['n_samples'] * 1000  # KB
        
        return df
        
    def _save_results(self, df: pd.DataFrame):
        """Save benchmark results to files.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as CSV
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save summary as JSON
        summary = self.create_summary_report(df)
        json_path = self.output_dir / f"benchmark_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to: {json_path}")
        
    def create_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary report from results.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame.
            
        Returns
        -------
        Dict[str, Any]
            Summary report.
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'hardware': self.hardware_info,
            'n_benchmarks': len(df),
            'backends_tested': list(df['backend'].unique()),
            'datasets_tested': list(df['task_name'].unique()),
        }
        
        # Performance by backend
        backend_summary = {}
        for backend in df['backend'].unique():
            backend_df = df[df['backend'] == backend]
            backend_summary[backend] = {
                'avg_training_time': backend_df['training_time'].mean(),
                'avg_prediction_time': backend_df['prediction_time'].mean(),
                'avg_memory_usage_mb': backend_df['memory_usage_mb'].mean(),
                'avg_accuracy': backend_df['accuracy_score'].mean(),
                'n_failures': backend_df['error'].notna().sum(),
                'best_scenario': backend_df.loc[backend_df['accuracy_score'].idxmax(), 'task_name']
                if not backend_df.empty else None
            }
            
        summary['backend_performance'] = backend_summary
        
        # Best backend by scenario
        best_by_scenario = {}
        for task_name in df['task_name'].unique():
            task_df = df[df['task_name'] == task_name]
            if not task_df.empty:
                # Best by accuracy
                best_acc = task_df.loc[task_df['accuracy_score'].idxmax()]
                # Fastest
                fastest = task_df.loc[task_df['training_time'].idxmin()]
                # Most memory efficient
                mem_efficient = task_df.loc[task_df['memory_usage_mb'].idxmin()]
                
                best_by_scenario[task_name] = {
                    'best_accuracy': {
                        'backend': best_acc['backend'],
                        'score': best_acc['accuracy_score']
                    },
                    'fastest': {
                        'backend': fastest['backend'],
                        'time': fastest['training_time']
                    },
                    'memory_efficient': {
                        'backend': mem_efficient['backend'],
                        'memory_mb': mem_efficient['memory_usage_mb']
                    }
                }
                
        summary['best_by_scenario'] = best_by_scenario
        
        # Overall rankings
        rankings = self._calculate_rankings(df)
        summary['overall_rankings'] = rankings
        
        return summary
        
    def _calculate_rankings(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Calculate overall backend rankings.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame.
            
        Returns
        -------
        Dict[str, List[str]]
            Rankings by different criteria.
        """
        # Filter out failed runs
        df_clean = df[df['error'].isna()].copy()
        
        if df_clean.empty:
            return {}
            
        # Normalize metrics for fair comparison
        df_clean['norm_accuracy'] = (df_clean['accuracy_score'] - df_clean['accuracy_score'].min()) / \
                                    (df_clean['accuracy_score'].max() - df_clean['accuracy_score'].min() + 1e-10)
        df_clean['norm_speed'] = 1 - (df_clean['training_time'] - df_clean['training_time'].min()) / \
                                 (df_clean['training_time'].max() - df_clean['training_time'].min() + 1e-10)
        df_clean['norm_memory'] = 1 - (df_clean['memory_usage_mb'] - df_clean['memory_usage_mb'].min()) / \
                                  (df_clean['memory_usage_mb'].max() - df_clean['memory_usage_mb'].min() + 1e-10)
        
        # Calculate composite scores
        backend_scores = {}
        for backend in df_clean['backend'].unique():
            backend_df = df_clean[df_clean['backend'] == backend]
            
            # Weighted composite score
            accuracy_weight = 0.5
            speed_weight = 0.3
            memory_weight = 0.2
            
            composite_score = (
                backend_df['norm_accuracy'].mean() * accuracy_weight +
                backend_df['norm_speed'].mean() * speed_weight +
                backend_df['norm_memory'].mean() * memory_weight
            )
            
            backend_scores[backend] = {
                'composite': composite_score,
                'accuracy': backend_df['norm_accuracy'].mean(),
                'speed': backend_df['norm_speed'].mean(),
                'memory': backend_df['norm_memory'].mean()
            }
            
        # Create rankings
        rankings = {
            'overall': sorted(backend_scores.keys(), 
                            key=lambda x: backend_scores[x]['composite'], 
                            reverse=True),
            'accuracy': sorted(backend_scores.keys(), 
                             key=lambda x: backend_scores[x]['accuracy'], 
                             reverse=True),
            'speed': sorted(backend_scores.keys(), 
                          key=lambda x: backend_scores[x]['speed'], 
                          reverse=True),
            'memory': sorted(backend_scores.keys(), 
                           key=lambda x: backend_scores[x]['memory'], 
                           reverse=True)
        }
        
        return rankings
        
    def plot_results(self, df: Optional[pd.DataFrame] = None):
        """Plot benchmark results.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            Results DataFrame. If None, uses self.results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if df is None:
            df = self._create_summary_dataframe()
            
        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Training time comparison
        ax = axes[0, 0]
        sns.boxplot(data=df, x='backend', y='training_time', ax=ax)
        ax.set_title('Training Time by Backend')
        ax.set_ylabel('Time (seconds)')
        ax.set_yscale('log')
        
        # 2. Memory usage comparison
        ax = axes[0, 1]
        sns.boxplot(data=df, x='backend', y='memory_usage_mb', ax=ax)
        ax.set_title('Memory Usage by Backend')
        ax.set_ylabel('Memory (MB)')
        
        # 3. Accuracy comparison
        ax = axes[0, 2]
        sns.boxplot(data=df, x='backend', y='accuracy_score', ax=ax)
        ax.set_title('Accuracy by Backend')
        ax.set_ylabel('Accuracy Score')
        
        # 4. Performance vs dataset size
        ax = axes[1, 0]
        for backend in df['backend'].unique():
            backend_df = df[df['backend'] == backend]
            ax.scatter(backend_df['n_samples'], backend_df['training_time'], 
                      label=backend, alpha=0.7)
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Scalability: Time vs Dataset Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        
        # 5. Categorical features impact
        ax = axes[1, 1]
        for backend in df['backend'].unique():
            backend_df = df[df['backend'] == backend]
            ax.scatter(backend_df['n_categorical'], backend_df['accuracy_score'], 
                      label=backend, alpha=0.7)
        ax.set_xlabel('Number of Categorical Features')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Performance with Categorical Features')
        ax.legend()
        
        # 6. Speed vs Accuracy tradeoff
        ax = axes[1, 2]
        for backend in df['backend'].unique():
            backend_df = df[df['backend'] == backend]
            ax.scatter(backend_df['training_time'], backend_df['accuracy_score'], 
                      label=backend, s=100, alpha=0.7)
        ax.set_xlabel('Training Time (s)')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Speed vs Accuracy Tradeoff')
        ax.set_xscale('log')
        ax.legend()
        
        plt.suptitle('Gradient Boosting Benchmark Results', fontsize=16)
        plt.tight_layout()
        
        if self.save_results:
            plot_path = self.output_dir / f"benchmark_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
            
        plt.show()


def run_quick_benchmark(use_gpu: bool = False) -> pd.DataFrame:
    """Run a quick benchmark with default settings.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU if available.
        
    Returns
    -------
    pd.DataFrame
        Benchmark results.
        
    Examples
    --------
    >>> from mlpy.benchmarks import run_quick_benchmark
    >>> results = run_quick_benchmark()
    >>> print(results.groupby('backend')['accuracy_score'].mean())
    """
    benchmark = GradientBoostingBenchmark(
        n_estimators=50,  # Fewer iterations for quick test
        use_gpu=use_gpu,
        save_results=False,
        verbose=True
    )
    
    # Create a few test scenarios
    scenarios = [
        {'n_samples': 1000, 'n_features': 20, 'n_categorical': 5},
        {'n_samples': 5000, 'n_features': 50, 'n_categorical': 10},
    ]
    
    for i, scenario in enumerate(scenarios):
        task, name = benchmark.create_synthetic_dataset(
            task_type='classification',
            **scenario
        )
        benchmark.benchmark_task(task, f"Quick_Test_{i+1}")
        
    df = benchmark._create_summary_dataframe()
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK BENCHMARK SUMMARY")
    print("=" * 60)
    
    for backend in df['backend'].unique():
        backend_df = df[df['backend'] == backend]
        print(f"\n{backend.upper()}:")
        print(f"  Avg Training Time: {backend_df['training_time'].mean():.3f}s")
        print(f"  Avg Memory Usage: {backend_df['memory_usage_mb'].mean():.1f}MB")
        print(f"  Avg Accuracy: {backend_df['accuracy_score'].mean():.4f}")
        
    return df


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark = GradientBoostingBenchmark(
        n_estimators=100,
        use_gpu=False,
        save_results=True,
        verbose=True
    )
    
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Plot results
    benchmark.plot_results(results_df)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    summary = benchmark.create_summary_report(results_df)
    
    if 'overall_rankings' in summary:
        print("\nOVERALL RANKINGS:")
        for metric, ranking in summary['overall_rankings'].items():
            print(f"  {metric.capitalize()}: {' > '.join(ranking)}")
            
    print("\nResults saved to:", benchmark.output_dir)