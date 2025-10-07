"""
Advanced benchmarking system for MLPY.

This module provides comprehensive benchmarking capabilities for comparing
multiple learners across tasks, with support for statistical analysis,
visualization, and ranking similar to mlr3benchmark.
"""

import time
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from datetime import datetime
import json

from .tasks import Task
from .learners import Learner
from .resamplings import Resampling
from .measures import Measure
from .resample import resample, ResampleResult
from .utils.logging import get_logger
from .parallel import Backend, get_backend
from .callbacks import Callback

logger = get_logger(__name__)


@dataclass
class BenchmarkDesign:
    """
    Design for benchmark experiments.
    
    Defines the grid of tasks, learners, resamplings, and measures
    to evaluate in the benchmark.
    
    Parameters
    ----------
    tasks : List[Task]
        Tasks to evaluate
    learners : List[Learner]
        Learners to compare
    resamplings : List[Resampling]
        Resampling strategies
    measures : List[Measure]
        Performance measures
    paired : bool
        Whether to use paired resampling (same splits for all learners)
    """
    tasks: List[Task]
    learners: List[Learner]
    resamplings: List[Resampling]
    measures: List[Measure]
    paired: bool = True
    
    def __post_init__(self):
        """Validate and prepare design."""
        # Ensure lists
        if not isinstance(self.tasks, list):
            self.tasks = [self.tasks]
        if not isinstance(self.learners, list):
            self.learners = [self.learners]
        if not isinstance(self.resamplings, list):
            self.resamplings = [self.resamplings]
        if not isinstance(self.measures, list):
            self.measures = [self.measures]
        
        # Validate non-empty
        if not self.tasks:
            raise ValueError("At least one task required")
        if not self.learners:
            raise ValueError("At least one learner required")
        if not self.resamplings:
            raise ValueError("At least one resampling required")
        if not self.measures:
            raise ValueError("At least one measure required")
    
    @property
    def n_experiments(self) -> int:
        """Total number of experiments in the design."""
        return len(self.tasks) * len(self.learners) * len(self.resamplings)
    
    def grid(self) -> List[Tuple[Task, Learner, Resampling]]:
        """Generate experiment grid."""
        return list(product(self.tasks, self.learners, self.resamplings))


@dataclass
class BenchmarkScore:
    """Single benchmark score with metadata."""
    task_id: str
    learner_id: str
    resampling_id: str
    measure_id: str
    iteration: int
    score: float
    train_time: float = 0.0
    predict_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkResult:
    """
    Advanced benchmark result with comprehensive analysis capabilities.
    
    Stores and analyzes results from benchmark experiments.
    """
    
    def __init__(self, design: BenchmarkDesign):
        self.design = design
        self.scores: List[BenchmarkScore] = []
        self.errors: Dict[Tuple[str, str, str], Exception] = {}
        self.start_time = None
        self.end_time = None
        self.metadata = {}
    
    def add_score(self, score: BenchmarkScore) -> None:
        """Add a single score."""
        self.scores.append(score)
    
    def add_error(
        self, 
        task_id: str, 
        learner_id: str, 
        resampling_id: str, 
        error: Exception
    ) -> None:
        """Record an error."""
        self.errors[(task_id, learner_id, resampling_id)] = error
    
    def to_dataframe(self, wide: bool = False) -> pd.DataFrame:
        """
        Convert results to DataFrame.
        
        Parameters
        ----------
        wide : bool
            If True, return wide format with learners as columns
            
        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        # Create long format dataframe
        data = []
        for score in self.scores:
            data.append({
                'task': score.task_id,
                'learner': score.learner_id,
                'resampling': score.resampling_id,
                'measure': score.measure_id,
                'iteration': score.iteration,
                'score': score.score,
                'train_time': score.train_time,
                'predict_time': score.predict_time
            })
        
        df = pd.DataFrame(data)
        
        if wide and not df.empty:
            # Pivot to wide format
            df = df.pivot_table(
                index=['task', 'resampling', 'measure', 'iteration'],
                columns='learner',
                values='score'
            ).reset_index()
        
        return df
    
    def aggregate(
        self,
        measure: Optional[str] = None,
        group_by: List[str] = ['task', 'learner'],
        aggr_func: Union[str, Callable] = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate scores.
        
        Parameters
        ----------
        measure : str, optional
            Specific measure to aggregate. If None, uses all.
        group_by : List[str]
            Columns to group by
        aggr_func : str or callable
            Aggregation function ('mean', 'std', 'median', etc.)
            
        Returns
        -------
        pd.DataFrame
            Aggregated results
        """
        df = self.to_dataframe()
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter by measure if specified
        if measure:
            df = df[df['measure'] == measure]
        
        # Group and aggregate
        if isinstance(aggr_func, str):
            result = df.groupby(group_by)['score'].agg(aggr_func).reset_index()
        else:
            result = df.groupby(group_by)['score'].agg(aggr_func).reset_index()
        
        return result
    
    def rank_learners(
        self,
        measure: str,
        minimize: bool = False,
        method: str = 'average'
    ) -> pd.DataFrame:
        """
        Rank learners by performance.
        
        Parameters
        ----------
        measure : str
            Measure to rank by
        minimize : bool
            Whether lower scores are better
        method : str
            Ranking method for scipy.stats.rankdata
            
        Returns
        -------
        pd.DataFrame
            Ranking results
        """
        from scipy.stats import rankdata
        
        # Aggregate scores by task and learner
        agg_df = self.aggregate(
            measure=measure,
            group_by=['task', 'learner'],
            aggr_func='mean'
        )
        
        if agg_df.empty:
            return pd.DataFrame()
        
        # Rank within each task
        rankings = []
        for task in agg_df['task'].unique():
            task_df = agg_df[agg_df['task'] == task].copy()
            
            # Rank (lower is better if minimize=True)
            if minimize:
                task_df['rank'] = rankdata(task_df['score'], method=method)
            else:
                task_df['rank'] = rankdata(-task_df['score'], method=method)
            
            rankings.append(task_df)
        
        rankings_df = pd.concat(rankings, ignore_index=True)
        
        # Calculate average rank across tasks
        avg_ranks = rankings_df.groupby('learner').agg({
            'rank': 'mean',
            'score': 'mean'
        }).round(3)
        avg_ranks = avg_ranks.sort_values('rank')
        avg_ranks['final_rank'] = range(1, len(avg_ranks) + 1)
        
        return avg_ranks
    
    def statistical_test(
        self,
        measure: str,
        test: str = 'friedman',
        posthoc: str = 'nemenyi',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical tests to compare learners.
        
        Parameters
        ----------
        measure : str
            Measure to test
        test : str
            Statistical test ('friedman', 'kruskal', 'anova')
        posthoc : str
            Post-hoc test ('nemenyi', 'dunn', 'tukey')
        alpha : float
            Significance level
            
        Returns
        -------
        dict
            Test results
        """
        from scipy import stats
        
        # Get scores by learner and task
        df = self.to_dataframe()
        df = df[df['measure'] == measure]
        
        if df.empty:
            return {'error': 'No data for measure'}
        
        # Prepare data for testing
        learners = df['learner'].unique()
        tasks = df['task'].unique()
        
        # Create matrix: rows=tasks, cols=learners
        score_matrix = []
        for task in tasks:
            task_scores = []
            for learner in learners:
                scores = df[(df['task'] == task) & (df['learner'] == learner)]['score']
                if len(scores) > 0:
                    task_scores.append(scores.mean())
                else:
                    task_scores.append(np.nan)
            score_matrix.append(task_scores)
        
        score_matrix = np.array(score_matrix)
        
        # Perform main test
        results = {'test': test, 'alpha': alpha}
        
        if test == 'friedman':
            # Friedman test for repeated measures
            stat, p_value = stats.friedmanchisquare(*score_matrix.T)
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < alpha
            
        elif test == 'kruskal':
            # Kruskal-Wallis test
            groups = [score_matrix[:, i] for i in range(len(learners))]
            stat, p_value = stats.kruskal(*groups)
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < alpha
            
        elif test == 'anova':
            # One-way ANOVA
            groups = [score_matrix[:, i] for i in range(len(learners))]
            stat, p_value = stats.f_oneway(*groups)
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < alpha
        
        # Perform post-hoc test if significant
        if results.get('significant', False) and posthoc:
            results['posthoc'] = self._posthoc_test(
                score_matrix, learners, posthoc, alpha
            )
        
        return results
    
    def _posthoc_test(
        self,
        score_matrix: np.ndarray,
        learners: List[str],
        method: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Perform post-hoc pairwise comparisons."""
        try:
            import scikit_posthocs as sp
            
            if method == 'nemenyi':
                # Nemenyi test
                p_matrix = sp.posthoc_nemenyi_friedman(score_matrix.T)
            elif method == 'dunn':
                # Dunn test
                p_matrix = sp.posthoc_dunn(score_matrix.T)
            else:
                return {'error': f'Unknown posthoc method: {method}'}
            
            # Convert to dictionary
            results = {
                'method': method,
                'p_values': p_matrix.to_dict(),
                'significant_pairs': []
            }
            
            # Find significant pairs
            for i in range(len(learners)):
                for j in range(i + 1, len(learners)):
                    if p_matrix.iloc[i, j] < alpha:
                        results['significant_pairs'].append(
                            (learners[i], learners[j], p_matrix.iloc[i, j])
                        )
            
            return results
            
        except ImportError:
            return {'error': 'scikit-posthocs not installed'}
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 60)
        lines.append("BENCHMARK RESULTS SUMMARY")
        lines.append("=" * 60)
        
        # Basic info
        n_tasks = len(self.design.tasks)
        n_learners = len(self.design.learners)
        n_resamplings = len(self.design.resamplings)
        n_measures = len(self.design.measures)
        
        lines.append(f"Tasks: {n_tasks}")
        lines.append(f"Learners: {n_learners}")
        lines.append(f"Resamplings: {n_resamplings}")
        lines.append(f"Measures: {n_measures}")
        lines.append(f"Total scores: {len(self.scores)}")
        lines.append(f"Errors: {len(self.errors)}")
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            lines.append(f"Duration: {duration:.2f} seconds")
        
        lines.append("")
        
        # Performance summary
        for measure in self.design.measures:
            lines.append(f"Measure: {measure.id}")
            rankings = self.rank_learners(measure.id, minimize=measure.minimize)
            
            if not rankings.empty:
                lines.append("  Learner Rankings:")
                for learner, row in rankings.iterrows():
                    lines.append(f"    {row['final_rank']}. {learner}: "
                               f"avg_score={row['score']:.4f}, "
                               f"avg_rank={row['rank']:.2f}")
            lines.append("")
        
        # Errors
        if self.errors:
            lines.append("Errors encountered:")
            for (task, learner, resampling), error in self.errors.items():
                lines.append(f"  {task}/{learner}/{resampling}: {str(error)}")
        
        return "\n".join(lines)
    
    def save(self, path: str) -> None:
        """Save results to file."""
        # Convert to serializable format
        data = {
            'design': {
                'tasks': [t.id for t in self.design.tasks],
                'learners': [l.id for l in self.design.learners],
                'resamplings': [r.id for r in self.design.resamplings],
                'measures': [m.id for m in self.design.measures]
            },
            'scores': self.to_dataframe().to_dict('records'),
            'errors': {str(k): str(v) for k, v in self.errors.items()},
            'metadata': self.metadata
        }
        
        # Save based on extension
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.endswith('.csv'):
            self.to_dataframe().to_csv(path, index=False)
        else:
            # Default to pickle
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)


def benchmark_grid(
    tasks: Union[Task, List[Task]],
    learners: Union[Learner, List[Learner]],
    resamplings: Union[Resampling, List[Resampling]],
    measures: Union[Measure, List[Measure]],
    paired: bool = True
) -> BenchmarkDesign:
    """
    Create benchmark design from grid specification.
    
    Parameters
    ----------
    tasks : Task or List[Task]
        Tasks to evaluate
    learners : Learner or List[Learner]
        Learners to compare
    resamplings : Resampling or List[Resampling]
        Resampling strategies
    measures : Measure or List[Measure]
        Performance measures
    paired : bool
        Use paired resampling
        
    Returns
    -------
    BenchmarkDesign
        Benchmark design object
        
    Examples
    --------
    >>> design = benchmark_grid(
    ...     tasks=task,
    ...     learners=[learner1, learner2, learner3],
    ...     resamplings=cv_5fold,
    ...     measures=['accuracy', 'auc']
    ... )
    """
    # Ensure lists
    if not isinstance(tasks, list):
        tasks = [tasks]
    if not isinstance(learners, list):
        learners = [learners]
    if not isinstance(resamplings, list):
        resamplings = [resamplings]
    if not isinstance(measures, list):
        measures = [measures]
    
    # Convert string measures to Measure objects
    from .measures import create_measure
    measures = [
        create_measure(m) if isinstance(m, str) else m
        for m in measures
    ]
    
    return BenchmarkDesign(
        tasks=tasks,
        learners=learners,
        resamplings=resamplings,
        measures=measures,
        paired=paired
    )


def benchmark(
    design: BenchmarkDesign,
    store_models: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
    verbose: int = 1,
    callbacks: Optional[List[Callback]] = None
) -> BenchmarkResult:
    """
    Execute benchmark experiments.
    
    Parameters
    ----------
    design : BenchmarkDesign
        Benchmark design
    store_models : bool
        Whether to store trained models
    parallel : bool
        Use parallel execution
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    callbacks : List[Callback], optional
        Callbacks for monitoring
        
    Returns
    -------
    BenchmarkResult
        Benchmark results
        
    Examples
    --------
    >>> from mlpy import benchmark, benchmark_grid
    >>> 
    >>> # Create design
    >>> design = benchmark_grid(
    ...     tasks=[task1, task2],
    ...     learners=[rf, xgb, lgb],
    ...     resamplings=cv_5fold,
    ...     measures=['accuracy', 'auc', 'f1']
    ... )
    >>> 
    >>> # Run benchmark
    >>> results = benchmark(design, parallel=True)
    >>> 
    >>> # Analyze results
    >>> print(results.summary())
    >>> rankings = results.rank_learners('accuracy')
    >>> stats = results.statistical_test('accuracy')
    """
    result = BenchmarkResult(design)
    result.start_time = time.time()
    
    if verbose:
        print(f"Starting benchmark with {design.n_experiments} experiments")
    
    # Get backend for parallel execution
    if parallel:
        from .parallel.utils import _create_backend
        try:
            backend = _create_backend('joblib', n_jobs=n_jobs)
        except ValueError:
            # Fallback to threading if joblib not available
            backend = _create_backend('threading', n_jobs=n_jobs)
    else:
        from .parallel.utils import _create_backend
        backend = _create_backend('sequential')
    
    # Generate experiment grid
    experiments = design.grid()
    
    # Execute experiments
    def run_experiment(task, learner, resampling):
        """Run single experiment."""
        scores_collected = []
        errors_collected = []
        
        try:
            # Instantiate resampling if needed
            if not resampling.is_instantiated:
                resampling.instantiate(task)
            
            # Run resampling
            res = resample(
                task=task,
                learner=learner,
                resampling=resampling,
                measures=design.measures,
                store_models=store_models
            )
            
            # Extract scores
            for measure in design.measures:
                for i, score in enumerate(res.scores.get(measure.id, [])):
                    benchmark_score = BenchmarkScore(
                        task_id=task.id,
                        learner_id=learner.id,
                        resampling_id=resampling.id,
                        measure_id=measure.id,
                        iteration=i,
                        score=score,
                        train_time=res.train_times[i] if i < len(res.train_times) else 0,
                        predict_time=res.predict_times[i] if i < len(res.predict_times) else 0
                    )
                    scores_collected.append(benchmark_score)
            
            if verbose > 1:
                print(f"  Completed: {task.id}/{learner.id}/{resampling.id}")
                
        except Exception as e:
            errors_collected.append((task.id, learner.id, resampling.id, e))
            if verbose:
                print(f"  Error in {task.id}/{learner.id}/{resampling.id}: {e}")
        
        return scores_collected, errors_collected
    
    # Run experiments
    if parallel and n_jobs != 1:
        # Parallel execution
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_experiment)(task, learner, resampling)
                for task, learner, resampling in experiments
            )
            
            # Collect results
            for scores_list, errors_list in results:
                for score in scores_list:
                    result.add_score(score)
                for task_id, learner_id, resampling_id, error in errors_list:
                    result.add_error(task_id, learner_id, resampling_id, error)
                    
        except ImportError:
            # Fallback to sequential if joblib not available
            for task, learner, resampling in experiments:
                scores_list, errors_list = run_experiment(task, learner, resampling)
                for score in scores_list:
                    result.add_score(score)
                for task_id, learner_id, resampling_id, error in errors_list:
                    result.add_error(task_id, learner_id, resampling_id, error)
    else:
        # Sequential execution
        for task, learner, resampling in experiments:
            scores_list, errors_list = run_experiment(task, learner, resampling)
            for score in scores_list:
                result.add_score(score)
            for task_id, learner_id, resampling_id, error in errors_list:
                result.add_error(task_id, learner_id, resampling_id, error)
    
    result.end_time = time.time()
    
    if verbose:
        duration = result.end_time - result.start_time
        print(f"Benchmark completed in {duration:.2f} seconds")
        print(f"Total scores: {len(result.scores)}")
        print(f"Errors: {len(result.errors)}")
    
    return result


def compare_learners(
    task: Task,
    learners: List[Learner],
    cv_folds: int = 5,
    measures: Optional[List[str]] = None,
    test: str = 'friedman',
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Quick comparison of multiple learners on a task.
    
    Convenience function for rapid learner comparison with statistical testing.
    
    Parameters
    ----------
    task : Task
        Task to evaluate on
    learners : List[Learner]
        Learners to compare
    cv_folds : int
        Number of CV folds
    measures : List[str], optional
        Measures to use (default: task-appropriate)
    test : str
        Statistical test to use
    show_plot : bool
        Whether to show comparison plot
        
    Returns
    -------
    dict
        Comparison results
        
    Examples
    --------
    >>> from mlpy import compare_learners
    >>> from mlpy.learners import learner_xgboost, learner_lightgbm, learner_ranger
    >>> 
    >>> results = compare_learners(
    ...     task=task,
    ...     learners=[
    ...         learner_xgboost(),
    ...         learner_lightgbm(),
    ...         learner_ranger()
    ...     ],
    ...     cv_folds=5,
    ...     measures=['accuracy', 'auc']
    ... )
    >>> 
    >>> print(results['rankings'])
    >>> print(results['statistical_test'])
    """
    from .resamplings import ResamplingCV
    
    # Default measures based on task type
    if measures is None:
        if task.task_type == 'classif':
            measures = ['accuracy', 'auc']
        else:
            measures = ['rmse', 'mae']
    
    # Create benchmark design
    design = benchmark_grid(
        tasks=task,
        learners=learners,
        resamplings=ResamplingCV(folds=cv_folds),
        measures=measures
    )
    
    # Run benchmark
    result = benchmark(design, verbose=0)
    
    # Prepare output
    output = {
        'design': design,
        'result': result,
        'scores': result.to_dataframe(),
        'rankings': {},
        'statistical_tests': {}
    }
    
    # Rank learners for each measure
    for measure in measures:
        output['rankings'][measure] = result.rank_learners(measure)
        output['statistical_tests'][measure] = result.statistical_test(measure, test=test)
    
    # Create plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(measures), figsize=(6*len(measures), 5))
            if len(measures) == 1:
                axes = [axes]
            
            for ax, measure in zip(axes, measures):
                # Box plot of scores
                df = result.to_dataframe()
                df_measure = df[df['measure'] == measure]
                
                data_to_plot = []
                labels = []
                for learner in learners:
                    scores = df_measure[df_measure['learner'] == learner.id]['score']
                    if len(scores) > 0:
                        data_to_plot.append(scores)
                        labels.append(learner.id)
                
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(f'{measure} Comparison')
                ax.set_ylabel('Score')
                ax.set_xlabel('Learner')
                ax.grid(True, alpha=0.3)
                
                # Add significance markers if available
                if output['statistical_tests'][measure].get('significant'):
                    ax.text(0.02, 0.98, 'p < 0.05', transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='yellow', alpha=0.5))
            
            plt.suptitle(f'Learner Comparison on {task.id}')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
    
    # Print summary
    print("\nLearner Comparison Summary")
    print("=" * 50)
    for measure in measures:
        print(f"\n{measure.upper()}:")
        rankings = output['rankings'][measure]
        for learner, row in rankings.iterrows():
            print(f"  {row['final_rank']}. {learner}: {row['score']:.4f}")
        
        stat_test = output['statistical_tests'][measure]
        if stat_test.get('significant'):
            print(f"  Statistical test: p={stat_test['p_value']:.4f} (significant)")
        else:
            print(f"  Statistical test: p={stat_test.get('p_value', 'N/A'):.4f} (not significant)")
    
    return output