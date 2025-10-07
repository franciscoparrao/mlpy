"""
Example of Multi-Model Benchmarking in MLPY.

This example demonstrates the advanced benchmarking system for comparing
multiple learners across tasks with statistical analysis and visualization.
"""

import numpy as np
import pandas as pd
import warnings
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import (
    learner_xgboost,
    learner_lightgbm,
    learner_catboost
)
from mlpy.learners.baseline import LearnerBaseline, LearnerClassifFeatureless, LearnerRegrFeatureless
from mlpy.learners.sklearn_wrapper import learner_sklearn
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.measures import create_measure
from mlpy.benchmark_advanced import (
    benchmark,
    benchmark_grid,
    compare_learners,
    BenchmarkDesign,
    BenchmarkResult
)


def create_sample_tasks():
    """Create sample tasks for benchmarking."""
    np.random.seed(42)
    
    # Binary classification task
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5) > 0
    
    df_binary = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_binary['target'] = y.astype(int)
    
    task_binary = TaskClassif(
        data=df_binary,
        target='target',
        id='binary_classification'
    )
    
    # Multi-class classification task
    y_multi = np.random.choice(['A', 'B', 'C'], n_samples)
    df_multi = df_binary.copy()
    df_multi['target'] = y_multi
    
    task_multi = TaskClassif(
        data=df_multi,
        target='target',
        id='multiclass_classification'
    )
    
    # Regression task
    y_reg = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(n_samples)
    df_reg = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_reg['target'] = y_reg
    
    task_reg = TaskRegr(
        data=df_reg,
        target='target',
        id='regression'
    )
    
    return task_binary, task_multi, task_reg


def example_basic_benchmark():
    """Basic benchmark example."""
    print("=" * 60)
    print("BASIC BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Create tasks
    task_binary, _, _ = create_sample_tasks()
    
    # Create learners
    learners = [
        LearnerBaseline(id='baseline'),
        LearnerClassifFeatureless(method='mode', id='featureless_mode'),
        LearnerClassifFeatureless(method='weighted', id='featureless_weighted')
    ]
    
    # Create benchmark design
    design = benchmark_grid(
        tasks=task_binary,
        learners=learners,
        resamplings=ResamplingCV(folds=3),
        measures=['accuracy', 'auc', 'f1']
    )
    
    print(f"\nBenchmark Design:")
    print(f"  Tasks: {len(design.tasks)}")
    print(f"  Learners: {len(design.learners)}")
    print(f"  Resamplings: {len(design.resamplings)}")
    print(f"  Measures: {len(design.measures)}")
    print(f"  Total experiments: {design.n_experiments}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    result = benchmark(design, parallel=False, verbose=0)
    
    # Show results
    print("\n" + result.summary())
    
    return result


def example_multi_task_benchmark():
    """Benchmark across multiple tasks."""
    print("\n" + "=" * 60)
    print("MULTI-TASK BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Create all tasks
    tasks = list(create_sample_tasks())
    
    # Create learners with different configurations
    learners = [
        LearnerBaseline(id='baseline'),
        LearnerClassifFeatureless(method='mode', id='featureless_mode'),
        LearnerClassifFeatureless(method='weighted', id='featureless_weighted')
    ]
    
    # Create design with multiple resamplings
    design = benchmark_grid(
        tasks=tasks,
        learners=learners,
        resamplings=[
            ResamplingCV(folds=5, id='cv5'),
            ResamplingHoldout(ratio=0.8, id='holdout')
        ],
        measures=['accuracy', 'auc', 'rmse', 'mae']
    )
    
    print(f"\nMulti-Task Design:")
    print(f"  Tasks: {[t.id for t in design.tasks]}")
    print(f"  Learners: {[l.id for l in design.learners]}")
    print(f"  Resamplings: {[r.id for r in design.resamplings]}")
    print(f"  Total experiments: {design.n_experiments}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    result = benchmark(design, parallel=False, verbose=0)
    
    # Aggregate results by task
    print("\nResults by Task:")
    for task in design.tasks:
        print(f"\n{task.id}:")
        
        # Get appropriate measure
        if task.task_type == 'classif':
            measure = 'accuracy'
        else:
            measure = 'rmse'
        
        # Get rankings for this task
        df = result.to_dataframe()
        task_df = df[(df['task'] == task.id) & (df['measure'] == measure)]
        
        if not task_df.empty:
            avg_scores = task_df.groupby('learner')['score'].mean()
            print(f"  Average {measure}:")
            for learner, score in avg_scores.items():
                print(f"    {learner}: {score:.4f}")
    
    return result


def example_statistical_comparison():
    """Statistical comparison of learners."""
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create task
    task, _, _ = create_sample_tasks()
    
    # Create learners
    learners = [
        LearnerBaseline(id='Baseline'),
        LearnerClassifFeatureless(method='mode', id='Mode'),
        LearnerClassifFeatureless(method='sample', id='Sample'),
        LearnerClassifFeatureless(method='weighted', id='Weighted')
    ]
    
    # Use compare_learners for quick comparison
    print("\nComparing learners with statistical tests...")
    results = compare_learners(
        task=task,
        learners=learners,
        cv_folds=5,
        measures=['accuracy', 'auc'],
        test='friedman',
        show_plot=False  # Set to True if matplotlib available
    )
    
    # Show statistical test results
    print("\nStatistical Test Results:")
    for measure, test_result in results['statistical_tests'].items():
        print(f"\n{measure}:")
        print(f"  Test: {test_result.get('test', 'N/A')}")
        print(f"  P-value: {test_result.get('p_value', 'N/A'):.4f}")
        print(f"  Significant: {test_result.get('significant', False)}")
        
        if test_result.get('significant') and 'posthoc' in test_result:
            posthoc = test_result['posthoc']
            if 'significant_pairs' in posthoc:
                print(f"  Significant pairs:")
                for pair in posthoc['significant_pairs']:
                    print(f"    {pair[0]} vs {pair[1]}: p={pair[2]:.4f}")
    
    return results


def example_custom_aggregation():
    """Custom aggregation and analysis of benchmark results."""
    print("\n" + "=" * 60)
    print("CUSTOM AGGREGATION EXAMPLE")
    print("=" * 60)
    
    # Create task and learners
    task, _, _ = create_sample_tasks()
    
    learners = [
        LearnerClassifFeatureless(method='mode', id='mode'),
        LearnerClassifFeatureless(method='sample', id='sample'),
        LearnerClassifFeatureless(method='weighted', id='weighted')
    ]
    
    # Benchmark
    design = benchmark_grid(
        tasks=task,
        learners=learners,
        resamplings=ResamplingCV(folds=5),
        measures=['accuracy', 'auc', 'f1']
    )
    
    result = benchmark(design, verbose=0)
    
    # Custom aggregations
    print("\nCustom Aggregations:")
    
    # 1. Mean and std by learner
    print("\nMean ± Std by Learner:")
    df = result.to_dataframe()
    for learner in design.learners:
        learner_df = df[df['learner'] == learner.id]
        for measure in ['accuracy', 'auc', 'f1']:
            measure_df = learner_df[learner_df['measure'] == measure]
            if not measure_df.empty:
                mean_score = measure_df['score'].mean()
                std_score = measure_df['score'].std()
                print(f"  {learner.id} - {measure}: {mean_score:.3f} ± {std_score:.3f}")
    
    # 2. Training time analysis
    print("\nTraining Time Analysis:")
    time_df = df.groupby('learner')['train_time'].agg(['mean', 'sum'])
    print(time_df)
    
    # 3. Best configuration per measure
    print("\nBest Configuration per Measure:")
    for measure in ['accuracy', 'auc', 'f1']:
        rankings = result.rank_learners(measure)
        if not rankings.empty:
            best_learner = rankings.index[0]
            best_score = rankings.iloc[0]['score']
            print(f"  {measure}: {best_learner} ({best_score:.4f})")
    
    return result


def example_export_results():
    """Export benchmark results to different formats."""
    print("\n" + "=" * 60)
    print("EXPORT RESULTS EXAMPLE")
    print("=" * 60)
    
    # Run simple benchmark
    task, _, _ = create_sample_tasks()
    
    design = benchmark_grid(
        tasks=task,
        learners=[
            LearnerBaseline(id='baseline'),
            LearnerClassifFeatureless(method='mode', id='mode')
        ],
        resamplings=ResamplingCV(folds=3),
        measures=['accuracy', 'auc']
    )
    
    result = benchmark(design, verbose=0)
    
    # Export to different formats
    print("\nExporting results...")
    
    # 1. DataFrame (wide format)
    df_wide = result.to_dataframe(wide=True)
    print("\nWide format DataFrame:")
    print(df_wide.head())
    
    # 2. Aggregated results
    agg_df = result.aggregate(
        measure='accuracy',
        group_by=['learner'],
        aggr_func='mean'
    )
    print("\nAggregated results:")
    print(agg_df)
    
    # 3. Rankings
    rankings = result.rank_learners('accuracy')
    print("\nLearner rankings:")
    print(rankings)
    
    # 4. Save to files (uncomment to actually save)
    # result.save('benchmark_results.json')
    # result.save('benchmark_results.csv')
    # result.save('benchmark_results.pkl')
    
    print("\nResults can be saved to JSON, CSV, or pickle format")
    
    return result


def example_parallel_benchmark():
    """Parallel execution of benchmark."""
    print("\n" + "=" * 60)
    print("PARALLEL BENCHMARK EXAMPLE")
    print("=" * 60)
    
    # Create larger task for parallel benefit
    np.random.seed(42)
    n_samples = 5000
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['target'] = y
    
    task = TaskClassif(data=df, target='target', id='large_task')
    
    # Multiple learner configurations
    learners = [
        LearnerBaseline(id='baseline1'),
        LearnerBaseline(id='baseline2'),
        LearnerClassifFeatureless(method='mode', id='mode'),
        LearnerClassifFeatureless(method='sample', id='sample'),
        LearnerClassifFeatureless(method='weighted', id='weighted1'),
        LearnerClassifFeatureless(method='weighted', id='weighted2')
    ]
    
    design = benchmark_grid(
        tasks=task,
        learners=learners,
        resamplings=ResamplingCV(folds=3),
        measures=['accuracy']
    )
    
    print(f"\nLarge-scale benchmark:")
    print(f"  Task size: {n_samples} x {n_features}")
    print(f"  Number of learners: {len(learners)}")
    print(f"  Total experiments: {design.n_experiments}")
    
    # Run parallel
    import time
    
    print("\nRunning parallel benchmark...")
    start = time.time()
    result_parallel = benchmark(design, parallel=True, n_jobs=2, verbose=0)
    parallel_time = time.time() - start
    
    print(f"Parallel execution time: {parallel_time:.2f} seconds")
    print(f"Scores collected: {len(result_parallel.scores)}")
    
    # Show best configuration
    rankings = result_parallel.rank_learners('accuracy')
    if not rankings.empty:
        print(f"\nBest configuration: {rankings.index[0]}")
        print(f"Best score: {rankings.iloc[0]['score']:.4f}")
    
    return result_parallel


def main():
    """Run all benchmark examples."""
    print("MLPY MULTI-MODEL BENCHMARKING EXAMPLES")
    print("=" * 60)
    
    # Run examples
    result1 = example_basic_benchmark()
    result2 = example_multi_task_benchmark()
    result3 = example_statistical_comparison()
    result4 = example_custom_aggregation()
    result5 = example_export_results()
    
    # Parallel example (optional - may take longer)
    # result6 = example_parallel_benchmark()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 60)
    
    print("\nKey Features Demonstrated:")
    print("1. Basic benchmarking with multiple learners")
    print("2. Multi-task benchmarking across different problem types")
    print("3. Statistical comparison with significance tests")
    print("4. Custom aggregation and analysis")
    print("5. Export results to various formats")
    print("6. Parallel execution for large-scale benchmarks")
    
    print("\nMLPY's benchmarking system provides:")
    print("- Flexible experiment design")
    print("- Statistical testing (Friedman, Kruskal-Wallis, ANOVA)")
    print("- Post-hoc tests (Nemenyi, Dunn)")
    print("- Multiple aggregation strategies")
    print("- Comprehensive result analysis")
    print("- Export to multiple formats")
    
    return {
        'basic': result1,
        'multi_task': result2,
        'statistical': result3,
        'custom': result4,
        'export': result5
    }


if __name__ == "__main__":
    results = main()