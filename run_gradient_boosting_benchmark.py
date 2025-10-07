"""
Script para ejecutar benchmark comparativo de Gradient Boosting en MLPY.
Compara XGBoost, LightGBM y CatBoost en diferentes escenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Verificar disponibilidad de librerías
libraries_status = {
    'xgboost': False,
    'lightgbm': False,
    'catboost': False
}

try:
    import xgboost
    libraries_status['xgboost'] = True
    print("[OK] XGBoost disponible")
except ImportError:
    print("[X] XGBoost no disponible")

try:
    import lightgbm
    libraries_status['lightgbm'] = True
    print("[OK] LightGBM disponible")
except ImportError:
    print("[X] LightGBM no disponible")

try:
    import catboost
    libraries_status['catboost'] = True
    print("[OK] CatBoost disponible")
except ImportError:
    print("[X] CatBoost no disponible")

# Importar MLPY
from mlpy.data.data import Data
from mlpy.tasks.task import TaskClassif, TaskRegr
from mlpy.measures.classification import MeasureClassifAccuracy, MeasureClassifAUC
from mlpy.measures.regression import MeasureRegrRMSE, MeasureRegrMAE

# Importar gradient boosting
from mlpy.learners.gradient_boosting import GradientBoostingLearner, GBOptimizationProfile

# Importar wrappers específicos si están disponibles
if libraries_status['xgboost']:
    from mlpy.learners.xgboost_wrapper import LearnerXGBoostClassif, LearnerXGBoostRegr
    
if libraries_status['lightgbm']:
    from mlpy.learners.lightgbm_wrapper import LearnerLightGBMClassif, LearnerLightGBMRegr
    
if libraries_status['catboost']:
    from mlpy.learners.catboost_wrapper import LearnerCatBoostClassif, LearnerCatBoostRegr


def create_test_datasets():
    """Crea diferentes tipos de datasets para el benchmark."""
    
    datasets = []
    
    # 1. Dataset pequeño numérico
    print("\nCreando dataset pequeño numérico...")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    df1 = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
    df1['target'] = y
    datasets.append(('small_numeric', df1, 'classification'))
    
    # 2. Dataset mediano con categóricas
    print("Creando dataset mediano con categóricas...")
    n_samples = 5000
    np.random.seed(42)
    df2 = pd.DataFrame({
        'num_1': np.random.randn(n_samples),
        'num_2': np.random.randn(n_samples) * 10,
        'num_3': np.random.exponential(2, n_samples),
        'cat_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'cat_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'cat_3': np.random.choice(['Red', 'Green', 'Blue'], n_samples),
        'cat_4': pd.Categorical(np.random.choice(['Small', 'Medium', 'Large'], n_samples)),
    })
    # Target con lógica basada en features
    df2['target'] = (
        (df2['num_1'] > 0).astype(int) + 
        (df2['cat_1'] == 'A').astype(int) + 
        (df2['cat_2'] == 'X').astype(int)
    ) > 1
    df2['target'] = df2['target'].astype(int)
    datasets.append(('medium_categorical', df2, 'classification'))
    
    # 3. Dataset con valores faltantes
    print("Creando dataset con valores faltantes...")
    X, y = make_classification(n_samples=3000, n_features=30, random_state=42)
    df3 = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(30)])
    # Introducir 15% de valores faltantes
    mask = np.random.random(df3.shape) < 0.15
    df3[mask] = np.nan
    df3['target'] = y
    datasets.append(('missing_values', df3, 'classification'))
    
    # 4. Dataset de regresión
    print("Creando dataset de regresión...")
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=4000,
        n_features=25,
        n_informative=20,
        noise=0.1,
        random_state=42
    )
    df4 = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(25)])
    df4['target'] = y
    datasets.append(('regression', df4, 'regression'))
    
    # 5. Dataset grande
    print("Creando dataset grande...")
    X, y = make_classification(
        n_samples=20000,
        n_features=50,
        n_informative=35,
        n_classes=3,
        random_state=42
    )
    df5 = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(50)])
    df5['target'] = y
    datasets.append(('large_multiclass', df5, 'classification'))
    
    return datasets


def benchmark_single_dataset(name, df, task_type, n_estimators=100):
    """Ejecuta benchmark en un solo dataset."""
    
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"Type: {task_type}")
    
    # Detectar features categóricas
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'target' in cat_features:
        cat_features.remove('target')
    
    print(f"Categorical features: {len(cat_features)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"{'='*60}")
    
    # Crear Data y Task de MLPY
    data = Data(
        id=f"data_{name}",
        backend=df,
        target_names=['target']
    )
    
    if task_type == 'classification':
        task = TaskClassif(
            id=f"task_{name}",
            backend=data,
            target='target'
        )
        measure = MeasureClassifAccuracy()
        metric_name = "Accuracy"
    else:
        task = TaskRegr(
            id=f"task_{name}",
            backend=data,
            target='target'
        )
        measure = MeasureRegrRMSE()
        metric_name = "RMSE"
    
    results = []
    
    # Benchmark XGBoost
    if libraries_status['xgboost']:
        print("\n→ Testing XGBoost...")
        try:
            if task_type == 'classification':
                learner = LearnerXGBoostClassif(n_estimators=n_estimators, max_depth=6)
            else:
                learner = LearnerXGBoostRegr(n_estimators=n_estimators, max_depth=6)
            
            # Entrenar
            start_time = time.time()
            learner.train(task)
            train_time = time.time() - start_time
            
            # Predecir
            start_time = time.time()
            predictions = learner.predict(task)
            predict_time = time.time() - start_time
            
            # Evaluar
            score = predictions.score(measure)
            
            results.append({
                'Backend': 'XGBoost',
                'Train Time (s)': round(train_time, 3),
                'Predict Time (s)': round(predict_time, 3),
                metric_name: round(score, 4)
            })
            
            print(f"  [OK] Train: {train_time:.3f}s | Predict: {predict_time:.3f}s | {metric_name}: {score:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)[:50]}")
            results.append({
                'Backend': 'XGBoost',
                'Train Time (s)': None,
                'Predict Time (s)': None,
                metric_name: None
            })
    
    # Benchmark LightGBM
    if libraries_status['lightgbm']:
        print("\n→ Testing LightGBM...")
        try:
            if task_type == 'classification':
                learner = LearnerLightGBMClassif(
                    n_estimators=n_estimators,
                    num_leaves=31,
                    categorical_features='auto' if cat_features else None
                )
            else:
                learner = LearnerLightGBMRegr(
                    n_estimators=n_estimators,
                    num_leaves=31
                )
            
            # Entrenar
            start_time = time.time()
            learner.train(task)
            train_time = time.time() - start_time
            
            # Predecir
            start_time = time.time()
            predictions = learner.predict(task)
            predict_time = time.time() - start_time
            
            # Evaluar
            score = predictions.score(measure)
            
            results.append({
                'Backend': 'LightGBM',
                'Train Time (s)': round(train_time, 3),
                'Predict Time (s)': round(predict_time, 3),
                metric_name: round(score, 4)
            })
            
            print(f"  [OK] Train: {train_time:.3f}s | Predict: {predict_time:.3f}s | {metric_name}: {score:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)[:50]}")
            results.append({
                'Backend': 'LightGBM',
                'Train Time (s)': None,
                'Predict Time (s)': None,
                metric_name: None
            })
    
    # Benchmark CatBoost
    if libraries_status['catboost']:
        print("\n→ Testing CatBoost...")
        try:
            if task_type == 'classification':
                learner = LearnerCatBoostClassif(
                    n_estimators=n_estimators,
                    max_depth=6,
                    cat_features='auto' if cat_features else None,
                    verbose=False
                )
            else:
                learner = LearnerCatBoostRegr(
                    n_estimators=n_estimators,
                    max_depth=6,
                    verbose=False
                )
            
            # Entrenar
            start_time = time.time()
            learner.train(task)
            train_time = time.time() - start_time
            
            # Predecir
            start_time = time.time()
            predictions = learner.predict(task)
            predict_time = time.time() - start_time
            
            # Evaluar
            score = predictions.score(measure)
            
            results.append({
                'Backend': 'CatBoost',
                'Train Time (s)': round(train_time, 3),
                'Predict Time (s)': round(predict_time, 3),
                metric_name: round(score, 4)
            })
            
            print(f"  [OK] Train: {train_time:.3f}s | Predict: {predict_time:.3f}s | {metric_name}: {score:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)[:50]}")
            results.append({
                'Backend': 'CatBoost',
                'Train Time (s)': None,
                'Predict Time (s)': None,
                metric_name: None
            })
    
    # Test interfaz unificada con auto-selección
    print("\n→ Testing Unified Interface (Auto-selection)...")
    try:
        learner = GradientBoostingLearner(
            backend='auto',
            n_estimators=n_estimators,
            auto_optimize=True,
            verbose=False
        )
        
        # Entrenar
        start_time = time.time()
        learner.train(task)
        train_time = time.time() - start_time
        
        # Ver qué backend se seleccionó
        backend_info = learner.get_backend_info()
        selected = backend_info['selected_backend']
        
        # Predecir
        start_time = time.time()
        predictions = learner.predict(task)
        predict_time = time.time() - start_time
        
        # Evaluar
        score = predictions.score(measure)
        
        results.append({
            'Backend': f'Auto ({selected})',
            'Train Time (s)': round(train_time, 3),
            'Predict Time (s)': round(predict_time, 3),
            metric_name: round(score, 4)
        })
        
        print(f"  [OK] Selected: {selected}")
        print(f"  [OK] Train: {train_time:.3f}s | Predict: {predict_time:.3f}s | {metric_name}: {score:.4f}")
        
    except Exception as e:
        print(f"  [ERROR] Error: {str(e)[:50]}")
        results.append({
            'Backend': 'Auto',
            'Train Time (s)': None,
            'Predict Time (s)': None,
            metric_name: None
        })
    
    return pd.DataFrame(results)


def main():
    """Función principal del benchmark."""
    
    print("\n" + "="*80)
    print(" MLPY GRADIENT BOOSTING BENCHMARK ".center(80))
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Libraries available: {[k for k, v in libraries_status.items() if v]}")
    
    # Crear datasets de prueba
    print("\n" + "-"*80)
    print("Creating test datasets...")
    print("-"*80)
    datasets = create_test_datasets()
    
    # Ejecutar benchmarks
    all_results = []
    
    for name, df, task_type in datasets:
        result_df = benchmark_single_dataset(name, df, task_type, n_estimators=100)
        result_df['Dataset'] = name
        all_results.append(result_df)
    
    # Combinar resultados
    print("\n" + "="*80)
    print(" FINAL RESULTS ".center(80))
    print("="*80)
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Reorganizar columnas
    cols = ['Dataset', 'Backend', 'Train Time (s)', 'Predict Time (s)']
    metric_cols = [col for col in final_df.columns if col not in cols]
    final_df = final_df[cols + metric_cols]
    
    print("\n", final_df.to_string(index=False))
    
    # Calcular estadísticas agregadas
    print("\n" + "="*80)
    print(" AGGREGATE STATISTICS ".center(80))
    print("="*80)
    
    # Por backend
    backend_stats = final_df.groupby('Backend').agg({
        'Train Time (s)': 'mean',
        'Predict Time (s)': 'mean'
    }).round(3)
    
    print("\nAverage Times by Backend:")
    print(backend_stats.to_string())
    
    # Mejor rendimiento por dataset
    print("\n" + "-"*40)
    print("Best Performer by Dataset:")
    print("-"*40)
    
    for dataset in final_df['Dataset'].unique():
        dataset_df = final_df[final_df['Dataset'] == dataset].copy()
        
        # Encontrar el más rápido
        fastest = dataset_df.loc[dataset_df['Train Time (s)'].idxmin(), 'Backend'] if dataset_df['Train Time (s)'].notna().any() else 'N/A'
        
        # Encontrar el más preciso (para clasificación es mayor mejor, para regresión es menor mejor)
        metric_col = [col for col in dataset_df.columns if col not in ['Dataset', 'Backend', 'Train Time (s)', 'Predict Time (s)']][0]
        
        if 'RMSE' in metric_col:  # Para regresión, menor es mejor
            best_score_idx = dataset_df[metric_col].idxmin() if dataset_df[metric_col].notna().any() else None
        else:  # Para clasificación, mayor es mejor
            best_score_idx = dataset_df[metric_col].idxmax() if dataset_df[metric_col].notna().any() else None
        
        best_accuracy = dataset_df.loc[best_score_idx, 'Backend'] if best_score_idx is not None else 'N/A'
        
        print(f"\n{dataset}:")
        print(f"  Fastest: {fastest}")
        print(f"  Best {metric_col}: {best_accuracy}")
    
    # Guardar resultados
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n[OK] Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE ".center(80))
    print("="*80)
    
    # Resumen final
    print("\nSummary:")
    available = [k for k, v in libraries_status.items() if v]
    if len(available) == 3:
        print("[OK] All three gradient boosting libraries were tested successfully")
    else:
        print(f"[WARNING] Only {len(available)} libraries were available: {available}")
    
    print(f"[OK] Tested on {len(datasets)} different datasets")
    print(f"[OK] Total benchmark runs: {len(final_df)}")
    
    return final_df


if __name__ == "__main__":
    results = main()