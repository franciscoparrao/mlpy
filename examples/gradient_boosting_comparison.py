"""
Ejemplo de comparación entre XGBoost, LightGBM y CatBoost usando MLPY.

Este script demuestra:
1. Uso directo de cada backend
2. Uso de la interfaz unificada con selección automática
3. Benchmark comparativo de rendimiento
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# MLPY imports
from mlpy.data import Data
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC, MeasureRegrRMSE
from mlpy.resampling import ResamplingCV

# Gradient Boosting imports
from mlpy.learners import learner_gradient_boosting
from mlpy.learners.gradient_boosting import GBOptimizationProfile

# Import backends directamente si están disponibles
try:
    from mlpy.learners.xgboost_wrapper import learner_xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost no disponible. Instalar con: pip install xgboost")

try:
    from mlpy.learners.lightgbm_wrapper import learner_lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM no disponible. Instalar con: pip install lightgbm")

try:
    from mlpy.learners.catboost_wrapper import learner_catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost no disponible. Instalar con: pip install catboost")


def create_sample_dataset(dataset_type='mixed'):
    """Crea un dataset de ejemplo con diferentes características.
    
    Parameters
    ----------
    dataset_type : str
        'numeric': Solo features numéricas
        'categorical': Muchas features categóricas
        'mixed': Mezcla de numéricas y categóricas
        'missing': Con valores faltantes
        'text': Con features de texto (solo CatBoost)
    """
    np.random.seed(42)
    n_samples = 5000
    
    if dataset_type == 'numeric':
        # Dataset puramente numérico - XGBoost debería ser rápido
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(20)])
        df['target'] = y
        
    elif dataset_type == 'categorical':
        # Dataset con muchas categóricas - CatBoost debería brillar
        data = {
            'cat_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'cat_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'cat_3': np.random.choice(['Red', 'Green', 'Blue', 'Yellow'], n_samples),
            'cat_4': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
            'cat_5': np.random.choice(['Type1', 'Type2', 'Type3', 'Type4', 'Type5'], n_samples),
            'num_1': np.random.randn(n_samples),
            'num_2': np.random.randn(n_samples) * 10,
            'num_3': np.random.exponential(2, n_samples),
        }
        df = pd.DataFrame(data)
        
        # Target basado en interacciones categóricas
        df['target'] = (
            (df['cat_1'] == 'A').astype(int) * 2 +
            (df['cat_2'] == 'X').astype(int) +
            (df['cat_3'] == 'Red').astype(int) +
            np.random.randn(n_samples) * 0.1
        )
        df['target'] = (df['target'] > df['target'].median()).astype(int)
        
    elif dataset_type == 'mixed':
        # Dataset mixto balanceado
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
            'education': np.random.choice(['HS', 'College', 'Graduate', 'PhD'], n_samples),
            'employment': np.random.choice(['Full', 'Part', 'Self', 'None'], n_samples),
            'num_accounts': np.random.poisson(3, n_samples),
            'balance': np.random.exponential(10000, n_samples),
        }
        df = pd.DataFrame(data)
        
        # Target con lógica compleja
        df['target'] = (
            (df['income'] > 60000).astype(int) +
            (df['credit_score'] > 700).astype(int) +
            (df['education'].isin(['Graduate', 'PhD'])).astype(int) +
            np.random.randn(n_samples) * 0.5
        )
        df['target'] = (df['target'] > 1.5).astype(int)
        
    elif dataset_type == 'missing':
        # Dataset con valores faltantes - LightGBM/CatBoost manejan nativamente
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
        
        # Introducir 20% de valores faltantes
        mask = np.random.random(df.shape) < 0.2
        df[mask] = np.nan
        df['target'] = y
        
    elif dataset_type == 'text':
        # Dataset con features de texto - Solo CatBoost puede manejar nativamente
        texts = [
            'excellent product highly recommend',
            'terrible experience would not buy again',
            'average quality nothing special',
            'amazing value for money',
            'poor quality disappointed'
        ]
        
        data = {
            'review': np.random.choice(texts, n_samples),
            'rating': np.random.randint(1, 6, n_samples),
            'length': np.random.randint(10, 500, n_samples),
            'category': np.random.choice(['Electronics', 'Books', 'Clothing'], n_samples)
        }
        df = pd.DataFrame(data)
        
        # Target basado en el sentimiento
        df['target'] = df['review'].apply(
            lambda x: 1 if any(word in x for word in ['excellent', 'amazing']) else 0
        )
        
    # Convertir categóricas a tipo category
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'target':
            df[col] = df[col].astype('category')
            
    return df


def compare_backends_directly():
    """Compara los backends directamente sin la interfaz unificada."""
    
    print("=" * 80)
    print("COMPARACIÓN DIRECTA DE BACKENDS")
    print("=" * 80)
    
    # Crear dataset mixto
    df = create_sample_dataset('mixed')
    
    # Crear Data y Task de MLPY
    data = Data(
        id="comparison_data",
        backend=df,
        target_names=['target']
    )
    
    task = TaskClassif(
        id="comparison_task",
        backend=data,
        target='target'
    )
    
    print(f"\nDataset: {len(df)} samples, {len(df.columns)-1} features")
    print(f"Features categóricas: {df.select_dtypes(include='category').columns.tolist()}")
    print(f"Balance de clases: {df['target'].value_counts().to_dict()}")
    
    results = {}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "-" * 40)
        print("XGBoost")
        print("-" * 40)
        
        learner_xgb = learner_xgboost(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic'
        )
        
        start = time.time()
        learner_xgb.train(task)
        train_time = time.time() - start
        
        predictions = learner_xgb.predict(task)
        accuracy = predictions.score(MeasureClassifAccuracy())
        
        results['XGBoost'] = {
            'train_time': train_time,
            'accuracy': accuracy
        }
        
        print(f"Tiempo de entrenamiento: {train_time:.3f}s")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Feature importance
        if hasattr(learner_xgb, 'feature_importances'):
            importance = learner_xgb.feature_importances
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 features:")
                for feat, imp in top_features:
                    print(f"  - {feat}: {imp:.3f}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n" + "-" * 40)
        print("LightGBM")
        print("-" * 40)
        
        learner_lgb = learner_lightgbm(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            objective='binary',
            categorical_features='auto'  # Detección automática
        )
        
        start = time.time()
        learner_lgb.train(task)
        train_time = time.time() - start
        
        predictions = learner_lgb.predict(task)
        accuracy = predictions.score(MeasureClassifAccuracy())
        
        results['LightGBM'] = {
            'train_time': train_time,
            'accuracy': accuracy
        }
        
        print(f"Tiempo de entrenamiento: {train_time:.3f}s")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Feature importance
        if hasattr(learner_lgb, 'feature_importances'):
            importance = learner_lgb.feature_importances
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 features:")
                for feat, imp in top_features:
                    print(f"  - {feat}: {imp:.3f}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\n" + "-" * 40)
        print("CatBoost")
        print("-" * 40)
        
        learner_cb = learner_catboost(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='Logloss',
            cat_features='auto',  # Detección automática
            verbose=False
        )
        
        start = time.time()
        learner_cb.train(task)
        train_time = time.time() - start
        
        predictions = learner_cb.predict(task)
        accuracy = predictions.score(MeasureClassifAccuracy())
        
        results['CatBoost'] = {
            'train_time': train_time,
            'accuracy': accuracy
        }
        
        print(f"Tiempo de entrenamiento: {train_time:.3f}s")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Feature importance
        if hasattr(learner_cb, 'feature_importances'):
            importance = learner_cb.feature_importances
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 features:")
                for feat, imp in top_features:
                    print(f"  - {feat}: {imp:.3f}")
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    if results:
        # Encontrar el mejor
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        fastest = min(results.items(), key=lambda x: x[1]['train_time'])
        
        print(f"\n✓ Mejor accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"✓ Más rápido: {fastest[0]} ({fastest[1]['train_time']:.3f}s)")
    
    return results


def demonstrate_unified_interface():
    """Demuestra la interfaz unificada con selección automática."""
    
    print("\n" + "=" * 80)
    print("INTERFAZ UNIFICADA CON SELECCIÓN AUTOMÁTICA")
    print("=" * 80)
    
    scenarios = {
        'numeric': "Dataset numérico (XGBoost debería ser seleccionado)",
        'categorical': "Dataset categórico (CatBoost debería ser seleccionado)",
        'missing': "Dataset con missing values (LightGBM o CatBoost)"
    }
    
    for scenario_type, description in scenarios.items():
        print(f"\n{description}")
        print("-" * 60)
        
        # Crear dataset
        df = create_sample_dataset(scenario_type)
        
        # Crear Data y Task
        data = Data(
            id=f"data_{scenario_type}",
            backend=df,
            target_names=['target']
        )
        
        task = TaskClassif(
            id=f"task_{scenario_type}",
            backend=data,
            target='target'
        )
        
        # Usar interfaz unificada con auto-selección
        learner = learner_gradient_boosting(
            backend='auto',  # Selección automática
            n_estimators=50,
            auto_optimize=True,
            verbose=True
        )
        
        # Entrenar
        learner.train(task)
        
        # Ver qué backend se seleccionó
        backend_info = learner.get_backend_info()
        print(f"\nBackend seleccionado: {backend_info['selected_backend']}")
        
        # Evaluar
        predictions = learner.predict(task)
        accuracy = predictions.score(MeasureClassifAccuracy())
        print(f"Accuracy: {accuracy:.4f}")


def demonstrate_optimization_profiles():
    """Demuestra diferentes perfiles de optimización."""
    
    print("\n" + "=" * 80)
    print("PERFILES DE OPTIMIZACIÓN")
    print("=" * 80)
    
    # Crear dataset
    df = create_sample_dataset('mixed')
    data = Data(id="opt_data", backend=df, target_names=['target'])
    task = TaskClassif(id="opt_task", backend=data, target='target')
    
    profiles = [
        ("Optimizado para VELOCIDAD", GBOptimizationProfile(
            optimize_for='speed',
            handle_categorical=True
        )),
        ("Optimizado para ACCURACY", GBOptimizationProfile(
            optimize_for='accuracy',
            handle_categorical=True
        )),
        ("Optimizado para MEMORIA", GBOptimizationProfile(
            optimize_for='memory',
            handle_categorical=True
        )),
        ("Con UNCERTAINTY", GBOptimizationProfile(
            enable_uncertainty=True,
            handle_categorical=True
        ))
    ]
    
    for profile_name, profile in profiles:
        print(f"\n{profile_name}")
        print("-" * 40)
        
        learner = learner_gradient_boosting(
            optimization_profile=profile,
            n_estimators=50,
            verbose=True
        )
        
        start = time.time()
        learner.train(task)
        train_time = time.time() - start
        
        backend_info = learner.get_backend_info()
        print(f"Backend seleccionado: {backend_info['selected_backend']}")
        print(f"Tiempo de entrenamiento: {train_time:.3f}s")
        
        predictions = learner.predict(task)
        accuracy = predictions.score(MeasureClassifAccuracy())
        print(f"Accuracy: {accuracy:.4f}")


def run_mini_benchmark():
    """Ejecuta un mini-benchmark rápido."""
    
    print("\n" + "=" * 80)
    print("MINI-BENCHMARK RÁPIDO")
    print("=" * 80)
    
    try:
        from mlpy.benchmarks.gradient_boosting_benchmark import GradientBoostingBenchmark
    except ImportError:
        print("No se pudo importar el módulo de benchmark")
        return
    
    # Crear benchmark
    benchmark = GradientBoostingBenchmark(
        n_estimators=50,
        save_results=False,
        verbose=True
    )
    
    # Crear datasets de prueba
    datasets = [
        (1000, 20, 0),   # Pequeño, numérico
        (5000, 50, 10),  # Mediano, mixto
        (10000, 30, 25), # Mediano, muy categórico
    ]
    
    all_results = []
    
    for n_samples, n_features, n_categorical in datasets:
        print(f"\nDataset: {n_samples} samples, {n_features} features, {n_categorical} categorical")
        
        task, name = benchmark.create_synthetic_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_categorical=n_categorical,
            task_type='classification'
        )
        
        results = benchmark.benchmark_task(task, name)
        all_results.extend(results)
    
    # Crear resumen
    print("\n" + "=" * 80)
    print("RESUMEN DEL BENCHMARK")
    print("=" * 80)
    
    # Agrupar por backend
    backend_summary = {}
    for result in all_results:
        if result.backend not in backend_summary:
            backend_summary[result.backend] = {
                'times': [],
                'accuracies': [],
                'memory': []
            }
        
        if result.error is None:
            backend_summary[result.backend]['times'].append(result.training_time)
            backend_summary[result.backend]['accuracies'].append(result.accuracy_score)
            backend_summary[result.backend]['memory'].append(result.memory_usage_mb)
    
    # Imprimir resumen
    for backend, metrics in backend_summary.items():
        if metrics['times']:
            print(f"\n{backend.upper()}:")
            print(f"  Tiempo promedio: {np.mean(metrics['times']):.3f}s")
            print(f"  Accuracy promedio: {np.mean(metrics['accuracies']):.4f}")
            print(f"  Memoria promedio: {np.mean(metrics['memory']):.1f} MB")
    
    # Determinar ganador
    if backend_summary:
        best_accuracy = max(
            backend_summary.items(),
            key=lambda x: np.mean(x[1]['accuracies']) if x[1]['accuracies'] else 0
        )
        print(f"\n✓ Mejor accuracy promedio: {best_accuracy[0]}")
        
        fastest = min(
            backend_summary.items(),
            key=lambda x: np.mean(x[1]['times']) if x[1]['times'] else float('inf')
        )
        print(f"✓ Más rápido en promedio: {fastest[0]}")


if __name__ == "__main__":
    print("DEMOSTRACIÓN DE GRADIENT BOOSTING EN MLPY")
    print("=" * 80)
    print("Este script demuestra las capacidades de gradient boosting en MLPY:")
    print("1. Comparación directa entre XGBoost, LightGBM y CatBoost")
    print("2. Interfaz unificada con selección automática")
    print("3. Perfiles de optimización")
    print("4. Mini-benchmark de rendimiento")
    print("=" * 80)
    
    # 1. Comparación directa
    results_direct = compare_backends_directly()
    
    # 2. Interfaz unificada
    demonstrate_unified_interface()
    
    # 3. Perfiles de optimización
    demonstrate_optimization_profiles()
    
    # 4. Mini-benchmark
    run_mini_benchmark()
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("\nConclusiones:")
    print("- XGBoost: Rápido y confiable para datos numéricos")
    print("- LightGBM: Más eficiente en memoria y velocidad con datasets grandes")
    print("- CatBoost: Superior con features categóricas y texto")
    print("- La interfaz unificada selecciona automáticamente el mejor backend")
    print("=" * 80)