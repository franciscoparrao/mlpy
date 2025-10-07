"""
Test del sistema de benchmark avanzado con modelos reales de sklearn, XGBoost, LightGBM y CatBoost.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.resamplings import ResamplingCV
from mlpy.measures import create_measure
from mlpy.benchmark_advanced import benchmark, benchmark_grid, compare_learners

# Importar los wrappers de learners
from mlpy.learners.sklearn_wrapper import learner_sklearn

# Importar modelos de sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

# Intentar importar gradient boosting libraries
try:
    from mlpy.learners import learner_xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost no disponible")

try:
    from mlpy.learners import learner_lightgbm
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM no disponible")

try:
    from mlpy.learners import learner_catboost
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost no disponible")


def crear_datos_ejemplo():
    """Crear datos de ejemplo para clasificación y regresión."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # Generar features
    X = np.random.randn(n_samples, n_features)
    
    # Clasificación binaria con patrón no lineal
    y_classif = ((X[:, 0]**2 + X[:, 1]**2) > 1.5) | (X[:, 2] > 0.5)
    y_classif = y_classif.astype(int)
    
    # Regresión con relación no lineal
    y_regr = (
        2 * X[:, 0] + 
        X[:, 1]**2 - 
        3 * X[:, 2] + 
        0.5 * X[:, 3] * X[:, 4] +
        np.random.randn(n_samples) * 0.5
    )
    
    # Crear DataFrames
    df_classif = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_classif['target'] = y_classif
    
    df_regr = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df_regr['target'] = y_regr
    
    # Crear tasks
    task_classif = TaskClassif(
        data=df_classif,
        target='target',
        id='clasificacion_binaria'
    )
    
    task_regr = TaskRegr(
        data=df_regr,
        target='target',
        id='regresion'
    )
    
    return task_classif, task_regr


def benchmark_clasificacion():
    """Benchmark de modelos de clasificación."""
    print("\n" + "="*60)
    print("BENCHMARK DE CLASIFICACIÓN")
    print("="*60)
    
    task_classif, _ = crear_datos_ejemplo()
    
    # Crear learners de sklearn
    learners = [
        learner_sklearn(
            DecisionTreeClassifier(max_depth=3, random_state=42),
            id='decision_tree'
        ),
        learner_sklearn(
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            id='random_forest'
        ),
        learner_sklearn(
            LogisticRegression(max_iter=1000, random_state=42),
            id='logistic_regression'
        ),
        learner_sklearn(
            SVC(kernel='rbf', probability=True, random_state=42),
            id='svm_rbf'
        )
    ]
    
    # Agregar gradient boosting si están disponibles
    if HAS_XGBOOST:
        learners.append(
            learner_xgboost(n_estimators=50, max_depth=3, id='xgboost')
        )
    
    if HAS_LIGHTGBM:
        learners.append(
            learner_lightgbm(n_estimators=50, max_depth=3, id='lightgbm', verbose=-1)
        )
    
    if HAS_CATBOOST:
        learners.append(
            learner_catboost(iterations=50, depth=3, id='catboost', verbose=False)
        )
    
    print(f"\nModelos a comparar: {[l.id for l in learners]}")
    
    # Crear diseño de benchmark
    design = benchmark_grid(
        tasks=task_classif,
        learners=learners,
        resamplings=ResamplingCV(folds=5),
        measures=['accuracy', 'f1', 'auc']
    )
    
    print(f"\nDiseño del benchmark:")
    print(f"  Tasks: {len(design.tasks)}")
    print(f"  Learners: {len(design.learners)}")
    print(f"  Resamplings: {len(design.resamplings)}")
    print(f"  Medidas: {[m.id for m in design.measures]}")
    print(f"  Total experimentos: {design.n_experiments}")
    
    # Ejecutar benchmark
    print("\nEjecutando benchmark...")
    result = benchmark(design, parallel=False, verbose=0)
    
    print(f"\nResultados:")
    print(f"  Scores recolectados: {len(result.scores)}")
    print(f"  Errores: {len(result.errors)}")
    
    # Mostrar rankings por medida
    for measure_id in ['accuracy', 'f1', 'auc']:
        print(f"\n{measure_id.upper()} Rankings:")
        rankings = result.rank_learners(measure_id)
        if not rankings.empty:
            for idx, row in rankings.iterrows():
                print(f"  {int(row['final_rank'])}. {idx}: {row['score']:.4f}")
    
    # Análisis estadístico
    print("\nAnálisis Estadístico (Friedman test):")
    stat_result = result.statistical_test('accuracy', test='friedman')
    if 'p_value' in stat_result:
        print(f"  p-value: {stat_result['p_value']:.4f}")
        print(f"  Significativo: {'Sí' if stat_result.get('significant', False) else 'No'}")
    
    return result


def benchmark_regresion():
    """Benchmark de modelos de regresión."""
    print("\n" + "="*60)
    print("BENCHMARK DE REGRESIÓN")
    print("="*60)
    
    _, task_regr = crear_datos_ejemplo()
    
    # Crear learners de sklearn
    learners = [
        learner_sklearn(
            DecisionTreeRegressor(max_depth=5, random_state=42),
            id='decision_tree_regr'
        ),
        learner_sklearn(
            RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            id='random_forest_regr'
        ),
        learner_sklearn(
            Ridge(alpha=1.0, random_state=42),
            id='ridge'
        ),
        learner_sklearn(
            SVR(kernel='rbf'),
            id='svr_rbf'
        )
    ]
    
    # Agregar gradient boosting si están disponibles
    if HAS_XGBOOST:
        learners.append(
            learner_xgboost(n_estimators=50, max_depth=3, objective='reg:squarederror', id='xgboost_regr')
        )
    
    if HAS_LIGHTGBM:
        learners.append(
            learner_lightgbm(n_estimators=50, max_depth=3, objective='regression', id='lightgbm_regr', verbose=-1)
        )
    
    if HAS_CATBOOST:
        learners.append(
            learner_catboost(iterations=50, depth=3, loss_function='RMSE', id='catboost_regr', verbose=False)
        )
    
    print(f"\nModelos a comparar: {[l.id for l in learners]}")
    
    # Crear diseño de benchmark
    design = benchmark_grid(
        tasks=task_regr,
        learners=learners,
        resamplings=ResamplingCV(folds=5),
        measures=['rmse', 'mae', 'r2']
    )
    
    print(f"\nDiseño del benchmark:")
    print(f"  Total experimentos: {design.n_experiments}")
    
    # Ejecutar benchmark
    print("\nEjecutando benchmark...")
    result = benchmark(design, parallel=False, verbose=0)
    
    print(f"\nResultados:")
    print(f"  Scores recolectados: {len(result.scores)}")
    
    # Mostrar rankings
    print(f"\nRMSE Rankings (menor es mejor):")
    rankings = result.rank_learners('rmse', minimize=True)
    if not rankings.empty:
        for idx, row in rankings.iterrows():
            print(f"  {int(row['final_rank'])}. {idx}: {row['score']:.4f}")
    
    print(f"\nR2 Rankings (mayor es mejor):")
    rankings_r2 = result.rank_learners('r2', minimize=False)
    if not rankings_r2.empty:
        for idx, row in rankings_r2.iterrows():
            print(f"  {int(row['final_rank'])}. {idx}: {row['score']:.4f}")
    
    return result


def comparacion_rapida():
    """Usar compare_learners para comparación rápida con gráficos."""
    print("\n" + "="*60)
    print("COMPARACIÓN RÁPIDA CON ANÁLISIS ESTADÍSTICO")
    print("="*60)
    
    task_classif, _ = crear_datos_ejemplo()
    
    # Seleccionar algunos modelos para comparación
    learners = [
        learner_sklearn(DecisionTreeClassifier(max_depth=3), id='DT'),
        learner_sklearn(RandomForestClassifier(n_estimators=50), id='RF'),
        learner_sklearn(LogisticRegression(max_iter=1000), id='LR')
    ]
    
    if HAS_XGBOOST:
        learners.append(learner_xgboost(n_estimators=50, id='XGB'))
    
    print(f"\nComparando: {[l.id for l in learners]}")
    
    # Comparación rápida
    results = compare_learners(
        task=task_classif,
        learners=learners,
        cv_folds=5,
        measures=['accuracy', 'auc', 'f1'],
        test='friedman',
        show_plot=False  # Cambiar a True si tienes matplotlib
    )
    
    return results


def main():
    """Función principal."""
    print("SISTEMA DE BENCHMARK AVANZADO MLPY")
    print("Probando con modelos reales de ML")
    
    # Benchmark de clasificación
    result_classif = benchmark_clasificacion()
    
    # Benchmark de regresión
    result_regr = benchmark_regresion()
    
    # Comparación rápida
    result_compare = comparacion_rapida()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    print("\nEl sistema de benchmark avanzado permite:")
    print("✓ Comparar múltiples modelos")
    print("✓ Evaluar en múltiples tareas")
    print("✓ Usar diferentes estrategias de resampling")
    print("✓ Calcular múltiples métricas")
    print("✓ Realizar análisis estadístico")
    print("✓ Generar rankings automáticos")
    print("✓ Exportar resultados")
    
    return result_classif, result_regr, result_compare


if __name__ == "__main__":
    results = main()