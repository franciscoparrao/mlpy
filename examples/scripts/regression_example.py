"""
Ejemplo de Regresión con MLPY
==============================

Este script demuestra un flujo de regresión usando MLPY.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Importar MLPY
from mlpy.tasks import TaskRegr
from mlpy.learners import LearnerRegrFeatureless
from mlpy.learners.sklearn import learner_sklearn
from mlpy.measures import MeasureRegrMSE, MeasureRegrR2, MeasureRegrMAE
from mlpy.resamplings import ResamplingCV
from mlpy.pipelines import linear_pipeline
from mlpy.pipelines.operators import PipeOpScale
from mlpy import resample, benchmark

# Para modelos de sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def main():
    print("="*60)
    print("Ejemplo de Regresión con MLPY")
    print("="*60)
    
    # 1. Cargar datos
    print("\n1. Cargando dataset California Housing...")
    housing = fetch_california_housing()
    
    # Usar solo una muestra para que sea más rápido
    n_samples = 2000
    indices = np.random.choice(len(housing.data), n_samples, replace=False)
    
    df = pd.DataFrame(
        housing.data[indices], 
        columns=housing.feature_names
    )
    df['price'] = housing.target[indices]
    
    print(f"   - Shape: {df.shape}")
    print(f"   - Target range: [{df['price'].min():.2f}, {df['price'].max():.2f}]")
    
    # 2. Crear tarea
    print("\n2. Creando tarea de regresión...")
    task = TaskRegr(
        data=df,
        target='price',
        id='housing_regression'
    )
    print(f"   - Features: {task.n_features}")
    print(f"   - Observaciones: {task.n_obs}")
    
    # 3. Definir learners
    print("\n3. Configurando modelos...")
    
    # Baseline
    baseline = LearnerRegrFeatureless(id='baseline_mean', method='mean')
    
    # Modelos lineales
    linear_models = [
        learner_sklearn(LinearRegression(), id='linear'),
        learner_sklearn(Ridge(alpha=1.0), id='ridge'),
        learner_sklearn(Lasso(alpha=0.1), id='lasso'),
        learner_sklearn(ElasticNet(alpha=0.1), id='elastic_net')
    ]
    
    # Modelos de árbol
    tree_models = [
        learner_sklearn(
            DecisionTreeRegressor(max_depth=10, random_state=42),
            id='decision_tree'
        ),
        learner_sklearn(
            RandomForestRegressor(n_estimators=50, random_state=42),
            id='random_forest'
        ),
        learner_sklearn(
            GradientBoostingRegressor(n_estimators=50, random_state=42),
            id='gradient_boosting'
        )
    ]
    
    # Pipeline con escalado
    pipeline_rf = linear_pipeline([
        PipeOpScale(method='standard'),
        learner_sklearn(
            RandomForestRegressor(n_estimators=50, random_state=42),
            id='rf_scaled'
        )
    ])
    
    all_learners = [baseline] + linear_models + tree_models + [pipeline_rf]
    print(f"   - {len(all_learners)} modelos configurados")
    
    # 4. Configurar evaluación
    print("\n4. Configurando evaluación...")
    cv = ResamplingCV(folds=5)
    measures = [
        MeasureRegrMSE(),
        MeasureRegrMAE(),
        MeasureRegrR2()
    ]
    
    # 5. Evaluar modelos individuales (para mostrar progreso)
    print("\n5. Evaluando modelos...")
    results = {}
    
    for learner in all_learners:
        print(f"\n   Evaluando {learner.id}...")
        result = resample(task, learner, cv, measures)
        results[learner.id] = result
        
        agg = result.aggregate()
        print(f"   - MSE: {agg['regr.mse']['mean']:.3f}")
        print(f"   - MAE: {agg['regr.mae']['mean']:.3f}")
        print(f"   - R²:  {agg['regr.rsq']['mean']:.3f}")
    
    # 6. Benchmark completo
    print("\n6. Ejecutando benchmark completo...")
    bench_result = benchmark(
        tasks=[task],
        learners=all_learners,
        resampling=cv,
        measures=measures
    )
    
    # 7. Mostrar rankings
    print("\n7. RANKINGS DE MODELOS")
    print("-"*60)
    
    print("\nRanking por R² (mayor es mejor):")
    print(bench_result.rank_learners('regr.rsq'))
    
    print("\nRanking por MSE (menor es mejor):")
    print(bench_result.rank_learners('regr.mse'))
    
    # 8. Visualización
    print("\n8. Generando visualizaciones...")
    
    # Preparar datos
    model_names = [l.id for l in all_learners]
    r2_scores = []
    mse_scores = []
    
    agg_r2 = bench_result.aggregate('regr.rsq')
    agg_mse = bench_result.aggregate('regr.mse')
    
    for learner in all_learners:
        r2_scores.append(agg_r2.loc['housing_regression', learner.id])
        mse_scores.append(agg_mse.loc['housing_regression', learner.id])
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico R²
    bars1 = ax1.barh(model_names, r2_scores)
    ax1.set_xlabel('R² Score')
    ax1.set_title('Comparación de Modelos - R² (mayor es mejor)')
    ax1.grid(axis='x', alpha=0.3)
    
    # Colorear baseline
    bars1[0].set_color('red')
    
    # Añadir valores
    for bar, score in zip(bars1, r2_scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center')
    
    # Gráfico MSE
    bars2 = ax2.barh(model_names, mse_scores)
    ax2.set_xlabel('MSE')
    ax2.set_title('Comparación de Modelos - MSE (menor es mejor)')
    ax2.grid(axis='x', alpha=0.3)
    
    # Colorear baseline
    bars2[0].set_color('red')
    
    # Añadir valores
    for bar, score in zip(bars2, mse_scores):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('housing_regression_results.png', dpi=150, bbox_inches='tight')
    print("   - Gráfico guardado como 'housing_regression_results.png'")
    
    # 9. Análisis del mejor modelo
    best_id = bench_result.rank_learners('regr.rsq').iloc[0]['learner']
    best_learner = next(l for l in all_learners if l.id == best_id)
    
    print(f"\n9. ANÁLISIS DEL MEJOR MODELO: {best_id}")
    print("-"*60)
    
    # Entrenar en todo el dataset
    best_learner.train(task)
    
    # Hacer predicciones en una muestra
    sample_indices = np.random.choice(task.n_obs, 5, replace=False)
    
    print("\nEjemplos de predicciones:")
    print(f"{'Real':>10} {'Predicción':>12} {'Error':>10}")
    print("-"*35)
    
    for idx in sample_indices:
        sample_data = df.iloc[[idx]].drop('price', axis=1)
        pred = best_learner.predict_newdata(sample_data, task)
        
        real_value = df.iloc[idx]['price']
        pred_value = pred.response[0]
        error = abs(real_value - pred_value)
        
        print(f"{real_value:>10.2f} {pred_value:>12.2f} {error:>10.2f}")
    
    # 10. Feature importance (si está disponible)
    if hasattr(best_learner, 'importance'):
        importance = best_learner.importance()
        if importance is not None:
            print("\n10. IMPORTANCIA DE FEATURES")
            print("-"*35)
            
            feat_imp = pd.DataFrame({
                'feature': task.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(feat_imp)
            
            # Visualizar
            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp['feature'], feat_imp['importance'])
            plt.xlabel('Importancia')
            plt.title(f'Importancia de Features - {best_id}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            print("\n   - Gráfico guardado como 'feature_importance.png'")
    
    print("\n¡Ejemplo completado con éxito!")
    plt.show()


if __name__ == "__main__":
    main()