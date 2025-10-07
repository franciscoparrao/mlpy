"""
Ejemplo de Clasificación con MLPY
==================================

Este script demuestra un flujo completo de clasificación usando MLPY.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Importar MLPY
from mlpy.tasks import TaskClassif
from mlpy.learners import LearnerClassifFeatureless
from mlpy.learners.sklearn import learner_sklearn
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1, MeasureClassifAUC
from mlpy.resamplings import ResamplingCV
from mlpy import resample, benchmark

# Para modelos de sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def main():
    print("="*60)
    print("Ejemplo de Clasificación con MLPY")
    print("="*60)
    
    # 1. Cargar y preparar datos
    print("\n1. Cargando dataset Wine...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['wine_type'] = wine.target_names[wine.target]
    
    print(f"   - Shape: {df.shape}")
    print(f"   - Clases: {df['wine_type'].unique()}")
    
    # 2. Crear tarea
    print("\n2. Creando tarea de clasificación...")
    task = TaskClassif(
        data=df,
        target='wine_type',
        id='wine_classification'
    )
    print(f"   - Features: {task.n_features}")
    print(f"   - Observaciones: {task.n_obs}")
    
    # 3. Definir learners
    print("\n3. Configurando modelos...")
    learners = [
        # Baseline
        LearnerClassifFeatureless(id='baseline', method='mode'),
        
        # Modelos de sklearn
        learner_sklearn(
            LogisticRegression(max_iter=1000, random_state=42),
            id='logistic_regression'
        ),
        learner_sklearn(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            id='decision_tree'
        ),
        learner_sklearn(
            RandomForestClassifier(n_estimators=100, random_state=42),
            id='random_forest'
        ),
        learner_sklearn(
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            id='gradient_boosting'
        ),
        learner_sklearn(
            SVC(kernel='rbf', probability=True, random_state=42),
            id='svm_rbf'
        )
    ]
    
    print(f"   - {len(learners)} modelos configurados")
    
    # 4. Configurar evaluación
    print("\n4. Configurando evaluación...")
    cv = ResamplingCV(folds=5, stratify=True)
    measures = [
        MeasureClassifAccuracy(),
        MeasureClassifF1(average='macro'),
        MeasureClassifAUC(average='macro')
    ]
    print(f"   - Estrategia: {cv.folds}-fold CV estratificado")
    print(f"   - Métricas: {[m.id for m in measures]}")
    
    # 5. Ejecutar benchmark
    print("\n5. Ejecutando benchmark...")
    print("   Esto puede tomar unos momentos...\n")
    
    bench_result = benchmark(
        tasks=[task],
        learners=learners,
        resampling=cv,
        measures=measures
    )
    
    # 6. Mostrar resultados
    print("\n6. RESULTADOS")
    print("-"*60)
    
    # Rankings
    for measure in measures:
        print(f"\nRanking por {measure.id}:")
        rankings = bench_result.rank_learners(measure.id)
        print(rankings)
    
    # Tabla de scores
    print("\n\nTabla de Accuracy:")
    print(bench_result.score_table('classif.acc'))
    
    # 7. Visualización
    print("\n7. Generando visualización...")
    
    # Preparar datos para gráfico
    learner_names = [l.id for l in learners]
    accuracies = []
    f1_scores = []
    
    agg_acc = bench_result.aggregate('classif.acc')
    agg_f1 = bench_result.aggregate('classif.f1')
    
    for learner in learners:
        accuracies.append(agg_acc.loc['wine_classification', learner.id])
        f1_scores.append(agg_f1.loc['wine_classification', learner.id])
    
    # Crear gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    bars1 = ax1.bar(range(len(learner_names)), accuracies)
    ax1.set_xticks(range(len(learner_names)))
    ax1.set_xticklabels(learner_names, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Comparación de Accuracy')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Colorear baseline diferente
    bars1[0].set_color('red')
    
    # Añadir valores
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1-Score
    bars2 = ax2.bar(range(len(learner_names)), f1_scores)
    ax2.set_xticks(range(len(learner_names)))
    ax2.set_xticklabels(learner_names, rotation=45, ha='right')
    ax2.set_ylabel('F1-Score (macro)')
    ax2.set_title('Comparación de F1-Score')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Colorear baseline diferente
    bars2[0].set_color('red')
    
    # Añadir valores
    for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('wine_classification_results.png', dpi=150, bbox_inches='tight')
    print("   - Gráfico guardado como 'wine_classification_results.png'")
    
    # 8. Mejor modelo
    best_learner_id = bench_result.rank_learners('classif.acc').iloc[0]['learner']
    best_learner = next(l for l in learners if l.id == best_learner_id)
    
    print(f"\n8. Mejor modelo: {best_learner_id}")
    
    # Entrenar en todo el dataset
    best_learner.train(task)
    
    # Mostrar algunas predicciones
    sample_indices = [0, 50, 100]
    print("\n   Ejemplos de predicciones:")
    for idx in sample_indices:
        sample_data = df.iloc[[idx]].drop('wine_type', axis=1)
        pred = best_learner.predict_newdata(
            newdata=sample_data,
            task=task,
            predict_type='prob'
        )
        
        true_class = df.iloc[idx]['wine_type']
        pred_class = pred.response[0]
        prob = pred.prob[0].max()
        
        print(f"   - Muestra {idx}: Real='{true_class}', "
              f"Predicción='{pred_class}' (prob={prob:.2f})")
    
    print("\n¡Ejemplo completado con éxito!")
    plt.show()


if __name__ == "__main__":
    main()