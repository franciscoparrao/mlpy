Tutorial: Clasificación
=======================

Este tutorial te guiará a través de un flujo completo de clasificación usando MLPY.

Preparación de Datos
-------------------

Comenzaremos con el dataset Wine de scikit-learn:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.datasets import load_wine
   
   # Cargar datos
   wine = load_wine()
   df = pd.DataFrame(wine.data, columns=wine.feature_names)
   df['wine_class'] = wine.target_names[wine.target]
   
   print(f"Dataset shape: {df.shape}")
   print(f"Classes: {df['wine_class'].unique()}")
   print(f"\nPrimeras filas:")
   print(df.head())

Crear una Tarea de Clasificación
--------------------------------

.. code-block:: python

   from mlpy.tasks import TaskClassif
   
   # Crear tarea
   task = TaskClassif(
       data=df,
       target='wine_class',
       id='wine_classification'
   )
   
   # Explorar la tarea
   print(f"Número de features: {task.n_features}")
   print(f"Número de observaciones: {task.n_obs}")
   print(f"Clases: {task.class_names}")
   print(f"Features: {task.feature_names[:5]}...")  # Primeras 5

Modelo Baseline
--------------

Siempre es buena práctica comenzar con un modelo baseline:

.. code-block:: python

   from mlpy.learners import LearnerClassifFeatureless
   from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
   from mlpy.resamplings import ResamplingCV
   from mlpy import resample
   
   # Learner baseline
   baseline = LearnerClassifFeatureless(
       id='baseline',
       method='mode'  # Predice la clase más frecuente
   )
   
   # Evaluar
   cv = ResamplingCV(folds=5, stratify=True)
   measures = [MeasureClassifAccuracy(), MeasureClassifF1(average='macro')]
   
   result_baseline = resample(
       task=task,
       learner=baseline,
       resampling=cv,
       measures=measures
   )
   
   print("Resultados Baseline:")
   for measure_id, scores in result_baseline.aggregate().items():
       print(f"{measure_id}: {scores['mean']:.3f} ± {scores['std']:.3f}")

Modelos de Scikit-learn
----------------------

Ahora probemos varios modelos de scikit-learn:

.. code-block:: python

   from mlpy.learners.sklearn import learner_sklearn
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   
   # Crear learners
   learners = {
       'decision_tree': learner_sklearn(
           DecisionTreeClassifier(max_depth=5, random_state=42),
           id='decision_tree'
       ),
       'random_forest': learner_sklearn(
           RandomForestClassifier(n_estimators=100, random_state=42),
           id='random_forest'
       ),
       'gradient_boosting': learner_sklearn(
           GradientBoostingClassifier(n_estimators=100, random_state=42),
           id='gradient_boosting'
       ),
       'svm': learner_sklearn(
           SVC(kernel='rbf', probability=True, random_state=42),
           id='svm'
       ),
       'logistic_regression': learner_sklearn(
           LogisticRegression(max_iter=1000, random_state=42),
           id='logistic_regression'
       )
   }
   
   # Evaluar cada modelo
   results = {}
   for name, learner in learners.items():
       print(f"\nEvaluando {name}...")
       results[name] = resample(
           task=task,
           learner=learner,
           resampling=cv,
           measures=measures
       )

Comparación de Modelos
---------------------

Usar benchmark para comparación sistemática:

.. code-block:: python

   from mlpy import benchmark
   
   # Incluir baseline en la comparación
   all_learners = [baseline] + list(learners.values())
   
   # Benchmark
   bench_result = benchmark(
       tasks=[task],
       learners=all_learners,
       resampling=cv,
       measures=measures
   )
   
   # Ver rankings
   print("\nRanking por Accuracy:")
   print(bench_result.rank_learners('classif.acc'))
   
   print("\nRanking por F1-Score:")
   print(bench_result.rank_learners('classif.f1'))

Pipeline con Preprocesamiento
----------------------------

Crear un pipeline que incluya preprocesamiento:

.. code-block:: python

   from mlpy.pipelines import linear_pipeline
   from mlpy.pipelines.operators import PipeOpScale, PipeOpSelect
   
   # Pipeline: seleccionar features → escalar → clasificar
   pipeline = linear_pipeline([
       PipeOpSelect(
           selector_type='variance',
           threshold=0.1
       ),
       PipeOpScale(
           method='standard'
       ),
       learner_sklearn(
           RandomForestClassifier(n_estimators=100),
           id='rf_pipeline'
       )
   ])
   
   # Evaluar pipeline
   result_pipeline = resample(
       task=task,
       learner=pipeline,
       resampling=cv,
       measures=measures
   )
   
   print("\nResultados del Pipeline:")
   for measure_id, scores in result_pipeline.aggregate().items():
       print(f"{measure_id}: {scores['mean']:.3f} ± {scores['std']:.3f}")

Optimización de Hiperparámetros
-------------------------------

Optimizar hiperparámetros del mejor modelo:

.. code-block:: python

   from mlpy.automl import TunerGridSearch, ParamSet
   
   # Definir espacio de búsqueda para Random Forest
   param_set = ParamSet({
       'n_estimators': [50, 100, 200],
       'max_depth': [5, 10, 15, None],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
   })
   
   # Crear tuner
   tuner = TunerGridSearch(
       param_set=param_set,
       measure=MeasureClassifAccuracy(),
       resampling=ResamplingCV(folds=3)  # CV interno
   )
   
   # Base learner
   rf_base = learner_sklearn(
       RandomForestClassifier(random_state=42),
       id='rf_tuned'
   )
   
   # Optimizar
   print("Optimizando hiperparámetros...")
   best_learner = tuner.tune(task, rf_base)
   
   print(f"\nMejores parámetros: {best_learner.model.get_params()}")
   
   # Evaluar modelo optimizado
   result_tuned = resample(
       task=task,
       learner=best_learner,
       resampling=cv,
       measures=measures
   )
   
   print("\nResultados del modelo optimizado:")
   for measure_id, scores in result_tuned.aggregate().items():
       print(f"{measure_id}: {scores['mean']:.3f} ± {scores['std']:.3f}")

Análisis de Predicciones
-----------------------

Analizar las predicciones en detalle:

.. code-block:: python

   # Entrenar en todo el dataset
   best_learner.train(task)
   
   # Predecir
   predictions = best_learner.predict(task, predict_type='prob')
   
   # Matriz de confusión
   cm = predictions.confusion_matrix()
   print("\nMatriz de Confusión:")
   print(cm)
   
   # Probabilidades por clase
   prob_df = pd.DataFrame(
       predictions.prob,
       columns=task.class_names
   )
   prob_df['true_class'] = predictions.truth
   prob_df['predicted_class'] = predictions.response
   
   # Ver casos con baja confianza
   prob_df['max_prob'] = prob_df[task.class_names].max(axis=1)
   low_confidence = prob_df[prob_df['max_prob'] < 0.7]
   
   print(f"\nCasos con baja confianza (<70%): {len(low_confidence)}")
   if len(low_confidence) > 0:
       print(low_confidence.head())

Visualización de Resultados
--------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Comparar modelos
   models = list(results.keys())
   accuracies = []
   f1_scores = []
   
   for model in models:
       agg = results[model].aggregate()
       accuracies.append(agg['classif.acc']['mean'])
       f1_scores.append(agg['classif.f1']['mean'])
   
   # Gráfico de barras
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Accuracy
   bars1 = ax1.bar(models, accuracies)
   ax1.set_ylabel('Accuracy')
   ax1.set_title('Comparación de Accuracy')
   ax1.set_ylim(0, 1)
   
   # Añadir baseline como línea horizontal
   baseline_acc = result_baseline.aggregate()['classif.acc']['mean']
   ax1.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
   ax1.legend()
   
   # F1-Score
   bars2 = ax2.bar(models, f1_scores)
   ax2.set_ylabel('F1-Score (macro)')
   ax2.set_title('Comparación de F1-Score')
   ax2.set_ylim(0, 1)
   
   # Rotar etiquetas
   ax1.tick_params(axis='x', rotation=45)
   ax2.tick_params(axis='x', rotation=45)
   
   plt.tight_layout()
   plt.show()

Guardar el Mejor Modelo
----------------------

.. code-block:: python

   import pickle
   
   # Guardar el learner entrenado
   with open('best_wine_classifier.pkl', 'wb') as f:
       pickle.dump(best_learner, f)
   
   print("Modelo guardado como 'best_wine_classifier.pkl'")
   
   # Para cargar después:
   # with open('best_wine_classifier.pkl', 'rb') as f:
   #     loaded_learner = pickle.load(f)

Conclusiones
-----------

En este tutorial aprendiste a:

1. Crear tareas de clasificación en MLPY
2. Evaluar modelos baseline
3. Usar learners de scikit-learn
4. Comparar múltiples modelos con benchmark
5. Crear pipelines con preprocesamiento
6. Optimizar hiperparámetros
7. Analizar predicciones en detalle
8. Visualizar resultados

MLPY proporciona una interfaz consistente y componible que facilita experimentar con diferentes modelos y estrategias de evaluación.