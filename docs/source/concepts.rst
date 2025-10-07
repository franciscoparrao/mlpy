Conceptos Principales
====================

MLPY está construido alrededor de varios conceptos clave que trabajan juntos para proporcionar una experiencia de ML unificada y flexible.

Arquitectura General
-------------------

.. code-block:: text

   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │    Task     │────▶│   Learner   │────▶│ Prediction  │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                    │                    │
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ DataBackend │     │ Resampling  │     │   Measure   │
   └─────────────┘     └─────────────┘     └─────────────┘

Task (Tarea)
-----------

Una **Task** encapsula:

- Los datos del problema (features y target)
- Metadatos sobre el problema (tipo, roles de columnas)
- Información sobre filas/columnas activas

**Tipos de Task:**

- ``TaskClassif``: Para clasificación
- ``TaskRegr``: Para regresión
- ``TaskUnsupervised``: Para aprendizaje no supervisado (futuro)

**Ejemplo:**

.. code-block:: python

   from mlpy.tasks import TaskClassif
   
   task = TaskClassif(
       data=df,
       target='species',
       features=['sepal_length', 'sepal_width'],  # Opcional
       id='iris_task'
   )
   
   # Acceder a información
   print(task.n_features)  # Número de features
   print(task.n_obs)       # Número de observaciones
   print(task.class_names) # Clases únicas

DataBackend
----------

El **DataBackend** es una abstracción sobre el almacenamiento de datos que permite:

- Soporte para diferentes formatos (pandas, numpy, etc.)
- Operaciones eficientes sin copiar datos
- Composición de backends (stacking, cbinding)

Normalmente no interactúas directamente con DataBackend, Task lo maneja internamente.

Learner (Aprendiz)
-----------------

Un **Learner** es la abstracción unificada para algoritmos de ML:

**Características:**

- Interfaz consistente (train/predict)
- Gestión de estado (entrenado/no entrenado)
- Manejo de errores robusto
- Soporte para diferentes tipos de predicción

**Tipos de Learners:**

.. code-block:: python

   # Learner nativo
   from mlpy.learners import LearnerClassifFeatureless
   learner = LearnerClassifFeatureless(method='mode')
   
   # Wrapper de scikit-learn
   from mlpy.learners.sklearn import learner_sklearn
   from sklearn.ensemble import RandomForestClassifier
   
   rf = RandomForestClassifier(n_estimators=100)
   learner = learner_sklearn(rf)

Prediction (Predicción)
----------------------

Una **Prediction** encapsula los resultados de un learner:

- Predicciones (response)
- Probabilidades (para clasificación)
- Valores verdaderos (si están disponibles)
- Métodos útiles de análisis

.. code-block:: python

   # Entrenar y predecir
   learner.train(task)
   prediction = learner.predict(task)
   
   # Analizar predicciones
   print(prediction.response)      # Clases predichas
   print(prediction.prob)          # Probabilidades
   print(prediction.confusion_matrix())  # Matriz de confusión

Measure (Medida)
---------------

Las **Measures** evalúan el rendimiento de las predicciones:

**Para Clasificación:**

- ``MeasureClassifAccuracy``: Exactitud
- ``MeasureClassifF1``: F1-Score
- ``MeasureClassifAUC``: Área bajo la curva ROC
- ``MeasureClassifCE``: Error de clasificación

**Para Regresión:**

- ``MeasureRegrMSE``: Error cuadrático medio
- ``MeasureRegrR2``: Coeficiente de determinación
- ``MeasureRegrMAE``: Error absoluto medio

.. code-block:: python

   from mlpy.measures import MeasureClassifAccuracy
   
   measure = MeasureClassifAccuracy()
   score = measure.score(prediction)

Resampling (Remuestreo)
----------------------

**Resampling** define estrategias para evaluar modelos:

**Estrategias disponibles:**

- ``ResamplingHoldout``: División simple train/test
- ``ResamplingCV``: Cross-validation k-fold
- ``ResamplingLOO``: Leave-one-out
- ``ResamplingBootstrap``: Bootstrap con OOB
- ``ResamplingRepeatedCV``: CV repetido

.. code-block:: python

   from mlpy.resamplings import ResamplingCV
   
   # 5-fold CV estratificado
   cv = ResamplingCV(folds=5, stratify=True)
   
   # Instanciar para obtener splits fijos
   cv_instance = cv.instantiate(task)
   for i, (train_ids, test_ids) in enumerate(cv_instance):
       print(f"Fold {i}: {len(train_ids)} train, {len(test_ids)} test")

Pipeline
--------

Los **Pipelines** permiten encadenar operaciones:

.. code-block:: python

   from mlpy.pipelines import linear_pipeline
   from mlpy.pipelines.operators import PipeOpScale, PipeOpImpute
   
   # Pipeline: imputar → escalar → predecir
   pipe = linear_pipeline([
       PipeOpImpute(strategy='mean'),
       PipeOpScale(method='standard'),
       learner
   ])
   
   # Usar como learner normal
   pipe.train(task)
   predictions = pipe.predict(task)

Funciones de Alto Nivel
----------------------

MLPY proporciona funciones convenientes para tareas comunes:

**resample()**
   Evalúa un learner usando una estrategia de resampling:

   .. code-block:: python

      from mlpy import resample
      
      result = resample(
          task=task,
          learner=learner,
          resampling=ResamplingCV(folds=5),
          measures=[MeasureClassifAccuracy()]
      )

**benchmark()**
   Compara múltiples learners en múltiples tasks:

   .. code-block:: python

      from mlpy import benchmark
      
      result = benchmark(
          tasks=[task1, task2],
          learners=[learner1, learner2],
          resampling=ResamplingCV(folds=3),
          measures=[MeasureClassifAccuracy()]
      )

**tune()**
   Optimiza hiperparámetros:

   .. code-block:: python

      from mlpy.automl import TunerGridSearch, ParamSet
      
      param_set = ParamSet({
          'n_estimators': [50, 100, 200],
          'max_depth': [5, 10, None]
      })
      
      tuner = TunerGridSearch(param_set)
      best_learner = tuner.tune(task, learner, cv, measure)

Composabilidad
-------------

Una característica clave de MLPY es la composabilidad. Todos los componentes están diseñados para trabajar juntos:

.. code-block:: python

   # Componer un flujo complejo
   from mlpy.callbacks import CallbackProgress
   from mlpy.parallel import BackendMultiprocessing
   
   # Pipeline con AutoML
   pipeline = create_auto_pipeline(task, base_learner=learner)
   
   # Benchmark paralelo con callbacks
   with BackendMultiprocessing(n_jobs=4):
       result = benchmark(
           tasks=tasks,
           learners=[pipeline],
           resampling=cv,
           measures=measures,
           callbacks=[CallbackProgress()]
       )

Esta composabilidad permite construir flujos de trabajo ML complejos manteniendo el código limpio y mantenible.