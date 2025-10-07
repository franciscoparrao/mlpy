Inicio Rápido
=============

Esta guía te ayudará a comenzar con MLPY en minutos.

Tu Primer Modelo con MLPY
------------------------

Vamos a crear un clasificador simple usando el dataset Iris:

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_iris
   
   # Importar MLPY
   from mlpy.tasks import TaskClassif
   from mlpy.learners.sklearn import learner_sklearn
   from mlpy.resamplings import ResamplingCV
   from mlpy.measures import MeasureClassifAccuracy
   from mlpy import resample
   
   # Cargar datos
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['species'] = iris.target_names[iris.target]
   
   # 1. Crear una tarea
   task = TaskClassif(
       data=df,
       target='species',
       id='iris'
   )
   
   # 2. Crear un learner
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   learner = learner_sklearn(rf, id='random_forest')
   
   # 3. Definir estrategia de evaluación
   cv = ResamplingCV(folds=5, stratify=True)
   
   # 4. Evaluar el modelo
   result = resample(
       task=task,
       learner=learner,
       resampling=cv,
       measures=[MeasureClassifAccuracy()]
   )
   
   # 5. Ver resultados
   print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
   print(f"Std Dev: {result.aggregate()['classif.acc']['std']:.3f}")

Conceptos Clave
--------------

**Task (Tarea)**
   Encapsula los datos y metadatos de un problema de ML. Hay dos tipos principales:
   
   - ``TaskClassif``: Para problemas de clasificación
   - ``TaskRegr``: Para problemas de regresión

**Learner (Aprendiz)**
   Interfaz unificada para algoritmos de ML. MLPY proporciona:
   
   - Learners nativos (baseline, debug)
   - Wrappers para scikit-learn
   - Interfaz para crear learners personalizados

**Measure (Medida)**
   Métricas para evaluar el rendimiento:
   
   - Clasificación: accuracy, F1, AUC, etc.
   - Regresión: MSE, RMSE, R², etc.

**Resampling (Remuestreo)**
   Estrategias para evaluación robusta:
   
   - Cross-validation
   - Holdout
   - Bootstrap
   - Leave-one-out

Ejemplo de Regresión
-------------------

.. code-block:: python

   from mlpy.tasks import TaskRegr
   from mlpy.measures import MeasureRegrMSE, MeasureRegrR2
   from sklearn.datasets import make_regression
   from sklearn.linear_model import Ridge
   
   # Generar datos sintéticos
   X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
   df = pd.DataFrame(X, columns=[f'x{i}' for i in range(20)])
   df['target'] = y
   
   # Crear tarea de regresión
   task = TaskRegr(data=df, target='target')
   
   # Crear learner
   ridge = Ridge(alpha=1.0)
   learner = learner_sklearn(ridge, id='ridge')
   
   # Evaluar con múltiples métricas
   result = resample(
       task=task,
       learner=learner,
       resampling=ResamplingCV(folds=10),
       measures=[MeasureRegrMSE(), MeasureRegrR2()]
   )
   
   # Mostrar resultados
   scores = result.aggregate()
   print(f"MSE: {scores['regr.mse']['mean']:.3f}")
   print(f"R²: {scores['regr.rsq']['mean']:.3f}")

Comparando Múltiples Modelos
----------------------------

MLPY facilita la comparación sistemática de modelos:

.. code-block:: python

   from mlpy import benchmark
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.svm import SVC
   
   # Definir múltiples learners
   learners = [
       learner_sklearn(RandomForestClassifier(), id='rf'),
       learner_sklearn(DecisionTreeClassifier(), id='tree'),
       learner_sklearn(SVC(probability=True), id='svm')
   ]
   
   # Benchmark en una tarea
   bench_result = benchmark(
       tasks=[task],
       learners=learners,
       resampling=ResamplingCV(folds=5),
       measures=[MeasureClassifAccuracy()]
   )
   
   # Ver rankings
   print(bench_result.rank_learners('classif.acc'))

Creando Pipelines
----------------

MLPY soporta pipelines complejos de preprocesamiento:

.. code-block:: python

   from mlpy.pipelines import linear_pipeline
   from mlpy.pipelines.operators import PipeOpScale, PipeOpImpute
   
   # Crear pipeline: imputar -> escalar -> clasificar
   pipeline = linear_pipeline([
       PipeOpImpute(strategy='mean'),
       PipeOpScale(method='standard'),
       learner
   ])
   
   # El pipeline funciona como un learner normal
   result = resample(
       task=task,
       learner=pipeline,
       resampling=cv,
       measures=[MeasureClassifAccuracy()]
   )

Próximos Pasos
-------------

- Explora la :doc:`user_guide/tasks` para aprender sobre manejo de datos
- Lee sobre :doc:`user_guide/learners` para entender los modelos disponibles
- Consulta :doc:`user_guide/pipelines` para flujos de trabajo avanzados
- Prueba :doc:`tutorials/classification` para un ejemplo completo