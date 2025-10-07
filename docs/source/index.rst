.. MLPY documentation master file

MLPY - Machine Learning Framework for Python
============================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

MLPY es un framework moderno de aprendizaje automático para Python inspirado en `mlr3 <https://mlr3.mlr-org.com/>`_. 
Proporciona una interfaz unificada, componible y extensible para tareas de machine learning.

Características Principales
--------------------------

- 🎯 **API Unificada**: Interfaz consistente para diferentes tareas de ML
- 🧩 **Diseño Modular**: Bloques de construcción componibles para flujos de trabajo complejos
- 🔧 **Extensible**: Fácil añadir learners, medidas y pasos de preprocesamiento personalizados
- 📊 **Evaluación Completa**: Estrategias de resampling y medidas de rendimiento integradas
- 🚀 **Python Moderno**: Type hints completos, soporte async y características de Python 3.8+
- 🔗 **Integración**: Integración transparente con scikit-learn y otras librerías de ML
- 🎛️ **AutoML**: Tuning de hiperparámetros y feature engineering automático
- ⚡ **Paralelización**: Múltiples backends para computación paralela
- 📈 **Visualización**: Gráficos integrados para análisis de resultados

Instalación Rápida
-----------------

.. code-block:: bash

   # Instalación básica
   pip install mlpy

   # Con todas las dependencias opcionales
   pip install mlpy[all]

   # Para desarrollo
   pip install mlpy[dev]

Ejemplo Rápido
-------------

.. code-block:: python

   import mlpy
   from mlpy.tasks import TaskClassif
   from mlpy.learners.sklearn import learner_sklearn
   from mlpy.resamplings import ResamplingCV
   from mlpy.measures import MeasureClassifAccuracy
   from sklearn.ensemble import RandomForestClassifier

   # Crear una tarea de clasificación
   task = TaskClassif(data=df, target="species")

   # Crear un learner
   rf = RandomForestClassifier(n_estimators=100)
   learner = learner_sklearn(rf, id="rf")

   # Evaluar usando cross-validation
   resampling = ResamplingCV(folds=5)
   result = mlpy.resample(
       task=task,
       learner=learner,
       resampling=resampling,
       measures=[MeasureClassifAccuracy()]
   )

   # Ver resultados
   print(result.aggregate())

Contenido
---------

.. toctree::
   :maxdepth: 2
   :caption: Primeros Pasos

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Guía del Usuario

   user_guide/tasks
   user_guide/learners
   user_guide/measures
   user_guide/resampling
   user_guide/pipelines
   user_guide/automl
   user_guide/parallel
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: Integraciones

   tgpy_integration

.. toctree::
   :maxdepth: 2
   :caption: Tutoriales

   tutorials/classification
   tutorials/regression
   tutorials/pipelines
   tutorials/benchmarking
   tutorials/automl

.. toctree::
   :maxdepth: 2
   :caption: Referencia API

   api/tasks
   api/learners
   api/measures
   api/resampling
   api/pipelines
   api/automl
   api/parallel
   api/callbacks

.. toctree::
   :maxdepth: 1
   :caption: Desarrollo

   contributing
   changelog
   license

Índices y Tablas
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`