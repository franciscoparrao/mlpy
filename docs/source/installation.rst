Instalación
===========

Requisitos
----------

MLPY requiere Python 3.8 o superior. Las dependencias principales son:

- numpy >= 1.19.0
- pandas >= 1.1.0
- scikit-learn >= 0.24.0

Instalación desde PyPI
---------------------

La forma más fácil de instalar MLPY es usando pip:

.. code-block:: bash

   pip install mlpy

Instalación con Dependencias Opcionales
--------------------------------------

MLPY incluye varias dependencias opcionales para funcionalidades extendidas:

**Todas las dependencias opcionales:**

.. code-block:: bash

   pip install mlpy[all]

**Para visualización:**

.. code-block:: bash

   pip install mlpy[viz]

Incluye matplotlib y seaborn para gráficos.

**Para interpretabilidad:**

.. code-block:: bash

   pip install mlpy[interpret]

Incluye SHAP y LIME para interpretación de modelos.

**Para paralelización avanzada:**

.. code-block:: bash

   pip install mlpy[parallel]

Incluye joblib para backends de paralelización adicionales.

Instalación desde el Código Fuente
---------------------------------

Para instalar la versión de desarrollo:

.. code-block:: bash

   git clone https://github.com/mlpy-project/mlpy.git
   cd mlpy
   pip install -e .

Para desarrollo con todas las herramientas:

.. code-block:: bash

   pip install -e .[dev]

Esto incluye:

- pytest para tests
- pytest-cov para cobertura
- black para formateo de código
- flake8 para linting
- sphinx para documentación
- pre-commit para hooks de git

Verificar la Instalación
-----------------------

Para verificar que MLPY está instalado correctamente:

.. code-block:: python

   import mlpy
   print(mlpy.__version__)

   # Verificar componentes principales
   from mlpy.tasks import TaskClassif
   from mlpy.learners import LearnerClassifFeatureless
   from mlpy.measures import MeasureClassifAccuracy
   from mlpy.resamplings import ResamplingCV

   print("¡MLPY instalado correctamente!")

Solución de Problemas
--------------------

**ImportError con scikit-learn:**

Si encuentras errores al importar wrappers de scikit-learn, asegúrate de tener scikit-learn instalado:

.. code-block:: bash

   pip install scikit-learn>=0.24.0

**Problemas con NumPy:**

En algunos sistemas, puede ser necesario actualizar NumPy:

.. code-block:: bash

   pip install --upgrade numpy

**Problemas en Windows:**

Si encuentras problemas en Windows, considera usar Anaconda:

.. code-block:: bash

   conda install numpy pandas scikit-learn
   pip install mlpy