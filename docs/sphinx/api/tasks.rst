============================
Tasks API Reference
============================

Tasks are the fundamental abstraction in MLPY that represent machine learning problems. They encapsulate data, target variables, and metadata, providing a consistent interface for different types of ML problems.

.. currentmodule:: mlpy.tasks

Overview
========

The Task system in MLPY provides:

* **Unified interface** for classification, regression, clustering, and spatial problems
* **Automatic validation** of data consistency and format
* **Metadata tracking** for reproducibility
* **Backend flexibility** supporting pandas, dask, and vaex
* **Spatial support** for geographic and spatial machine learning

Quick Example
=============

.. code-block:: python

   from mlpy.tasks import TaskClassif
   import pandas as pd

   # Load your data
   df = pd.read_csv('my_data.csv')

   # Create a classification task
   task = TaskClassif(
       data=df,
       target='species',
       id='iris_classification'
   )

   print(f"Task: {task.id}")
   print(f"Type: {task.task_type}")
   print(f"Classes: {task.n_classes}")
   print(f"Features: {task.n_features}")
   print(f"Samples: {task.n_obs}")

Base Classes
============

Task
----

.. autoclass:: Task
   :members:
   :undoc-members:
   :show-inheritance:

   The base class for all MLPY tasks. Provides common functionality for data handling, validation, and metadata management.

   **Key Attributes:**

   .. attribute:: data
      
      The underlying dataset (pandas DataFrame, dask DataFrame, or vaex DataFrame)

   .. attribute:: target
      
      Name of the target column

   .. attribute:: task_type
      
      Type of ML task ('classif', 'regr', 'cluster', etc.)

   .. attribute:: id
      
      Unique identifier for the task

   .. attribute:: n_obs
      
      Number of observations (rows) in the dataset

   .. attribute:: n_features
      
      Number of features (columns) excluding the target

   .. attribute:: X
      
      Feature matrix (data without target column)

   .. attribute:: y
      
      Target values

   **Key Methods:**

   .. automethod:: subset
   .. automethod:: copy
   .. automethod:: get_train_test_split
   .. automethod:: get_cross_validation_folds

Classification Tasks
====================

TaskClassif
-----------

.. autoclass:: TaskClassif
   :members:
   :undoc-members:
   :show-inheritance:

   Task for classification problems where the target variable contains discrete categories.

   **Additional Attributes:**

   .. attribute:: n_classes
      
      Number of unique classes in the target

   .. attribute:: class_names
      
      Names of the classes

   .. attribute:: class_distribution
      
      Distribution of samples across classes

   **Example:**

   .. code-block:: python

      from mlpy.tasks import TaskClassif
      import pandas as pd

      # Binary classification
      df_binary = pd.DataFrame({
          'feature1': [1, 2, 3, 4, 5],
          'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
          'target': ['yes', 'no', 'yes', 'no', 'yes']
      })

      task_binary = TaskClassif(data=df_binary, target='target')
      print(f"Binary classification: {task_binary.n_classes} classes")

      # Multi-class classification
      df_multi = pd.DataFrame({
          'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4],
          'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.2], 
          'species': ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor']
      })

      task_multi = TaskClassif(data=df_multi, target='species')
      print(f"Multi-class: {task_multi.n_classes} classes")
      print(f"Classes: {task_multi.class_names}")

Regression Tasks
================

TaskRegr
--------

.. autoclass:: TaskRegr
   :members:
   :undoc-members:
   :show-inheritance:

   Task for regression problems where the target variable is continuous.

   **Additional Attributes:**

   .. attribute:: target_stats
      
      Descriptive statistics of the target variable (mean, std, min, max, etc.)

   **Example:**

   .. code-block:: python

      from mlpy.tasks import TaskRegr
      import pandas as pd
      import numpy as np

      # Create regression data
      df_regr = pd.DataFrame({
          'square_feet': [1200, 1500, 1800, 2000, 2200],
          'bedrooms': [2, 3, 3, 4, 4],
          'age': [5, 10, 15, 20, 25],
          'price': [200000, 250000, 300000, 350000, 400000]
      })

      task_regr = TaskRegr(data=df_regr, target='price')
      print(f"Target stats: {task_regr.target_stats}")

Spatial Tasks
=============

TaskSpatial
-----------

.. autoclass:: TaskSpatial
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for spatial machine learning tasks that incorporate geographic coordinates.

   **Additional Attributes:**

   .. attribute:: coords_cols
      
      Names of columns containing coordinates (e.g., ['longitude', 'latitude'])

   .. attribute:: spatial_bounds
      
      Bounding box of the spatial data

TaskClassifSpatial
------------------

.. autoclass:: TaskClassifSpatial
   :members:
   :undoc-members:
   :show-inheritance:

   Classification task with spatial coordinates.

   **Example:**

   .. code-block:: python

      from mlpy.tasks import TaskClassifSpatial
      import pandas as pd

      # Spatial classification data
      df_spatial = pd.DataFrame({
          'longitude': [-122.4, -122.3, -122.2, -122.1],
          'latitude': [37.8, 37.9, 37.7, 37.6],
          'elevation': [100, 150, 200, 250],
          'land_use': ['urban', 'forest', 'urban', 'water']
      })

      task_spatial = TaskClassifSpatial(
          data=df_spatial,
          target='land_use',
          coords_cols=['longitude', 'latitude']
      )

TaskRegrSpatial
---------------

.. autoclass:: TaskRegrSpatial
   :members:
   :undoc-members:
   :show-inheritance:

   Regression task with spatial coordinates.

Utility Functions
=================

create_task
-----------

.. autofunction:: create_task

   Automatically creates the appropriate task type based on the target variable.

   **Parameters:**

   * **data** (*DataFrame*) -- Input dataset
   * **target** (*str*) -- Name of target column
   * **task_type** (*str, optional*) -- Force specific task type
   * **spatial** (*bool, optional*) -- Whether to create spatial task
   * **coords_cols** (*list, optional*) -- Coordinate columns for spatial tasks

   **Returns:**

   * **Task** -- Appropriate task instance

   **Example:**

   .. code-block:: python

      from mlpy.tasks import create_task
      import pandas as pd

      # Automatic task type detection
      df = pd.DataFrame({
          'feature1': [1, 2, 3, 4, 5],
          'target_continuous': [1.1, 2.2, 3.3, 4.4, 5.5],
          'target_categorical': ['A', 'B', 'A', 'B', 'A']
      })

      # Will create TaskRegr
      task_regr = create_task(df, target='target_continuous')

      # Will create TaskClassif  
      task_classif = create_task(df, target='target_categorical')

validate_task
-------------

.. autofunction:: validate_task

   Validates that a task is properly constructed and ready for ML.

Task Factory
============

TaskFactory
-----------

.. autoclass:: TaskFactory
   :members:
   :undoc-members:

   Factory class for creating tasks with advanced configuration.

   **Example:**

   .. code-block:: python

      from mlpy.tasks import TaskFactory

      factory = TaskFactory()

      # Create with custom validation rules
      task = factory.create(
          data=df,
          target='target',
          validation_rules=['check_missing', 'check_variance'],
          auto_fix=True
      )

Common Patterns
===============

Train/Test Splitting
---------------------

.. code-block:: python

   from mlpy.tasks import TaskClassif
   from sklearn.model_selection import train_test_split

   # Create task
   task = TaskClassif(data=df, target='species')

   # Get indices for splitting
   train_idx, test_idx = train_test_split(
       range(task.n_obs),
       test_size=0.2,
       stratify=task.y,
       random_state=42
   )

   # Create train and test tasks
   task_train = task.subset(train_idx)
   task_test = task.subset(test_idx)

Cross-Validation
-----------------

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold

   # Create cross-validation folds
   cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

   for fold, (train_idx, val_idx) in enumerate(cv.split(task.X, task.y)):
       task_train = task.subset(train_idx)
       task_val = task.subset(val_idx)
       
       print(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}")

Working with Different Backends
--------------------------------

.. code-block:: python

   import pandas as pd
   import dask.dataframe as dd
   
   # Pandas backend (default)
   df_pandas = pd.read_csv('data.csv')
   task_pandas = TaskClassif(data=df_pandas, target='species')

   # Dask backend for large datasets
   df_dask = dd.read_csv('large_data.csv')
   task_dask = TaskClassif(data=df_dask, target='species')

   # Both have the same interface
   print(f"Pandas task: {task_pandas.n_obs} samples")
   print(f"Dask task: {task_dask.n_obs} samples")

Best Practices
==============

1. **Always validate your data** before creating tasks:

   .. code-block:: python

      from mlpy.validation import validate_task_data

      validation = validate_task_data(df, target='species')
      if not validation['valid']:
          print("Issues found:")
          for error in validation['errors']:
              print(f"  - {error}")

2. **Use descriptive task IDs** for better tracking:

   .. code-block:: python

      task = TaskClassif(
          data=df,
          target='species',
          id='iris_classification_v1.2_cleaned'
      )

3. **Leverage spatial tasks** for geographic data:

   .. code-block:: python

      # For any data with coordinates
      task_spatial = TaskClassifSpatial(
          data=df,
          target='land_cover',
          coords_cols=['lon', 'lat']
      )

4. **Use task.copy()** for experiments:

   .. code-block:: python

      # Original task
      task_original = TaskClassif(data=df, target='species')

      # Copy for experimentation
      task_experiment = task_original.copy(id='experiment_1')

See Also
========

* :doc:`learners` - Training models with tasks
* :doc:`measures` - Evaluating task performance  
* :doc:`validation` - Validating task data
* :doc:`../user_guide/concepts` - Conceptual overview of tasks