TGPY Integration
================

MLPY provides seamless integration with **Transport Gaussian Process (TGPY)**, 
a powerful library for Gaussian Process regression with transport maps.

Installation
------------

To use TGPY with MLPY, you need to install TGPY from the provided source:

.. code-block:: bash

    # Install dependencies
    pip install ipython torch

    # Install TGPY
    cd tgpy-master
    pip install -e .

Overview
--------

Transport Gaussian Processes extend traditional GPs by incorporating transport 
maps that can model complex, non-Gaussian distributions. MLPY's wrapper makes 
it easy to use TGPY within the MLPY ecosystem.

Key Features
~~~~~~~~~~~~

- **Multiple kernel types**: Squared Exponential (SE), Matern, etc.
- **Transport types**: Marginal, Radial, and Covariance transports
- **GPU acceleration**: Optional CUDA support for faster computation
- **Uncertainty quantification**: Full posterior distributions
- **Mini-batch training**: Efficient for large datasets

Basic Usage
-----------

Simple Regression
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mlpy.tasks import TaskRegr
    from mlpy.learners import LearnerTGPRegressor
    from mlpy.measures import MeasureRegrRMSE
    
    # Create a regression task
    task = TaskRegr(
        id="gp_example",
        data=your_dataframe,
        target="target_column"
    )
    
    # Create TGPY learner
    learner = LearnerTGPRegressor(
        kernel='SE',              # Squared Exponential kernel
        lengthscale=1.0,         # Kernel lengthscale
        variance=1.0,            # Kernel variance
        noise=0.1,               # Observation noise
        n_iterations=100,        # Training iterations
        learning_rate=0.01       # Optimization learning rate
    )
    
    # Train the model
    learner.train(task)
    
    # Make predictions
    predictions = learner.predict(task)
    
    # Get predictions with uncertainty
    learner.predict_type = "se"
    predictions_with_uncertainty = learner.predict(task)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Advanced TGPY configuration
    learner = LearnerTGPRegressor(
        kernel='Matern',         # Matern kernel
        transport='marginal',    # Marginal transport
        batch_size=0.5,         # Use 50% of data per batch
        use_gpu=True,           # Enable GPU acceleration
        n_iterations=200        # More iterations
    )

Transport Types
---------------

TGPY supports different transport map types:

1. **Marginal Transport**
   
   - Learns univariate transformations
   - Good for data with marginal non-Gaussianity
   - Lower computational cost

2. **Covariance Transport**
   
   - Full covariance structure
   - Better for multivariate dependencies
   - Higher expressiveness

3. **Radial Transport**
   
   - Radial basis transformations
   - Good for radially symmetric patterns

Example with different transports:

.. code-block:: python

    # Compare different transport types
    transports = ['marginal', 'covariance', 'radial']
    
    for transport_type in transports:
        learner = LearnerTGPRegressor(
            transport=transport_type,
            n_iterations=100
        )
        
        # Evaluate with cross-validation
        results = benchmark(
            tasks=[task],
            learners=[learner],
            resampling=ResamplingCV(folds=5),
            measures=[MeasureRegrRMSE()]
        )

Integration with MLPY Pipeline
------------------------------

TGPY learners work seamlessly with MLPY's pipeline system:

.. code-block:: python

    from mlpy.pipelines import Pipeline, PipeOpScale, PipeOpPCA
    
    # Create pipeline with preprocessing
    pipeline = Pipeline(
        ops=[
            PipeOpScale(),      # Standardize features
            PipeOpPCA(n_components=5)  # Reduce dimensions
        ],
        learner=LearnerTGPRegressor(
            kernel='SE',
            transport='covariance'
        )
    )
    
    # Train pipeline
    pipeline.train(task)

Benchmarking
------------

Compare TGPY with other methods:

.. code-block:: python

    from mlpy.benchmark import benchmark
    from mlpy.learners import (
        LearnerTGPRegressor,
        LearnerRandomForestRegressor,
        LearnerLinearRegression
    )
    
    learners = [
        LearnerTGPRegressor(id="tgp_se", kernel='SE'),
        LearnerTGPRegressor(id="tgp_matern", kernel='Matern'),
        LearnerRandomForestRegressor(id="rf"),
        LearnerLinearRegression(id="linear")
    ]
    
    results = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingCV(folds=5),
        measures=[MeasureRegrRMSE(), MeasureRegrR2()]
    )

Hyperparameter Tuning
---------------------

Use MLPY's AutoML capabilities with TGPY:

.. code-block:: python

    from mlpy.automl import AutoML
    
    # Define search space
    search_space = {
        'kernel': ['SE', 'Matern'],
        'lengthscale': [0.1, 0.5, 1.0, 2.0],
        'variance': [0.5, 1.0, 2.0],
        'noise': [0.01, 0.1, 0.5],
        'transport': ['marginal', 'covariance']
    }
    
    # Run hyperparameter optimization
    automl = AutoML(
        learner_class=LearnerTGPRegressor,
        search_space=search_space,
        n_trials=50
    )
    
    best_learner = automl.train(task)

Visualization
-------------

Visualize GP predictions with uncertainty:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get predictions with uncertainty
    learner.predict_type = "se"
    pred = learner.predict(task)
    
    # Extract data
    X = task.data(cols=task.feature_names).values
    y_true = task.truth()
    y_pred = pred.response
    y_std = pred.se
    
    # Plot for 1D case
    if X.shape[1] == 1:
        plt.figure(figsize=(10, 6))
        
        # Sort for plotting
        idx = np.argsort(X[:, 0])
        X_sorted = X[idx, 0]
        
        # Plot data
        plt.scatter(X[:, 0], y_true, alpha=0.5, label='Data')
        plt.plot(X_sorted, y_pred[idx], 'r-', label='GP mean')
        
        # Confidence intervals
        plt.fill_between(
            X_sorted,
            y_pred[idx] - 2*y_std[idx],
            y_pred[idx] + 2*y_std[idx],
            alpha=0.2,
            color='red',
            label='95% confidence'
        )
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('TGPY Predictions with Uncertainty')
        plt.show()

Best Practices
--------------

1. **Data Preprocessing**
   
   - Standardize features for better kernel performance
   - Remove outliers that might affect GP training

2. **Kernel Selection**
   
   - SE kernel: smooth functions
   - Matern kernel: less smooth, more flexible
   - Start with SE and experiment

3. **Computational Considerations**
   
   - Use mini-batches for large datasets
   - Enable GPU for datasets with >1000 samples
   - Consider inducing points for very large datasets

4. **Transport Selection**
   
   - Start with covariance transport
   - Try marginal for faster computation
   - Use radial for specific symmetries

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **ImportError: No module named 'IPython'**
   
   Solution: Install IPython
   
   .. code-block:: bash
   
       pip install ipython

2. **CUDA/GPU errors**
   
   Solution: Disable GPU or install appropriate CUDA version
   
   .. code-block:: python
   
       learner = LearnerTGPRegressor(use_gpu=False)

3. **Memory issues with large datasets**
   
   Solution: Use mini-batch training
   
   .. code-block:: python
   
       learner = LearnerTGPRegressor(batch_size=0.1)  # 10% of data

API Reference
-------------

.. autoclass:: mlpy.learners.LearnerTGPRegressor
   :members:
   :inherited-members:
   :show-inheritance:

Further Resources
-----------------

- TGPY Repository: ``tgpy-master/``
- TGPY Notebooks: ``tgpy-master/notebooks/``
- Paper: *Transport Gaussian Processes for Regression*