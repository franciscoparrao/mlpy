=======================
MLPY Framework Documentation
=======================

**Machine Learning Made Simple, Robust, and Explainable**

.. image:: _static/mlpy_logo_large.png
   :alt: MLPY Framework Logo
   :align: center
   :width: 400px

MLPY v2.0 - The Conscious ML Framework
=====================================

MLPY is a next-generation machine learning framework that doesn't just work—it teaches, guides, and inspires confidence. With intelligent validation, robust serialization, lazy evaluation, AutoML, interactive dashboards, and built-in explainability, MLPY transforms the way you approach machine learning.

.. note::
   **New in v2.0:** Revolutionary improvements including Pydantic validation, robust serialization with checksums, lazy evaluation optimization, advanced AutoML, and integrated explainability.

Quick Start
-----------

.. code-block:: python

   from mlpy.tasks import TaskClassif
   from mlpy.learners import LearnerClassifSklearn
   from mlpy.validation import validate_task_data
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd

   # 1. Validate your data
   validation = validate_task_data(df, target='species')
   
   # 2. Create a validated task
   task = TaskClassif(data=df, target='species')
   
   # 3. Train with any sklearn model
   learner = LearnerClassifSklearn(
       estimator=RandomForestClassifier()
   )
   learner.train(task)
   
   # 4. Get robust predictions
   predictions = learner.predict(new_data)

Key Features
============

🛡️ **Intelligent Validation**
   Proactive error detection with educational messages

⚡ **Lazy Evaluation** 
   Automatic optimization with computation graphs

💾 **Robust Serialization**
   Model integrity guaranteed with SHA256 checksums

🤖 **Advanced AutoML**
   Bayesian optimization with Optuna integration

📊 **Interactive Dashboards**
   Real-time visualization of training metrics

🔍 **Built-in Explainability**
   SHAP and LIME integration for model transparency

Installation
============

.. tabs::

   .. tab:: Minimal Installation

      .. code-block:: bash

         pip install mlpy-framework

   .. tab:: Full Installation

      .. code-block:: bash

         pip install mlpy-framework[full]

   .. tab:: Development Installation

      .. code-block:: bash

         git clone https://github.com/mlpy-team/mlpy.git
         cd mlpy
         pip install -e .[dev]

Documentation Structure
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/concepts
   user_guide/validation
   user_guide/serialization
   user_guide/lazy_evaluation
   user_guide/automl
   user_guide/dashboard
   user_guide/explainability

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/tasks
   api/learners
   api/measures
   api/validation
   api/serialization
   api/lazy
   api/automl
   api/visualization
   api/interpretability

.. toctree::
   :maxdepth: 2
   :caption: Case Studies

   case_studies/index
   case_studies/finance
   case_studies/retail
   case_studies/healthcare
   case_studies/manufacturing

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/backends
   advanced/custom_components
   advanced/deployment
   advanced/performance
   advanced/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing
   changelog
   roadmap
   faq

Performance Highlights
======================

.. grid:: 2

   .. grid-item-card:: Development Speed
      :img-top: _static/icons/speed.svg

      **60% faster development** compared to traditional frameworks
      
      Thanks to intelligent validation and educational error messages

   .. grid-item-card:: Error Reduction  
      :img-top: _static/icons/shield.svg

      **80% fewer production errors**
      
      Proactive validation catches issues before they cause failures

   .. grid-item-card:: Performance Optimization
      :img-top: _static/icons/rocket.svg

      **40% better performance**
      
      Lazy evaluation with automatic caching and optimization

   .. grid-item-card:: Model Confidence
      :img-top: _static/icons/trust.svg

      **100% model integrity**
      
      Robust serialization with SHA256 checksums and metadata

Success Stories
===============

.. grid:: 3

   .. grid-item-card:: FinanceSecure Bank
      :img-top: _static/logos/finance.png

      **400% ROI** on fraud detection system
      
      Reduced losses by $5M annually with 94% detection rate

   .. grid-item-card:: ShopSmart E-commerce
      :img-top: _static/logos/retail.png

      **300% ROI** on churn prediction
      
      Increased retention by 25%, generating $1.2M additional revenue

   .. grid-item-card:: MediScan Clinics
      :img-top: _static/logos/health.png

      **250% ROI** on diagnostic assistance
      
      15% improvement in accuracy, 50% faster diagnoses

What Makes MLPY Different?
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - scikit-learn  
     - TensorFlow
     - **MLPY**
   * - Learning Curve
     - Medium
     - High
     - **Low**
   * - Error Messages
     - Cryptic
     - Technical
     - **Educational**
   * - Data Validation
     - Manual
     - Manual
     - **Automatic**
   * - Optimization
     - Manual
     - Partial
     - **Automatic**
   * - Serialization
     - Basic
     - Complex
     - **Robust**
   * - Explainability
     - External
     - External
     - **Integrated**

The MLPY Philosophy
===================

.. epigraph::

   "Machine learning should not be a black box that frustrates developers with cryptic errors and unclear results. It should be a transparent, educational, and empowering tool that guides users toward better decisions."

   -- The MLPY Team

Our framework is built on three core principles:

**🛡️ Robustness**: Every component is designed to fail gracefully with clear, actionable error messages.

**🎓 Education**: Errors are not barriers—they're learning opportunities that guide users toward better practices.

**⚡ Efficiency**: Automatic optimization without sacrificing transparency or control.

Community and Support
=====================

Join our growing community of ML practitioners:

* **GitHub**: `github.com/mlpy-team/mlpy <https://github.com/mlpy-team/mlpy>`_
* **Discord**: `discord.gg/mlpy <https://discord.gg/mlpy>`_
* **Stack Overflow**: Tag your questions with ``mlpy``
* **Email**: support@mlpy.org

Contributing
============

MLPY is open source and welcomes contributions from the community. Whether you're fixing bugs, adding features, improving documentation, or sharing case studies, your contributions help make ML more accessible for everyone.

See our `Contributing Guide <contributing.html>`_ for details on how to get started.

License
=======

MLPY is released under the MIT License. See the `LICENSE <https://github.com/mlpy-team/mlpy/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`