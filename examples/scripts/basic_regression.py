"""
Basic Regression Workflow
===============================

This example demonstrates a simple regression workflow with MLPY:
1. Load data
2. Create task
3. Train learner
4. Make predictions
5. Evaluate performance
"""

import pandas as pd
import sklearn.datasets

from mlpy.tasks import TaskRegr
from mlpy.learners.sklearn import LearnerLinearRegression
from mlpy.measures import MeasureRegrRMSE

def main():
    # Load dataset
    data = sklearn.datasets.load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = pd.Series(data.target)

    # Create task
    task = TaskRegr(df, target="target", id="diabetes_basic")

    # Train learner
    learner = LearnerLinearRegression()
    learner.train(task)

    # Make predictions
    preds = learner.predict(task)

    # Evaluate performance
    measure = MeasureRegrRMSE()
    rmse = measure.score(preds, df["target"])
    print(f"RMSE: {rmse:.2f}")

    # Other statistics
    target_mean = df["target"].mean()
    target_std = df["target"].std()
    print(f"Target mean: {target_mean:.2f}")
    print(f"Target standard deviation: {target_std:.2f}")

if __name__ == "__main__":
    main()