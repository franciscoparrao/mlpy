"""
Basic Classification Workflow
===============================

This example demonstrates a simple classification workflow with MLPY:
1. Load data
2. Create task
3. Train learner
4. Make predictions
5. Evaluate performance
"""

import pandas as pd
from sklearn import datasets

from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy


def main():
    print("Loading Iris dataset...")
    iris = datasets.load_iris()

    # Build DataFrame with features and target
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target_names[iris.target]

    # Create task
    task = TaskClassif(data=df, target="target", id="iris_basic")

    # Train learner
    learner = LearnerRandomForestClassifier(n_estimators=100, random_state=42)
    learner.train(task)

    # Predict on whole dataset
    pred = learner.predict(task)

    # Evaluate
    acc = MeasureClassifAccuracy()
    score = acc.score(pred, task)
    print(f"\nAccuracy: {score:.3f}")

    print("Done!")

if __name__ == "__main__":
    main()
