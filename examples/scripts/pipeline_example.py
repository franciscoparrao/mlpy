"""
Pipeline Example
================

This script demonstrates how to use pipelines in MLPY:
1. Preprocess data (e.g., normalization).
2. Train a model within a pipeline.
3. Evaluate the pipeline's performance.
"""

import pandas as pd
from sklearn import datasets
from mlpy.tasks import TaskClassif
from mlpy.pipelines import linear_pipeline, GraphLearner, PipeOpScale, PipeOpLearner
from mlpy.learners.sklearn import LearnerLogisticRegression
from mlpy.measures import MeasureClassifAccuracy

def main():
    print("Loading Iris dataset...")
    iris = datasets.load_iris()

    # Build DataFrame with features and target
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Create task
    task = TaskClassif(data=df, target="target", id="iris_pipeline")

    # Define pipeline with preprocessing and model
    pipeline = linear_pipeline([
        PipeOpScale(),
        PipeOpLearner(LearnerLogisticRegression(random_state=42))
    ])

    graph_learner = GraphLearner(graph=pipeline)

    # Train pipeline
    print("Training pipeline...")
    graph_learner.train(task)

    # Predict and evaluate
    pred = graph_learner.predict(task)
    acc = MeasureClassifAccuracy()
    score = acc.score(pred, task)
    print(f"\nPipeline Accuracy: {score:.3f}")

    print("Done!")

if __name__ == "__main__":
    main()