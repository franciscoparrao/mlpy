"""
Large Dataset Handling Example
==============================

This script demonstrates how to handle large datasets in MLPY:
1. Generate a large synthetic dataset.
2. Use mini-batch training to process the data in chunks.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerLogisticRegression
from mlpy.measures import MeasureClassifAccuracy

def main():
    print("Generating large synthetic dataset...")
    X, y = make_classification(n_samples=100000, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Create task
    task = TaskClassif(data=df, target="target", id="large_dataset")

    # Define learner
    learner = LearnerLogisticRegression(random_state=42)

    # Mini-batch training
    batch_size = 10000
    n_batches = len(df) // batch_size
    print(f"Training in {n_batches} batches...")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_task = TaskClassif(data=batch, target="target", id=f"batch_{i}")
        learner.train(batch_task)
        print(f"Batch {i // batch_size + 1}/{n_batches} trained.")

    # Evaluate on the full dataset
    pred = learner.predict(task)
    acc = MeasureClassifAccuracy()
    score = acc.score(pred, task)
    print(f"\nAccuracy on full dataset: {score:.3f}")

    print("Done!")

if __name__ == "__main__":
    main()