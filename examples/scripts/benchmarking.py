"""
Benchmarking Multiple Learners
===============================

Compare multiple learners on the same dataset.
"""

import pandas as pd
from sklearn import datasets

from mlpy import benchmark
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import (
    LearnerRandomForestClassifier,
    LearnerLogisticRegression,
    LearnerSVM,
    LearnerAdaBoost  # using LearnerAdaBoost as available alias
)
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.resamplings import ResamplingCV


def main():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    task = TaskClassif(data=df, target="target", id="iris_bench")

    learners = [
        LearnerRandomForestClassifier(random_state=42),
        LearnerLogisticRegression(random_state=42),
        LearnerSVM(predict_type="prob"),
        LearnerAdaBoost(random_state=42)
    ]

    print("Benchmarking learners...")
    result = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy(), MeasureClassifF1(average='weighted')]
    )

    print("\nResults:")

    rank = result.rank_learners()
    print("\nLearner Rankings (by mean accuracy):")
    print(rank)

    try:
        df_res = result.as_data_frame()
        print(df_res)
    except Exception:
        # Fallback: print repr
        print(result)


if __name__ == "__main__":
    main()
