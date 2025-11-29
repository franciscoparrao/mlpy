"""
Cross-Validation Example
=========================

Demonstrates how to use cross-validation for model evaluation.
"""

import pandas as pd
from sklearn import datasets

from mlpy import resample
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.resamplings import ResamplingCV


def main():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    task = TaskClassif(data=df, target="target", id="iris_cv")

    learner = LearnerRandomForestClassifier(n_estimators=100, random_state=42)

    print("Running 5-fold cross-validation...")
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[
            MeasureClassifAccuracy(),
            MeasureClassifF1(average='weighted')
        ]
    )

    print(f"\nCross-Validation Results:")
    print(f"  Accuracy: {result.score('classif.acc', average='mean'):.3f} "
          f"± {result.score('classif.acc', average='std'):.3f}")
    print(f"  F1-Score: {result.score('classif.f1', average='mean'):.3f} "
          f"± {result.score('classif.f1', average='std'):.3f}")


if __name__ == "__main__":
    main()
