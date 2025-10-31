"""
Spatial measures for MLPY.

This module provides performance measures specifically designed for 
spatial classification and regression tasks, accounting for spatial 
autocorrelation and geographic structure in the data.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    matthews_corrcoef
)
from .base import MeasureClassif


class MeasureSpatialAccuracy(MeasureClassif):
    """Accuracy measure for spatial classification tasks.

    This measure computes standard accuracy with `sklearn.metrics.accuracy_score`
    and marks it as compatible with spatial tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.spatial import MeasureSpatialAccuracy
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> MeasureSpatialAccuracy().score(y_true, y_pred)
    0.75
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.acc',
            minimize=False,
            predict_type='response',
            task_type='classif_spatial'  # Compatible con tareas espaciales
        )
        self.name = 'Spatial Accuracy'
        self.range = (0, 1)
        
    def _score(self, prediction, task=None):
        """Calculate accuracy for spatial tasks."""
        return accuracy_score(prediction.truth, prediction.response)


class MeasureSpatialAUC(MeasureClassif):
    """Area Under the ROC Curve (AUC) for spatial classification.

    Uses ``sklearn.metrics.roc_auc_score``. For binary problems, expects
    probabilities for the positive class; for multiclass, computes weighted
    one-vs-rest AUC.

    Examples
    --------
    Binary (non-perfect AUC):

    >>> import numpy as np, pandas as pd
    >>> from mlpy.measures.spatial import MeasureSpatialAUC
    >>> from mlpy.predictions import PredictionClassif
    >>> y_true = np.array([0, 0, 1, 1])
    >>> prob = pd.DataFrame({0: [0.9, 0.3, 0.4, 0.1], 1: [0.1, 0.7, 0.6, 0.9]})
    >>> pred = PredictionClassif(task=None, learner_id='demo', row_ids=[0,1,2,3], truth=y_true, prob=prob)
    >>> round(MeasureSpatialAUC().score(pred), 2)
    0.75
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.auc',
            minimize=False,
            predict_type='prob',
            task_type='classif_spatial'
        )
        self.name = 'Spatial AUC'
        self.range = (0, 1)
    
    def _score(self, prediction, task=None):
        """Calculate AUC for spatial binary classification."""
        if not hasattr(prediction, 'prob') or prediction.prob is None:
            return np.nan
            
        # Binary classification
        if prediction.prob.shape[1] == 2:
            return roc_auc_score(prediction.truth, prediction.prob.iloc[:, 1])
        
        # Multiclass classification
        try:
            return roc_auc_score(
                prediction.truth, 
                prediction.prob, 
                multi_class='ovr',
                average='weighted'
            )
        except:
            return np.nan


class MeasureSpatialF1(MeasureClassif):
    """F1 score for spatial classification.

    Harmonic mean of precision and recall. Uses weighted averaging for
    multiclass problems.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.spatial import MeasureSpatialF1
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> round(MeasureSpatialF1().score(y_true, y_pred), 2)
    0.8
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.f1',
            minimize=False,
            predict_type='response',
            task_type='classif_spatial'
        )
        self.name = 'Spatial F1 Score'
        self.range = (0, 1)
    
    def _score(self, prediction, task=None):
        """Calculate F1 score."""
        n_classes = len(np.unique(prediction.truth))
        average = 'weighted' if n_classes > 2 else 'binary'
        return f1_score(prediction.truth, prediction.response, average=average)


class MeasureSpatialPrecision(MeasureClassif):
    """Precision for spatial classification.

    Fraction of predicted positives that are correct. Uses weighted averaging
    for multiclass problems.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.spatial import MeasureSpatialPrecision
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> round(MeasureSpatialPrecision().score(y_true, y_pred), 2)
    0.67
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.precision',
            minimize=False,
            predict_type='response',
            task_type='classif_spatial'
        )
        self.name = 'Spatial Precision'
        self.range = (0, 1)
    
    def _score(self, prediction, task=None):
        """Calculate precision."""
        n_classes = len(np.unique(prediction.truth))
        average = 'weighted' if n_classes > 2 else 'binary'
        return precision_score(
            prediction.truth,
            prediction.response,
            average=average,
            zero_division=0
        )


class MeasureSpatialRecall(MeasureClassif):
    """Recall for spatial classification.

    Fraction of actual positives that are recovered. Uses weighted averaging
    for multiclass problems.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.spatial import MeasureSpatialRecall
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> MeasureSpatialRecall().score(y_true, y_pred)
    1.0
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.recall',
            minimize=False,
            predict_type='response',
            task_type='classif_spatial'
        )
        self.name = 'Spatial Recall'
        self.range = (0, 1)
    
    def _score(self, prediction, task=None):
        """Calculate recall."""
        n_classes = len(np.unique(prediction.truth))
        average = 'weighted' if n_classes > 2 else 'binary'
        return recall_score(
            prediction.truth,
            prediction.response,
            average=average,
            zero_division=0
        )


class MeasureSpatialMCC(MeasureClassif):
    """Matthews correlation coefficient for spatial classification.

    Balanced measure even with class imbalance; ranges from -1 (total
    disagreement) to 1 (perfect agreement).

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.spatial import MeasureSpatialMCC
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> round(MeasureSpatialMCC().score(y_true, y_pred), 3)
    0.577
    """
    
    def __init__(self):
        super().__init__(
            id='spatial.mcc',
            minimize=False,
            predict_type='response',
            task_type='classif_spatial'
        )
        self.name = 'Spatial Matthews Correlation'
        self.range = (-1, 1)
    
    def _score(self, prediction, task=None):
        """Calculate MCC."""
        return matthews_corrcoef(prediction.truth, prediction.response)


# Aliases for convenience
SpatialAccuracy = MeasureSpatialAccuracy
SpatialAUC = MeasureSpatialAUC  
SpatialF1 = MeasureSpatialF1
SpatialPrecision = MeasureSpatialPrecision
SpatialRecall = MeasureSpatialRecall
SpatialMCC = MeasureSpatialMCC