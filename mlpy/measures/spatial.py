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
    
    This measure computes standard accuracy but is explicitly marked
    as compatible with spatial task types.
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
    """Area Under ROC Curve for spatial classification tasks."""
    
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
            return roc_auc_score(prediction.truth, prediction.prob[:, 1])
        
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
    """F1 Score for spatial classification tasks."""
    
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
    """Precision for spatial classification tasks."""
    
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
    """Recall for spatial classification tasks."""
    
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
    """Matthews Correlation Coefficient for spatial classification tasks."""
    
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