"""Classification measures for MLPY."""

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from .base import MeasureClassif, MeasureSimple, register_measure
from ..predictions import PredictionClassif


@register_measure
class MeasureClassifAccuracy(MeasureClassif):
    """Classification accuracy measure.

    Wraps the standard accuracy metric with MLPY's unified interface.
    Computes the fraction (or count) of correctly classified samples.
    
    Parameters
    ----------
    normalize : bool, default=True
        If True, return fraction of correctly classified samples.
        If False, return number of correctly classified samples.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.classification import MeasureClassifAccuracy
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> float(MeasureClassifAccuracy().score(y_true, y_pred))
    0.75

    Notes
    -----
    Requires response predictions (class labels).

    See Also
    --------
    MeasureClassifCE : Classification error (1 - accuracy)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__(
            id='classif.acc',
            minimize=False,
            range=(0, 1) if normalize else (0, np.inf),
            predict_type='response'
        )
        self.normalize = normalize
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate accuracy score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        correct = prediction.truth[mask] == prediction.response[mask]
        if self.normalize:
            return correct.mean()
        else:
            return correct.sum()


@register_measure
class MeasureClassifCE(MeasureClassif):
    """Classification error (complement of accuracy).

    Computes the fraction of incorrectly classified samples. Equal to
    ``1 - accuracy`` when ``normalize=True`` in accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.classification import MeasureClassifCE
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> float(MeasureClassifCE().score(y_true, y_pred))
    0.25

    Notes
    -----
    Requires response predictions (class labels).

    See Also
    --------
    MeasureClassifAccuracy : Accuracy measure
    """
    
    def __init__(self):
        super().__init__(
            id='classif.ce',
            minimize=True,
            range=(0, 1),
            predict_type='response'
        )
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate classification error."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
            
        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan
            
        incorrect = prediction.truth[mask] != prediction.response[mask]
        return incorrect.mean()


@register_measure
class MeasureClassifAUC(MeasureClassif):
    """Area Under the ROC Curve for binary classification.

    Measures the ability of the model to rank positive instances higher
    than negative ones, using probability predictions.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mlpy.measures.classification import MeasureClassifAUC
    >>> from mlpy.predictions import PredictionClassif
    >>> y_true = np.array([0, 0, 1, 1])
    >>> prob = pd.DataFrame({0: [0.9, 0.35, 0.4, 0.1], 1: [0.1, 0.65, 0.6, 0.9]})
    >>> pred = PredictionClassif(task=None, learner_id='demo', row_ids=[0,1,2,3], truth=y_true, prob=prob)
    >>> MeasureClassifAUC().score(pred)
    0.75

    Notes
    -----
    - Requires probability predictions.
    - Only applicable to binary classification.

    See Also
    --------
    MeasureClassifLogLoss : Probabilistic calibration loss
    """
    
    def __init__(self):
        super().__init__(
            id='classif.auc',
            minimize=False,
            range=(0, 1),
            predict_type='prob',
            properties={'binary'}
        )
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate AUC score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
        if prediction.prob is None:
            raise ValueError("AUC requires probability predictions")
            
        # Get unique classes
        classes = np.unique(prediction.truth[~pd.isna(prediction.truth)])
        if len(classes) != 2:
            raise ValueError(f"AUC requires binary classification, got {len(classes)} classes")
            
        # Get positive class probabilities
        pos_label = classes[1]  # Convention: second class is positive
        # Access probability by column name, not index
        if hasattr(prediction.prob, 'iloc'):
            # DataFrame: get probabilities for positive class
            prob_cols = list(prediction.prob.columns)
            pos_col_idx = prob_cols.index(pos_label) if pos_label in prob_cols else 1
            prob_pos = prediction.prob.iloc[:, pos_col_idx]
        else:
            # NumPy array
            prob_pos = prediction.prob[:, 1]
        
        mask = ~(pd.isna(prediction.truth) | pd.isna(prob_pos))
        if not mask.any():
            return np.nan
            
        return sklearn_metrics.roc_auc_score(
            prediction.truth[mask], 
            prob_pos[mask]
        )


@register_measure  
class MeasureClassifLogLoss(MeasureClassif):
    """Logarithmic loss (cross-entropy loss).

    Penalizes confident and wrong probability predictions more strongly.
    Lower is better; perfect predictions yield a score near 0.
    
    Parameters
    ----------
    eps : float, default=1e-15
        Small value to clip probabilities away from 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mlpy.measures.classification import MeasureClassifLogLoss
    >>> from mlpy.predictions import PredictionClassif
    >>> y_true = np.array([0, 0, 1, 1])
    >>> prob = pd.DataFrame({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
    >>> pred = PredictionClassif(task=None, learner_id='demo', row_ids=[0,1,2,3], truth=y_true, prob=prob)
    >>> score = MeasureClassifLogLoss().score(pred)
    >>> float(round(score, 3))
    0.164

    Notes
    -----
    Requires probability predictions.

    See Also
    --------
    MeasureClassifAUC : Ranking-based probability metric
    """
    
    def __init__(self, eps: float = 1e-15):
        super().__init__(
            id='classif.logloss',
            minimize=True,
            range=(0, np.inf),
            predict_type='prob'
        )
        self.eps = eps
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate log loss."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")
        if prediction.prob is None:
            raise ValueError("LogLoss requires probability predictions")
            
        # Get unique classes from truth
        unique_classes = np.unique(prediction.truth[~pd.isna(prediction.truth)])
        n_classes = len(unique_classes)
        
        # Create class to index mapping
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        
        # Get indices for true classes
        truth_indices = np.array([class_to_idx.get(x, -1) for x in prediction.truth])
        
        # Filter out missing values
        mask = truth_indices >= 0
        if not mask.any():
            return np.nan
            
        # Get valid indices and probabilities
        valid_indices = truth_indices[mask]
        
        # Convert DataFrame to numpy array if needed
        if hasattr(prediction.prob, 'values'):
            prob_array = prediction.prob.values[mask]
        else:
            prob_array = prediction.prob[mask]
        
        # Clip probabilities to avoid log(0)
        valid_probs = np.clip(prob_array, self.eps, 1 - self.eps)
        
        # Calculate log loss
        n = len(valid_indices)
        log_probs = np.log(valid_probs[np.arange(n), valid_indices])
        return -log_probs.mean()


@register_measure
class MeasureClassifF1(MeasureClassif):
    """F1 score (harmonic mean of precision and recall).
    
    Wraps sklearn's ``f1_score`` with MLPY's unified interface. Balances
    precision and recall into a single number; useful when both false
    positives and false negatives matter, especially on imbalanced data.
    
    Parameters
    ----------
    average : str, default='binary'
        Averaging method for multiclass:
        - 'binary': Only for binary classification
        - 'micro': Calculate globally
        - 'macro': Calculate for each class and average
        - 'weighted': Calculate for each class and weight by support
    pos_label : str or int, optional
        The positive class for binary classification.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.classification import MeasureClassifF1
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> round(MeasureClassifF1().score(y_true, y_pred), 3)
    0.667

    Notes
    -----
    Defaults to 'binary' average; for multiclass, this measure auto-switches
    to 'weighted' if average='binary' and more than two classes are detected.

    See Also
    --------
    MeasureClassifPrecision : Positive predictive value
    MeasureClassifRecall : Sensitivity (true positive rate)
    """
    
    def __init__(self, average: str = 'binary', pos_label=None):
        super().__init__(
            id='classif.f1',
            minimize=False,
            range=(0, 1),
            predict_type='response',
            average=average
        )
        self.pos_label = pos_label
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate F1 score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")

        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan

        # Detect number of classes
        unique_classes = np.unique(prediction.truth[mask])
        n_classes = len(unique_classes)

        # Auto-adjust average parameter based on number of classes
        average = self.average
        pos_label = self.pos_label

        if average == 'binary':
            if n_classes == 2:
                # Binary classification - use binary average
                pos_label = self.pos_label if self.pos_label is not None else unique_classes[1]
            else:
                # Multiclass - auto-switch to weighted
                average = 'weighted'
                pos_label = None

        return sklearn_metrics.f1_score(
            prediction.truth[mask],
            prediction.response[mask],
            average=average,
            pos_label=pos_label,
            zero_division=0
        )


@register_measure  
class MeasureClassifPrecision(MeasureClassif):
    """Precision score (positive predictive value).
    
    Wraps sklearn's ``precision_score`` with MLPY's unified interface.
    Proportion of predicted positives that are correct; useful when the
    cost of false positives is high.
    
    Parameters
    ----------
    average : str, default='binary'
        Averaging method for multiclass.
    pos_label : str or int, optional
        The positive class for binary classification.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.classification import MeasureClassifPrecision
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> MeasureClassifPrecision().score(y_true, y_pred)
    1.0

    Notes
    -----
    Defaults to 'binary' average; for multiclass, this measure auto-switches
    to 'weighted' if average='binary' and more than two classes are detected.

    See Also
    --------
    MeasureClassifRecall : Sensitivity (true positive rate)
    MeasureClassifF1 : Harmonic mean of precision and recall
    """
    
    def __init__(self, average: str = 'binary', pos_label=None):
        super().__init__(
            id='classif.precision',
            minimize=False,
            range=(0, 1),
            predict_type='response',
            average=average
        )
        self.pos_label = pos_label
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate precision score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")

        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan

        # Detect number of classes
        unique_classes = np.unique(prediction.truth[mask])
        n_classes = len(unique_classes)

        # Auto-adjust average parameter based on number of classes
        average = self.average
        pos_label = self.pos_label

        if average == 'binary':
            if n_classes == 2:
                # Binary classification - use binary average
                pos_label = self.pos_label if self.pos_label is not None else unique_classes[1]
            else:
                # Multiclass - auto-switch to weighted
                average = 'weighted'
                pos_label = None

        return sklearn_metrics.precision_score(
            prediction.truth[mask],
            prediction.response[mask],
            average=average,
            pos_label=pos_label,
            zero_division=0
        )


@register_measure
class MeasureClassifRecall(MeasureClassif):
    """Recall score (sensitivity, true positive rate).
    
    Wraps sklearn's ``recall_score`` with MLPY's unified interface.
    Proportion of actual positives correctly identified; useful when the
    cost of false negatives is high.
    
    Parameters
    ----------
    average : str, default='binary'
        Averaging method for multiclass.
    pos_label : str or int, optional
        The positive class for binary classification.

    Examples
    --------
    >>> import numpy as np
    >>> from mlpy.measures.classification import MeasureClassifRecall
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> MeasureClassifRecall().score(y_true, y_pred)
    0.5

    Notes
    -----
    Defaults to 'binary' average; for multiclass, this measure auto-switches
    to 'weighted' if average='binary' and more than two classes are detected.

    See Also
    --------
    MeasureClassifPrecision : Positive predictive value
    MeasureClassifF1 : Harmonic mean of precision and recall
    """
    
    def __init__(self, average: str = 'binary', pos_label=None):
        super().__init__(
            id='classif.recall',
            minimize=False,
            range=(0, 1),
            predict_type='response',
            average=average
        )
        self.pos_label = pos_label
        
    def _score(self, prediction: PredictionClassif, task=None, **kwargs) -> float:
        """Calculate recall score."""
        if prediction.truth is None:
            raise ValueError("Prediction must have truth values for scoring")

        mask = ~(pd.isna(prediction.truth) | pd.isna(prediction.response))
        if not mask.any():
            return np.nan

        # Detect number of classes
        unique_classes = np.unique(prediction.truth[mask])
        n_classes = len(unique_classes)

        # Auto-adjust average parameter based on number of classes
        average = self.average
        pos_label = self.pos_label

        if average == 'binary':
            if n_classes == 2:
                # Binary classification - use binary average
                pos_label = self.pos_label if self.pos_label is not None else unique_classes[1]
            else:
                # Multiclass - auto-switch to weighted
                average = 'weighted'
                pos_label = None

        return sklearn_metrics.recall_score(
            prediction.truth[mask],
            prediction.response[mask],
            average=average,
            pos_label=pos_label,
            zero_division=0
        )


# Create some convenience measures with specific averaging
register_measure(lambda: MeasureSimple(
    'classif.bacc',
    score_func=sklearn_metrics.balanced_accuracy_score,
    task_type='classif',
    minimize=False,
    range=(0, 1),
    predict_type='response'
))()

register_measure(lambda: MeasureSimple(
    'classif.mcc',
    score_func=sklearn_metrics.matthews_corrcoef,
    task_type='classif', 
    minimize=False,
    range=(-1, 1),
    predict_type='response'
))()

# Aliases
from ..utils.registry import mlpy_measures as _mlpy_measures
MeasureClassifBalancedAccuracy = lambda: _mlpy_measures['classif.bacc']
MeasureClassifMCC = lambda: _mlpy_measures['classif.mcc']