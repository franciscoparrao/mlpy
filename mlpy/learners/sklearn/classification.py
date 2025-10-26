"""Scikit-learn classification model wrappers."""

from typing import Optional

from .base import LearnerClassifSKLearn

# Import sklearn models with error handling
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class LearnerLogisticRegression(LearnerClassifSKLearn):
    """Logistic Regression classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    penalty : str, optional
        Penalty norm ('l1', 'l2', 'elasticnet', 'none').
    C : float, optional
        Inverse of regularization strength.
    solver : str, optional
        Algorithm to use in optimization.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for LogisticRegression.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=LogisticRegression,
            id=id or "logreg",
            predict_type=predict_type,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )


class LearnerDecisionTree(LearnerClassifSKLearn):
    """Decision Tree classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    criterion : str, optional
        Function to measure split quality.
    max_depth : int, optional
        Maximum depth of tree.
    min_samples_split : int, optional
        Minimum samples required to split.
    min_samples_leaf : int, optional
        Minimum samples required at leaf.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for DecisionTreeClassifier.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=DecisionTreeClassifier,
            id=id or "decision_tree",
            predict_type=predict_type,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )


class LearnerRandomForest(LearnerClassifSKLearn):
    """Random Forest classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    n_estimators : int, optional
        Number of trees.
    criterion : str, optional
        Function to measure split quality.
    max_depth : int, optional
        Maximum depth of trees.
    min_samples_split : int, optional
        Minimum samples required to split.
    min_samples_leaf : int, optional
        Minimum samples required at leaf.
    max_features : str or float, optional
        Number of features to consider for best split.
    random_state : int, optional
        Random seed.
    n_jobs : int, optional
        Number of parallel jobs.
    **kwargs
        Additional parameters for RandomForestClassifier.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=RandomForestClassifier,
            id=id or "random_forest",
            predict_type=predict_type,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )


class LearnerGradientBoosting(LearnerClassifSKLearn):
    """Gradient Boosting classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    n_estimators : int, optional
        Number of boosting stages.
    learning_rate : float, optional
        Learning rate.
    max_depth : int, optional
        Maximum depth of trees.
    subsample : float, optional
        Fraction of samples for fitting base learners.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for GradientBoostingClassifier.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=GradientBoostingClassifier,
            id=id or "gradient_boosting",
            predict_type=predict_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )

class LearnerAdaBoost(LearnerClassifSKLearn):
    """ AdaBoost classifier wrapper

    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    estimator : object, optional
        Base estimator to boost.
    n_estimators : int, optional
        Maximum number of estimators.
    learning_rate : float, optional
        Weighting applied to each classifier.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional parameters for AdaBoostClassifier
    """

    def __init__(
      self,
      id: Optional[str] = None,
      predict_type: str = "response",
      estimator: Optional[object] = None,
      n_estimators: int = 50,
      learning_rate: float = 1.0,
      random_state: Optional[int] = None,
      **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")

        super().__init__(
            estimator_class=AdaBoostClassifier,
            id=id or "adaboost",
            predict_type=predict_type,
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

class LearnerExtraTrees(LearnerClassifSKLearn):
    """ Extra Trees classifier wrapper

    Parameters
    ----------

    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    n_estimators : int, optional
        Number of trees.
    criterion : str, optional
        Function to measure split quality.
    max_depth : int, optional
        Maximum depth of trees.
    min_samples_split : int, optional
        Minimum samples required to split.
    min_samples_leaf : int, optional
        Minimum samples required at leaf.
    max_features : str or float, optional
        Number of features to consider when looking for the best split. Use
        'sqrt', 'log2', an int, or a float in (0, 1].
    random_state : int, optional
        Random seed.
    n_jobs : int, optional
        Number of parallel jobs to run for both fit and predict.
    **kwargs
        Additional parameters for ExtraTreesClassifier.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")

        super().__init__(
            estimator_class=ExtraTreesClassifier,
            id=id or "extra_trees",
            predict_type=predict_type,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

class LearnerSVM(LearnerClassifSKLearn):
    """Support Vector Machine classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    C : float, optional
        Regularization parameter.
    kernel : str, optional
        Kernel type.
    degree : int, optional
        Degree for polynomial kernel.
    gamma : str or float, optional
        Kernel coefficient.
    probability : bool, optional
        Enable probability estimates.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for SVC.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: str = 'scale',
        probability: bool = False,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        # Enable probability if predict_type is "prob"
        if predict_type == "prob":
            probability = True
            
        super().__init__(
            estimator_class=SVC,
            id=id or "svm",
            predict_type=predict_type,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )


class LearnerKNN(LearnerClassifSKLearn):
    """K-Nearest Neighbors classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    n_neighbors : int, optional
        Number of neighbors.
    weights : str, optional
        Weight function.
    algorithm : str, optional
        Algorithm to compute nearest neighbors.
    metric : str, optional
        Distance metric.
    **kwargs
        Additional parameters for KNeighborsClassifier.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        metric: str = 'minkowski',
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=KNeighborsClassifier,
            id=id or "knn",
            predict_type=predict_type,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            **kwargs
        )


class LearnerNaiveBayes(LearnerClassifSKLearn):
    """Gaussian Naive Bayes classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    var_smoothing : float, optional
        Portion of largest variance added to variances.
    **kwargs
        Additional parameters for GaussianNB.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        var_smoothing: float = 1e-9,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=GaussianNB,
            id=id or "naive_bayes",
            predict_type=predict_type,
            var_smoothing=var_smoothing,
            **kwargs
        )


class LearnerMLPClassifier(LearnerClassifSKLearn):
    """Multi-layer Perceptron classifier wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    predict_type : str, optional
        Type of prediction ("response" or "prob").
    hidden_layer_sizes : tuple, optional
        Number of neurons in each hidden layer.
    activation : str, optional
        Activation function.
    solver : str, optional
        Weight optimization solver.
    alpha : float, optional
        L2 penalty parameter.
    learning_rate : str, optional
        Learning rate schedule.
    max_iter : int, optional
        Maximum iterations.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for MLPClassifier.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        predict_type: str = "response",
        hidden_layer_sizes: tuple = (100,),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        learning_rate: str = 'constant',
        max_iter: int = 200,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=MLPClassifier,
            id=id or "mlp",
            predict_type=predict_type,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )