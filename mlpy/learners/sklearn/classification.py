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

    Wraps sklearn's LogisticRegression with MLPY's unified interface.
    Logistic Regression is a linear model for classification that predicts probabilities 
    using the sigmoid function. Suitable for binary and multiclass classification tasks.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    penalty : {"l1", "l2", "elasticnet", "none"}, default="l2"
        Norm used in the penalization.
    C : float, default=1.0
        Inverse of regularization strength.
    solver : {"lbfgs", "liblinear", "saga", ...}, default="lbfgs"
        Optimization algorithm.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.linear_model.LogisticRegression``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerLogisticRegression
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerLogisticRegression(max_iter=200)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    This is a wrapper around ``sklearn.linear_model.LogisticRegression``.
    For more details, see the sklearn documentation.

    See Also
    --------
    LearnerSVM : Support Vector Machine classifier
    LearnerDecisionTree : Decision tree classifier
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

    Wraps sklearn's DecisionTreeClassifier with MLPY's unified interface.
    Decision Trees recursively partition the feature space with axis-aligned splits
    to learn simple rules. Suitable for non-linear relationships, mixed feature
    types, and interpretable models.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Function to measure the quality of a split.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.tree.DecisionTreeClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerDecisionTree
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerDecisionTree(max_depth=3)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> len(pred.response) == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.tree.DecisionTreeClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Breiman, Friedman, Olshen, and Stone (1984).
           Classification and Regression Trees. Wadsworth.

    See Also
    --------
    LearnerRandomForest : Random forest classifier
    LearnerExtraTrees : Extremely randomized trees classifier
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

    Wraps sklearn's RandomForestClassifier with MLPY's unified interface.
    Random Forests build an ensemble of decision trees on bootstrapped samples
    with feature randomness and aggregate their votes. Suitable for robust
    performance with minimal tuning and for estimating feature importance.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Function to measure the quality of a split.
    max_depth : int or None, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {"sqrt", "log2"} or float, default="sqrt"
        Number of features to consider when looking for the best split.
    random_state : int, default=None
        Random seed for reproducibility.
    n_jobs : int, default=None
        Number of parallel jobs to run.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.RandomForestClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerRandomForest
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerRandomForest(n_estimators=200, random_state=0)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> hasattr(pred, "prob")
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.RandomForestClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

    See Also
    --------
    LearnerDecisionTree : Single decision tree classifier
    LearnerExtraTrees : Extremely randomized trees classifier
    LearnerGradientBoosting : Gradient boosting classifier
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

    Wraps sklearn's GradientBoostingClassifier with MLPY's unified interface. Builds an additive ensemble of
    shallow trees that sequentially correct previous errors. Suitable for
    tabular data with complex patterns; strong accuracy but sensitive to
    hyperparameters.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of individual regression trees.
    subsample : float, default=1.0
        Fraction of samples used for fitting the base learners.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.GradientBoostingClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerGradientBoosting
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerGradientBoosting(n_estimators=150, learning_rate=0.05)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> np.unique(pred.response).size <= task.n_classes
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.GradientBoostingClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Friedman, J. H. (2001). Greedy Function Approximation:
           A Gradient Boosting Machine. Annals of Statistics.

    See Also
    --------
    LearnerAdaBoost : Adaptive boosting classifier
    LearnerRandomForest : Random forest classifier
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
    """AdaBoost classifier wrapper.

    Wraps sklearn's AdaBoostClassifier with MLPY's unified interface. Reweights misclassified samples to focus
    on hard cases and combines many weak learners. Suitable with simple base
    estimators; can be sensitive to noise and outliers.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    estimator : object, default=None
        Base estimator to boost. If None, uses a decision stump.
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.AdaBoostClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerAdaBoost
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerAdaBoost(n_estimators=75)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.AdaBoostClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Freund, Y. and Schapire, R. E. (1997). A Decision-Theoretic
           Generalization of On-Line Learning and an Application to Boosting.

    See Also
    --------
    LearnerGradientBoosting : Gradient boosting classifier
    LearnerDecisionTree : Decision tree classifier
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
    """Extremely Randomized Trees (Extra Trees) classifier wrapper.

    Wraps sklearn's ExtraTreesClassifier with MLPY's unified interface. Uses fully random feature/threshold
    splits across many trees to reduce variance. Suitable for fast, robust
    models on high-dimensional data.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Function to measure the quality of a split.
    max_depth : int or None, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {"sqrt", "log2"} or float, default="sqrt"
        Number of features to consider when looking for the best split.
    random_state : int, default=None
        Random seed for reproducibility.
    n_jobs : int, default=None
        Number of parallel jobs to run.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.ExtraTreesClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerExtraTrees
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerExtraTrees(n_estimators=300, random_state=42)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> bool(np.isnan(pred.response).sum() == 0)
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.ExtraTreesClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Geurts, P., Ernst, D., and Wehenkel, L. (2006).
           Extremely Randomized Trees. Machine Learning, 63, 3–42.

    See Also
    --------
    LearnerRandomForest : Random forest classifier
    LearnerDecisionTree : Decision tree classifier
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

    Wraps sklearn's SVC with MLPY's unified interface. Maximizes the margin between
    classes and can use kernels for non-linear boundaries. Suitable for
    high-dimensional data; probability estimates available when enabled.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    C : float, default=1.0
        Regularization parameter.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ("poly").
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel coefficient for "rbf", "poly" and "sigmoid".
    probability : bool, default=False
        Enable probability estimates. Automatically set to True when
        ``predict_type="prob"``.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to ``sklearn.svm.SVC``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerSVM
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerSVM(kernel="rbf", predict_type="prob")
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.prob is not None
    True

    Notes
    -----
    Wrapper around ``sklearn.svm.SVC``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Cortes, C. and Vapnik, V. (1995). Support-Vector Networks.

    See Also
    --------
    LearnerLogisticRegression : Linear classifier with probabilistic output
    LearnerKNN : k-nearest neighbors classifier
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

    Wraps sklearn's KNeighborsClassifier with MLPY's unified interface. Predicts by majority vote of the
    closest points in feature space. Suitable for small to medium datasets;
    sensitive to feature scaling and the choice of k.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used to compute the nearest neighbors.
    metric : str, default="minkowski"
        Distance metric to use for the tree.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.neighbors.KNeighborsClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerKNN
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerKNN(n_neighbors=7)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> bool(np.isnan(pred.response).any())
    False

    Notes
    -----
    Wrapper around ``sklearn.neighbors.KNeighborsClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Cover, T. and Hart, P. (1967). Nearest neighbor pattern
           classification. IEEE Transactions on Information Theory.

    See Also
    --------
    LearnerSVM : Support Vector Machine classifier
    LearnerLogisticRegression : Logistic regression classifier
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

    Wraps sklearn's GaussianNB with MLPY's unified interface. A probabilistic classifier assuming conditional
    independence with Gaussian likelihoods. Suitable as a fast baseline, often
    effective with limited data and high-dimensional features.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features added to variances.
    **kwargs
        Additional estimator parameters forwarded to ``sklearn.naive_bayes.GaussianNB``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerNaiveBayes
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerNaiveBayes()
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response[:3].shape[0] == 3
    True

    Notes
    -----
    Wrapper around ``sklearn.naive_bayes.GaussianNB``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Mitchell, TM. (1997). Machine Learning. McGraw-Hill. (Naive Bayes)

    See Also
    --------
    LearnerLogisticRegression : Logistic regression classifier
    LearnerKNN : k-nearest neighbors classifier
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
    """Multi-layer Perceptron (neural network) classifier wrapper.

    Wraps sklearn's MLPClassifier with MLPY's unified interface. A feedforward neural
    network trained with backpropagation for classification. Suitable for
    non-linear decision boundaries; may require tuning and feature scaling.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    predict_type : {"response", "prob"}, default="response"
        Type of prediction to produce.
    hidden_layer_sizes : tuple, default=(100,)
        Number of neurons in each hidden layer.
    activation : {"identity", "logistic", "tanh", "relu"}, default="relu"
        Activation function for the hidden layer.
    solver : {"lbfgs", "sgd", "adam"}, default="adam"
        The solver for weight optimization.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    learning_rate : {"constant", "invscaling", "adaptive"}, default="constant"
        Learning rate schedule for weight updates.
    max_iter : int, default=200
        Maximum number of iterations.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.neural_network.MLPClassifier``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskClassif
    >>> from mlpy.learners.sklearn import LearnerMLPClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> df['target'] = iris.target
    >>> task = TaskClassif(data=df, target='target')
    >>> learner = LearnerMLPClassifier(max_iter=300, random_state=0)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.size == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.neural_network.MLPClassifier``.
    For more details, see the sklearn documentation.

    References
    ----------
    .. [1] Rumelhart, Hinton, and Williams (1986). Learning Representations
           by Back-Propagating Errors. Nature.

    See Also
    --------
    LearnerSVM : Support Vector Machine classifier
    LearnerRandomForest : Random forest classifier
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