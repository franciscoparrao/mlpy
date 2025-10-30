"""Scikit-learn regression model wrappers."""

from typing import Optional

from .base import LearnerRegrSKLearn

# Import sklearn models with error handling
try:
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class LearnerLinearRegression(LearnerRegrSKLearn):
    """Linear Regression wrapper.

    Wraps sklearn's LinearRegression with MLPY's unified interface. Ordinary
    least squares linear model; a strong baseline when relationships are
    approximately linear.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    copy_X : bool, default=True
        Whether to copy X.
    n_jobs : int, default=None
        Number of parallel jobs.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.linear_model.LinearRegression``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerLinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerLinearRegression()
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> len(pred.response) == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.linear_model.LinearRegression``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Seber, G. A. F., and Lee, A. J. (2003).
        Linear Regression Analysis. Wiley.

    See Also
    --------
    LearnerRidge : Ridge regression (L2 penalty)
    LearnerLasso : Lasso regression (L1 penalty)
    LearnerElasticNet : Elastic net regression (L1+L2)
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=LinearRegression,
            id=id or "linear_regression",
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            **kwargs
        )


class LearnerRidge(LearnerRegrSKLearn):
    """Ridge Regression wrapper.

    Wraps sklearn's Ridge with MLPY's unified interface. Linear regression with
    L2 regularization to reduce variance and handle multicollinearity.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    alpha : float, default=1.0
        Regularization strength (L2 penalty).
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    solver : {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"}, default="auto"
        Solver to use.
    max_iter : int, default=None
        Maximum iterations for iterative solvers.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.linear_model.Ridge``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerRidge
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerRidge(alpha=0.5)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.linear_model.Ridge``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Hoerl, A. E., and Kennard, R. W. (1970).
        Ridge Regression: Biased Estimation for Nonorthogonal Problems.
        Technometrics, 12(1), 55–67.

    See Also
    --------
    LearnerLasso : L1-regularized regression
    LearnerElasticNet : Combined L1/L2 regularization
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = 'auto',
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=Ridge,
            id=id or "ridge",
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )


class LearnerLasso(LearnerRegrSKLearn):
    """Lasso Regression wrapper.

    Wraps sklearn's Lasso with MLPY's unified interface. Linear regression with
    L1 regularization that can drive coefficients to zero for feature selection.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    alpha : float, default=1.0
        Regularization strength (L1 penalty).
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    selection : {"cyclic", "random"}, default="cyclic"
        Coordinate descent selection strategy.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.linear_model.Lasso``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerLasso
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerLasso(alpha=0.01, max_iter=2000)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.size == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.linear_model.Lasso``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Tibshirani, R. (1996). Regression Shrinkage and Selection
        via the Lasso. Journal of the Royal Statistical Society:
        Series B (Methodological), 58(1), 267–288.

    See Also
    --------
    LearnerRidge : L2-regularized regression
    LearnerElasticNet : Combined L1/L2 regularization
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        selection: str = 'cyclic',
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=Lasso,
            id=id or "lasso",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state,
            **kwargs
        )


class LearnerElasticNet(LearnerRegrSKLearn):
    """Elastic Net Regression wrapper.

    Wraps sklearn's ElasticNet with MLPY's unified interface. Linear regression
    with combined L1/L2 penalties; balances sparsity and stability.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    alpha : float, default=1.0
        Overall regularization strength.
    l1_ratio : float, default=0.5
        The mixing parameter between L1 (1.0) and L2 (0.0).
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    selection : {"cyclic", "random"}, default="cyclic"
        Coordinate descent selection strategy.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.linear_model.ElasticNet``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerElasticNet
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerElasticNet(alpha=0.1, l1_ratio=0.7)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.linear_model.ElasticNet``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Zou, H., and Hastie, T. (2005). Regularization and variable
        selection via the elastic net. Journal of the Royal Statistical
        Society: Series B (Statistical Methodology), 67(2), 301–320.

    See Also
    --------
    LearnerRidge : L2-regularized regression
    LearnerLasso : L1-regularized regression
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        selection: str = 'cyclic',
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=ElasticNet,
            id=id or "elastic_net",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            selection=selection,
            random_state=random_state,
            **kwargs
        )


class LearnerDecisionTreeRegressor(LearnerRegrSKLearn):
    """Decision Tree Regressor wrapper.

    Wraps sklearn's DecisionTreeRegressor with MLPY's unified interface.
    Recursively partitions the feature space using axis-aligned splits; handles
    non-linear relationships and mixed feature types.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"
        Function to measure split quality.
    splitter : {"best", "random"}, default="best"
        Strategy used to choose the split at each node.
    max_depth : int, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.tree.DecisionTreeRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerDecisionTreeRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerDecisionTreeRegressor(max_depth=4)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response[:5].shape[0] == 5
    True

    Notes
    -----
    Wrapper around ``sklearn.tree.DecisionTreeRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. (1984).
        Classification and Regression Trees. Wadsworth.

    See Also
    --------
    LearnerRandomForestRegressor : Random forest regressor
    LearnerExtraTreesRegressor : Extremely randomized trees regressor
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        criterion: str = 'squared_error',
        splitter: str = 'best',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=DecisionTreeRegressor,
            id=id or "decision_tree_regr",
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )


class LearnerRandomForestRegressor(LearnerRegrSKLearn):
    """Random Forest Regressor wrapper.

    Wraps sklearn's RandomForestRegressor with MLPY's unified interface. An
    ensemble of decision trees trained on bootstrapped samples with feature
    randomness; strong default performance and robust to overfitting.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"squared_error", "absolute_error", "poisson"}, default="squared_error"
        Function to measure split quality.
    max_depth : int, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    max_features : float or str, default=1.0
        Features considered when looking for the best split.
    random_state : int, default=None
        Random seed.
    n_jobs : int, default=None
        Number of parallel jobs.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.RandomForestRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerRandomForestRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerRandomForestRegressor(n_estimators=200, random_state=0)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> hasattr(pred, "response")
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.RandomForestRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Breiman, L. (2001). Random Forests. Machine Learning,
        45(1), 5–32.

    See Also
    --------
    LearnerDecisionTreeRegressor : Single decision tree regressor
    LearnerExtraTreesRegressor : Extremely randomized trees regressor
    LearnerGradientBoostingRegressor : Gradient boosting regressor
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        n_estimators: int = 100,
        criterion: str = 'squared_error',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=RandomForestRegressor,
            id=id or "random_forest_regr",
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


class LearnerGradientBoostingRegressor(LearnerRegrSKLearn):
    """Gradient Boosting Regressor wrapper.

    Wraps sklearn's GradientBoostingRegressor with MLPY's unified interface.
    Builds an additive ensemble of shallow trees, sequentially correcting
    previous errors; strong on tabular data but sensitive to hyperparameters.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    loss : {"squared_error", "absolute_error", "huber", "quantile"}, default="squared_error"
        Loss function to optimize.
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree.
    n_estimators : int, default=100
        Number of boosting stages.
    subsample : float, default=1.0
        Fraction of samples used for fitting the base learners.
    criterion : {"friedman_mse", "squared_error"}, default="friedman_mse"
        Function to measure split quality.
    min_samples_split : int, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf.
    max_depth : int, default=3
        Maximum depth of the individual trees.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.GradientBoostingRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerGradientBoostingRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerGradientBoostingRegressor(n_estimators=150, learning_rate=0.05)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> bool(np.isfinite(pred.response).all())
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.GradientBoostingRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Friedman, J. H. (2001). Greedy Function Approximation:
        A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232.

    See Also
    --------
    LearnerAdaBoostRegressor : Adaptive boosting regressor
    LearnerRandomForestRegressor : Random forest regressor
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        loss: str = 'squared_error',
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = 'friedman_mse',
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=GradientBoostingRegressor,
            id=id or "gradient_boosting_regr",
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

class LearnerAdaBoostRegressor(LearnerRegrSKLearn):
    """AdaBoost Regressor wrapper.

    Wraps sklearn's AdaBoostRegressor with MLPY's unified interface. Iteratively
    reweights errors to focus on hard cases; works well with simple base
    estimators; can be sensitive to noise.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    estimator : object, default=None
        Base estimator to boost; if None, uses a decision tree stump.
    n_estimators : int, default=50
        Number of boosting stages.
    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration.
    random_state : int, default=None
        Random seed.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.AdaBoostRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerAdaBoostRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerAdaBoostRegressor(n_estimators=75)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.AdaBoostRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Drucker, H. (1997). Improving Regressors using Boosting Techniques.
        ICML.


    See Also
    --------
    LearnerGradientBoostingRegressor : Gradient boosting regressor
    LearnerDecisionTreeRegressor : Decision tree regressor
    """

    def __init__(
        self,
        id: Optional[str] = None,
        estimator: Optional[object] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")

        super().__init__(
            estimator_class=AdaBoostRegressor,
            id=id or "adaboost_regr",
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

class LearnerExtraTreesRegressor(LearnerRegrSKLearn):
    """Extra Trees Regressor wrapper.

    Wraps sklearn's ExtraTreesRegressor with MLPY's unified interface. Uses
    random feature/threshold splits across many trees to reduce variance; fast
    and robust for high-dimensional data.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        Function to measure split quality.
    max_depth : int, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    max_features : float, default=1.0
        Features considered when looking for the best split.
    random_state : int, default=None
        Random seed.
    n_jobs : int, default=None
        Number of parallel jobs.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.ensemble.ExtraTreesRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerExtraTreesRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerExtraTreesRegressor(n_estimators=300, random_state=42)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> bool(np.isnan(pred.response).sum() == 0)
    True

    Notes
    -----
    Wrapper around ``sklearn.ensemble.ExtraTreesRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Geurts, P., Ernst, D., and Wehenkel, L. (2006).
        Extremely Randomized Trees. Machine Learning, 63, 3–42.

    See Also
    --------
    LearnerRandomForestRegressor : Random forest regressor
    LearnerDecisionTreeRegressor : Decision tree regressor
    """

    def __init__(
        self,
        id: Optional[str] = None,
        n_estimators: int = 100,
        criterion: str = 'squared_error',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")

        super().__init__(
            estimator_class=ExtraTreesRegressor,
            id=id or "extra_trees_regr",
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

class LearnerSVR(LearnerRegrSKLearn):
    """Support Vector Regression wrapper.

    Wraps sklearn's SVR with MLPY's unified interface. Maximizes margin with an
    epsilon-insensitive loss; kernels enable non-linear relationships.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        Kernel type.
    degree : int, default=3
        Degree for the polynomial kernel.
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel coefficient.
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model.
    **kwargs
        Additional estimator parameters forwarded to ``sklearn.svm.SVR``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerSVR
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerSVR(kernel="rbf", C=2.0, epsilon=0.2)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.shape[0] == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.svm.SVR``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Vapnik, V. N. (1995). The Nature of Statistical Learning Theory.
        Springer.
    .. [2] Smola, A. J., and Schölkopf, B. (2004). A Tutorial on Support
        Vector Regression. Statistics and Computing, 14(3), 199–222.

    See Also
    --------
    LearnerKNNRegressor : k-nearest neighbors regressor
    LearnerLinearRegression : Linear regression baseline
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        kernel: str = 'rbf',
        degree: int = 3,
        gamma: str = 'scale',
        C: float = 1.0,
        epsilon: float = 0.1,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=SVR,
            id=id or "svr",
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            C=C,
            epsilon=epsilon,
            **kwargs
        )


class LearnerKNNRegressor(LearnerRegrSKLearn):
    """K-Nearest Neighbors Regressor wrapper.

    Wraps sklearn's KNeighborsRegressor with MLPY's unified interface. Predicts
    by averaging the targets of the closest neighbors; sensitive to scaling and
    the choice of k.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    n_neighbors : int, default=5
        Number of neighbors.
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size for tree-based algorithms.
    metric : str, default="minkowski"
        Distance metric.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.neighbors.KNeighborsRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerKNNRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerKNNRegressor(n_neighbors=7)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> bool(np.isfinite(pred.response).all())
    True

    Notes
    -----
    Wrapper around ``sklearn.neighbors.KNeighborsRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Cover, T. and Hart, P. (1967). Nearest neighbor pattern
        classification. IEEE Transactions on Information Theory.

    See Also
    --------
    LearnerSVR : Support Vector Regression
    LearnerLinearRegression : Linear regression baseline
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        metric: str = 'minkowski',
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=KNeighborsRegressor,
            id=id or "knn_regr",
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            **kwargs
        )


class LearnerMLPRegressor(LearnerRegrSKLearn):
    """Multi-layer Perceptron (neural network) Regressor wrapper.

    Wraps sklearn's MLPRegressor with MLPY's unified interface. A feedforward
    neural network trained with backpropagation; models non-linear relationships
    but may require tuning and feature scaling.

    Parameters
    ----------
    id : str, default=None
        Unique identifier for the learner.
    hidden_layer_sizes : tuple, default=(100,)
        Number of neurons in each hidden layer.
    activation : {"identity", "logistic", "tanh", "relu"}, default="relu"
        Activation function for the hidden layer.
    solver : {"lbfgs", "sgd", "adam"}, default="adam"
        The solver for weight optimization.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    batch_size : {"auto"} or int, default="auto"
        Size of minibatches.
    learning_rate : {"constant", "invscaling", "adaptive"}, default="constant"
        Learning rate schedule for weight updates.
    learning_rate_init : float, default=0.001
        Initial learning rate for weight updates.
    max_iter : int, default=200
        Maximum number of iterations.
    random_state : int, default=None
        Random seed for reproducibility.
    **kwargs
        Additional estimator parameters forwarded to
        ``sklearn.neural_network.MLPRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks import TaskRegr
    >>> from mlpy.learners.sklearn import LearnerMLPRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> task = TaskRegr(data=df, target='target')
    >>> learner = LearnerMLPRegressor(max_iter=300, random_state=0)
    >>> _ = learner.train(task)
    >>> pred = learner.predict(task)
    >>> pred.response.size == len(df)
    True

    Notes
    -----
    Wrapper around ``sklearn.neural_network.MLPRegressor``.
    See the scikit-learn documentation.

    References
    ----------
    .. [1] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986).
        Learning Representations by Back-Propagating Errors. Nature.

    See Also
    --------
    LearnerSVR : Kernel-based regression
    LearnerRandomForestRegressor : Ensemble tree-based regression
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        hidden_layer_sizes: tuple = (100,),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        batch_size: str = 'auto',
        learning_rate: str = 'constant',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for this learner")
            
        super().__init__(
            estimator_class=MLPRegressor,
            id=id or "mlp_regr",
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )