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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    fit_intercept : bool, optional
        Whether to calculate intercept.
    copy_X : bool, optional
        Whether to copy X or overwrite.
    n_jobs : int, optional
        Number of parallel jobs.
    **kwargs
        Additional parameters for LinearRegression.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    alpha : float, optional
        Regularization strength.
    fit_intercept : bool, optional
        Whether to calculate intercept.
    solver : str, optional
        Solver to use.
    max_iter : int, optional
        Maximum iterations for iterative solvers.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for Ridge.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    alpha : float, optional
        Regularization strength.
    fit_intercept : bool, optional
        Whether to calculate intercept.
    max_iter : int, optional
        Maximum iterations.
    selection : str, optional
        Selection method ('cyclic' or 'random').
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for Lasso.
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
    """ElasticNet Regression wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        Mix parameter between L1 and L2 penalty.
    fit_intercept : bool, optional
        Whether to calculate intercept.
    max_iter : int, optional
        Maximum iterations.
    selection : str, optional
        Selection method ('cyclic' or 'random').
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for ElasticNet.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    criterion : str, optional
        Function to measure split quality.
    splitter : str, optional
        Strategy to choose split.
    max_depth : int, optional
        Maximum depth of tree.
    min_samples_split : int, optional
        Minimum samples required to split.
    min_samples_leaf : int, optional
        Minimum samples required at leaf.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for DecisionTreeRegressor.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
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
        Additional parameters for RandomForestRegressor.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    loss : str, optional
        Loss function to optimize.
    learning_rate : float, optional
        Learning rate.
    n_estimators : int, optional
        Number of boosting stages.
    subsample : float, optional
        Fraction of samples for fitting base learners.
    criterion : str, optional
        Function to measure split quality.
    min_samples_split : int, optional
        Minimum samples required to split.
    min_samples_leaf : int, optional
        Minimum samples required at leaf.
    max_depth : int, optional
        Maximum depth of trees.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for GradientBoostingRegressor.
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
    """ AdaBoost Regressor wrapper.

    Parameters
    ----------
    id : str, optional
        Unique identifier.
    estimator : object, optional
        Base estimator to boost.
    n_estimators : int, optional
        Number of boosting stages.
    learning_rate : float, optional
        Learning rate.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for AdaBoostRegressor.
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
    """ Extra Trees Regressor wrapper.

    Parameters
    ----------
    id : str, optional
        Unique identifier.
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
    max_features : float or str, optional
        Number of features to consider when looking for the best split. Use a
        float in (0, 1], an int, or strings like 'sqrt'/'log2'.
    random_state : int, optional
        Random seed.
    n_jobs : int, optional
        Number of parallel jobs.
    **kwargs
        Additional parameters for ExtraTreesRegressor.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    kernel : str, optional
        Kernel type.
    degree : int, optional
        Degree for polynomial kernel.
    gamma : str or float, optional
        Kernel coefficient.
    C : float, optional
        Regularization parameter.
    epsilon : float, optional
        Epsilon in epsilon-SVR model.
    **kwargs
        Additional parameters for SVR.
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
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    n_neighbors : int, optional
        Number of neighbors.
    weights : str, optional
        Weight function.
    algorithm : str, optional
        Algorithm to compute nearest neighbors.
    leaf_size : int, optional
        Leaf size for tree algorithms.
    metric : str, optional
        Distance metric.
    **kwargs
        Additional parameters for KNeighborsRegressor.
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
    """Multi-layer Perceptron Regressor wrapper.
    
    Parameters
    ----------
    id : str, optional
        Unique identifier.
    hidden_layer_sizes : tuple, optional
        Number of neurons in each hidden layer.
    activation : str, optional
        Activation function.
    solver : str, optional
        Weight optimization solver.
    alpha : float, optional
        L2 penalty parameter.
    batch_size : str or int, optional
        Size of minibatches.
    learning_rate : str, optional
        Learning rate schedule.
    learning_rate_init : float, optional
        Initial learning rate.
    max_iter : int, optional
        Maximum iterations.
    random_state : int, optional
        Random seed.
    **kwargs
        Additional parameters for MLPRegressor.
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