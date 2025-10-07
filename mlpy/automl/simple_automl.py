"""
Simple AutoML interface for MLPY.

Provides a unified, easy-to-use interface for automated machine learning
that combines all MLPY components into a cohesive workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import warnings
import time
from pathlib import Path

from ..tasks import TaskClassif, TaskRegr, Task
from ..learners import LearnerClassifSklearn, LearnerRegrSklearn
from ..measures import MeasureClassifAccuracy, MeasureRegrMSE
from ..resamplings import ResamplingCV, ResamplingHoldout
from ..pipelines import Graph, GraphLearner, PipeOpScale, PipeOpImpute, PipeOpFilter
from ..filters import filter_features
from .feature_engineering import AutoFeaturesNumeric, AutoFeaturesCategorical, AutoFeaturesInteraction
from .tuning import TunerRandom, ParamSet, ParamInt, ParamFloat, ParamCategorical
from .meta_learning import MetaLearner, DatasetCharacteristics


@dataclass
class AutoMLResult:
    """Result of AutoML training.
    
    Attributes
    ----------
    best_learner : GraphLearner
        The best pipeline found.
    best_score : float
        Score of the best pipeline.
    leaderboard : pd.DataFrame
        All tried configurations with scores.
    feature_importance : pd.Series, optional
        Feature importance from best model.
    task : Task
        Original task.
    training_time : float
        Total training time in seconds.
    meta_info : Dict[str, Any], optional
        Meta-learning information and insights.
    """
    best_learner: GraphLearner
    best_score: float
    leaderboard: pd.DataFrame
    feature_importance: Optional[pd.Series] = None
    task: Optional[Task] = None
    training_time: float = 0.0
    meta_info: Optional[Dict[str, Any]] = None
    
    def predict(self, data: Union[pd.DataFrame, Task]):
        """Make predictions with the best model."""
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to Task
            task_class = type(self.task)
            pred_task = task_class(
                data=data,
                id="prediction",
                label="Prediction Data"
            )
        else:
            pred_task = data
            
        return self.best_learner.predict(pred_task)
    
    def plot_leaderboard(self, top_n=10, figsize=(10, 6)):
        """Plot model performance leaderboard."""
        import matplotlib.pyplot as plt
        
        top_models = self.leaderboard.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_models)), top_models['score'])
        plt.yticks(range(len(top_models)), top_models['model'])
        plt.xlabel('Score')
        plt.title(f'Top {top_n} Models')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    def save(self, path: str):
        """Save the AutoML result."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a saved AutoML result."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class SimpleAutoML:
    """Simple AutoML interface for MLPY.
    
    This class provides an easy-to-use interface for automated machine learning
    that handles:
    - Data preprocessing
    - Feature engineering
    - Feature selection
    - Model selection
    - Hyperparameter tuning
    - Model evaluation
    
    Parameters
    ----------
    time_limit : int, default=300
        Time limit in seconds for AutoML search.
    max_models : int, default=50
        Maximum number of models to try.
    feature_engineering : bool, default=True
        Whether to perform automatic feature engineering.
    feature_selection : bool, default=True
        Whether to perform feature selection.
    cross_validation : int, default=5
        Number of CV folds for model evaluation.
    test_size : float, default=0.2
        Fraction of data to hold out for testing.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : bool, default=True
        Whether to print progress information.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.automl import SimpleAutoML
    >>> 
    >>> # Load your data
    >>> data = pd.read_csv('data.csv')
    >>> 
    >>> # Create and run AutoML
    >>> automl = SimpleAutoML(time_limit=600)
    >>> result = automl.fit(data, target='target_column')
    >>> 
    >>> # Make predictions
    >>> predictions = result.predict(test_data)
    >>> 
    >>> # View results
    >>> print(f"Best score: {result.best_score:.3f}")
    >>> result.plot_leaderboard()
    """
    
    def __init__(
        self,
        time_limit: int = 300,
        max_models: int = 50,
        feature_engineering: bool = True,
        feature_selection: bool = True,
        cross_validation: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
        meta_learning: bool = True
    ):
        self.time_limit = time_limit
        self.max_models = max_models
        self.feature_engineering = feature_engineering
        self.feature_selection = feature_selection
        self.cross_validation = cross_validation
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.meta_learning = meta_learning
        
        # Initialize meta-learner if enabled
        if self.meta_learning:
            self._meta_learner = MetaLearner()
        else:
            self._meta_learner = None
        
        # Runtime state
        self._leaderboard = []
        self._start_time = None
        
    def fit(
        self, 
        data: pd.DataFrame, 
        target: str,
        task_type: Optional[str] = None
    ) -> AutoMLResult:
        """Fit AutoML on the provided data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data with features and target.
        target : str
            Name of the target column.
        task_type : str, optional
            Task type: 'classification' or 'regression'.
            If None, will be inferred.
            
        Returns
        -------
        AutoMLResult
            Complete results of the AutoML search.
        """
        self._start_time = time.time()
        self._leaderboard = []
        
        if self.verbose:
            print("🚀 Starting SimpleAutoML...")
            print(f"⏰ Time limit: {self.time_limit}s")
            print(f"🎯 Target: {target}")
            print(f"📊 Data shape: {data.shape}")
        
        # 1. Create task
        task = self._create_task(data, target, task_type)
        
        if self.verbose:
            print(f"📋 Task type: {task.__class__.__name__}")
            print(f"🔧 Features: {len(task.feature_names)}")
        
        # 1.5. Meta-learning analysis (if enabled)
        meta_info = None
        if self._meta_learner is not None:
            if self.verbose:
                print("🧠 Analyzing dataset characteristics...")
            meta_info = self._analyze_with_meta_learning(task)
        
        # 2. Split data
        train_task, test_task = self._split_data(task, meta_info)
        
        # 3. Build pipelines and search
        best_learner, best_score = self._search_pipelines(train_task)
        
        # 4. Final evaluation
        final_score = self._final_evaluation(best_learner, test_task)
        
        # 5. Feature importance
        feature_importance = self._get_feature_importance(best_learner, train_task)
        
        # 6. Create result
        result = AutoMLResult(
            best_learner=best_learner,
            best_score=final_score,
            leaderboard=pd.DataFrame(self._leaderboard).sort_values('score', ascending=False),
            feature_importance=feature_importance,
            task=task,
            training_time=time.time() - self._start_time,
            meta_info=meta_info
        )
        
        if self.verbose:
            print(f"\n✅ AutoML completed!")
            print(f"⭐ Best score: {final_score:.4f}")
            print(f"⏱️  Total time: {result.training_time:.1f}s")
            print(f"🔍 Models tried: {len(self._leaderboard)}")
        
        return result
        
    def _create_task(self, data: pd.DataFrame, target: str, task_type: Optional[str]) -> Task:
        """Create appropriate task from data."""
        if task_type is None:
            # Infer task type
            target_series = data[target]
            if pd.api.types.is_numeric_dtype(target_series):
                n_unique = target_series.nunique()
                if n_unique <= 10 and n_unique < len(target_series) * 0.05:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
            else:
                task_type = 'classification'
        
        if task_type == 'classification':
            return TaskClassif(
                data=data,
                target=target,
                id="automl_task",
                label="AutoML Task"
            )
        else:
            return TaskRegr(
                data=data,
                target=target,
                id="automl_task", 
                label="AutoML Task"
            )
    
    def _split_data(self, task: Task, meta_info: Optional[Dict[str, Any]] = None) -> tuple:
        """Split data into train/test, optionally using meta-learning recommendations."""
        # Use meta-learning recommendations if available
        if meta_info and 'cv_strategy' in meta_info:
            cv_strategy = meta_info['cv_strategy']
            if cv_strategy['method'] == 'holdout':
                ratio = cv_strategy.get('ratio', self.test_size)
                stratify = cv_strategy.get('stratify', isinstance(task, TaskClassif))
            else:
                ratio = self.test_size
                stratify = isinstance(task, TaskClassif)
        else:
            ratio = self.test_size
            stratify = isinstance(task, TaskClassif)
        
        holdout = ResamplingHoldout(ratio=ratio, stratify=stratify)
        instance = holdout.instantiate(task)
        
        train_indices = instance.train_set(0)  # First (and only) iteration (0-based)
        test_indices = instance.test_set(0)
        
        train_task = task.filter(train_indices)
        test_task = task.filter(test_indices)
        
        return train_task, test_task
    
    def _search_pipelines(self, train_task: Task) -> tuple:
        """Search for best pipeline configuration."""
        if self.verbose:
            print(f"\n🔍 Starting pipeline search...")
        
        best_learner = None
        best_score = -float('inf') if isinstance(train_task, TaskRegr) else 0.0
        models_tried = 0
        
        # Define base learners to try
        base_learners = self._get_base_learners(train_task)
        
        # Define preprocessing combinations
        preprocessing_configs = self._get_preprocessing_configs(train_task)
        
        for learner_name, learner_class, param_set in base_learners:
            if self._time_exceeded() or models_tried >= self.max_models:
                break
                
            for prep_config in preprocessing_configs:
                if self._time_exceeded() or models_tried >= self.max_models:
                    break
                    
                try:
                    # Build pipeline
                    graph_learner = self._build_pipeline(
                        train_task, learner_class, prep_config
                    )
                    
                    # Evaluate
                    score = self._evaluate_pipeline(graph_learner, train_task)
                    
                    # Track results
                    self._leaderboard.append({
                        'model': f"{learner_name}_{prep_config['name']}",
                        'score': score,
                        'learner': learner_name,
                        'preprocessing': prep_config['name']
                    })
                    
                    # Check if best
                    if self._is_better_score(score, best_score, train_task):
                        best_score = score
                        best_learner = graph_learner
                        
                        if self.verbose:
                            print(f"⭐ New best: {score:.4f} ({learner_name}_{prep_config['name']})")
                    
                    models_tried += 1
                    
                    if self.verbose and models_tried % 10 == 0:
                        elapsed = time.time() - self._start_time
                        print(f"   Tried {models_tried} models in {elapsed:.1f}s")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed {learner_name}_{prep_config['name']}: {str(e)[:50]}")
                    continue
        
        return best_learner, best_score
    
    def _get_base_learners(self, task: Task) -> List[tuple]:
        """Get list of base learners to try."""
        learners = []
        
        if isinstance(task, TaskClassif):
            # Classification learners
            learners.extend([
                ("RandomForest", LearnerClassifSklearn, 
                 {"classifier": "RandomForestClassifier", "n_estimators": 100, "random_state": self.random_state}),
                ("XGBoost", LearnerClassifSklearn,
                 {"classifier": "XGBClassifier", "random_state": self.random_state}),
                ("LogisticRegression", LearnerClassifSklearn,
                 {"classifier": "LogisticRegression", "random_state": self.random_state}),
                ("SVM", LearnerClassifSklearn,
                 {"classifier": "SVC", "random_state": self.random_state, "probability": True}),
                ("GradientBoosting", LearnerClassifSklearn,
                 {"classifier": "GradientBoostingClassifier", "random_state": self.random_state}),
            ])
        else:
            # Regression learners
            learners.extend([
                ("RandomForest", LearnerRegrSklearn,
                 {"regressor": "RandomForestRegressor", "n_estimators": 100, "random_state": self.random_state}),
                ("XGBoost", LearnerRegrSklearn,
                 {"regressor": "XGBRegressor", "random_state": self.random_state}),
                ("LinearRegression", LearnerRegrSklearn,
                 {"regressor": "LinearRegression"}),
                ("SVR", LearnerRegrSklearn,
                 {"regressor": "SVR"}),
                ("GradientBoosting", LearnerRegrSklearn,
                 {"regressor": "GradientBoostingRegressor", "random_state": self.random_state}),
            ])
        
        return learners
    
    def _get_preprocessing_configs(self, task: Task) -> List[Dict]:
        """Get preprocessing configurations to try."""
        configs = [
            {"name": "minimal", "impute": True, "scale": False, "feature_eng": False, "feature_sel": False},
            {"name": "scaled", "impute": True, "scale": True, "feature_eng": False, "feature_sel": False},
        ]
        
        if self.feature_selection and len(task.feature_names) > 10:
            configs.extend([
                {"name": "selected", "impute": True, "scale": True, "feature_eng": False, "feature_sel": True},
            ])
        
        if self.feature_engineering:
            configs.extend([
                {"name": "engineered", "impute": True, "scale": True, "feature_eng": True, "feature_sel": False},
            ])
            
            if self.feature_selection and len(task.feature_names) > 10:
                configs.append({
                    "name": "full", "impute": True, "scale": True, 
                    "feature_eng": True, "feature_sel": True
                })
        
        return configs
    
    def _build_pipeline(self, task: Task, learner_class, prep_config: Dict) -> GraphLearner:
        """Build a pipeline with specified configuration."""
        graph = Graph()
        current_id = "input"
        
        # Add preprocessing steps
        if prep_config.get("impute", False):
            graph.add_pipeop(PipeOpImpute(id="impute"))
            if current_id != "input":
                graph.add_edge("input", current_id, "impute", "input")
            current_id = "impute"
        
        if prep_config.get("scale", False):
            graph.add_pipeop(PipeOpScale(id="scale"))
            if current_id != "input":
                graph.add_edge(current_id, "output", "scale", "input")
            else:
                graph.add_edge("input", "", "scale", "input")
            current_id = "scale"
        
        if prep_config.get("feature_eng", False):
            # Add numeric feature engineering
            graph.add_pipeop(AutoFeaturesNumeric(id="feat_eng_num"))
            if current_id != "input":
                graph.add_edge(current_id, "output", "feat_eng_num", "input")
            else:
                graph.add_edge("input", "", "feat_eng_num", "input")
            current_id = "feat_eng_num"
        
        if prep_config.get("feature_sel", False):
            # Add feature selection
            n_features = min(50, max(5, len(task.feature_names) // 2))
            graph.add_pipeop(PipeOpFilter(id="filter", method="auto", k=n_features))
            if current_id != "input":
                graph.add_edge(current_id, "output", "filter", "input")
            else:
                graph.add_edge("input", "", "filter", "input")
            current_id = "filter"
        
        # Add learner
        learner = learner_class(id="learner")
        graph.add_pipeop(learner)
        
        if current_id != "input":
            graph.add_edge(current_id, "output", "learner", "input")
        else:
            graph.add_edge("input", "", "learner", "input")
        
        return GraphLearner(graph, id="automl_pipeline")
    
    def _evaluate_pipeline(self, graph_learner: GraphLearner, task: Task) -> float:
        """Evaluate pipeline using cross-validation."""
        if isinstance(task, TaskClassif):
            measure = MeasureClassifAccuracy()
        else:
            measure = MeasureRegrMSE()
        
        resampling = ResamplingCV(folds=self.cross_validation, stratify=isinstance(task, TaskClassif))
        
        # Simple cross-validation
        scores = []
        for i in range(self.cross_validation):
            instance = resampling.instantiate(task)
            train_indices = instance.train_set(i)
            test_indices = instance.test_set(i)
            
            train_subset = task.filter(train_indices)
            test_subset = task.filter(test_indices)
            
            # Train and predict
            graph_learner.train(train_subset)
            predictions = graph_learner.predict(test_subset)
            
            # Score
            score = measure.score(predictions, test_subset)
            scores.append(score)
        
        return np.mean(scores)
    
    def _final_evaluation(self, learner: GraphLearner, test_task: Task) -> float:
        """Final evaluation on test set."""
        predictions = learner.predict(test_task)
        
        if isinstance(test_task, TaskClassif):
            measure = MeasureClassifAccuracy()
        else:
            measure = MeasureRegrMSE()
        
        return measure.score(predictions, test_task)
    
    def _get_feature_importance(self, learner: GraphLearner, task: Task) -> Optional[pd.Series]:
        """Extract feature importance if available."""
        try:
            # Try to get from the last pipeop that might have importance
            for pipeop_id in reversed(learner.graph.ids()):
                pipeop = learner.graph.pipeops[pipeop_id]
                if hasattr(pipeop, 'get_importance'):
                    return pipeop.get_importance()
            return None
        except:
            return None
    
    def _is_better_score(self, new_score: float, current_best: float, task: Task) -> bool:
        """Check if new score is better than current best."""
        if isinstance(task, TaskClassif):
            # Higher is better for classification accuracy
            return new_score > current_best
        else:
            # Lower is better for regression MSE
            return new_score < current_best
    
    def _time_exceeded(self) -> bool:
        """Check if time limit exceeded."""
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.time_limit
    
    def _analyze_with_meta_learning(self, task: Task) -> Dict[str, Any]:
        """Analyze task with meta-learning and return recommendations."""
        if self._meta_learner is None:
            return {}
        
        # Extract characteristics
        characteristics = self._meta_learner.extract_characteristics(task)
        
        # Get comprehensive recommendations
        meta_summary = self._meta_learner.get_meta_summary(characteristics)
        
        if self.verbose:
            print(f"📊 Dataset characteristics:")
            print(f"   • Size: {characteristics.size_category} ({characteristics.n_samples:,} samples)")
            print(f"   • Features: {characteristics.n_features} ({characteristics.n_numeric} numeric)")
            print(f"   • Complexity: {characteristics.complexity_category}")
            
            if characteristics.task_type == "classif" and characteristics.n_classes:
                print(f"   • Classes: {characteristics.n_classes}")
            
            if meta_summary['insights']:
                print(f"🔍 Key insights:")
                for insight in meta_summary['insights'][:3]:  # Show top 3 insights
                    print(f"   • {insight}")
            
            # Show algorithm recommendations
            algorithms = meta_summary['algorithms'][:3]  # Top 3
            print(f"🎯 Recommended algorithms:")
            for alg, score in algorithms:
                print(f"   • {alg} (priority: {score:.1f})")
        
        return meta_summary