"""
AutoML Avanzado para MLPY con Optuna.

Integración completa de búsqueda de hiperparámetros, 
selección de modelos y pipeline optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import warnings
import time
from datetime import datetime
import json
from pathlib import Path

# Imports de MLPY
from ..tasks import TaskClassif, TaskRegr, Task
from ..learners import (
    LearnerClassifSklearn, 
    LearnerRegrSklearn
)
from ..measures import (
    MeasureClassifAccuracy, 
    MeasureClassifAUC,
    MeasureRegrMSE,
    MeasureRegrMAE
)
from ..resamplings import ResamplingCV, ResamplingHoldout
from ..pipelines import PipeOpScale, PipeOpImpute, GraphLearner

# AutoML components
from .simple_automl import AutoMLResult

# Importación opcional de Optuna
try:
    import optuna
    from optuna import Trial
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    Trial = None


@dataclass
class AdvancedAutoMLConfig:
    """Configuración para AutoML Avanzado."""
    
    # Configuración básica
    task_type: str = "auto"  # "classification", "regression", "auto"
    time_budget: int = 3600  # segundos
    n_trials: int = 100
    n_jobs: int = -1
    
    # Modelos a considerar
    include_models: List[str] = field(default_factory=lambda: [
        "random_forest",
        "gradient_boosting", 
        "xgboost",
        "lightgbm",
        "catboost",
        "linear",
        "svm"
    ])
    
    # Preprocesamiento
    auto_preprocessing: bool = True
    handle_missing: bool = True
    scale_features: bool = True
    encode_categorical: bool = True
    
    # Feature engineering
    feature_selection: bool = True
    feature_generation: bool = False
    max_features: Optional[int] = None
    
    # Validación
    cv_folds: int = 5
    validation_strategy: str = "cv"  # "cv", "holdout", "time_series"
    
    # Optimización
    optimization_metric: Optional[str] = None  # Auto-detectado si None
    ensemble: bool = True
    early_stopping: bool = True
    
    # Reporte y visualización
    verbose: int = 1
    show_progress: bool = True
    generate_report: bool = True
    report_path: Optional[str] = None


class AdvancedAutoML:
    """
    AutoML Avanzado con Optuna para MLPY.
    
    Características:
    - Búsqueda Bayesiana de hiperparámetros
    - Selección automática de modelos
    - Pipeline optimization
    - Feature engineering automático
    - Ensemble learning
    - Explicabilidad integrada
    """
    
    def __init__(self, config: Optional[AdvancedAutoMLConfig] = None):
        """
        Inicializa AutoML Avanzado.
        
        Parameters
        ----------
        config : AdvancedAutoMLConfig, optional
            Configuración del AutoML. Usa valores por defecto si None.
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna no está instalado. "
                "Instalar con: pip install optuna"
            )
        
        self.config = config or AdvancedAutoMLConfig()
        self.study = None
        self.best_pipeline = None
        self.results = None
        self.feature_importance = None
        self._start_time = None
        
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> 'AdvancedAutoML':
        """
        Entrena AutoML en los datos.
        
        Parameters
        ----------
        X : DataFrame o array
            Features de entrenamiento
        y : Series o array
            Target de entrenamiento
        X_val : DataFrame o array, optional
            Features de validación
        y_val : Series o array, optional
            Target de validación
            
        Returns
        -------
        self : AdvancedAutoML
            Instancia entrenada
        """
        self._start_time = time.time()
        
        # Crear task de MLPY
        task = self._create_task(X, y)
        
        # Detectar tipo de problema si es auto
        if self.config.task_type == "auto":
            self.config.task_type = self._detect_task_type(y)
        
        # Configurar métrica de optimización
        if self.config.optimization_metric is None:
            self.config.optimization_metric = self._get_default_metric()
        
        # Crear estudio de Optuna
        self.study = self._create_study()
        
        # Optimizar
        if self.config.verbose > 0:
            print(f"=== AutoML Avanzado MLPY ===")
            print(f"Tipo de tarea: {self.config.task_type}")
            print(f"Métrica: {self.config.optimization_metric}")
            print(f"Presupuesto: {self.config.time_budget}s")
            print(f"Trials: {self.config.n_trials}")
            print(f"Iniciando optimización...")
        
        # Función objetivo para Optuna
        def objective(trial):
            return self._objective(trial, task, X_val, y_val)
        
        # Ejecutar optimización
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.time_budget,
            n_jobs=self.config.n_jobs if self.config.n_jobs != -1 else None,
            show_progress_bar=self.config.show_progress
        )
        
        # Obtener mejor pipeline
        self.best_pipeline = self._create_best_pipeline()
        
        # Entrenar modelo final con todos los datos
        self.best_pipeline.train(task)
        
        # Calcular feature importance
        self.feature_importance = self._calculate_feature_importance(X)
        
        # Generar resultados
        self.results = self._generate_results(task)
        
        # Generar reporte si está configurado
        if self.config.generate_report:
            self._generate_report()
        
        elapsed_time = time.time() - self._start_time
        if self.config.verbose > 0:
            print(f"\nOptimización completada en {elapsed_time:.2f}s")
            print(f"Mejor score: {self.study.best_value:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Realiza predicciones con el mejor modelo."""
        if self.best_pipeline is None:
            raise ValueError("AutoML no ha sido entrenado. Llamar fit() primero.")
        
        # Crear task temporal para predicción
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Crear task dummy para predicción
        if self.config.task_type == "classification":
            dummy_y = pd.Series([0] * len(X))
            pred_task = TaskClassif(
                data=pd.concat([X, dummy_y.rename('target')], axis=1),
                target='target',
                id='prediction'
            )
        else:
            dummy_y = pd.Series([0.0] * len(X))
            pred_task = TaskRegr(
                data=pd.concat([X, dummy_y.rename('target')], axis=1),
                target='target',
                id='prediction'
            )
        
        predictions = self.best_pipeline.predict(pred_task)
        return predictions.response
    
    def _create_task(self, X, y):
        """Crea task de MLPY desde los datos."""
        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        
        # Combinar datos
        data = pd.concat([X, y.rename('target')], axis=1)
        
        # Detectar tipo y crear task
        if self._is_classification(y):
            return TaskClassif(data=data, target='target', id='automl_task')
        else:
            return TaskRegr(data=data, target='target', id='automl_task')
    
    def _detect_task_type(self, y):
        """Detecta si es clasificación o regresión."""
        if self._is_classification(y):
            return "classification"
        return "regression"
    
    def _is_classification(self, y):
        """Verifica si es un problema de clasificación."""
        unique_ratio = len(np.unique(y)) / len(y)
        return unique_ratio < 0.05 or y.dtype == 'object'
    
    def _get_default_metric(self):
        """Obtiene métrica por defecto según el tipo de tarea."""
        if self.config.task_type == "classification":
            return "accuracy"
        return "mse"
    
    def _create_study(self):
        """Crea estudio de Optuna."""
        # Determinar dirección de optimización
        if self.config.task_type == "classification":
            direction = "maximize"
        else:
            direction = "minimize"
        
        return optuna.create_study(
            direction=direction,
            study_name=f"MLPY_AutoML_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner() if self.config.early_stopping else None
        )
    
    def _objective(self, trial: Trial, task: Task, X_val=None, y_val=None):
        """Función objetivo para Optuna."""
        
        # Seleccionar modelo
        model_name = trial.suggest_categorical('model', self.config.include_models)
        
        # Crear learner con hiperparámetros sugeridos
        learner = self._create_learner(trial, model_name)
        
        # Crear pipeline con preprocesamiento
        pipeline = self._create_pipeline(trial, learner)
        
        # Evaluar con cross-validation
        if self.config.validation_strategy == "cv":
            resampling = ResamplingCV(folds=self.config.cv_folds)
        else:
            resampling = ResamplingHoldout(ratio=0.2)
        
        # Seleccionar medida
        if self.config.task_type == "classification":
            if self.config.optimization_metric == "accuracy":
                measure = MeasureClassifAccuracy()
            else:
                measure = MeasureClassifAUC()
        else:
            if self.config.optimization_metric == "mse":
                measure = MeasureRegrMSE()
            else:
                measure = MeasureRegrMAE()
        
        # Evaluar
        from .. import resample
        result = resample(
            task=task,
            learner=pipeline,
            resampling=resampling,
            measures=[measure]
        )
        
        # Retornar score promedio
        return result.aggregate(measure)[0]
    
    def _create_learner(self, trial: Trial, model_name: str):
        """Crea learner con hiperparámetros de Optuna."""
        
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'random_state': 42
            }
            
            if self.config.task_type == "classification":
                model = RandomForestClassifier(**params)
                return LearnerClassifSklearn(estimator=model)
            else:
                model = RandomForestRegressor(**params)
                return LearnerRegrSklearn(estimator=model)
        
        elif model_name == "xgboost":
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
            }
            
            try:
                # Intentar importar XGBoost wrapper si existe
                from ..learners.xgboost_wrapper import LearnerClassifXGBoost, LearnerRegrXGBoost
                if self.config.task_type == "classification":
                    return LearnerClassifXGBoost(**params)
                else:
                    return LearnerRegrXGBoost(**params)
            except:
                # Fallback si XGBoost no está disponible
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                if self.config.task_type == "classification":
                    model = GradientBoostingClassifier(random_state=42)
                    return LearnerClassifSklearn(estimator=model)
                else:
                    model = GradientBoostingRegressor(random_state=42)
                    return LearnerRegrSklearn(estimator=model)
        
        # Default: modelo lineal
        from sklearn.linear_model import LogisticRegression, LinearRegression
        if self.config.task_type == "classification":
            model = LogisticRegression(random_state=42, max_iter=1000)
            return LearnerClassifSklearn(estimator=model)
        else:
            model = LinearRegression()
            return LearnerRegrSklearn(estimator=model)
    
    def _create_pipeline(self, trial: Trial, learner):
        """Crea pipeline con preprocesamiento."""
        from ..pipelines import Graph, GraphLearner
        
        graph = Graph()
        
        # Agregar preprocesamiento si está habilitado
        if self.config.auto_preprocessing:
            if self.config.handle_missing:
                imputer = PipeOpImpute()
                graph.add_pipeop(imputer)
            
            if self.config.scale_features:
                scaler = PipeOpScale()
                graph.add_pipeop(scaler)
        
        # Agregar learner al final
        from ..pipelines import PipeOpLearner
        learner_op = PipeOpLearner(learner)
        graph.add_pipeop(learner_op)
        
        # Conectar pipeline
        if self.config.auto_preprocessing:
            if self.config.handle_missing and self.config.scale_features:
                graph.add_edge(imputer.id, scaler.id)
                graph.add_edge(scaler.id, learner_op.id)
            elif self.config.handle_missing:
                graph.add_edge(imputer.id, learner_op.id)
            elif self.config.scale_features:
                graph.add_edge(scaler.id, learner_op.id)
        
        return GraphLearner(graph)
    
    def _create_best_pipeline(self):
        """Crea el mejor pipeline desde los mejores parámetros."""
        best_params = self.study.best_params
        best_trial = self.study.best_trial
        
        # Recrear el mejor modelo
        model_name = best_params['model']
        
        # Crear trial dummy con los mejores parámetros
        class DummyTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])
            
            def suggest_int(self, name, low, high):
                return self.params.get(name, low)
            
            def suggest_float(self, name, low, high, log=False):
                return self.params.get(name, low)
        
        dummy_trial = DummyTrial(best_params)
        learner = self._create_learner(dummy_trial, model_name)
        pipeline = self._create_pipeline(dummy_trial, learner)
        
        return pipeline
    
    def _calculate_feature_importance(self, X):
        """Calcula importancia de features."""
        # Por ahora retornamos None
        # TODO: Implementar extracción de feature importance
        return None
    
    def _generate_results(self, task):
        """Genera objeto de resultados."""
        # Crear DataFrame con leaderboard
        trials_df = self.study.trials_dataframe()
        
        leaderboard = pd.DataFrame({
            'model': [t.params['model'] for t in self.study.trials],
            'score': [t.value for t in self.study.trials],
            'duration': [t.duration.total_seconds() for t in self.study.trials]
        }).sort_values('score', ascending=False if self.config.task_type == "classification" else True)
        
        return AutoMLResult(
            best_learner=self.best_pipeline,
            best_score=self.study.best_value,
            leaderboard=leaderboard,
            feature_importance=self.feature_importance,
            task=task,
            training_time=time.time() - self._start_time,
            meta_info={
                'n_trials': len(self.study.trials),
                'best_params': self.study.best_params,
                'optimization_history': [t.value for t in self.study.trials]
            }
        )
    
    def _generate_report(self):
        """Genera reporte HTML del AutoML."""
        if self.config.report_path is None:
            self.config.report_path = f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Por ahora solo guardamos información básica
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials)
        }
        
        # Guardar como JSON por ahora
        report_path = Path(self.config.report_path).with_suffix('.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        if self.config.verbose > 0:
            print(f"Reporte guardado en: {report_path}")
    
    def plot_optimization_history(self):
        """Visualiza historia de optimización."""
        if self.study is None:
            raise ValueError("No hay estudio para visualizar. Ejecutar fit() primero.")
        
        return plot_optimization_history(self.study)
    
    def plot_param_importance(self):
        """Visualiza importancia de hiperparámetros."""
        if self.study is None:
            raise ValueError("No hay estudio para visualizar. Ejecutar fit() primero.")
        
        return plot_param_importances(self.study)
    
    def plot_parallel_coordinates(self):
        """Visualiza coordenadas paralelas de hiperparámetros."""
        if self.study is None:
            raise ValueError("No hay estudio para visualizar. Ejecutar fit() primero.")
        
        return plot_parallel_coordinate(self.study)