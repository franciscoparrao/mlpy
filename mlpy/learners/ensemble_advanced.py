"""
Ensemble Learners Avanzados para MLPY.

Este módulo implementa algoritmos de ensemble más sofisticados con
auto-tuning, optimización Bayesiana y técnicas avanzadas.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

from .ensemble import LearnerEnsemble
from ..base import Learner
from ...tasks import Task, TaskClassif, TaskRegr
from ...predictions import PredictionClassif, PredictionRegr
from ...resamplings import ResamplingCV, ResamplingHoldout
from ...validation.validators import validate_task_data
from ...core.lazy import LazyEvaluationContext
from ...interpretability import create_explanation


class LearnerAdaptiveEnsemble(LearnerEnsemble):
    """
    Ensemble Adaptativo que ajusta automáticamente pesos basado en performance.
    
    Características:
    - Pesos dinámicos basados en performance de validación
    - Auto-selección de mejores learners
    - Optimización Bayesiana de hiperparámetros
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        adaptation_metric: str = 'accuracy',
        selection_threshold: float = 0.1,
        auto_tune: bool = True,
        tune_trials: int = 50,
        cv_folds: int = 5,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, **kwargs)
        
        self.adaptation_metric = adaptation_metric
        self.selection_threshold = selection_threshold
        self.auto_tune = auto_tune
        self.tune_trials = tune_trials
        self.cv_folds = cv_folds
        
        self.learner_weights = None
        self.learner_performances = None
        self.selected_learners = None
        self.tuning_history = []
    
    def _evaluate_learner_performance(self, task: Task) -> Dict[int, float]:
        """Evaluar performance individual de cada learner."""
        performances = {}
        
        # Usar cross-validation para evaluación robusta
        cv = ResamplingCV(folds=self.cv_folds, stratify=isinstance(task, TaskClassif))
        cv_instance = cv.instantiate(task)
        
        for i, learner in enumerate(self.base_learners):
            fold_scores = []
            
            for fold in range(self.cv_folds):
                train_idx = cv_instance.train_set(fold)
                test_idx = cv_instance.test_set(fold)
                
                train_task = task.filter(train_idx)
                test_task = task.filter(test_idx)
                
                # Entrenar y evaluar
                fold_learner = deepcopy(learner)
                fold_learner.train(train_task)
                pred = fold_learner.predict(test_task)
                
                # Calcular métrica
                if isinstance(task, TaskClassif):
                    if self.adaptation_metric == 'accuracy':
                        score = accuracy_score(test_task.truth(), pred.response)
                    else:
                        score = accuracy_score(test_task.truth(), pred.response)
                else:
                    if self.adaptation_metric == 'mse':
                        score = -mean_squared_error(test_task.truth(), pred.response)
                    else:
                        score = -mean_squared_error(test_task.truth(), pred.response)
                
                fold_scores.append(score)
            
            performances[i] = np.mean(fold_scores)
        
        return performances
    
    def _select_best_learners(self, performances: Dict[int, float]) -> List[int]:
        """Seleccionar mejores learners basado en performance."""
        # Ordenar por performance
        sorted_learners = sorted(performances.items(), key=lambda x: x[1], reverse=True)
        
        # Seleccionar learners que estén dentro del threshold del mejor
        best_score = sorted_learners[0][1]
        threshold = best_score - self.selection_threshold
        
        selected = []
        for learner_idx, score in sorted_learners:
            if score >= threshold:
                selected.append(learner_idx)
        
        # Mínimo 2 learners
        if len(selected) < 2:
            selected = [idx for idx, _ in sorted_learners[:2]]
        
        return selected
    
    def _optimize_weights(self, task: Task, selected_learners: List[int]) -> np.ndarray:
        """Optimizar pesos usando Optuna."""
        if not self.auto_tune:
            # Pesos uniformes
            return np.ones(len(selected_learners)) / len(selected_learners)
        
        def objective(trial):
            # Generar pesos
            weights = []
            for i in range(len(selected_learners) - 1):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            # Último peso para que sumen 1
            weights.append(max(0.0, 1.0 - sum(weights)))
            weights = np.array(weights)
            
            # Normalizar
            if weights.sum() == 0:
                weights = np.ones(len(selected_learners)) / len(selected_learners)
            else:
                weights = weights / weights.sum()
            
            # Evaluar ensemble con estos pesos
            return self._evaluate_weighted_ensemble(task, selected_learners, weights)
        
        # Optimización
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.tune_trials, show_progress_bar=False)
        
        # Extraer mejores pesos
        best_params = study.best_params
        weights = []
        for i in range(len(selected_learners) - 1):
            weights.append(best_params.get(f'weight_{i}', 0.0))
        weights.append(max(0.0, 1.0 - sum(weights)))
        weights = np.array(weights)
        
        # Normalizar
        if weights.sum() == 0:
            weights = np.ones(len(selected_learners)) / len(selected_learners)
        else:
            weights = weights / weights.sum()
        
        self.tuning_history = study.trials_dataframe()
        
        return weights
    
    def _evaluate_weighted_ensemble(
        self, 
        task: Task, 
        learner_indices: List[int], 
        weights: np.ndarray
    ) -> float:
        """Evaluar performance del ensemble con pesos específicos."""
        cv = ResamplingCV(folds=3, stratify=isinstance(task, TaskClassif))  # CV rápido
        cv_instance = cv.instantiate(task)
        
        fold_scores = []
        
        for fold in range(3):
            train_idx = cv_instance.train_set(fold)
            test_idx = cv_instance.test_set(fold)
            
            train_task = task.filter(train_idx)
            test_task = task.filter(test_idx)
            
            # Entrenar learners seleccionados
            trained_learners = []
            for idx in learner_indices:
                learner = deepcopy(self.base_learners[idx])
                learner.train(train_task)
                trained_learners.append(learner)
            
            # Predicciones ponderadas
            if isinstance(task, TaskClassif):
                # Votación ponderada
                all_preds = []
                for learner in trained_learners:
                    pred = learner.predict(test_task)
                    all_preds.append(pred.response)
                
                # Combinar con pesos
                ensemble_preds = []
                for sample_idx in range(len(test_task.truth())):
                    votes = {}
                    for learner_idx, preds in enumerate(all_preds):
                        pred = str(preds[sample_idx])
                        if pred not in votes:
                            votes[pred] = 0
                        votes[pred] += weights[learner_idx]
                    
                    best_class = max(votes, key=votes.get)
                    ensemble_preds.append(best_class)
                
                score = accuracy_score(test_task.truth(), ensemble_preds)
            else:
                # Promedio ponderado
                predictions = np.zeros(len(test_task.truth()))
                for i, learner in enumerate(trained_learners):
                    pred = learner.predict(test_task)
                    predictions += weights[i] * np.array(pred.response)
                
                score = -mean_squared_error(test_task.truth(), predictions)
            
            fold_scores.append(score)
        
        return np.mean(fold_scores)
    
    def train(self, task: Task) -> 'LearnerAdaptiveEnsemble':
        """Entrenar ensemble adaptativo."""
        with LazyEvaluationContext():
            # Validación
            validation = validate_task_data(task.data, target=task.target)
            if not validation['valid']:
                from ...core.exceptions import MLPYValidationError
                raise MLPYValidationError("Adaptive ensemble training failed validation")
            
            self._check_task_compatibility(task)
            self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
            
            # Paso 1: Evaluar performance individual
            print("Evaluating base learner performances...")
            self.learner_performances = self._evaluate_learner_performance(task)
            
            # Paso 2: Seleccionar mejores learners
            self.selected_learners = self._select_best_learners(self.learner_performances)
            print(f"Selected {len(self.selected_learners)} best learners out of {len(self.base_learners)}")
            
            # Paso 3: Optimizar pesos
            print("Optimizing ensemble weights...")
            self.learner_weights = self._optimize_weights(task, self.selected_learners)
            
            # Paso 4: Entrenar learners seleccionados en datos completos
            self._trained_learners = []
            for idx in self.selected_learners:
                learner = deepcopy(self.base_learners[idx])
                learner.train(task)
                self._trained_learners.append(learner)
            
            self._model = self._trained_learners  # Mark as trained
            
            return self
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Hacer predicciones con ensemble adaptativo."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if isinstance(task, TaskClassif):
            return self._predict_classif(task)
        else:
            return self._predict_regr(task)
    
    def _predict_classif(self, task: TaskClassif) -> PredictionClassif:
        """Predicciones de clasificación."""
        n_samples = task.nrow
        
        # Obtener predicciones de learners entrenados
        all_preds = []
        for learner in self._trained_learners:
            pred = learner.predict(task)
            all_preds.append(pred.response)
        
        # Votación ponderada
        predictions = []
        for sample_idx in range(n_samples):
            votes = {}
            for learner_idx, preds in enumerate(all_preds):
                pred = str(preds[sample_idx])
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += self.learner_weights[learner_idx]
            
            best_class = max(votes, key=votes.get)
            predictions.append(best_class)
        
        return PredictionClassif(
            task=task,
            learner_id=self.id or "adaptive_ensemble",
            row_ids=list(range(n_samples)),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=predictions
        )
    
    def _predict_regr(self, task: TaskRegr) -> PredictionRegr:
        """Predicciones de regresión."""
        # Promedio ponderado
        predictions = np.zeros(task.nrow)
        
        for i, learner in enumerate(self._trained_learners):
            pred = learner.predict(task)
            predictions += self.learner_weights[i] * np.array(pred.response)
        
        return PredictionRegr(
            task=task,
            learner_id=self.id or "adaptive_ensemble",
            row_ids=list(range(task.nrow)),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=predictions
        )
    
    def explain(self, method='learner_contribution', **kwargs):
        """
        Explicar el ensemble adaptativo.
        
        Parameters:
        -----------
        method : str
            Método de explicación ('learner_contribution', 'weight_analysis')
        """
        if method == 'learner_contribution':
            return self._explain_learner_contribution(**kwargs)
        elif method == 'weight_analysis':
            return self._explain_weight_analysis(**kwargs)
        else:
            return create_explanation(self, method=method, **kwargs)
    
    def _explain_learner_contribution(self, **kwargs):
        """Explicar contribución de cada learner."""
        if self.learner_weights is None:
            return {"error": "Model not trained"}
        
        contributions = {}
        for i, idx in enumerate(self.selected_learners):
            learner = self.base_learners[idx]
            contributions[f"learner_{idx}_{learner.__class__.__name__}"] = {
                'weight': self.learner_weights[i],
                'performance': self.learner_performances[idx],
                'selected': True
            }
        
        # Agregar learners no seleccionados
        for idx, performance in self.learner_performances.items():
            if idx not in self.selected_learners:
                learner = self.base_learners[idx]
                contributions[f"learner_{idx}_{learner.__class__.__name__}"] = {
                    'weight': 0.0,
                    'performance': performance,
                    'selected': False
                }
        
        return {
            'method': 'learner_contribution',
            'contributions': contributions,
            'total_learners': len(self.base_learners),
            'selected_learners': len(self.selected_learners)
        }
    
    def _explain_weight_analysis(self, **kwargs):
        """Análisis de pesos optimizados."""
        if self.tuning_history is None or len(self.tuning_history) == 0:
            return {"error": "No tuning history available"}
        
        analysis = {
            'method': 'weight_analysis',
            'final_weights': self.learner_weights.tolist(),
            'optimization_trials': len(self.tuning_history),
            'best_score': self.tuning_history['value'].max(),
            'convergence': {
                'early_best': self.tuning_history['value'][:10].max(),
                'final_best': self.tuning_history['value'].max(),
                'improvement': self.tuning_history['value'].max() - self.tuning_history['value'][:10].max()
            }
        }
        
        return analysis


class LearnerBayesianEnsemble(LearnerEnsemble):
    """
    Ensemble Bayesiano que modela incertidumbre en predicciones.
    
    Utiliza múltiples modelos entrenados con bootstrap para capturar
    incertidumbre epistémica y aleatoria.
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        n_bootstrap: int = 100,
        bootstrap_ratio: float = 0.8,
        uncertainty_method: str = 'variance',
        confidence_level: float = 0.95,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, **kwargs)
        
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ratio = bootstrap_ratio
        self.uncertainty_method = uncertainty_method
        self.confidence_level = confidence_level
        
        self.bootstrap_models = []
        self.uncertainty_estimates = None
    
    def train(self, task: Task) -> 'LearnerBayesianEnsemble':
        """Entrenar ensemble Bayesiano con bootstrap."""
        with LazyEvaluationContext():
            self._check_task_compatibility(task)
            self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
            
            print(f"Training Bayesian ensemble with {self.n_bootstrap} bootstrap samples...")
            
            # Generar modelos bootstrap
            self.bootstrap_models = []
            
            for i in range(self.n_bootstrap):
                # Bootstrap sampling
                n_samples = int(task.nrow * self.bootstrap_ratio)
                bootstrap_indices = np.random.choice(
                    task.nrow, 
                    size=n_samples, 
                    replace=True
                )
                
                bootstrap_task = task.filter(bootstrap_indices)
                
                # Entrenar modelo en muestra bootstrap
                for learner in self.base_learners:
                    bootstrap_learner = deepcopy(learner)
                    bootstrap_learner.train(bootstrap_task)
                    self.bootstrap_models.append(bootstrap_learner)
                
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
            
            self._model = self.bootstrap_models  # Mark as trained
            return self
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Predicciones Bayesianas con estimación de incertidumbre."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Obtener predicciones de todos los modelos bootstrap
        all_predictions = []
        
        for model in self.bootstrap_models:
            pred = model.predict(task)
            all_predictions.append(pred.response)
        
        all_predictions = np.array(all_predictions)
        
        if isinstance(task, TaskClassif):
            return self._predict_classif_bayesian(task, all_predictions)
        else:
            return self._predict_regr_bayesian(task, all_predictions)
    
    def _predict_classif_bayesian(
        self, 
        task: TaskClassif, 
        all_predictions: np.ndarray
    ) -> PredictionClassif:
        """Predicciones de clasificación Bayesianas."""
        n_samples = task.nrow
        n_classes = len(task.class_names)
        
        # Calcular probabilidades empíricas
        class_probabilities = np.zeros((n_samples, n_classes))
        final_predictions = []
        
        for sample_idx in range(n_samples):
            sample_preds = all_predictions[:, sample_idx]
            
            # Contar votos para cada clase
            votes = np.zeros(n_classes)
            for pred in sample_preds:
                if str(pred) in task.class_names:
                    class_idx = task.class_names.index(str(pred))
                    votes[class_idx] += 1
            
            # Normalizar a probabilidades
            class_probabilities[sample_idx] = votes / len(sample_preds)
            
            # Predicción final: clase con mayor probabilidad
            final_pred_idx = np.argmax(class_probabilities[sample_idx])
            final_predictions.append(task.class_names[final_pred_idx])
        
        # Calcular incertidumbre
        self.uncertainty_estimates = self._calculate_classification_uncertainty(
            class_probabilities
        )
        
        return PredictionClassif(
            task=task,
            learner_id=self.id or "bayesian_ensemble",
            row_ids=list(range(n_samples)),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=final_predictions,
            prob=class_probabilities
        )
    
    def _predict_regr_bayesian(
        self, 
        task: TaskRegr, 
        all_predictions: np.ndarray
    ) -> PredictionRegr:
        """Predicciones de regresión Bayesianas."""
        # Estadísticas de predicciones
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # Intervalos de confianza
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(all_predictions, upper_percentile, axis=0)
        
        # Almacenar estimaciones de incertidumbre
        self.uncertainty_estimates = {
            'mean': mean_predictions,
            'std': std_predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'confidence_level': self.confidence_level
        }
        
        return PredictionRegr(
            task=task,
            learner_id=self.id or "bayesian_ensemble",
            row_ids=list(range(task.nrow)),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=mean_predictions
        )
    
    def _calculate_classification_uncertainty(
        self, 
        class_probabilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calcular métricas de incertidumbre para clasificación."""
        # Entropía de Shannon
        entropy = -np.sum(
            class_probabilities * np.log(class_probabilities + 1e-10), 
            axis=1
        )
        
        # Varianza de probabilidades
        max_probs = np.max(class_probabilities, axis=1)
        confidence = max_probs
        uncertainty = 1 - confidence
        
        # Diferencia entre top 2 clases
        sorted_probs = np.sort(class_probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        return {
            'entropy': entropy,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'margin': margin,
            'class_probabilities': class_probabilities
        }
    
    def get_prediction_intervals(
        self, 
        confidence_level: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Obtener intervalos de predicción.
        
        Parameters:
        -----------
        confidence_level : float, optional
            Nivel de confianza (por defecto usa el del modelo)
            
        Returns:
        --------
        dict : Intervalos de predicción y métricas de incertidumbre
        """
        if self.uncertainty_estimates is None:
            raise ValueError("No uncertainty estimates available. Make predictions first.")
        
        return self.uncertainty_estimates
    
    def explain(self, method='uncertainty_analysis', **kwargs):
        """Explicar incertidumbre del ensemble Bayesiano."""
        if method == 'uncertainty_analysis':
            return self._explain_uncertainty(**kwargs)
        else:
            return create_explanation(self, method=method, **kwargs)
    
    def _explain_uncertainty(self, **kwargs):
        """Análisis de incertidumbre."""
        if self.uncertainty_estimates is None:
            return {"error": "No uncertainty estimates available"}
        
        analysis = {
            'method': 'uncertainty_analysis',
            'model_type': self._task_type,
            'n_bootstrap_models': len(self.bootstrap_models),
            'bootstrap_ratio': self.bootstrap_ratio
        }
        
        if self._task_type == 'classif':
            uncertainty = self.uncertainty_estimates
            analysis.update({
                'mean_entropy': np.mean(uncertainty['entropy']),
                'mean_confidence': np.mean(uncertainty['confidence']),
                'mean_margin': np.mean(uncertainty['margin']),
                'high_uncertainty_samples': np.sum(uncertainty['uncertainty'] > 0.5),
                'low_confidence_samples': np.sum(uncertainty['confidence'] < 0.7)
            })
        else:
            uncertainty = self.uncertainty_estimates
            analysis.update({
                'mean_std': np.mean(uncertainty['std']),
                'mean_interval_width': np.mean(
                    uncertainty['upper_bound'] - uncertainty['lower_bound']
                ),
                'confidence_level': uncertainty['confidence_level']
            })
        
        return analysis


class LearnerCascadeEnsemble(LearnerEnsemble):
    """
    Ensemble en Cascada - modelos se ejecutan secuencialmente.
    
    Cada modelo decide si es lo suficientemente confiado para hacer
    la predicción final, o si debe pasar al siguiente modelo más complejo.
    """
    
    def __init__(
        self,
        base_learners: List[Learner],
        confidence_thresholds: Optional[List[float]] = None,
        complexity_order: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(base_learners=base_learners, **kwargs)
        
        # Thresholds de confianza para cada modelo
        if confidence_thresholds is None:
            # Thresholds crecientes: modelos simples necesitan más confianza
            self.confidence_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5][:len(base_learners)]
        else:
            self.confidence_thresholds = confidence_thresholds
        
        # Orden de complejidad (opcional)
        if complexity_order is None:
            self.complexity_order = list(range(len(base_learners)))
        else:
            self.complexity_order = complexity_order
        
        self.cascade_statistics = None
    
    def train(self, task: Task) -> 'LearnerCascadeEnsemble':
        """Entrenar modelos en cascada."""
        with LazyEvaluationContext():
            self._check_task_compatibility(task)
            self._task_type = "classif" if isinstance(task, TaskClassif) else "regr"
            
            # Entrenar todos los modelos
            self._trained_learners = []
            for i in self.complexity_order:
                learner = deepcopy(self.base_learners[i])
                learner.train(task)
                self._trained_learners.append(learner)
            
            self._model = self._trained_learners  # Mark as trained
            return self
    
    def predict(self, task: Task) -> Union[PredictionClassif, PredictionRegr]:
        """Predicciones en cascada."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        n_samples = task.nrow
        final_predictions = [None] * n_samples
        prediction_stage = [-1] * n_samples  # Qué modelo hizo la predicción
        confidence_scores = [0.0] * n_samples
        
        # Procesar cada muestra a través de la cascada
        remaining_indices = list(range(n_samples))
        
        for stage, (learner, threshold) in enumerate(
            zip(self._trained_learners, self.confidence_thresholds)
        ):
            if not remaining_indices:
                break
            
            # Crear sub-task con muestras restantes
            sub_task = task.filter(remaining_indices)
            pred = learner.predict(sub_task)
            
            # Evaluar confianza
            if isinstance(task, TaskClassif):
                confidences = self._calculate_classification_confidence(pred)
            else:
                confidences = self._calculate_regression_confidence(pred)
            
            # Decidir qué predicciones aceptar
            new_remaining = []
            for i, (idx, conf) in enumerate(zip(remaining_indices, confidences)):
                if conf >= threshold or stage == len(self._trained_learners) - 1:
                    # Aceptar predicción
                    final_predictions[idx] = pred.response[i]
                    prediction_stage[idx] = stage
                    confidence_scores[idx] = conf
                else:
                    # Pasar al siguiente modelo
                    new_remaining.append(idx)
            
            remaining_indices = new_remaining
        
        # Estadísticas de cascada
        self.cascade_statistics = {
            'predictions_per_stage': [
                len([s for s in prediction_stage if s == stage])
                for stage in range(len(self._trained_learners))
            ],
            'mean_confidence_per_stage': [
                np.mean([confidence_scores[i] for i, s in enumerate(prediction_stage) if s == stage])
                for stage in range(len(self._trained_learners))
                if len([s for s in prediction_stage if s == stage]) > 0
            ]
        }
        
        # Crear predicción final
        if isinstance(task, TaskClassif):
            return PredictionClassif(
                task=task,
                learner_id=self.id or "cascade_ensemble",
                row_ids=list(range(n_samples)),
                truth=task.truth() if hasattr(task, 'truth') else None,
                response=final_predictions
            )
        else:
            return PredictionRegr(
                task=task,
                learner_id=self.id or "cascade_ensemble",
                row_ids=list(range(n_samples)),
                truth=task.truth() if hasattr(task, 'truth') else None,
                response=final_predictions
            )
    
    def _calculate_classification_confidence(self, pred: PredictionClassif) -> List[float]:
        """Calcular confianza para clasificación."""
        if hasattr(pred, 'prob') and pred.prob is not None:
            # Usar probabilidades máximas
            max_probs = np.max(pred.prob, axis=1)
            return max_probs.tolist()
        else:
            # Si no hay probabilidades, usar confianza constante baja
            return [0.5] * len(pred.response)
    
    def _calculate_regression_confidence(self, pred: PredictionRegr) -> List[float]:
        """Calcular confianza para regresión (simplificado)."""
        # Para regresión, usar confianza basada en varianza de residuos
        # (implementación simplificada)
        predictions = np.array(pred.response)
        
        # Confidence proxy: inverso de la desviación absoluta de la mediana
        mad = np.median(np.abs(predictions - np.median(predictions)))
        confidence = 1.0 / (1.0 + mad)
        
        return [confidence] * len(predictions)
    
    def get_cascade_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cascada."""
        if self.cascade_statistics is None:
            raise ValueError("No cascade statistics available. Make predictions first.")
        
        return self.cascade_statistics
    
    def explain(self, method='cascade_analysis', **kwargs):
        """Explicar comportamiento de la cascada."""
        if method == 'cascade_analysis':
            return self._explain_cascade(**kwargs)
        else:
            return create_explanation(self, method=method, **kwargs)
    
    def _explain_cascade(self, **kwargs):
        """Análisis del comportamiento de cascada."""
        if self.cascade_statistics is None:
            return {"error": "No cascade statistics available"}
        
        stats = self.cascade_statistics
        total_predictions = sum(stats['predictions_per_stage'])
        
        analysis = {
            'method': 'cascade_analysis',
            'total_models': len(self._trained_learners),
            'total_predictions': total_predictions,
            'efficiency': {
                'early_exit_rate': sum(stats['predictions_per_stage'][:-1]) / total_predictions,
                'predictions_per_stage_ratio': [
                    count / total_predictions for count in stats['predictions_per_stage']
                ]
            },
            'confidence_thresholds': self.confidence_thresholds,
            'mean_confidence_per_stage': stats.get('mean_confidence_per_stage', [])
        }
        
        return analysis


# Factory function para ensemble avanzados
def create_advanced_ensemble(
    method: str,
    base_learners: List[Learner],
    **kwargs
) -> LearnerEnsemble:
    """
    Crear ensemble avanzado.
    
    Parameters:
    -----------
    method : str
        Tipo de ensemble ('adaptive', 'bayesian', 'cascade')
    base_learners : List[Learner]
        Lista de learners base
    **kwargs
        Argumentos adicionales específicos del método
        
    Returns:
    --------
    LearnerEnsemble
        Ensemble configurado
    """
    methods = {
        'adaptive': LearnerAdaptiveEnsemble,
        'bayesian': LearnerBayesianEnsemble,
        'cascade': LearnerCascadeEnsemble
    }
    
    if method not in methods:
        raise ValueError(
            f"Unknown advanced ensemble method: {method}. "
            f"Choose from {list(methods.keys())}"
        )
    
    return methods[method](base_learners=base_learners, **kwargs)


__all__ = [
    'LearnerAdaptiveEnsemble',
    'LearnerBayesianEnsemble', 
    'LearnerCascadeEnsemble',
    'create_advanced_ensemble'
]