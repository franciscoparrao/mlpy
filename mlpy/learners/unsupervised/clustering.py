"""
Algoritmos de Clustering avanzados para MLPY.

Este módulo implementa algoritmos de clustering con auto-tuning de parámetros,
validación automática y explicabilidad integrada.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List, Tuple
from sklearn.cluster import DBSCAN, SpectralClustering, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import optuna
from optuna.samplers import TPESampler

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

from ..base import Learner
from ...validation.validators import validate_task_data
from ...core.lazy import LazyEvaluationContext
from ...interpretability import create_explanation


class LearnerClustering(Learner):
    """
    Clase base para algoritmos de clustering con funcionalidades MLPY.
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        auto_scale: bool = True,
        auto_tune: bool = True,
        tune_metric: str = 'silhouette',
        tune_trials: int = 50,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_clusters = n_clusters
        self.auto_scale = auto_scale
        self.auto_tune = auto_tune
        self.tune_metric = tune_metric
        self.tune_trials = tune_trials
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.feature_names = None
        self.tuning_history = []
    
    def _validate_clustering_data(self, task):
        """Validación específica para clustering."""
        # Usar validación básica de MLPY
        validation = validate_task_data(task.data, target=None)
        
        if not validation['valid']:
            from ...core.exceptions import MLPYValidationError
            error_msg = "Clustering failed validation:\\n"
            for error in validation['errors']:
                error_msg += f"  - {error}\\n"
            error_msg += "\\nFor clustering:\\n"
            error_msg += "  • Remove or impute missing values\\n"
            error_msg += "  • Consider feature scaling\\n"
            error_msg += "  • Check for sufficient samples (min 10)\\n"
            error_msg += "  • Remove constant features"
            raise MLPYValidationError(error_msg)
    
    def _prepare_data(self, X):
        """Preparar datos para clustering."""
        if hasattr(X, 'values'):
            X = X.values
        
        if self.auto_scale:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return X_scaled
    
    def _evaluate_clustering(self, X, labels):
        """Evaluar calidad del clustering."""
        if len(np.unique(labels)) <= 1:
            return -1  # Clustering inválido
        
        try:
            if self.tune_metric == 'silhouette':
                return silhouette_score(X, labels)
            elif self.tune_metric == 'calinski_harabasz':
                return calinski_harabasz_score(X, labels)
            elif self.tune_metric == 'davies_bouldin':
                return -davies_bouldin_score(X, labels)  # Negativo porque menor es mejor
            else:
                return silhouette_score(X, labels)
        except:
            return -1
    
    def fit(self, task):
        """Ajustar el modelo de clustering."""
        with LazyEvaluationContext():
            # Validación
            self._validate_clustering_data(task)
            
            # Preparar datos
            X = task.X if hasattr(task, 'X') else task.data
            X_scaled = self._prepare_data(X)
            
            # Auto-tuning si está habilitado
            if self.auto_tune:
                X_scaled = self._auto_tune(X_scaled)
            
            # Ajustar modelo
            self.labels_ = self.model.fit_predict(X_scaled)
            
            # Calcular centros si es posible
            if hasattr(self.model, 'cluster_centers_'):
                self.cluster_centers_ = self.model.cluster_centers_
            else:
                self._compute_cluster_centers(X_scaled, self.labels_)
            
            # Guardar nombres de features
            self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            
            return self
    
    def _compute_cluster_centers(self, X, labels):
        """Computar centros de clusters manualmente."""
        unique_labels = np.unique(labels)
        self.cluster_centers_ = []
        
        for label in unique_labels:
            if label == -1:  # Outliers en DBSCAN
                continue
            mask = labels == label
            center = X[mask].mean(axis=0)
            self.cluster_centers_.append(center)
        
        self.cluster_centers_ = np.array(self.cluster_centers_)
    
    def predict(self, X):
        """Predecir clusters para nuevos datos."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self._prepare_data(X)
        
        # Para algunos algoritmos, necesitamos asignar al cluster más cercano
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # Asignar al centro más cercano
            from scipy.spatial.distance import cdist
            distances = cdist(X_scaled, self.cluster_centers_)
            return np.argmin(distances, axis=1)
    
    def explain(self, method='feature_importance', **kwargs):
        """
        Explicar el clustering.
        
        Parameters:
        -----------
        method : str
            Método de explicación ('feature_importance', 'cluster_profile')
        
        Returns:
        --------
        dict : Explicación del clustering
        """
        if method == 'feature_importance':
            return self._explain_feature_importance(**kwargs)
        elif method == 'cluster_profile':
            return self._explain_cluster_profile(**kwargs)
        else:
            return create_explanation(self, method=method, **kwargs)
    
    def _explain_feature_importance(self, **kwargs):
        """Explicar importancia de features en clustering."""
        if self.cluster_centers_ is None:
            return {"error": "No cluster centers available"}
        
        # Calcular varianza de cada feature entre centros
        feature_variance = np.var(self.cluster_centers_, axis=0)
        feature_importance = feature_variance / feature_variance.sum()
        
        explanation = {
            'method': 'feature_importance',
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'cluster_centers': self.cluster_centers_
        }
        
        return explanation
    
    def _explain_cluster_profile(self, **kwargs):
        """Explicar perfil de cada cluster."""
        if self.cluster_centers_ is None:
            return {"error": "No cluster centers available"}
        
        profiles = {}
        for i, center in enumerate(self.cluster_centers_):
            profile = {
                'cluster_id': i,
                'size': np.sum(self.labels_ == i),
                'center': center,
                'feature_values': {}
            }
            
            if self.feature_names:
                for j, feature in enumerate(self.feature_names):
                    profile['feature_values'][feature] = center[j]
            
            profiles[f'cluster_{i}'] = profile
        
        return {
            'method': 'cluster_profile',
            'profiles': profiles,
            'n_clusters': len(profiles)
        }


class LearnerDBSCAN(LearnerClustering):
    """
    DBSCAN Clustering con auto-tuning de parámetros.
    
    Características:
    - Auto-tuning de eps y min_samples
    - Manejo inteligente de outliers
    - Explicabilidad de clusters encontrados
    """
    
    def __init__(
        self,
        eps: Union[float, str] = 'auto',
        min_samples: Union[int, str] = 'auto',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.eps = eps
        self.min_samples = min_samples
    
    def _auto_tune(self, X):
        """Auto-tuning específico para DBSCAN."""
        def objective(trial):
            eps = trial.suggest_float('eps', 0.1, 2.0)
            min_samples = trial.suggest_int('min_samples', 2, min(20, len(X)//5))
            
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            
            # Penalizar si todos son outliers o un solo cluster
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= 1:
                return -1
            
            return self._evaluate_clustering(X, labels)
        
        # Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.tune_trials, show_progress_bar=False)
        
        # Usar mejores parámetros
        best_params = study.best_params
        self.eps = best_params['eps']
        self.min_samples = best_params['min_samples']
        
        # Guardar historial
        self.tuning_history = study.trials_dataframe()
        
        # Crear modelo final
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )
        
        return X
    
    def fit(self, task):
        """Ajustar DBSCAN con auto-tuning."""
        with LazyEvaluationContext():
            # Validación
            self._validate_clustering_data(task)
            
            # Preparar datos
            X = task.X if hasattr(task, 'X') else task.data
            X_scaled = self._prepare_data(X)
            
            # Auto-tuning si es necesario
            if self.eps == 'auto' or self.min_samples == 'auto':
                if self.eps == 'auto' and self.min_samples == 'auto':
                    X_scaled = self._auto_tune(X_scaled)
                else:
                    # Auto-tune solo parámetros específicos
                    if self.eps == 'auto':
                        self.eps = self._estimate_eps(X_scaled)
                    if self.min_samples == 'auto':
                        self.min_samples = max(2, len(X_scaled) // 50)
                    
                    self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            else:
                self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            
            # Ajustar
            self.labels_ = self.model.fit_predict(X_scaled)
            
            # Calcular centros excluyendo outliers
            mask = self.labels_ != -1
            if mask.any():
                self._compute_cluster_centers(X_scaled[mask], self.labels_[mask])
            
            # Guardar nombres de features
            self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            
            return self
    
    def _estimate_eps(self, X):
        """Estimar eps usando k-distance graph."""
        from sklearn.neighbors import NearestNeighbors
        
        k = 4  # Típicamente min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Tomar k-ésima distancia y encontrar el "knee"
        k_distances = np.sort(distances[:, k-1])
        
        # Método simple del knee: punto con mayor diferencia
        diffs = np.diff(k_distances)
        knee_idx = np.argmax(diffs)
        
        return k_distances[knee_idx]
    
    def get_outliers(self):
        """Obtener outliers detectados por DBSCAN."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        outlier_mask = self.labels_ == -1
        return np.where(outlier_mask)[0]


class LearnerGaussianMixture(LearnerClustering):
    """
    Gaussian Mixture Models con auto-tuning del número de componentes.
    """
    
    def __init__(
        self,
        n_components: Union[int, str] = 'auto',
        covariance_type: str = 'full',
        max_components: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_components = max_components
    
    def _auto_tune(self, X):
        """Auto-tuning para Gaussian Mixture."""
        def objective(trial):
            n_comp = trial.suggest_int('n_components', 2, min(self.max_components, len(X)//5))
            cov_type = trial.suggest_categorical('covariance_type', 
                                               ['full', 'tied', 'diag', 'spherical'])
            
            model = GaussianMixture(
                n_components=n_comp,
                covariance_type=cov_type,
                random_state=self.random_state
            )
            
            try:
                labels = model.fit_predict(X)
                return self._evaluate_clustering(X, labels)
            except:
                return -1
        
        # Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.tune_trials, show_progress_bar=False)
        
        # Usar mejores parámetros
        best_params = study.best_params
        self.n_components = best_params['n_components']
        self.covariance_type = best_params['covariance_type']
        
        # Crear modelo final
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        
        self.tuning_history = study.trials_dataframe()
        
        return X
    
    def fit(self, task):
        """Ajustar Gaussian Mixture Model."""
        # Auto-tuning si es necesario
        if self.n_components == 'auto' and self.auto_tune:
            return super().fit(task)
        else:
            if self.n_components == 'auto':
                # Estimación simple sin auto-tune
                X = task.X if hasattr(task, 'X') else task.data
                self.n_components = min(10, max(2, len(X) // 20))
            
            self.model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            
            return super().fit(task)


class LearnerSpectralClustering(LearnerClustering):
    """
    Spectral Clustering con auto-tuning de parámetros.
    """
    
    def __init__(
        self,
        n_clusters: Union[int, str] = 'auto',
        affinity: str = 'rbf',
        gamma: Union[float, str] = 'auto',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
    
    def _auto_tune(self, X):
        """Auto-tuning para Spectral Clustering."""
        def objective(trial):
            n_clust = trial.suggest_int('n_clusters', 2, min(10, len(X)//5))
            
            if self.affinity == 'rbf':
                gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)
            else:
                gamma = 1.0
            
            model = SpectralClustering(
                n_clusters=n_clust,
                affinity=self.affinity,
                gamma=gamma,
                random_state=self.random_state
            )
            
            try:
                labels = model.fit_predict(X)
                return self._evaluate_clustering(X, labels)
            except:
                return -1
        
        # Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.tune_trials, show_progress_bar=False)
        
        # Usar mejores parámetros
        best_params = study.best_params
        self.n_clusters = best_params['n_clusters']
        
        if 'gamma' in best_params:
            self.gamma = best_params['gamma']
        
        # Crear modelo final
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            random_state=self.random_state
        )
        
        self.tuning_history = study.trials_dataframe()
        
        return X


if _HAS_HDBSCAN:
    class LearnerHDBSCAN(LearnerClustering):
        """
        HDBSCAN Clustering - versión jerárquica de DBSCAN.
        """
        
        def __init__(
            self,
            min_cluster_size: Union[int, str] = 'auto',
            min_samples: Union[int, str] = 'auto',
            **kwargs
        ):
            super().__init__(**kwargs)
            
            self.min_cluster_size = min_cluster_size
            self.min_samples = min_samples
        
        def _auto_tune(self, X):
            """Auto-tuning para HDBSCAN."""
            def objective(trial):
                min_cluster_size = trial.suggest_int('min_cluster_size', 2, len(X)//5)
                min_samples = trial.suggest_int('min_samples', 1, min_cluster_size)
                
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                
                try:
                    labels = model.fit_predict(X)
                    if len(set(labels)) <= 1:
                        return -1
                    return self._evaluate_clustering(X, labels)
                except:
                    return -1
            
            # Optuna optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=self.tune_trials, show_progress_bar=False)
            
            # Usar mejores parámetros
            best_params = study.best_params
            self.min_cluster_size = best_params['min_cluster_size']
            self.min_samples = best_params['min_samples']
            
            # Crear modelo final
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            
            self.tuning_history = study.trials_dataframe()
            
            return X
else:
    class LearnerHDBSCAN(LearnerClustering):
        """HDBSCAN placeholder when library not available."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            raise ImportError(
                "HDBSCAN not available. Install with: pip install hdbscan"
            )


class LearnerMeanShift(LearnerClustering):
    """
    Mean Shift Clustering con bandwidth auto-tuning.
    """
    
    def __init__(
        self,
        bandwidth: Union[float, str] = 'auto',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
    
    def fit(self, task):
        """Ajustar Mean Shift."""
        if self.bandwidth == 'auto':
            # Estimar bandwidth automáticamente
            from sklearn.cluster import estimate_bandwidth
            X = task.X if hasattr(task, 'X') else task.data
            X_scaled = self._prepare_data(X)
            
            bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
            self.bandwidth = bandwidth if bandwidth > 0 else 1.0
        
        self.model = MeanShift(bandwidth=self.bandwidth)
        return super().fit(task)


class LearnerAffinityPropagation(LearnerClustering):
    """
    Affinity Propagation Clustering.
    """
    
    def __init__(
        self,
        damping: float = 0.5,
        preference: Union[float, str] = 'auto',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.damping = damping
        self.preference = preference
    
    def fit(self, task):
        """Ajustar Affinity Propagation."""
        X = task.X if hasattr(task, 'X') else task.data
        X_scaled = self._prepare_data(X)
        
        if self.preference == 'auto':
            # Usar mediana de similaridades como preference
            from sklearn.metrics import euclidean_distances
            distances = euclidean_distances(X_scaled)
            self.preference = -np.median(distances)
        
        self.model = AffinityPropagation(
            damping=self.damping,
            preference=self.preference,
            random_state=self.random_state
        )
        
        return super().fit(task)