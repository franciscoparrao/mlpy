"""
Integración con MLflow para tracking de experimentos.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import json

from .base import ExperimentTracker, TrackerConfig, ExperimentRun

logger = logging.getLogger(__name__)


@dataclass
class MLFlowConfig(TrackerConfig):
    """Configuración específica para MLflow.
    
    Attributes
    ----------
    tracking_uri : str
        URI del servidor MLflow.
    artifact_location : Optional[str]
        Ubicación para guardar artefactos.
    registry_uri : Optional[str]
        URI del model registry.
    autolog_config : Dict[str, Any]
        Configuración para autolog.
    """
    tracking_uri: str = "file:./mlruns"
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None
    autolog_config: Dict[str, Any] = field(default_factory=lambda: {
        'log_models': True,
        'log_input_examples': False,
        'log_model_signatures': True,
        'log_dataset': False,
        'disable': False,
        'exclusive': False,
        'disable_for_unsupported_versions': False,
        'silent': False
    })


class MLFlowTracker(ExperimentTracker):
    """Tracker de experimentos usando MLflow."""
    
    def __init__(self, config: MLFlowConfig):
        """Inicializa el tracker de MLflow.
        
        Parameters
        ----------
        config : MLFlowConfig
            Configuración del tracker.
        """
        self.mlflow = None
        self.client = None
        super().__init__(config)
    
    def _setup(self):
        """Configura MLflow."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            self.mlflow = mlflow
            
            # Configurar tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Configurar registry URI si existe
            if hasattr(self.config, 'registry_uri') and self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)
            
            # Crear cliente
            self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
            
            # Crear o obtener experimento
            try:
                experiment = self.client.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(
                        self.config.experiment_name,
                        artifact_location=getattr(self.config, 'artifact_location', None),
                        tags=self.config.tags
                    )
                else:
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(self.config.experiment_name)
                
            except Exception as e:
                logger.warning(f"Could not create/get experiment: {e}")
                experiment_id = "0"  # Default experiment
            
            # Configurar autolog si está habilitado
            if self.config.auto_log and hasattr(self.config, 'autolog_config'):
                self._setup_autolog()
            
            logger.info(f"MLflow tracker initialized with URI: {self.config.tracking_uri}")
            
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")
    
    def _setup_autolog(self):
        """Configura autolog de MLflow."""
        if not self.mlflow:
            return
        
        autolog_config = getattr(self.config, 'autolog_config', {})
        
        # Configurar autolog para diferentes frameworks
        try:
            # sklearn
            self.mlflow.sklearn.autolog(**autolog_config)
        except Exception:
            pass
        
        try:
            # pytorch
            self.mlflow.pytorch.autolog(**autolog_config)
        except Exception:
            pass
        
        try:
            # tensorflow/keras
            self.mlflow.tensorflow.autolog(**autolog_config)
        except Exception:
            pass
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ExperimentRun:
        """Inicia un nuevo run en MLflow.
        
        Parameters
        ----------
        run_name : Optional[str]
            Nombre del run.
        tags : Optional[Dict[str, str]]
            Tags adicionales para el run.
            
        Returns
        -------
        ExperimentRun
            Información del run iniciado.
        """
        if not self.mlflow:
            raise RuntimeError("MLflow not initialized")
        
        # Iniciar run
        run_name = run_name or self.config.run_name
        all_tags = {**self.config.tags, **(tags or {})}
        
        active_run = self.mlflow.start_run(
            run_name=run_name,
            tags=all_tags
        )
        
        # Crear ExperimentRun
        self.current_run = ExperimentRun(
            run_id=active_run.info.run_id,
            experiment_id=active_run.info.experiment_id,
            status="RUNNING",
            start_time=datetime.fromtimestamp(active_run.info.start_time / 1000),
            tags=all_tags
        )
        
        if self.config.verbose:
            logger.info(f"Started MLflow run: {self.current_run.run_id}")
        
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """Finaliza el run actual en MLflow.
        
        Parameters
        ----------
        status : str
            Estado final del run.
        """
        if not self.mlflow:
            return
        
        try:
            self.mlflow.end_run(status=status)
            
            if self.current_run:
                self.current_run.status = status
                self.current_run.end_time = datetime.now()
                
                if self.config.verbose:
                    logger.info(f"Ended MLflow run: {self.current_run.run_id}")
                
                self.current_run = None
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Registra parámetros en MLflow.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parámetros a registrar.
        """
        if not self.mlflow or not self.is_active:
            return
        
        try:
            # MLflow solo acepta strings como valores de parámetros
            str_params = {k: str(v) for k, v in params.items()}
            self.mlflow.log_params(str_params)
            
            if self.current_run:
                self.current_run.params.update(params)
            
        except Exception as e:
            logger.error(f"Error logging params to MLflow: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Registra métricas en MLflow.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Métricas a registrar.
        step : Optional[int]
            Paso del entrenamiento.
        """
        if not self.mlflow or not self.is_active:
            return
        
        try:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)
            
            if self.current_run:
                self.current_run.metrics.update(metrics)
            
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: Optional[str] = None
    ):
        """Registra un artefacto en MLflow.
        
        Parameters
        ----------
        artifact_path : Union[str, Path]
            Ruta del artefacto.
        artifact_type : Optional[str]
            Tipo de artefacto.
        """
        if not self.mlflow or not self.is_active:
            return
        
        try:
            artifact_path = Path(artifact_path)
            
            if artifact_path.is_file():
                self.mlflow.log_artifact(str(artifact_path))
            elif artifact_path.is_dir():
                self.mlflow.log_artifacts(str(artifact_path))
            
            if self.current_run:
                self.current_run.artifacts.append(str(artifact_path))
            
        except Exception as e:
            logger.error(f"Error logging artifact to MLflow: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un modelo en MLflow.
        
        Parameters
        ----------
        model : Any
            Modelo a registrar.
        model_name : str
            Nombre del modelo.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del modelo.
        """
        if not self.mlflow or not self.is_active:
            return
        
        try:
            # Detectar tipo de modelo y usar el flavor apropiado
            model_type = type(model).__module__
            
            if 'sklearn' in model_type:
                self.mlflow.sklearn.log_model(
                    model,
                    model_name,
                    registered_model_name=model_name if self.config.log_models else None
                )
            elif 'torch' in model_type:
                self.mlflow.pytorch.log_model(
                    model,
                    model_name,
                    registered_model_name=model_name if self.config.log_models else None
                )
            elif 'tensorflow' in model_type or 'keras' in model_type:
                self.mlflow.tensorflow.log_model(
                    model,
                    model_name,
                    registered_model_name=model_name if self.config.log_models else None
                )
            else:
                # Usar pickle genérico
                import pickle
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                    pickle.dump(model, tmp)
                    self.log_artifact(tmp.name, artifact_type='model')
            
            # Log metadata
            if metadata:
                self.log_params({f"model_{k}": v for k, v in metadata.items()})
            
            logger.info(f"Logged model '{model_name}' to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
    
    def log_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un dataset en MLflow.
        
        Parameters
        ----------
        dataset : Any
            Dataset a registrar.
        dataset_name : str
            Nombre del dataset.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del dataset.
        """
        if not self.mlflow or not self.is_active:
            return
        
        try:
            # Guardar dataset como artefacto
            import pandas as pd
            
            if isinstance(dataset, pd.DataFrame):
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    dataset.to_csv(tmp.name, index=False)
                    self.log_artifact(tmp.name, artifact_type='dataset')
            else:
                # Intentar pickle
                import pickle
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                    pickle.dump(dataset, tmp)
                    self.log_artifact(tmp.name, artifact_type='dataset')
            
            # Log metadata
            if metadata:
                self.log_params({f"dataset_{k}": v for k, v in metadata.items()})
            
            # Log dataset info
            self.log_params({
                'dataset_name': dataset_name,
                'dataset_type': type(dataset).__name__,
                'dataset_shape': str(getattr(dataset, 'shape', 'unknown'))
            })
            
        except Exception as e:
            logger.error(f"Error logging dataset to MLflow: {e}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Obtiene información de un run de MLflow.
        
        Parameters
        ----------
        run_id : str
            ID del run.
            
        Returns
        -------
        Optional[ExperimentRun]
            Información del run o None si no existe.
        """
        if not self.client:
            return None
        
        try:
            mlflow_run = self.client.get_run(run_id)
            
            return ExperimentRun(
                run_id=mlflow_run.info.run_id,
                experiment_id=mlflow_run.info.experiment_id,
                status=mlflow_run.info.status,
                start_time=datetime.fromtimestamp(mlflow_run.info.start_time / 1000),
                end_time=datetime.fromtimestamp(mlflow_run.info.end_time / 1000) if mlflow_run.info.end_time else None,
                metrics=mlflow_run.data.metrics,
                params=mlflow_run.data.params,
                tags=mlflow_run.data.tags
            )
        except Exception as e:
            logger.error(f"Error getting run from MLflow: {e}")
            return None
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[ExperimentRun]:
        """Lista runs del experimento en MLflow.
        
        Parameters
        ----------
        experiment_name : Optional[str]
            Nombre del experimento.
        filter_string : Optional[str]
            Filtro para los runs.
        max_results : int
            Número máximo de resultados.
            
        Returns
        -------
        List[ExperimentRun]
            Lista de runs.
        """
        if not self.client:
            return []
        
        try:
            # Obtener experiment_id
            if experiment_name:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if not experiment:
                    return []
                experiment_ids = [experiment.experiment_id]
            else:
                experiment_ids = [self.current_run.experiment_id] if self.current_run else ["0"]
            
            # Buscar runs
            mlflow_runs = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results
            )
            
            # Convertir a ExperimentRun
            runs = []
            for mlflow_run in mlflow_runs:
                runs.append(ExperimentRun(
                    run_id=mlflow_run.info.run_id,
                    experiment_id=mlflow_run.info.experiment_id,
                    status=mlflow_run.info.status,
                    start_time=datetime.fromtimestamp(mlflow_run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(mlflow_run.info.end_time / 1000) if mlflow_run.info.end_time else None,
                    metrics=mlflow_run.data.metrics,
                    params=mlflow_run.data.params,
                    tags=mlflow_run.data.tags
                ))
            
            return runs
            
        except Exception as e:
            logger.error(f"Error listing runs from MLflow: {e}")
            return []
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compara múltiples runs de MLflow.
        
        Parameters
        ----------
        run_ids : List[str]
            IDs de los runs a comparar.
        metrics : Optional[List[str]]
            Métricas específicas a comparar.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Comparación de runs.
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                run_data = {
                    'params': run.params,
                    'metrics': run.metrics if not metrics else {
                        k: v for k, v in run.metrics.items() if k in metrics
                    },
                    'status': run.status,
                    'duration': (run.end_time - run.start_time).total_seconds() if run.end_time else None
                }
                comparison[run_id] = run_data
        
        return comparison
    
    def get_best_run(
        self,
        metric: str,
        mode: str = 'min'
    ) -> Optional[ExperimentRun]:
        """Obtiene el mejor run según una métrica.
        
        Parameters
        ----------
        metric : str
            Métrica a optimizar.
        mode : str
            'min' o 'max'.
            
        Returns
        -------
        Optional[ExperimentRun]
            Mejor run o None.
        """
        runs = self.list_runs()
        
        if not runs:
            return None
        
        # Filtrar runs con la métrica
        valid_runs = [r for r in runs if metric in r.metrics]
        
        if not valid_runs:
            return None
        
        # Encontrar el mejor
        if mode == 'min':
            best_run = min(valid_runs, key=lambda r: r.metrics[metric])
        else:
            best_run = max(valid_runs, key=lambda r: r.metrics[metric])
        
        return best_run