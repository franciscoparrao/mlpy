"""
Integración con Weights & Biases para tracking de experimentos.
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
class WandBConfig(TrackerConfig):
    """Configuración específica para Weights & Biases.
    
    Attributes
    ----------
    project : str
        Nombre del proyecto en WandB.
    entity : Optional[str]
        Entidad/equipo en WandB.
    group : Optional[str]
        Grupo de runs relacionados.
    job_type : Optional[str]
        Tipo de trabajo (train, eval, etc.).
    mode : str
        Modo de operación ('online', 'offline', 'disabled').
    dir : Optional[str]
        Directorio para guardar archivos locales.
    resume : Union[bool, str]
        Si resumir un run previo.
    reinit : bool
        Si reinicializar wandb.
    config_exclude_keys : List[str]
        Claves a excluir de la configuración.
    config_include_keys : List[str]
        Claves a incluir en la configuración.
    """
    project: str = "mlpy-experiments"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    mode: str = "online"
    dir: Optional[str] = None
    resume: Union[bool, str] = False
    reinit: bool = False
    config_exclude_keys: List[str] = field(default_factory=list)
    config_include_keys: List[str] = field(default_factory=list)
    save_code: bool = True
    anonymous: Optional[str] = None


class WandBTracker(ExperimentTracker):
    """Tracker de experimentos usando Weights & Biases."""
    
    def __init__(self, config: WandBConfig):
        """Inicializa el tracker de WandB.
        
        Parameters
        ----------
        config : WandBConfig
            Configuración del tracker.
        """
        self.wandb = None
        self.run = None
        super().__init__(config)
    
    def _setup(self):
        """Configura Weights & Biases."""
        try:
            import wandb
            self.wandb = wandb
            
            # Login si es necesario
            if hasattr(self.config, 'anonymous') and self.config.anonymous:
                wandb.login(anonymous=self.config.anonymous)
            
            logger.info(f"WandB tracker initialized for project: {self.config.project}")
            
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ExperimentRun:
        """Inicia un nuevo run en WandB.
        
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
        if not self.wandb:
            raise RuntimeError("WandB not initialized")
        
        # Preparar configuración
        run_name = run_name or self.config.run_name
        all_tags = list({**self.config.tags, **(tags or {})}.values())
        
        # Configuración del run
        init_config = {
            'project': self.config.project,
            'name': run_name,
            'tags': all_tags,
            'config': {},
            'reinit': getattr(self.config, 'reinit', False),
            'mode': getattr(self.config, 'mode', 'online')
        }
        
        # Añadir configuración opcional
        if hasattr(self.config, 'entity') and self.config.entity:
            init_config['entity'] = self.config.entity
        if hasattr(self.config, 'group') and self.config.group:
            init_config['group'] = self.config.group
        if hasattr(self.config, 'job_type') and self.config.job_type:
            init_config['job_type'] = self.config.job_type
        if hasattr(self.config, 'dir') and self.config.dir:
            init_config['dir'] = self.config.dir
        if hasattr(self.config, 'resume'):
            init_config['resume'] = self.config.resume
        if hasattr(self.config, 'save_code') and self.config.save_code:
            init_config['save_code'] = self.config.save_code
        
        # Iniciar run
        self.run = self.wandb.init(**init_config)
        
        # Crear ExperimentRun
        self.current_run = ExperimentRun(
            run_id=self.run.id,
            experiment_id=self.run.project,
            status="RUNNING",
            start_time=datetime.now(),
            tags={t: "" for t in all_tags}  # WandB usa lista de tags
        )
        
        if self.config.verbose:
            logger.info(f"Started WandB run: {self.current_run.run_id}")
            logger.info(f"View run at: {self.run.get_url()}")
        
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """Finaliza el run actual en WandB.
        
        Parameters
        ----------
        status : str
            Estado final del run.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            # Marcar estado final
            if status == "FAILED":
                self.wandb.alert(
                    title="Run Failed",
                    text=f"Run {self.run.id} failed",
                    level=self.wandb.AlertLevel.ERROR
                )
            
            # Finalizar run
            self.run.finish(exit_code=0 if status == "FINISHED" else 1)
            
            if self.current_run:
                self.current_run.status = status
                self.current_run.end_time = datetime.now()
                
                if self.config.verbose:
                    logger.info(f"Ended WandB run: {self.current_run.run_id}")
                
                self.current_run = None
            
            self.run = None
            
        except Exception as e:
            logger.error(f"Error ending WandB run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Registra parámetros en WandB.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parámetros a registrar.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            # Filtrar parámetros si es necesario
            if hasattr(self.config, 'config_exclude_keys'):
                params = {k: v for k, v in params.items() 
                         if k not in self.config.config_exclude_keys}
            
            if hasattr(self.config, 'config_include_keys') and self.config.config_include_keys:
                params = {k: v for k, v in params.items() 
                         if k in self.config.config_include_keys}
            
            # Actualizar config del run
            self.run.config.update(params)
            
            if self.current_run:
                self.current_run.params.update(params)
            
        except Exception as e:
            logger.error(f"Error logging params to WandB: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Registra métricas en WandB.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Métricas a registrar.
        step : Optional[int]
            Paso del entrenamiento.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            log_dict = metrics.copy()
            if step is not None:
                log_dict['step'] = step
            
            self.wandb.log(log_dict, step=step)
            
            if self.current_run:
                self.current_run.metrics.update(metrics)
            
        except Exception as e:
            logger.error(f"Error logging metrics to WandB: {e}")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: Optional[str] = None
    ):
        """Registra un artefacto en WandB.
        
        Parameters
        ----------
        artifact_path : Union[str, Path]
            Ruta del artefacto.
        artifact_type : Optional[str]
            Tipo de artefacto.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            artifact_path = Path(artifact_path)
            artifact_type = artifact_type or "artifact"
            
            # Crear artefacto
            artifact = self.wandb.Artifact(
                name=artifact_path.stem,
                type=artifact_type
            )
            
            if artifact_path.is_file():
                artifact.add_file(str(artifact_path))
            elif artifact_path.is_dir():
                artifact.add_dir(str(artifact_path))
            
            # Log artefacto
            self.run.log_artifact(artifact)
            
            if self.current_run:
                self.current_run.artifacts.append(str(artifact_path))
            
        except Exception as e:
            logger.error(f"Error logging artifact to WandB: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un modelo en WandB.
        
        Parameters
        ----------
        model : Any
            Modelo a registrar.
        model_name : str
            Nombre del modelo.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del modelo.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            # Crear artefacto de modelo
            model_artifact = self.wandb.Artifact(
                name=model_name,
                type="model",
                metadata=metadata or {}
            )
            
            # Guardar modelo temporalmente
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / f"{model_name}.pkl"
                
                # Intentar diferentes métodos de guardado
                model_type = type(model).__module__
                
                if 'sklearn' in model_type:
                    import joblib
                    joblib.dump(model, model_path)
                elif 'torch' in model_type:
                    import torch
                    torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, 
                              model_path)
                elif 'tensorflow' in model_type or 'keras' in model_type:
                    model.save(str(model_path))
                else:
                    import pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                # Añadir archivo al artefacto
                model_artifact.add_file(str(model_path))
            
            # Log artefacto
            self.run.log_artifact(model_artifact)
            
            # Log metadata como parámetros
            if metadata:
                self.log_params({f"model_{k}": v for k, v in metadata.items()})
            
            logger.info(f"Logged model '{model_name}' to WandB")
            
        except Exception as e:
            logger.error(f"Error logging model to WandB: {e}")
    
    def log_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un dataset en WandB.
        
        Parameters
        ----------
        dataset : Any
            Dataset a registrar.
        dataset_name : str
            Nombre del dataset.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del dataset.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            # Crear artefacto de dataset
            dataset_artifact = self.wandb.Artifact(
                name=dataset_name,
                type="dataset",
                metadata=metadata or {}
            )
            
            # Guardar dataset temporalmente
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / f"{dataset_name}.csv"
                
                import pandas as pd
                if isinstance(dataset, pd.DataFrame):
                    dataset.to_csv(dataset_path, index=False)
                else:
                    # Intentar pickle
                    import pickle
                    dataset_path = Path(tmpdir) / f"{dataset_name}.pkl"
                    with open(dataset_path, 'wb') as f:
                        pickle.dump(dataset, f)
                
                # Añadir archivo al artefacto
                dataset_artifact.add_file(str(dataset_path))
            
            # Log artefacto
            self.run.log_artifact(dataset_artifact)
            
            # Log metadata
            if metadata:
                self.log_params({f"dataset_{k}": v for k, v in metadata.items()})
            
            # Log info del dataset
            self.log_params({
                'dataset_name': dataset_name,
                'dataset_type': type(dataset).__name__,
                'dataset_shape': str(getattr(dataset, 'shape', 'unknown'))
            })
            
        except Exception as e:
            logger.error(f"Error logging dataset to WandB: {e}")
    
    def log_table(
        self,
        table_name: str,
        columns: List[str],
        data: List[List[Any]]
    ):
        """Registra una tabla en WandB.
        
        Parameters
        ----------
        table_name : str
            Nombre de la tabla.
        columns : List[str]
            Nombres de las columnas.
        data : List[List[Any]]
            Datos de la tabla.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            table = self.wandb.Table(columns=columns, data=data)
            self.run.log({table_name: table})
        except Exception as e:
            logger.error(f"Error logging table to WandB: {e}")
    
    def log_image(
        self,
        image: Any,
        image_name: str,
        caption: Optional[str] = None
    ):
        """Registra una imagen en WandB.
        
        Parameters
        ----------
        image : Any
            Imagen (numpy array, PIL, matplotlib figure, etc.).
        image_name : str
            Nombre de la imagen.
        caption : Optional[str]
            Caption de la imagen.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            wandb_image = self.wandb.Image(image, caption=caption)
            self.run.log({image_name: wandb_image})
        except Exception as e:
            logger.error(f"Error logging image to WandB: {e}")
    
    def log_histogram(
        self,
        values: Any,
        hist_name: str,
        num_bins: Optional[int] = None
    ):
        """Registra un histograma en WandB.
        
        Parameters
        ----------
        values : Any
            Valores para el histograma.
        hist_name : str
            Nombre del histograma.
        num_bins : Optional[int]
            Número de bins.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            histogram = self.wandb.Histogram(values, num_bins=num_bins)
            self.run.log({hist_name: histogram})
        except Exception as e:
            logger.error(f"Error logging histogram to WandB: {e}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Obtiene información de un run de WandB.
        
        Parameters
        ----------
        run_id : str
            ID del run.
            
        Returns
        -------
        Optional[ExperimentRun]
            Información del run o None si no existe.
        """
        if not self.wandb:
            return None
        
        try:
            api = self.wandb.Api()
            run = api.run(f"{self.config.entity or api.default_entity}/{self.config.project}/{run_id}")
            
            return ExperimentRun(
                run_id=run.id,
                experiment_id=run.project,
                status="FINISHED" if run.state == "finished" else "RUNNING",
                start_time=datetime.fromisoformat(run.created_at),
                end_time=datetime.fromisoformat(run.heartbeat_at) if run.heartbeat_at else None,
                metrics=run.summary._json_dict,
                params=run.config,
                tags={t: "" for t in run.tags}
            )
        except Exception as e:
            logger.error(f"Error getting run from WandB: {e}")
            return None
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[ExperimentRun]:
        """Lista runs del proyecto en WandB.
        
        Parameters
        ----------
        experiment_name : Optional[str]
            Nombre del experimento (proyecto en WandB).
        filter_string : Optional[str]
            Filtro para los runs.
        max_results : int
            Número máximo de resultados.
            
        Returns
        -------
        List[ExperimentRun]
            Lista de runs.
        """
        if not self.wandb:
            return []
        
        try:
            api = self.wandb.Api()
            project_name = experiment_name or self.config.project
            project_path = f"{self.config.entity or api.default_entity}/{project_name}"
            
            # Obtener runs
            runs = api.runs(
                project_path,
                filters=filter_string,
                per_page=max_results
            )
            
            # Convertir a ExperimentRun
            experiment_runs = []
            for run in runs:
                experiment_runs.append(ExperimentRun(
                    run_id=run.id,
                    experiment_id=run.project,
                    status="FINISHED" if run.state == "finished" else "RUNNING",
                    start_time=datetime.fromisoformat(run.created_at),
                    end_time=datetime.fromisoformat(run.heartbeat_at) if run.heartbeat_at else None,
                    metrics=run.summary._json_dict,
                    params=run.config,
                    tags={t: "" for t in run.tags}
                ))
            
            return experiment_runs
            
        except Exception as e:
            logger.error(f"Error listing runs from WandB: {e}")
            return []
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compara múltiples runs de WandB.
        
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
    
    def watch(
        self,
        model: Any,
        criterion: Optional[Any] = None,
        log: str = "gradients",
        log_freq: int = 100
    ):
        """Observa un modelo para logging automático de gradientes/parámetros.
        
        Parameters
        ----------
        model : Any
            Modelo a observar (típicamente PyTorch).
        criterion : Optional[Any]
            Función de pérdida.
        log : str
            Qué loggear ('gradients', 'parameters', 'all', None).
        log_freq : int
            Frecuencia de logging.
        """
        if not self.wandb or not self.run:
            return
        
        try:
            self.wandb.watch(model, criterion, log=log, log_freq=log_freq)
            logger.info(f"Watching model for {log} logging")
        except Exception as e:
            logger.error(f"Error setting up model watch: {e}")