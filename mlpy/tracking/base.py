"""
Base classes para tracking de experimentos.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuración base para trackers.
    
    Attributes
    ----------
    experiment_name : str
        Nombre del experimento.
    run_name : Optional[str]
        Nombre del run específico.
    tracking_uri : Optional[str]
        URI del servidor de tracking.
    tags : Dict[str, str]
        Tags para el experimento.
    auto_log : bool
        Si hacer auto-logging de métricas.
    log_artifacts : bool
        Si guardar artefactos (modelos, plots, etc.).
    """
    experiment_name: str = "default"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    auto_log: bool = True
    log_artifacts: bool = True
    log_models: bool = True
    log_datasets: bool = False
    verbose: bool = True


@dataclass
class ExperimentRun:
    """Información de un run de experimento.
    
    Attributes
    ----------
    run_id : str
        ID único del run.
    experiment_id : str
        ID del experimento.
    status : str
        Estado del run ('RUNNING', 'FINISHED', 'FAILED').
    start_time : datetime
        Tiempo de inicio.
    end_time : Optional[datetime]
        Tiempo de finalización.
    metrics : Dict[str, float]
        Métricas del run.
    params : Dict[str, Any]
        Parámetros del run.
    tags : Dict[str, str]
        Tags del run.
    """
    run_id: str
    experiment_id: str
    status: str = "RUNNING"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class ExperimentTracker(ABC):
    """Clase base abstracta para tracking de experimentos."""
    
    def __init__(self, config: TrackerConfig):
        """Inicializa el tracker.
        
        Parameters
        ----------
        config : TrackerConfig
            Configuración del tracker.
        """
        self.config = config
        self.current_run: Optional[ExperimentRun] = None
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Configura el tracker."""
        pass
    
    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ExperimentRun:
        """Inicia un nuevo run.
        
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
        pass
    
    @abstractmethod
    def end_run(self, status: str = "FINISHED"):
        """Finaliza el run actual.
        
        Parameters
        ----------
        status : str
            Estado final del run.
        """
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Registra parámetros del experimento.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parámetros a registrar.
        """
        pass
    
    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Registra métricas del experimento.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Métricas a registrar.
        step : Optional[int]
            Paso del entrenamiento.
        """
        pass
    
    @abstractmethod
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: Optional[str] = None
    ):
        """Registra un artefacto.
        
        Parameters
        ----------
        artifact_path : Union[str, Path]
            Ruta del artefacto.
        artifact_type : Optional[str]
            Tipo de artefacto ('model', 'figure', 'data', etc.).
        """
        pass
    
    @abstractmethod
    def log_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un modelo.
        
        Parameters
        ----------
        model : Any
            Modelo a registrar.
        model_name : str
            Nombre del modelo.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del modelo.
        """
        pass
    
    @abstractmethod
    def log_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un dataset.
        
        Parameters
        ----------
        dataset : Any
            Dataset a registrar.
        dataset_name : str
            Nombre del dataset.
        metadata : Optional[Dict[str, Any]]
            Metadata adicional del dataset.
        """
        pass
    
    @abstractmethod
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Obtiene información de un run.
        
        Parameters
        ----------
        run_id : str
            ID del run.
            
        Returns
        -------
        Optional[ExperimentRun]
            Información del run o None si no existe.
        """
        pass
    
    @abstractmethod
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[ExperimentRun]:
        """Lista runs del experimento.
        
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
        pass
    
    @abstractmethod
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compara múltiples runs.
        
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
        pass
    
    def log_batch(
        self,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Registra múltiples elementos en batch.
        
        Parameters
        ----------
        metrics : Optional[Dict[str, float]]
            Métricas a registrar.
        params : Optional[Dict[str, Any]]
            Parámetros a registrar.
        tags : Optional[Dict[str, str]]
            Tags a registrar.
        """
        if metrics:
            self.log_metrics(metrics)
        if params:
            self.log_params(params)
        if tags and hasattr(self, 'log_tags'):
            self.log_tags(tags)
    
    def log_figure(
        self,
        figure: Any,
        figure_name: str,
        close: bool = True
    ):
        """Registra una figura/plot.
        
        Parameters
        ----------
        figure : Any
            Figura matplotlib o similar.
        figure_name : str
            Nombre de la figura.
        close : bool
            Si cerrar la figura después de guardar.
        """
        # Implementación por defecto
        import tempfile
        import matplotlib.pyplot as plt
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            figure.savefig(tmp.name)
            self.log_artifact(tmp.name, artifact_type='figure')
            
            if close:
                plt.close(figure)
    
    def log_text(
        self,
        text: str,
        file_name: str
    ):
        """Registra texto como artefacto.
        
        Parameters
        ----------
        text : str
            Texto a registrar.
        file_name : str
            Nombre del archivo.
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            self.log_artifact(tmp.name, artifact_type='text')
    
    def log_dict(
        self,
        dictionary: Dict[str, Any],
        file_name: str
    ):
        """Registra un diccionario como JSON.
        
        Parameters
        ----------
        dictionary : Dict[str, Any]
            Diccionario a registrar.
        file_name : str
            Nombre del archivo.
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(dictionary, tmp, indent=2, default=str)
            tmp.flush()
            self.log_artifact(tmp.name, artifact_type='json')
    
    def __enter__(self):
        """Context manager entry."""
        if self.current_run is None:
            self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.end_run(status=status)
    
    @property
    def is_active(self) -> bool:
        """Verifica si hay un run activo."""
        return self.current_run is not None and self.current_run.status == "RUNNING"


class DummyTracker(ExperimentTracker):
    """Tracker dummy para cuando no se quiere usar tracking real."""
    
    def _setup(self):
        """No requiere setup."""
        logger.info("Using DummyTracker - no actual tracking will be performed")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ExperimentRun:
        """Simula inicio de run."""
        import uuid
        
        self.current_run = ExperimentRun(
            run_id=str(uuid.uuid4()),
            experiment_id="dummy_experiment",
            tags=tags or {}
        )
        
        if self.config.verbose:
            logger.info(f"DummyTracker: Started run {self.current_run.run_id}")
        
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """Simula fin de run."""
        if self.current_run:
            self.current_run.status = status
            self.current_run.end_time = datetime.now()
            
            if self.config.verbose:
                logger.info(f"DummyTracker: Ended run {self.current_run.run_id}")
            
            self.current_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Simula logging de parámetros."""
        if self.current_run:
            self.current_run.params.update(params)
            
            if self.config.verbose:
                logger.debug(f"DummyTracker: Logged params: {params}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Simula logging de métricas."""
        if self.current_run:
            self.current_run.metrics.update(metrics)
            
            if self.config.verbose:
                logger.debug(f"DummyTracker: Logged metrics at step {step}: {metrics}")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: Optional[str] = None
    ):
        """Simula logging de artefacto."""
        if self.current_run:
            self.current_run.artifacts.append(str(artifact_path))
            
            if self.config.verbose:
                logger.debug(f"DummyTracker: Logged artifact: {artifact_path}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Simula logging de modelo."""
        if self.config.verbose:
            logger.debug(f"DummyTracker: Logged model: {model_name}")
    
    def log_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Simula logging de dataset."""
        if self.config.verbose:
            logger.debug(f"DummyTracker: Logged dataset: {dataset_name}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Simula obtención de run."""
        if self.current_run and self.current_run.run_id == run_id:
            return self.current_run
        return None
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[ExperimentRun]:
        """Simula listado de runs."""
        if self.current_run:
            return [self.current_run]
        return []
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Simula comparación de runs."""
        return {}