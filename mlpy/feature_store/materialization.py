"""
Materialización de features.

Gestiona la materialización y actualización de features computadas.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import logging
from pathlib import Path
import json
import schedule
import threading
import time

from .base import FeatureStore, FeatureGroup, FeatureView, FeatureDefinition
from .transformations import FeatureTransformation

logger = logging.getLogger(__name__)


class MaterializationStatus(Enum):
    """Estado de la materialización."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MaterializationJob:
    """Job de materialización de features.
    
    Attributes
    ----------
    job_id : str
        ID único del job.
    feature_view : str
        Vista de features a materializar.
    start_time : datetime
        Tiempo de inicio.
    end_time : Optional[datetime]
        Tiempo de fin.
    status : MaterializationStatus
        Estado del job.
    error : Optional[str]
        Mensaje de error si falló.
    rows_processed : int
        Número de filas procesadas.
    features_computed : List[str]
        Features computadas.
    """
    job_id: str
    feature_view: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: MaterializationStatus = MaterializationStatus.PENDING
    error: Optional[str] = None
    rows_processed: int = 0
    features_computed: List[str] = field(default_factory=list)
    
    def duration(self) -> Optional[timedelta]:
        """Calcula la duración del job.
        
        Returns
        -------
        Optional[timedelta]
            Duración si el job ha terminado.
        """
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "job_id": self.job_id,
            "feature_view": self.feature_view,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "error": self.error,
            "rows_processed": self.rows_processed,
            "features_computed": self.features_computed,
            "duration": str(self.duration()) if self.duration() else None
        }


class MaterializationScheduler:
    """Scheduler para materialización periódica de features.
    
    Gestiona la ejecución programada de jobs de materialización.
    
    Parameters
    ----------
    feature_store : FeatureStore
        Feature store a usar.
    job_history_path : str
        Ruta para guardar historial de jobs.
    """
    
    def __init__(
        self,
        feature_store: FeatureStore,
        job_history_path: str = "./materialization_history"
    ):
        self.feature_store = feature_store
        self.job_history_path = Path(job_history_path)
        self.job_history_path.mkdir(parents=True, exist_ok=True)
        
        self.jobs: Dict[str, MaterializationJob] = {}
        self.scheduled_tasks: Dict[str, Any] = {}
        self.running = False
        self.thread = None
        
        self._load_history()
    
    def _load_history(self):
        """Carga el historial de jobs."""
        history_file = self.job_history_path / "jobs.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                for job_data in history.get("jobs", []):
                    job = self._deserialize_job(job_data)
                    self.jobs[job.job_id] = job
    
    def _save_history(self):
        """Guarda el historial de jobs."""
        history_file = self.job_history_path / "jobs.json"
        history = {
            "jobs": [job.to_dict() for job in self.jobs.values()],
            "updated_at": datetime.now().isoformat()
        }
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _deserialize_job(self, data: Dict) -> MaterializationJob:
        """Deserializa un job desde diccionario."""
        return MaterializationJob(
            job_id=data["job_id"],
            feature_view=data["feature_view"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=MaterializationStatus(data["status"]),
            error=data.get("error"),
            rows_processed=data.get("rows_processed", 0),
            features_computed=data.get("features_computed", [])
        )
    
    def materialize_view(
        self,
        feature_view: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> MaterializationJob:
        """Materializa una vista de features.
        
        Parameters
        ----------
        feature_view : str
            Nombre de la vista.
        start_date : Optional[datetime]
            Fecha de inicio para datos.
        end_date : Optional[datetime]
            Fecha de fin para datos.
            
        Returns
        -------
        MaterializationJob
            Job de materialización.
        """
        # Crear job
        job_id = f"{feature_view}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = MaterializationJob(
            job_id=job_id,
            feature_view=feature_view
        )
        
        self.jobs[job_id] = job
        job.status = MaterializationStatus.RUNNING
        
        try:
            # Obtener vista
            view = self.feature_store.feature_views.get(feature_view)
            if not view:
                raise ValueError(f"Feature view {feature_view} not found")
            
            # Recopilar todas las features necesarias
            all_features = []
            for group_name in view.feature_groups:
                group = self.feature_store.get_feature_group(group_name)
                if group:
                    all_features.extend([f.name for f in group.features])
            
            # Filtrar por las features específicas de la vista
            if view.features:
                all_features = [f for f in all_features if f in view.features]
            
            # Obtener datos
            # Por simplicidad, usar todos los entity IDs disponibles
            # En producción, esto debería ser más sofisticado
            data = self._get_all_data_for_view(view, start_date, end_date)
            
            if data is not None and not data.empty:
                # Aplicar transformaciones si hay definiciones
                transformed_data = self._apply_transformations(data, view)
                
                # Guardar datos materializados
                output_path = self.job_history_path / "materialized" / feature_view
                output_path.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = output_path / f"{timestamp}.parquet"
                transformed_data.to_parquet(output_file, index=False)
                
                job.rows_processed = len(transformed_data)
                job.features_computed = list(transformed_data.columns)
                
                logger.info(f"Materialized {job.rows_processed} rows for view {feature_view}")
            else:
                logger.warning(f"No data found for view {feature_view}")
            
            job.status = MaterializationStatus.COMPLETED
            
        except Exception as e:
            job.status = MaterializationStatus.FAILED
            job.error = str(e)
            logger.error(f"Materialization failed for {feature_view}: {e}")
        
        finally:
            job.end_time = datetime.now()
            self._save_history()
        
        return job
    
    def _get_all_data_for_view(
        self,
        view: FeatureView,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Optional[pd.DataFrame]:
        """Obtiene todos los datos para una vista.
        
        Parameters
        ----------
        view : FeatureView
            Vista de features.
        start_date : Optional[datetime]
            Fecha de inicio.
        end_date : Optional[datetime]
            Fecha de fin.
            
        Returns
        -------
        Optional[pd.DataFrame]
            Datos combinados.
        """
        all_data = []
        
        for group_name in view.feature_groups:
            # Usar el método interno del feature store para leer datos
            if hasattr(self.feature_store, '_read_latest_features'):
                group_data = self.feature_store._read_latest_features(group_name)
                if group_data is not None:
                    # Filtrar por fechas si se especifican
                    if "_timestamp" in group_data.columns:
                        if start_date:
                            group_data = group_data[group_data["_timestamp"] >= start_date]
                        if end_date:
                            group_data = group_data[group_data["_timestamp"] <= end_date]
                    
                    all_data.append(group_data)
        
        if all_data:
            # Combinar todos los datos
            result = all_data[0]
            for df in all_data[1:]:
                # Merge por entity ID si es posible
                entity_col = f"{view.entity}_id"
                if entity_col in result.columns and entity_col in df.columns:
                    result = result.merge(df, on=entity_col, how="outer", suffixes=('', '_dup'))
                    # Eliminar columnas duplicadas
                    result = result.loc[:, ~result.columns.str.endswith('_dup')]
                else:
                    result = pd.concat([result, df], axis=1)
            
            return result
        
        return None
    
    def _apply_transformations(
        self,
        data: pd.DataFrame,
        view: FeatureView
    ) -> pd.DataFrame:
        """Aplica transformaciones a los datos.
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos originales.
        view : FeatureView
            Vista con transformaciones.
            
        Returns
        -------
        pd.DataFrame
            Datos transformados.
        """
        # Por ahora, retornar datos sin transformar
        # En una implementación completa, aplicaríamos las transformaciones definidas
        return data
    
    def schedule_materialization(
        self,
        feature_view: str,
        schedule_expr: str,
        **kwargs
    ):
        """Programa materialización periódica.
        
        Parameters
        ----------
        feature_view : str
            Vista a materializar.
        schedule_expr : str
            Expresión de schedule (ej: "daily", "hourly").
        **kwargs
            Argumentos adicionales para materialización.
        """
        def job_func():
            logger.info(f"Running scheduled materialization for {feature_view}")
            self.materialize_view(feature_view, **kwargs)
        
        # Programar con schedule library
        if schedule_expr == "daily":
            schedule.every().day.do(job_func)
        elif schedule_expr == "hourly":
            schedule.every().hour.do(job_func)
        elif schedule_expr.startswith("every_"):
            # Ej: "every_30_minutes"
            parts = schedule_expr.split("_")
            if len(parts) == 3 and parts[2] == "minutes":
                minutes = int(parts[1])
                schedule.every(minutes).minutes.do(job_func)
        else:
            # Expresión cron-like
            schedule.every().day.at(schedule_expr).do(job_func)
        
        self.scheduled_tasks[feature_view] = schedule_expr
        logger.info(f"Scheduled {feature_view} with expression: {schedule_expr}")
    
    def start(self):
        """Inicia el scheduler."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_scheduler)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Materialization scheduler started")
    
    def stop(self):
        """Detiene el scheduler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Materialization scheduler stopped")
    
    def _run_scheduler(self):
        """Loop principal del scheduler."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check cada minuto
    
    def get_job_status(self, job_id: str) -> Optional[MaterializationJob]:
        """Obtiene el estado de un job.
        
        Parameters
        ----------
        job_id : str
            ID del job.
            
        Returns
        -------
        Optional[MaterializationJob]
            El job si existe.
        """
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        feature_view: Optional[str] = None,
        status: Optional[MaterializationStatus] = None,
        limit: int = 100
    ) -> List[MaterializationJob]:
        """Lista jobs de materialización.
        
        Parameters
        ----------
        feature_view : Optional[str]
            Filtrar por vista.
        status : Optional[MaterializationStatus]
            Filtrar por estado.
        limit : int
            Límite de resultados.
            
        Returns
        -------
        List[MaterializationJob]
            Lista de jobs.
        """
        jobs = list(self.jobs.values())
        
        # Filtrar
        if feature_view:
            jobs = [j for j in jobs if j.feature_view == feature_view]
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Ordenar por tiempo de inicio (más recientes primero)
        jobs.sort(key=lambda x: x.start_time, reverse=True)
        
        return jobs[:limit]
    
    def get_materialized_data(
        self,
        feature_view: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Obtiene datos materializados.
        
        Parameters
        ----------
        feature_view : str
            Vista de features.
        timestamp : Optional[datetime]
            Timestamp específico o más reciente.
            
        Returns
        -------
        Optional[pd.DataFrame]
            Datos materializados.
        """
        materialized_path = self.job_history_path / "materialized" / feature_view
        
        if not materialized_path.exists():
            return None
        
        # Listar archivos parquet
        parquet_files = list(materialized_path.glob("*.parquet"))
        
        if not parquet_files:
            return None
        
        # Ordenar por timestamp en nombre
        parquet_files.sort(reverse=True)
        
        # Si hay timestamp específico, buscar el más cercano
        if timestamp:
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            # Buscar el archivo más cercano pero no posterior
            for file in parquet_files:
                if file.stem <= timestamp_str:
                    return pd.read_parquet(file)
        
        # Retornar el más reciente
        return pd.read_parquet(parquet_files[0])