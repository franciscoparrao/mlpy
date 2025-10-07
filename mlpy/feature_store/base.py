"""
Clases base para el Feature Store.

Define las abstracciones principales para gestión de features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path


class FeatureType(Enum):
    """Tipos de features soportados."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    EMBEDDING = "embedding"
    BINARY = "binary"
    JSON = "json"


class DataSource(Enum):
    """Fuentes de datos soportadas."""
    CSV = "csv"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    CUSTOM = "custom"


@dataclass
class Feature:
    """Definición de una feature individual.
    
    Attributes
    ----------
    name : str
        Nombre de la feature.
    dtype : FeatureType
        Tipo de dato de la feature.
    description : str
        Descripción de la feature.
    tags : Dict[str, str]
        Tags asociados a la feature.
    statistics : Dict[str, Any]
        Estadísticas de la feature (mean, std, etc).
    version : int
        Versión de la feature.
    created_at : datetime
        Fecha de creación.
    updated_at : datetime
        Última actualización.
    """
    name: str
    dtype: FeatureType
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def compute_statistics(self, data: pd.Series):
        """Calcula estadísticas de la feature.
        
        Parameters
        ----------
        data : pd.Series
            Datos de la feature.
        """
        self.statistics = {}
        
        if self.dtype == FeatureType.NUMERIC:
            self.statistics = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "nulls": int(data.isnull().sum()),
                "unique": int(data.nunique())
            }
        elif self.dtype == FeatureType.CATEGORICAL:
            self.statistics = {
                "unique": int(data.nunique()),
                "nulls": int(data.isnull().sum()),
                "mode": data.mode()[0] if not data.mode().empty else None,
                "value_counts": data.value_counts().to_dict()
            }
        elif self.dtype == FeatureType.BINARY:
            self.statistics = {
                "true_ratio": float((data == 1).mean()),
                "false_ratio": float((data == 0).mean()),
                "nulls": int(data.isnull().sum())
            }
    
    def validate(self, value: Any) -> bool:
        """Valida un valor contra el tipo de feature.
        
        Parameters
        ----------
        value : Any
            Valor a validar.
            
        Returns
        -------
        bool
            True si el valor es válido.
        """
        if pd.isna(value):
            return True  # Permitir nulos
        
        if self.dtype == FeatureType.NUMERIC:
            return isinstance(value, (int, float, np.number))
        elif self.dtype == FeatureType.CATEGORICAL:
            return isinstance(value, (str, int))
        elif self.dtype == FeatureType.BINARY:
            return value in [0, 1, True, False]
        elif self.dtype == FeatureType.TIMESTAMP:
            return isinstance(value, (datetime, pd.Timestamp))
        elif self.dtype == FeatureType.TEXT:
            return isinstance(value, str)
        elif self.dtype == FeatureType.EMBEDDING:
            return isinstance(value, (list, np.ndarray))
        elif self.dtype == FeatureType.JSON:
            return isinstance(value, (dict, list))
        
        return True


@dataclass
class FeatureGroup:
    """Grupo de features relacionadas.
    
    Attributes
    ----------
    name : str
        Nombre del grupo.
    features : List[Feature]
        Features en el grupo.
    entity : str
        Entidad a la que pertenecen (ej: "user", "product").
    source : DataSource
        Fuente de datos.
    description : str
        Descripción del grupo.
    tags : Dict[str, str]
        Tags del grupo.
    version : int
        Versión del grupo.
    created_at : datetime
        Fecha de creación.
    """
    name: str
    features: List[Feature]
    entity: str
    source: DataSource
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_feature(self, name: str) -> Optional[Feature]:
        """Obtiene una feature por nombre.
        
        Parameters
        ----------
        name : str
            Nombre de la feature.
            
        Returns
        -------
        Optional[Feature]
            La feature si existe.
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        return None
    
    def add_feature(self, feature: Feature):
        """Agrega una feature al grupo.
        
        Parameters
        ----------
        feature : Feature
            Feature a agregar.
        """
        if self.get_feature(feature.name) is None:
            self.features.append(feature)
            self.version += 1
    
    def remove_feature(self, name: str) -> bool:
        """Elimina una feature del grupo.
        
        Parameters
        ----------
        name : str
            Nombre de la feature.
            
        Returns
        -------
        bool
            True si se eliminó.
        """
        for i, feature in enumerate(self.features):
            if feature.name == name:
                self.features.pop(i)
                self.version += 1
                return True
        return False
    
    def get_schema(self) -> Dict[str, str]:
        """Obtiene el esquema del grupo.
        
        Returns
        -------
        Dict[str, str]
            Mapeo de nombre de feature a tipo.
        """
        return {f.name: f.dtype.value for f in self.features}


@dataclass
class FeatureView:
    """Vista de features para training o serving.
    
    Una vista combina features de múltiples grupos.
    
    Attributes
    ----------
    name : str
        Nombre de la vista.
    feature_groups : List[str]
        Nombres de los grupos incluidos.
    features : List[str]
        Features específicas a incluir.
    entity : str
        Entidad principal.
    ttl : Optional[timedelta]
        Time-to-live para cache.
    description : str
        Descripción de la vista.
    tags : Dict[str, str]
        Tags de la vista.
    created_at : datetime
        Fecha de creación.
    """
    name: str
    feature_groups: List[str]
    features: List[str]
    entity: str
    ttl: Optional[timedelta] = None
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_cached(self, last_update: datetime) -> bool:
        """Verifica si los datos están en cache válido.
        
        Parameters
        ----------
        last_update : datetime
            Última actualización de los datos.
            
        Returns
        -------
        bool
            True si el cache es válido.
        """
        if self.ttl is None:
            return False
        
        age = datetime.now() - last_update
        return age < self.ttl


@dataclass
class FeatureDefinition:
    """Definición completa de una feature con transformaciones.
    
    Attributes
    ----------
    name : str
        Nombre de la feature.
    source_features : List[str]
        Features fuente para la transformación.
    transformation : str
        Código o nombre de la transformación.
    dtype : FeatureType
        Tipo de dato resultante.
    description : str
        Descripción de la definición.
    parameters : Dict[str, Any]
        Parámetros para la transformación.
    """
    name: str
    source_features: List[str]
    transformation: str
    dtype: FeatureType
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la transformación a los datos.
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos con las features fuente.
            
        Returns
        -------
        pd.Series
            Feature transformada.
        """
        # Transformaciones básicas predefinidas
        if self.transformation == "sum":
            return data[self.source_features].sum(axis=1)
        elif self.transformation == "mean":
            return data[self.source_features].mean(axis=1)
        elif self.transformation == "max":
            return data[self.source_features].max(axis=1)
        elif self.transformation == "min":
            return data[self.source_features].min(axis=1)
        elif self.transformation == "count":
            return data[self.source_features].notna().sum(axis=1)
        elif self.transformation == "ratio":
            if len(self.source_features) == 2:
                return data[self.source_features[0]] / (data[self.source_features[1]] + 1e-10)
        elif self.transformation == "diff":
            if len(self.source_features) == 2:
                return data[self.source_features[0]] - data[self.source_features[1]]
        elif self.transformation == "concat":
            return data[self.source_features].astype(str).agg('_'.join, axis=1)
        elif self.transformation == "custom":
            # Evaluar código personalizado (¡cuidado con la seguridad!)
            if "code" in self.parameters:
                # Este es un ejemplo simplificado
                # En producción, usar un sandbox seguro
                local_vars = {"data": data, "pd": pd, "np": np}
                exec(self.parameters["code"], {}, local_vars)
                return local_vars.get("result", pd.Series())
        
        return pd.Series()


class FeatureStore(ABC):
    """Clase base abstracta para Feature Store.
    
    Define la interfaz para implementaciones de Feature Store.
    """
    
    def __init__(self, name: str = "feature_store"):
        """Inicializa el Feature Store.
        
        Parameters
        ----------
        name : str
            Nombre del feature store.
        """
        self.name = name
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_views: Dict[str, FeatureView] = {}
    
    @abstractmethod
    def register_feature_group(self, feature_group: FeatureGroup) -> bool:
        """Registra un grupo de features.
        
        Parameters
        ----------
        feature_group : FeatureGroup
            Grupo a registrar.
            
        Returns
        -------
        bool
            True si se registró exitosamente.
        """
        pass
    
    @abstractmethod
    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Obtiene un grupo de features.
        
        Parameters
        ----------
        name : str
            Nombre del grupo.
            
        Returns
        -------
        Optional[FeatureGroup]
            El grupo si existe.
        """
        pass
    
    @abstractmethod
    def create_feature_view(self, feature_view: FeatureView) -> bool:
        """Crea una vista de features.
        
        Parameters
        ----------
        feature_view : FeatureView
            Vista a crear.
            
        Returns
        -------
        bool
            True si se creó exitosamente.
        """
        pass
    
    @abstractmethod
    def get_features(
        self,
        entity_ids: List[Any],
        features: List[str],
        timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Obtiene features para entidades específicas.
        
        Parameters
        ----------
        entity_ids : List[Any]
            IDs de las entidades.
        features : List[str]
            Nombres de las features.
        timestamp : Optional[datetime]
            Timestamp para features temporales.
            
        Returns
        -------
        pd.DataFrame
            DataFrame con las features.
        """
        pass
    
    @abstractmethod
    def write_features(
        self,
        feature_group: str,
        data: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Escribe features al store.
        
        Parameters
        ----------
        feature_group : str
            Nombre del grupo.
        data : pd.DataFrame
            Datos a escribir.
        timestamp : Optional[datetime]
            Timestamp de los datos.
            
        Returns
        -------
        bool
            True si se escribió exitosamente.
        """
        pass
    
    @abstractmethod
    def get_historical_features(
        self,
        entity_ids: List[Any],
        features: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Obtiene features históricas.
        
        Parameters
        ----------
        entity_ids : List[Any]
            IDs de las entidades.
        features : List[str]
            Nombres de las features.
        start_time : datetime
            Inicio del período.
        end_time : datetime
            Fin del período.
            
        Returns
        -------
        pd.DataFrame
            DataFrame con las features históricas.
        """
        pass