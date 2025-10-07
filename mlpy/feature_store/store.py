"""
Implementación local del Feature Store.

Proporciona un Feature Store basado en sistema de archivos local.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dataclasses import asdict

from .base import (
    FeatureStore,
    Feature,
    FeatureGroup,
    FeatureView,
    FeatureDefinition,
    FeatureType,
    DataSource
)

logger = logging.getLogger(__name__)


class LocalFeatureStore(FeatureStore):
    """Feature Store basado en sistema de archivos local.
    
    Almacena features en formato Parquet para eficiencia.
    
    Parameters
    ----------
    storage_path : str
        Ruta donde almacenar los datos.
    name : str
        Nombre del feature store.
    """
    
    def __init__(self, storage_path: str = "./feature_store", name: str = "local_feature_store"):
        super().__init__(name)
        self.storage_path = Path(storage_path)
        self._initialize_storage()
        self._load_metadata()
    
    def _initialize_storage(self):
        """Inicializa la estructura de directorios."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Directorios principales
        self.groups_path = self.storage_path / "feature_groups"
        self.groups_path.mkdir(exist_ok=True)
        
        self.views_path = self.storage_path / "feature_views"
        self.views_path.mkdir(exist_ok=True)
        
        self.data_path = self.storage_path / "data"
        self.data_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.storage_path / "metadata.json"
    
    def _load_metadata(self):
        """Carga metadata del store."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Cargar feature groups
            for group_name in metadata.get("feature_groups", []):
                group_file = self.groups_path / f"{group_name}.json"
                if group_file.exists():
                    with open(group_file, 'r') as f:
                        group_data = json.load(f)
                    self.feature_groups[group_name] = self._deserialize_feature_group(group_data)
            
            # Cargar feature views
            for view_name in metadata.get("feature_views", []):
                view_file = self.views_path / f"{view_name}.json"
                if view_file.exists():
                    with open(view_file, 'r') as f:
                        view_data = json.load(f)
                    self.feature_views[view_name] = self._deserialize_feature_view(view_data)
    
    def _save_metadata(self):
        """Guarda metadata del store."""
        metadata = {
            "name": self.name,
            "feature_groups": list(self.feature_groups.keys()),
            "feature_views": list(self.feature_views.keys()),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _serialize_feature(self, feature: Feature) -> Dict:
        """Serializa una feature a diccionario."""
        data = asdict(feature)
        data["dtype"] = feature.dtype.value
        data["created_at"] = feature.created_at.isoformat()
        data["updated_at"] = feature.updated_at.isoformat()
        return data
    
    def _deserialize_feature(self, data: Dict) -> Feature:
        """Deserializa una feature desde diccionario."""
        data["dtype"] = FeatureType(data["dtype"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return Feature(**data)
    
    def _serialize_feature_group(self, group: FeatureGroup) -> Dict:
        """Serializa un feature group."""
        data = {
            "name": group.name,
            "features": [self._serialize_feature(f) for f in group.features],
            "entity": group.entity,
            "source": group.source.value,
            "description": group.description,
            "tags": group.tags,
            "version": group.version,
            "created_at": group.created_at.isoformat()
        }
        return data
    
    def _deserialize_feature_group(self, data: Dict) -> FeatureGroup:
        """Deserializa un feature group."""
        features = [self._deserialize_feature(f) for f in data["features"]]
        return FeatureGroup(
            name=data["name"],
            features=features,
            entity=data["entity"],
            source=DataSource(data["source"]),
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def _serialize_feature_view(self, view: FeatureView) -> Dict:
        """Serializa una feature view."""
        data = {
            "name": view.name,
            "feature_groups": view.feature_groups,
            "features": view.features,
            "entity": view.entity,
            "ttl": view.ttl.total_seconds() if view.ttl else None,
            "description": view.description,
            "tags": view.tags,
            "created_at": view.created_at.isoformat()
        }
        return data
    
    def _deserialize_feature_view(self, data: Dict) -> FeatureView:
        """Deserializa una feature view."""
        ttl = timedelta(seconds=data["ttl"]) if data.get("ttl") else None
        return FeatureView(
            name=data["name"],
            feature_groups=data["feature_groups"],
            features=data["features"],
            entity=data["entity"],
            ttl=ttl,
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
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
        try:
            # Agregar al diccionario
            self.feature_groups[feature_group.name] = feature_group
            
            # Guardar a disco
            group_file = self.groups_path / f"{feature_group.name}.json"
            with open(group_file, 'w') as f:
                json.dump(self._serialize_feature_group(feature_group), f, indent=2)
            
            # Crear directorio para datos
            group_data_path = self.data_path / feature_group.name
            group_data_path.mkdir(exist_ok=True)
            
            # Actualizar metadata
            self._save_metadata()
            
            logger.info(f"Registered feature group: {feature_group.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature group: {e}")
            return False
    
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
        return self.feature_groups.get(name)
    
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
        try:
            # Validar que los grupos existen
            for group_name in feature_view.feature_groups:
                if group_name not in self.feature_groups:
                    raise ValueError(f"Feature group {group_name} not found")
            
            # Agregar al diccionario
            self.feature_views[feature_view.name] = feature_view
            
            # Guardar a disco
            view_file = self.views_path / f"{feature_view.name}.json"
            with open(view_file, 'w') as f:
                json.dump(self._serialize_feature_view(feature_view), f, indent=2)
            
            # Actualizar metadata
            self._save_metadata()
            
            logger.info(f"Created feature view: {feature_view.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            return False
    
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
        try:
            if feature_group not in self.feature_groups:
                raise ValueError(f"Feature group {feature_group} not found")
            
            group = self.feature_groups[feature_group]
            
            # Validar esquema
            expected_features = {f.name for f in group.features}
            actual_features = set(data.columns)
            
            if not expected_features.issubset(actual_features):
                missing = expected_features - actual_features
                raise ValueError(f"Missing features: {missing}")
            
            # Agregar timestamp si no existe
            if timestamp is None:
                timestamp = datetime.now()
            
            data = data.copy()
            data["_timestamp"] = timestamp
            
            # Guardar datos
            group_data_path = self.data_path / feature_group
            
            # Usar timestamp como nombre de archivo
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            file_path = group_data_path / filename
            
            data.to_parquet(file_path, index=False)
            
            # Actualizar estadísticas de las features
            for feature in group.features:
                if feature.name in data.columns:
                    feature.compute_statistics(data[feature.name])
            
            logger.info(f"Wrote {len(data)} rows to feature group {feature_group}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            return False
    
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
        result = pd.DataFrame()
        
        # Agrupar features por feature group
        features_by_group = {}
        for feature_name in features:
            for group_name, group in self.feature_groups.items():
                if any(f.name == feature_name for f in group.features):
                    if group_name not in features_by_group:
                        features_by_group[group_name] = []
                    features_by_group[group_name].append(feature_name)
                    break
        
        # Leer datos de cada grupo
        for group_name, group_features in features_by_group.items():
            group_data = self._read_latest_features(group_name, timestamp)
            
            if group_data is not None and not group_data.empty:
                # Filtrar por entity IDs si es posible
                group = self.feature_groups[group_name]
                entity_col = f"{group.entity}_id"
                
                if entity_col in group_data.columns:
                    group_data = group_data[group_data[entity_col].isin(entity_ids)]
                
                # Seleccionar features
                cols_to_select = [entity_col] if entity_col in group_data.columns else []
                cols_to_select.extend([f for f in group_features if f in group_data.columns])
                
                if cols_to_select:
                    group_data = group_data[cols_to_select]
                    
                    # Combinar con resultado
                    if result.empty:
                        result = group_data
                    else:
                        # Merge por entity ID si existe
                        if entity_col in group_data.columns and entity_col in result.columns:
                            result = result.merge(group_data, on=entity_col, how="outer")
                        else:
                            result = pd.concat([result, group_data], axis=1)
        
        return result
    
    def _read_latest_features(
        self,
        feature_group: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Lee las features más recientes de un grupo.
        
        Parameters
        ----------
        feature_group : str
            Nombre del grupo.
        timestamp : Optional[datetime]
            Timestamp máximo a considerar.
            
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame con las features.
        """
        group_data_path = self.data_path / feature_group
        
        if not group_data_path.exists():
            return None
        
        # Listar archivos parquet
        parquet_files = list(group_data_path.glob("*.parquet"))
        
        if not parquet_files:
            return None
        
        # Ordenar por timestamp en nombre de archivo
        parquet_files.sort(reverse=True)
        
        # Si hay timestamp, filtrar archivos
        if timestamp:
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            parquet_files = [f for f in parquet_files if f.stem <= timestamp_str]
        
        if not parquet_files:
            return None
        
        # Leer el más reciente
        latest_file = parquet_files[0]
        return pd.read_parquet(latest_file)
    
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
        all_data = []
        
        # Agrupar features por feature group
        features_by_group = {}
        for feature_name in features:
            for group_name, group in self.feature_groups.items():
                if any(f.name == feature_name for f in group.features):
                    if group_name not in features_by_group:
                        features_by_group[group_name] = []
                    features_by_group[group_name].append(feature_name)
                    break
        
        # Leer datos históricos de cada grupo
        for group_name, group_features in features_by_group.items():
            group_data_path = self.data_path / group_name
            
            if not group_data_path.exists():
                continue
            
            # Filtrar archivos por rango de tiempo
            start_str = start_time.strftime('%Y%m%d_%H%M%S')
            end_str = end_time.strftime('%Y%m%d_%H%M%S')
            
            parquet_files = [
                f for f in group_data_path.glob("*.parquet")
                if start_str <= f.stem <= end_str
            ]
            
            for file_path in parquet_files:
                data = pd.read_parquet(file_path)
                
                # Filtrar por entity IDs
                group = self.feature_groups[group_name]
                entity_col = f"{group.entity}_id"
                
                if entity_col in data.columns:
                    data = data[data[entity_col].isin(entity_ids)]
                
                # Seleccionar features
                cols_to_select = ["_timestamp"]
                if entity_col in data.columns:
                    cols_to_select.append(entity_col)
                cols_to_select.extend([f for f in group_features if f in data.columns])
                
                data = data[cols_to_select]
                all_data.append(data)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Ordenar por timestamp
            if "_timestamp" in result.columns:
                result = result.sort_values("_timestamp")
            return result
        
        return pd.DataFrame()


class FeatureRegistry:
    """Registro central de features y sus definiciones.
    
    Gestiona el catálogo de features disponibles.
    """
    
    def __init__(self):
        self.features: Dict[str, Feature] = {}
        self.definitions: Dict[str, FeatureDefinition] = {}
        self.lineage: Dict[str, List[str]] = {}  # Feature -> dependencias
    
    def register_feature(self, feature: Feature):
        """Registra una feature.
        
        Parameters
        ----------
        feature : Feature
            Feature a registrar.
        """
        self.features[feature.name] = feature
        logger.info(f"Registered feature: {feature.name}")
    
    def register_definition(self, definition: FeatureDefinition):
        """Registra una definición de feature.
        
        Parameters
        ----------
        definition : FeatureDefinition
            Definición a registrar.
        """
        self.definitions[definition.name] = definition
        
        # Actualizar lineage
        self.lineage[definition.name] = definition.source_features
        
        logger.info(f"Registered feature definition: {definition.name}")
    
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
        return self.features.get(name)
    
    def get_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Obtiene una definición por nombre.
        
        Parameters
        ----------
        name : str
            Nombre de la definición.
            
        Returns
        -------
        Optional[FeatureDefinition]
            La definición si existe.
        """
        return self.definitions.get(name)
    
    def get_lineage(self, feature: str, max_depth: int = 10) -> Dict[str, List[str]]:
        """Obtiene el lineage completo de una feature.
        
        Parameters
        ----------
        feature : str
            Nombre de la feature.
        max_depth : int
            Profundidad máxima a explorar.
            
        Returns
        -------
        Dict[str, List[str]]
            Árbol de dependencias.
        """
        def _get_deps(feat: str, depth: int = 0) -> Dict[str, Any]:
            if depth >= max_depth or feat not in self.lineage:
                return {}
            
            deps = {}
            for dep in self.lineage.get(feat, []):
                deps[dep] = _get_deps(dep, depth + 1)
            
            return deps
        
        return {feature: _get_deps(feature)}
    
    def search_features(
        self,
        tags: Optional[Dict[str, str]] = None,
        dtype: Optional[FeatureType] = None,
        pattern: Optional[str] = None
    ) -> List[Feature]:
        """Busca features por criterios.
        
        Parameters
        ----------
        tags : Optional[Dict[str, str]]
            Tags a buscar.
        dtype : Optional[FeatureType]
            Tipo de dato.
        pattern : Optional[str]
            Patrón en el nombre.
            
        Returns
        -------
        List[Feature]
            Features que coinciden.
        """
        results = []
        
        for feature in self.features.values():
            # Filtrar por tags
            if tags:
                if not all(feature.tags.get(k) == v for k, v in tags.items()):
                    continue
            
            # Filtrar por tipo
            if dtype and feature.dtype != dtype:
                continue
            
            # Filtrar por patrón
            if pattern and pattern.lower() not in feature.name.lower():
                continue
            
            results.append(feature)
        
        return results