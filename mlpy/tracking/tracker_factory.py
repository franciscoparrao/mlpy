"""
Factory para crear trackers de experimentos.
"""

from typing import Optional, Dict, Any, Union
import logging

from .base import ExperimentTracker, TrackerConfig, DummyTracker
from .mlflow_tracker import MLFlowTracker, MLFlowConfig
from .wandb_tracker import WandBTracker, WandBConfig

logger = logging.getLogger(__name__)

# Registro global de trackers
_TRACKERS: Dict[str, ExperimentTracker] = {}
_DEFAULT_TRACKER: Optional[str] = None


def create_tracker(
    tracker_type: str,
    config: Optional[Union[TrackerConfig, Dict[str, Any]]] = None,
    **kwargs
) -> ExperimentTracker:
    """Crea un tracker de experimentos.
    
    Parameters
    ----------
    tracker_type : str
        Tipo de tracker ('mlflow', 'wandb', 'dummy', 'none').
    config : Optional[Union[TrackerConfig, Dict[str, Any]]]
        Configuración del tracker.
    **kwargs
        Argumentos adicionales para la configuración.
        
    Returns
    -------
    ExperimentTracker
        Instancia del tracker.
    """
    # Convertir config dict a objeto si es necesario
    if isinstance(config, dict):
        config_dict = {**config, **kwargs}
    else:
        config_dict = kwargs
    
    # Crear tracker según tipo
    tracker_type = tracker_type.lower()
    
    if tracker_type in ['mlflow', 'ml_flow']:
        if config is None or isinstance(config, dict):
            config = MLFlowConfig(**config_dict)
        return MLFlowTracker(config)
    
    elif tracker_type in ['wandb', 'weights_and_biases', 'wb']:
        if config is None or isinstance(config, dict):
            config = WandBConfig(**config_dict)
        return WandBTracker(config)
    
    elif tracker_type in ['dummy', 'none', 'null']:
        if config is None or isinstance(config, dict):
            config = TrackerConfig(**config_dict)
        return DummyTracker(config)
    
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


def register_tracker(
    name: str,
    tracker: ExperimentTracker,
    set_as_default: bool = False
) -> None:
    """Registra un tracker globalmente.
    
    Parameters
    ----------
    name : str
        Nombre para el tracker.
    tracker : ExperimentTracker
        Instancia del tracker.
    set_as_default : bool
        Si establecer como tracker por defecto.
    """
    global _TRACKERS, _DEFAULT_TRACKER
    
    _TRACKERS[name] = tracker
    
    if set_as_default or _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = name
    
    logger.info(f"Registered tracker '{name}' (default={set_as_default})")


def get_tracker(name: Optional[str] = None) -> Optional[ExperimentTracker]:
    """Obtiene un tracker registrado.
    
    Parameters
    ----------
    name : Optional[str]
        Nombre del tracker. Si None, retorna el tracker por defecto.
        
    Returns
    -------
    Optional[ExperimentTracker]
        Tracker o None si no existe.
    """
    global _TRACKERS, _DEFAULT_TRACKER
    
    if name is None:
        name = _DEFAULT_TRACKER
    
    if name is None:
        return None
    
    return _TRACKERS.get(name)


def list_trackers() -> Dict[str, str]:
    """Lista los trackers registrados.
    
    Returns
    -------
    Dict[str, str]
        Diccionario con nombre y tipo de cada tracker.
    """
    return {
        name: type(tracker).__name__ 
        for name, tracker in _TRACKERS.items()
    }


def clear_trackers() -> None:
    """Limpia todos los trackers registrados."""
    global _TRACKERS, _DEFAULT_TRACKER
    
    # Finalizar runs activos
    for tracker in _TRACKERS.values():
        if tracker.is_active:
            tracker.end_run()
    
    _TRACKERS.clear()
    _DEFAULT_TRACKER = None
    
    logger.info("Cleared all registered trackers")


def set_default_tracker(name: str) -> None:
    """Establece el tracker por defecto.
    
    Parameters
    ----------
    name : str
        Nombre del tracker.
    """
    global _DEFAULT_TRACKER
    
    if name not in _TRACKERS:
        raise ValueError(f"Tracker '{name}' not registered")
    
    _DEFAULT_TRACKER = name
    logger.info(f"Set default tracker to '{name}'")