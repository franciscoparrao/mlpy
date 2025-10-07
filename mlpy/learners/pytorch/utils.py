"""
Utilidades para PyTorch en MLPY.

Funciones auxiliares para manejo de dispositivos, checkpoints y modelos.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Obtiene el dispositivo para PyTorch.
    
    Parameters
    ----------
    device : Optional[str]
        Dispositivo específico ('cuda', 'cpu', 'mps') o None para auto-detectar.
        
    Returns
    -------
    torch.device
        Dispositivo seleccionado.
    """
    if device is not None:
        # Dispositivo especificado por el usuario
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            return torch.device('cpu')
        elif device == 'mps' and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Using CPU.")
            return torch.device('cpu')
        return torch.device(device)
    
    # Auto-detectar mejor dispositivo disponible
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Cuenta el número de parámetros en un modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo PyTorch.
    trainable_only : bool
        Si contar solo parámetros entrenables.
        
    Returns
    -------
    int
        Número de parámetros.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_layers(model: nn.Module, layers_to_freeze: Optional[List[str]] = None):
    """Congela capas de un modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo PyTorch.
    layers_to_freeze : Optional[List[str]]
        Nombres de capas a congelar. Si None, congela todas.
    """
    if layers_to_freeze is None:
        # Congelar todas las capas
        for param in model.parameters():
            param.requires_grad = False
        logger.info(f"Froze all {count_parameters(model, False)} parameters")
    else:
        # Congelar capas específicas
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                frozen_count += param.numel()
        logger.info(f"Froze {frozen_count} parameters in layers: {layers_to_freeze}")


def unfreeze_layers(model: nn.Module, layers_to_unfreeze: Optional[List[str]] = None):
    """Descongela capas de un modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo PyTorch.
    layers_to_unfreeze : Optional[List[str]]
        Nombres de capas a descongelar. Si None, descongela todas.
    """
    if layers_to_unfreeze is None:
        # Descongelar todas las capas
        for param in model.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze all {count_parameters(model)} parameters")
    else:
        # Descongelar capas específicas
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
                unfrozen_count += param.numel()
        logger.info(f"Unfroze {unfrozen_count} parameters in layers: {layers_to_unfreeze}")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: Union[str, Path],
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
):
    """Guarda un checkpoint del modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo a guardar.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizador a guardar.
    epoch : int
        Época actual.
    path : Union[str, Path]
        Ruta donde guardar.
    metrics : Optional[Dict[str, float]]
        Métricas a guardar.
    **kwargs
        Datos adicionales a guardar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics or {},
        **kwargs
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Carga un checkpoint.
    
    Parameters
    ----------
    path : Union[str, Path]
        Ruta del checkpoint.
    model : Optional[nn.Module]
        Modelo donde cargar los pesos.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizador donde cargar el estado.
    device : Optional[torch.device]
        Dispositivo donde cargar.
        
    Returns
    -------
    Dict[str, Any]
        Datos del checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        if checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Optimizer loaded from {path}")
    
    return checkpoint


def get_gpu_memory_usage() -> Dict[str, float]:
    """Obtiene el uso de memoria GPU.
    
    Returns
    -------
    Dict[str, float]
        Información de memoria GPU en GB.
    """
    if not torch.cuda.is_available():
        return {}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        info[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
            "usage_percent": round((reserved / total) * 100, 1)
        }
    
    return info


def clear_gpu_cache():
    """Limpia la caché de GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def set_deterministic(seed: int = 42):
    """Configura PyTorch para resultados determinísticos.
    
    Parameters
    ----------
    seed : int
        Semilla aleatoria.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configuración determinística
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic mode with seed {seed}")


def model_summary(model: nn.Module) -> str:
    """Genera un resumen del modelo.
    
    Parameters
    ----------
    model : nn.Module
        Modelo PyTorch.
        
    Returns
    -------
    str
        Resumen del modelo.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Model Summary")
    lines.append("=" * 60)
    
    # Contar parámetros
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params
    
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append(f"Non-trainable parameters: {non_trainable_params:,}")
    lines.append("-" * 60)
    
    # Listar capas
    lines.append("Layers:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        lines.append(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Obtiene el learning rate actual del optimizador.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizador.
        
    Returns
    -------
    float
        Learning rate actual.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Establece el learning rate del optimizador.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizador.
    lr : float
        Nuevo learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info(f"Learning rate set to {lr}")


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Aplica gradient clipping.
    
    Parameters
    ----------
    model : nn.Module
        Modelo.
    max_norm : float
        Norma máxima para los gradientes.
        
    Returns
    -------
    float
        Norma total de los gradientes antes del clipping.
    """
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm)