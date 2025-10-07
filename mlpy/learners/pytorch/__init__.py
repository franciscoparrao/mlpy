"""
Integración con PyTorch para MLPY.

Este módulo proporciona learners basados en PyTorch para deep learning.
"""

from .base import (
    LearnerPyTorch,
    LearnerPyTorchClassif,
    LearnerPyTorchRegr
)

from .datasets import (
    MLPYDataset,
    MLPYDataLoader,
    create_data_loaders
)

from .models import (
    MLPNet,
    CNNClassifier,
    ResNetWrapper,
    TransformerModel,
    AutoEncoder
)

from .callbacks import (
    PyTorchCallback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoardLogger
)

from .utils import (
    get_device,
    count_parameters,
    freeze_layers,
    unfreeze_layers,
    save_checkpoint,
    load_checkpoint
)

from .pretrained import (
    load_pretrained_model,
    get_available_models,
    finetune_model
)

__all__ = [
    # Base
    'LearnerPyTorch',
    'LearnerPyTorchClassif',
    'LearnerPyTorchRegr',
    
    # Datasets
    'MLPYDataset',
    'MLPYDataLoader',
    'create_data_loaders',
    
    # Models
    'MLPNet',
    'CNNClassifier',
    'ResNetWrapper',
    'TransformerModel',
    'AutoEncoder',
    
    # Callbacks
    'PyTorchCallback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'TensorBoardLogger',
    
    # Utils
    'get_device',
    'count_parameters',
    'freeze_layers',
    'unfreeze_layers',
    'save_checkpoint',
    'load_checkpoint',
    
    # Pretrained
    'load_pretrained_model',
    'get_available_models',
    'finetune_model'
]