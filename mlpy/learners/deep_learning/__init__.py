"""
Modelos de Deep Learning avanzados para MLPY.

Este módulo proporciona implementaciones de algoritmos de deep learning
especializados con integración completa al ecosistema MLPY.
"""

from .rnn import (
    LearnerLSTM,
    LearnerGRU,
    LearnerBiLSTM,
    LearnerSeq2Seq
)

from .transformer import (
    LearnerTransformer,
    LearnerBERTClassifier,
    LearnerGPTGenerator,
    LearnerAttention
)

from .advanced import (
    LearnerVAE,
    LearnerGAN,
    LearnerWGAN,
    LearnerAutoencoder
)

from .cnn import (
    LearnerEfficientNet,
    LearnerViT,
    LearnerDenseNet,
    LearnerMobileNet
)

__all__ = [
    # RNN/LSTM
    'LearnerLSTM',
    'LearnerGRU', 
    'LearnerBiLSTM',
    'LearnerSeq2Seq',
    
    # Transformers
    'LearnerTransformer',
    'LearnerBERTClassifier',
    'LearnerGPTGenerator',
    'LearnerAttention',
    
    # Advanced/Generative
    'LearnerVAE',
    'LearnerGAN',
    'LearnerWGAN', 
    'LearnerAutoencoder',
    
    # CNN Advanced
    'LearnerEfficientNet',
    'LearnerViT',
    'LearnerDenseNet',
    'LearnerMobileNet'
]