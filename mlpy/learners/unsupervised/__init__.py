"""
Algoritmos de Unsupervised Learning para MLPY.

Este módulo proporciona implementaciones de algoritmos no supervisados
con integración completa al ecosistema MLPY.
"""

from .clustering import (
    LearnerDBSCAN,
    LearnerGaussianMixture,
    LearnerSpectralClustering,
    LearnerHDBSCAN,
    LearnerMeanShift,
    LearnerAffinityPropagation
)

from .dimension_reduction import (
    LearnerTSNE,
    LearnerUMAP,
    LearnerPCAKernel,
    LearnerICA,
    LearnerFactorAnalysis,
    LearnerManifoldLearning
)

from .anomaly_detection import (
    LearnerIsolationForest,
    LearnerOneClassSVM,
    LearnerLocalOutlierFactor,
    LearnerEllipticEnvelope,
    LearnerAnomalyAutoencoder,
    LearnerStatisticalOutlier
)

__all__ = [
    # Clustering
    'LearnerDBSCAN',
    'LearnerGaussianMixture',
    'LearnerSpectralClustering',
    'LearnerHDBSCAN',
    'LearnerMeanShift',
    'LearnerAffinityPropagation',
    
    # Dimension Reduction
    'LearnerTSNE',
    'LearnerUMAP',
    'LearnerPCAKernel',
    'LearnerICA',
    'LearnerFactorAnalysis',
    'LearnerManifoldLearning',
    
    # Anomaly Detection
    'LearnerIsolationForest',
    'LearnerOneClassSVM',
    'LearnerLocalOutlierFactor',
    'LearnerEllipticEnvelope',
    'LearnerAnomalyAutoencoder',
    'LearnerStatisticalOutlier'
]