"""
Modelos especializados de NLP para MLPY.

Este módulo proporciona learners para procesamiento de lenguaje natural
con integración completa al ecosistema MLPY.
"""

from .transformers import (
    LearnerBERTClassifier,
    LearnerBERTRegressor,
    LearnerGPTGenerator,
    LearnerRoBERTaClassifier,
    LearnerDistilBERTClassifier
)

from .embeddings import (
    LearnerWord2Vec,
    LearnerFastText,
    LearnerDoc2Vec,
    LearnerSentenceTransformer
)

from .tasks import (
    LearnerSentimentAnalysis,
    LearnerNamedEntityRecognition,
    LearnerTextClassification,
    LearnerLanguageDetection,
    LearnerTopicModeling
)

from .traditional import (
    LearnerTFIDFClassifier,
    LearnerNaiveBayesText,
    LearnerSVMText,
    LearnerLogisticRegressionText
)

__all__ = [
    # Transformers
    'LearnerBERTClassifier',
    'LearnerBERTRegressor',
    'LearnerGPTGenerator',
    'LearnerRoBERTaClassifier',
    'LearnerDistilBERTClassifier',
    
    # Embeddings
    'LearnerWord2Vec',
    'LearnerFastText',
    'LearnerDoc2Vec',
    'LearnerSentenceTransformer',
    
    # Specialized Tasks
    'LearnerSentimentAnalysis',
    'LearnerNamedEntityRecognition', 
    'LearnerTextClassification',
    'LearnerLanguageDetection',
    'LearnerTopicModeling',
    
    # Traditional Methods
    'LearnerTFIDFClassifier',
    'LearnerNaiveBayesText',
    'LearnerSVMText',
    'LearnerLogisticRegressionText'
]