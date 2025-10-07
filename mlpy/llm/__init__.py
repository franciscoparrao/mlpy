"""
MLPY LLM Integration Module
===========================

Large Language Model integration for enhanced ML workflows.
"""

from .base import (
    LLMConfig,
    LLMResponse,
    LLMProvider,
    BaseLLM,
    ModelType,
    Provider
)

from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    HuggingFaceProvider
)

from .prompts import (
    PromptTemplate,
    FewShotTemplate,
    ChatTemplate,
    ChainOfThoughtTemplate,
    PromptLibrary,
    PromptOptimizer
)

from .rag import (
    Document,
    SearchResult,
    VectorStore,
    DocumentLoader,
    RAGPipeline
)

from .chain import (
    ChainType,
    LLMChain,
    SequentialChain,
    ConversationChain,
    RouterChain,
    MapReduceChain,
    ChainBuilder
)

from .embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    GeminiEmbeddings,
    OllamaEmbeddings,
    CohereEmbeddings,
    EmbeddingManager
)

from .ml_assistant import MLAssistant

__all__ = [
    # Base
    'LLMConfig',
    'LLMResponse', 
    'LLMProvider',
    'BaseLLM',
    'ModelType',
    'Provider',
    
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'OllamaProvider',
    'HuggingFaceProvider',
    
    # Prompts
    'PromptTemplate',
    'FewShotTemplate',
    'ChatTemplate',
    'ChainOfThoughtTemplate',
    'PromptLibrary',
    'PromptOptimizer',
    
    # RAG
    'Document',
    'SearchResult',
    'VectorStore',
    'DocumentLoader',
    'RAGPipeline',
    
    # Chains
    'ChainType',
    'LLMChain',
    'SequentialChain',
    'ConversationChain',
    'RouterChain',
    'MapReduceChain',
    'ChainBuilder',
    
    # Embeddings
    'EmbeddingConfig',
    'EmbeddingProvider',
    'OpenAIEmbeddings',
    'HuggingFaceEmbeddings',
    'GeminiEmbeddings',
    'OllamaEmbeddings',
    'CohereEmbeddings',
    'EmbeddingManager',
    
    # ML Assistant
    'MLAssistant'
]