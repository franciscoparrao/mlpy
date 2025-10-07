"""
Embedding Providers and Management
===================================

Unified interface for text embeddings across providers.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import hashlib
import pickle

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    
    provider: str
    model: Optional[str] = None
    dimension: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    cache_embeddings: bool = True
    cache_dir: str = ".embeddings_cache"
    batch_size: int = 100
    normalize: bool = True


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding provider.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.cache = {} if config.cache_embeddings else None
        
        if config.cache_embeddings and config.cache_dir:
            Path(config.cache_dir).mkdir(exist_ok=True)
            self.cache_file = Path(config.cache_dir) / f"{config.provider}_{config.model}.pkl"
            self._load_cache()
    
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Provider-specific parameters
            
        Returns:
            Embeddings as numpy arrays
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Check cache
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if self.cache and text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)
        
        # Generate new embeddings if needed
        if texts_to_embed:
            new_embeddings = self._generate_embeddings(texts_to_embed, **kwargs)
            
            # Update results and cache
            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = emb
                if self.cache is not None:
                    self.cache[texts[idx]] = emb
            
            # Save cache
            if self.config.cache_embeddings:
                self._save_cache()
        
        # Normalize if requested
        if self.config.normalize:
            embeddings = [self._normalize(emb) for emb in embeddings]
        
        if single_input:
            return embeddings[0]
        return embeddings
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate embeddings (to be overridden)."""
        raise NotImplementedError
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if self.cache_file and self.cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache = {}
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize OpenAI embeddings."""
        if config is None:
            config = EmbeddingConfig(
                provider="openai",
                model="text-embedding-ada-002",
                dimension=1536
            )
        super().__init__(config)
        
        # Setup OpenAI client
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        if hasattr(openai, 'OpenAI'):
            self.client = openai.OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            self.client = openai
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate OpenAI embeddings."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                if hasattr(self.client, 'embeddings'):
                    response = self.client.embeddings.create(
                        model=self.config.model,
                        input=batch
                    )
                    batch_embeddings = [np.array(item.embedding) for item in response.data]
                else:
                    response = self.client.Embedding.create(
                        model=self.config.model,
                        input=batch
                    )
                    batch_embeddings = [np.array(item['embedding']) for item in response['data']]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                raise
        
        return embeddings


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace embeddings provider."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize HuggingFace embeddings."""
        if config is None:
            config = EmbeddingConfig(
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384
            )
        super().__init__(config)
        
        # Setup sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(config.model or "sentence-transformers/all-MiniLM-L6-v2")
        
        # Update dimension based on model
        self.config.dimension = self.model.get_sentence_embedding_dimension()
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate HuggingFace embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=kwargs.get('show_progress', False)
        )
        return [emb for emb in embeddings]


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embeddings provider."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize Ollama embeddings."""
        if config is None:
            config = EmbeddingConfig(
                provider="ollama",
                model="nomic-embed-text",
                dimension=768
            )
        super().__init__(config)
        
        import requests
        self.requests = requests
        self.base_url = config.api_base or "http://localhost:11434"
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate Ollama embeddings."""
        embeddings = []
        endpoint = f"{self.base_url}/api/embeddings"
        
        for text in texts:
            payload = {
                "model": self.config.model,
                "prompt": text
            }
            
            try:
                response = self.requests.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                embeddings.append(np.array(result['embedding']))
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                raise
        
        return embeddings


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embeddings provider."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize Cohere embeddings."""
        if config is None:
            config = EmbeddingConfig(
                provider="cohere",
                model="embed-english-v3.0",
                dimension=1024
            )
        super().__init__(config)
        
        try:
            import cohere
        except ImportError:
            raise ImportError("Cohere library required. Install with: pip install cohere")
        
        api_key = config.api_key or os.getenv('COHERE_API_KEY')
        if not api_key:
            raise ValueError("Cohere API key required")
        
        self.client = cohere.Client(api_key)
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate Cohere embeddings."""
        response = self.client.embed(
            texts=texts,
            model=self.config.model,
            input_type=kwargs.get('input_type', 'search_document')
        )
        
        return [np.array(emb) for emb in response.embeddings]


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embeddings provider."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize Gemini embeddings."""
        if config is None:
            config = EmbeddingConfig(
                provider="gemini",
                model="models/embedding-001",
                dimension=768
            )
        super().__init__(config)
        
        # Setup Gemini client
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI library required. Install with: pip install google-generativeai")
        
        api_key = config.api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Google/Gemini API key required")
        
        genai.configure(api_key=api_key)
        self.genai = genai
    
    def _generate_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """Generate Gemini embeddings."""
        embeddings = []
        model = self.config.model or 'models/embedding-001'
        
        for text in texts:
            try:
                result = self.genai.embed_content(
                    model=model,
                    content=text,
                    task_type=kwargs.get('task_type', 'retrieval_document')
                )
                embeddings.append(np.array(result['embedding']))
            except Exception as e:
                logger.error(f"Gemini embedding error: {e}")
                raise
        
        return embeddings


class EmbeddingManager:
    """Manager for multiple embedding providers."""
    
    def __init__(self, default_provider: str = "huggingface"):
        """
        Initialize embedding manager.
        
        Args:
            default_provider: Default provider to use
        """
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.default_provider = default_provider
    
    def add_provider(
        self,
        name: str,
        provider: Union[EmbeddingProvider, EmbeddingConfig, str]
    ):
        """
        Add embedding provider.
        
        Args:
            name: Provider name
            provider: Provider instance, config, or type string
        """
        if isinstance(provider, EmbeddingProvider):
            self.providers[name] = provider
        elif isinstance(provider, EmbeddingConfig):
            self.providers[name] = self._create_provider(provider)
        elif isinstance(provider, str):
            config = EmbeddingConfig(provider=provider)
            self.providers[name] = self._create_provider(config)
    
    def _create_provider(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create provider from config."""
        provider_map = {
            "openai": OpenAIEmbeddings,
            "huggingface": HuggingFaceEmbeddings,
            "gemini": GeminiEmbeddings,
            "ollama": OllamaEmbeddings,
            "cohere": CohereEmbeddings
        }
        
        provider_class = provider_map.get(config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        return provider_class(config)
    
    def embed(
        self,
        texts: Union[str, List[str]],
        provider: Optional[str] = None,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings using specified provider.
        
        Args:
            texts: Texts to embed
            provider: Provider name (uses default if None)
            **kwargs: Provider-specific parameters
            
        Returns:
            Embeddings
        """
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            # Auto-create provider
            self.add_provider(provider_name, provider_name)
        
        return self.providers[provider_name].embed(texts, **kwargs)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == "euclidean":
            return -np.linalg.norm(embedding1 - embedding2)
        elif metric == "dot":
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar(
        self,
        query: str,
        documents: List[str],
        k: int = 5,
        provider: Optional[str] = None,
        metric: str = "cosine"
    ) -> List[Tuple[int, float, str]]:
        """
        Find similar documents to query.
        
        Args:
            query: Query text
            documents: List of documents
            k: Number of results
            provider: Embedding provider
            metric: Similarity metric
            
        Returns:
            List of (index, score, document) tuples
        """
        # Get embeddings
        query_embedding = self.embed(query, provider=provider)
        doc_embeddings = self.embed(documents, provider=provider)
        
        # Compute similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            score = self.compute_similarity(query_embedding, doc_emb, metric)
            similarities.append((i, score, documents[i]))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]