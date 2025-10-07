"""
Base LLM Classes and Abstractions
=================================

Core interfaces for LLM integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    INSTRUCT = "instruct"


class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 60
    retry_attempts: int = 3
    streaming: bool = False
    
    # Provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k
        }
        if self.extra_params:
            config.update(self.extra_params)
        return config


@dataclass
class LLMResponse:
    """Response from LLM."""
    
    text: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def tokens_used(self) -> int:
        """Total tokens used."""
        if self.usage:
            return self.usage.get('total_tokens', 0)
        return 0
    
    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on usage."""
        if not self.usage:
            return 0.0
        
        # Rough cost estimates per 1K tokens
        cost_map = {
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.002,
            'claude-3': 0.025,
            'claude-instant': 0.001
        }
        
        model_key = self.model.lower()
        for key, cost in cost_map.items():
            if key in model_key:
                return (self.tokens_used / 1000) * cost
        
        return 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM provider.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self):
        """Setup the provider-specific client."""
        pass
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion for prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector(s)
        """
        pass
    
    def stream_complete(
        self,
        prompt: str,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Stream completion for prompt.
        
        Args:
            prompt: Input prompt
            callback: Function to call for each token
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        # Default implementation: non-streaming
        response = self.complete(prompt, **kwargs)
        if callback:
            callback(response.text)
        return response
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        # Simple approximation
        return len(text.split()) * 1.3
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        if not self.config.model:
            raise ValueError("Model name required")
        
        if self.config.temperature < 0 or self.config.temperature > 2:
            logger.warning(f"Unusual temperature: {self.config.temperature}")
        
        if self.config.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        
        return True


class BaseLLM:
    """Base class for high-level LLM operations."""
    
    def __init__(
        self,
        provider: Union[str, LLMProvider],
        config: Optional[LLMConfig] = None,
        **kwargs
    ):
        """
        Initialize LLM.
        
        Args:
            provider: Provider name or instance
            config: Configuration
            **kwargs: Additional config parameters
        """
        if isinstance(provider, str):
            # Create provider from string
            self.provider = self._create_provider(provider, config, **kwargs)
        else:
            self.provider = provider
        
        self.history: List[Dict[str, Any]] = []
        self.token_count = 0
        self.total_cost = 0.0
    
    def _create_provider(
        self,
        provider_name: str,
        config: Optional[LLMConfig],
        **kwargs
    ) -> LLMProvider:
        """Create provider instance from name."""
        # Import here to avoid circular imports
        from .providers import (
            OpenAIProvider,
            AnthropicProvider,
            GeminiProvider,
            OllamaProvider,
            HuggingFaceProvider
        )
        
        provider_map = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            'gemini': GeminiProvider,
            'ollama': OllamaProvider,
            'huggingface': HuggingFaceProvider
        }
        
        provider_class = provider_map.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Create config if not provided
        if config is None:
            config = LLMConfig(provider=provider_name, **kwargs)
        
        return provider_class(config)
    
    def complete(
        self,
        prompt: str,
        save_history: bool = True,
        **kwargs
    ) -> str:
        """
        Generate completion for prompt.
        
        Args:
            prompt: Input prompt
            save_history: Whether to save in history
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        response = self.provider.complete(prompt, **kwargs)
        
        if save_history:
            self.history.append({
                'type': 'completion',
                'prompt': prompt,
                'response': response.text,
                'timestamp': response.timestamp
            })
        
        self.token_count += response.tokens_used
        self.total_cost += response.cost_estimate
        
        return response.text
    
    def chat(
        self,
        message: str,
        role: str = "user",
        save_history: bool = True,
        **kwargs
    ) -> str:
        """
        Send chat message.
        
        Args:
            message: Message content
            role: Message role
            save_history: Whether to save in history
            **kwargs: Additional parameters
            
        Returns:
            Response text
        """
        # Build messages from history
        messages = []
        for item in self.history:
            if item['type'] == 'chat':
                messages.append({
                    'role': item.get('role', 'user'),
                    'content': item['prompt']
                })
                messages.append({
                    'role': 'assistant',
                    'content': item['response']
                })
        
        # Add current message
        messages.append({'role': role, 'content': message})
        
        # Get response
        response = self.provider.chat(messages, **kwargs)
        
        if save_history:
            self.history.append({
                'type': 'chat',
                'role': role,
                'prompt': message,
                'response': response.text,
                'timestamp': response.timestamp
            })
        
        self.token_count += response.tokens_used
        self.total_cost += response.cost_estimate
        
        return response.text
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_tokens': self.token_count,
            'total_cost': self.total_cost,
            'num_interactions': len(self.history),
            'history_size': len(json.dumps(self.history))
        }
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'history': self.history,
                'stats': self.get_usage_stats(),
                'config': self.provider.config.to_dict()
            }, f, indent=2)
    
    def load_conversation(self, filepath: str):
        """Load conversation history from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.history = data['history']
            stats = data.get('stats', {})
            self.token_count = stats.get('total_tokens', 0)
            self.total_cost = stats.get('total_cost', 0.0)