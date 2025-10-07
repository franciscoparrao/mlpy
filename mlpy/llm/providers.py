"""
LLM Provider Implementations
============================

Concrete implementations for different LLM providers.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
import warnings

from .base import LLMProvider, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def _setup_client(self):
        """Setup OpenAI client."""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        # Get API key
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        # Set up client
        if hasattr(openai, 'OpenAI'):
            # New API (>= 1.0)
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            self.is_new_api = True
        else:
            # Old API (< 1.0)
            openai.api_key = api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base
            self.client = openai
            self.is_new_api = False
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        start_time = time.time()
        
        # Merge kwargs with config
        params = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'top_p': self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty,
            'stop': self.config.stop_sequences
        }
        params.update(kwargs)
        
        try:
            if self.is_new_api:
                # New API
                if 'gpt' in self.config.model:
                    # Chat model
                    response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **params
                    )
                    text = response.choices[0].message.content
                    usage = response.usage.model_dump() if response.usage else None
                else:
                    # Completion model
                    response = self.client.completions.create(
                        prompt=prompt,
                        **params
                    )
                    text = response.choices[0].text
                    usage = response.usage.model_dump() if response.usage else None
            else:
                # Old API
                if 'gpt' in self.config.model:
                    response = self.client.ChatCompletion.create(
                        messages=[{"role": "user", "content": prompt}],
                        **params
                    )
                    text = response.choices[0].message.content
                else:
                    response = self.client.Completion.create(
                        prompt=prompt,
                        **params
                    )
                    text = response.choices[0].text
                
                usage = response.get('usage')
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model,
                provider="openai",
                usage=usage,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()
        
        params = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'top_p': self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty,
            'stop': self.config.stop_sequences
        }
        params.update(kwargs)
        
        try:
            if self.is_new_api:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **params
                )
                text = response.choices[0].message.content
                usage = response.usage.model_dump() if response.usage else None
            else:
                response = self.client.ChatCompletion.create(
                    messages=messages,
                    **params
                )
                text = response.choices[0].message.content
                usage = response.get('usage')
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model,
                provider="openai",
                usage=usage,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        # Use embedding model
        model = kwargs.get('model', 'text-embedding-ada-002')
        
        try:
            if self.is_new_api:
                response = self.client.embeddings.create(
                    input=text,
                    model=model
                )
                if isinstance(text, str):
                    return response.data[0].embedding
                else:
                    return [item.embedding for item in response.data]
            else:
                response = self.client.Embedding.create(
                    input=text,
                    model=model
                )
                if isinstance(text, str):
                    return response['data'][0]['embedding']
                else:
                    return [item['embedding'] for item in response['data']]
                    
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def _setup_client(self):
        """Setup Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        
        api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        start_time = time.time()
        
        # Claude uses messages API
        messages = [{"role": "user", "content": prompt}]
        
        params = {
            'model': self.config.model or 'claude-3-opus-20240229',
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'stop_sequences': self.config.stop_sequences
        }
        params.update(kwargs)
        
        try:
            response = self.client.messages.create(
                messages=messages,
                **params
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text from response
            text = ""
            for content in response.content:
                if hasattr(content, 'text'):
                    text += content.text
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model,
                provider="anthropic",
                usage={
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                } if hasattr(response, 'usage') else None,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()
        
        params = {
            'model': self.config.model or 'claude-3-opus-20240229',
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'stop_sequences': self.config.stop_sequences
        }
        params.update(kwargs)
        
        try:
            response = self.client.messages.create(
                messages=messages,
                **params
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            text = ""
            for content in response.content:
                if hasattr(content, 'text'):
                    text += content.text
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model,
                provider="anthropic",
                usage={
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                } if hasattr(response, 'usage') else None,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        # Anthropic doesn't have native embeddings
        # Use a fallback or raise error
        raise NotImplementedError("Anthropic doesn't provide embedding models. Use OpenAI or local models for embeddings.")


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""
    
    def _setup_client(self):
        """Setup Ollama client."""
        try:
            import requests
        except ImportError:
            raise ImportError("Requests library required for Ollama. Install with: pip install requests")
        
        self.base_url = self.config.api_base or "http://localhost:11434"
        self.requests = requests
        
        # Check if Ollama is running
        try:
            response = self.requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.warning("Ollama server not responding")
        except:
            logger.warning("Cannot connect to Ollama. Make sure it's running.")
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        start_time = time.time()
        
        endpoint = f"{self.base_url}/api/generate"
        
        payload = {
            'model': self.config.model or 'llama2',
            'prompt': prompt,
            'options': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'num_predict': self.config.max_tokens,
                'stop': self.config.stop_sequences or []
            }
        }
        payload.update(kwargs)
        
        try:
            response = self.requests.post(
                endpoint,
                json=payload,
                stream=False,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            # Parse streaming response
            full_text = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        full_text += data['response']
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                text=full_text.strip(),
                model=self.config.model,
                provider="ollama",
                usage={
                    'total_tokens': len(full_text.split())  # Approximation
                },
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()
        
        endpoint = f"{self.base_url}/api/chat"
        
        payload = {
            'model': self.config.model or 'llama2',
            'messages': messages,
            'options': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'num_predict': self.config.max_tokens
            }
        }
        payload.update(kwargs)
        
        try:
            response = self.requests.post(
                endpoint,
                json=payload,
                stream=False,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                text=result.get('message', {}).get('content', '').strip(),
                model=self.config.model,
                provider="ollama",
                usage={
                    'total_tokens': result.get('eval_count', 0) + result.get('prompt_eval_count', 0)
                },
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        endpoint = f"{self.base_url}/api/embeddings"
        
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        embeddings = []
        
        for t in texts:
            payload = {
                'model': kwargs.get('model', 'nomic-embed-text'),
                'prompt': t
            }
            
            try:
                response = self.requests.post(
                    endpoint,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings.append(result['embedding'])
                
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                raise
        
        return embeddings[0] if single_input else embeddings


class HuggingFaceProvider(LLMProvider):
    """HuggingFace models provider."""
    
    def _setup_client(self):
        """Setup HuggingFace client."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Transformers library not installed. Install with: pip install transformers torch")
        
        model_name = self.config.model or "microsoft/DialoGPT-medium"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Loaded HuggingFace model: {model_name} on {self.device}")
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        import torch
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider="huggingface",
            usage={
                'total_tokens': len(outputs[0])
            },
            latency_ms=latency_ms
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion."""
        # Convert messages to single prompt
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        prompt += "\nassistant: "
        
        return self.complete(prompt, **kwargs)
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        
        model_name = kwargs.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        if not hasattr(self, 'embed_model'):
            self.embed_model = SentenceTransformer(model_name)
        
        embeddings = self.embed_model.encode(text)
        
        return embeddings.tolist()


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def _setup_client(self):
        """Setup Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        
        # Get API key
        api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Google/Gemini API key required")
        
        # Configure client
        genai.configure(api_key=api_key)
        
        # Set up model
        model_name = self.config.model or 'gemini-pro'
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai
        
        logger.info(f"Initialized Gemini model: {model_name}")
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion."""
        start_time = time.time()
        
        # Configure generation
        generation_config = self.genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop_sequences=self.config.stop_sequences
        )
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            text = response.text if hasattr(response, 'text') else ""
            
            # Get token counts if available
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model or 'gemini-pro',
                provider="gemini",
                usage=usage,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion."""
        start_time = time.time()
        
        # Start chat session
        chat = self.model.start_chat(history=[])
        
        # Convert messages to Gemini format
        for message in messages[:-1]:  # All but last message as history
            role = 'user' if message['role'] == 'user' else 'model'
            chat.history.append({
                'role': role,
                'parts': [message['content']]
            })
        
        # Configure generation
        generation_config = self.genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )
        
        try:
            # Send last message
            last_message = messages[-1]['content']
            response = chat.send_message(
                last_message,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            text = response.text if hasattr(response, 'text') else ""
            
            # Get token counts
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                text=text.strip(),
                model=self.config.model or 'gemini-pro',
                provider="gemini",
                usage=usage,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise
    
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        # Use embedding model
        model_name = kwargs.get('model', 'models/embedding-001')
        
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        embeddings = []
        
        try:
            for t in texts:
                result = self.genai.embed_content(
                    model=model_name,
                    content=t,
                    task_type=kwargs.get('task_type', 'retrieval_document')
                )
                embeddings.append(result['embedding'])
            
            return embeddings[0] if single_input else embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise