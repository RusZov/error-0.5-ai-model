"""
LLM Client for interacting with OpenAI API or local models.
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 512, **kwargs):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        
        # Try to import openai
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, system_prompt: str = "", 
                 temperature: float = 0.7, top_p: float = 0.9,
                 **kwargs) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        if self.client is None:
            return "[ERROR: OpenAI client not initialized]"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    def generate_batch(self, prompts: List[str], n_repetitions: int = 1,
                       **kwargs) -> List[str]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            prompts: List of prompts
            n_repetitions: Number of repetitions per prompt
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        all_responses = []
        
        for prompt in prompts:
            for _ in range(n_repetitions):
                response = self.generate(prompt, **kwargs)
                all_responses.append(response)
        
        return all_responses


class LocalLLMClient(BaseLLMClient):
    """Client for local models using transformers."""
    
    def __init__(self, model_name: str = "microsoft/phi-2", 
                 device: str = "cpu", max_length: int = 512, **kwargs):
        """
        Initialize local LLM client.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ("cpu" or "cuda")
            max_length: Maximum sequence length
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        # Try to import transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            print("Model loaded successfully")
            
        except ImportError:
            print("Warning: transformers/torch not installed. Install with: pip install transformers torch")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, **kwargs) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        if self.model is None or self.tokenizer is None:
            return "[ERROR: Local model not initialized]"
        
        # Prepare input
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        try:
            import torch
            from transformers import TopKTopPWarper
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated text
            generated_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    def generate_batch(self, prompts: List[str], n_repetitions: int = 1,
                       **kwargs) -> List[str]:
        """
        Generate multiple responses for each prompt.
        
        Args:
            prompts: List of prompts
            n_repetitions: Number of repetitions per prompt
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        all_responses = []
        
        for prompt in prompts:
            for _ in range(n_repetitions):
                response = self.generate(prompt, **kwargs)
                all_responses.append(response)
        
        return all_responses


class LLMClient:
    """
    Factory class for creating LLM clients.
    """
    
    @staticmethod
    def create(provider: str = "openai", **kwargs) -> BaseLLMClient:
        """
        Create an LLM client based on provider type.
        
        Args:
            provider: Provider type ("openai" or "local")
            **kwargs: Provider-specific parameters
            
        Returns:
            LLM client instance
        """
        if provider == "openai":
            api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
            model = kwargs.get('model', 'gpt-3.5-turbo')
            max_tokens = kwargs.get('max_tokens', 512)
            return OpenAIClient(api_key=api_key, model=model, max_tokens=max_tokens)
        
        elif provider == "local":
            model_name = kwargs.get('model_name', 'microsoft/phi-2')
            device = kwargs.get('device', 'cpu')
            max_length = kwargs.get('max_length', 512)
            return LocalLLMClient(model_name=model_name, device=device, max_length=max_length)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'local'.")
