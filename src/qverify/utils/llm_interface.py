"""
LLM Interface utilities for QVERIFY.

This module provides unified interfaces for different LLM providers
including Anthropic (Claude), OpenAI, and local models.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.2
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self) -> None:
        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")


class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate n completions for the given prompt."""
        pass
    
    @abstractmethod
    def generate_with_system(
        self, 
        system: str, 
        prompt: str, 
        n: int = 1
    ) -> list[str]:
        """Generate with a system prompt."""
        pass


class AnthropicInterface(BaseLLMInterface):
    """Interface for Anthropic's Claude models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize()
    
    def _initialize(self) -> None:
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions using Claude."""
        responses = []
        
        for _ in range(n):
            try:
                message = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses.append(message.content[0].text)
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                responses.append("")
        
        return responses
    
    def generate_with_system(
        self, 
        system: str, 
        prompt: str, 
        n: int = 1
    ) -> list[str]:
        """Generate with system prompt using Claude."""
        responses = []
        
        for _ in range(n):
            try:
                message = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses.append(message.content[0].text)
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                responses.append("")
        
        return responses


class OpenAIInterface(BaseLLMInterface):
    """Interface for OpenAI models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize()
    
    def _initialize(self) -> None:
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                n=n,
                messages=[{"role": "user", "content": prompt}]
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return [""] * n
    
    def generate_with_system(
        self, 
        system: str, 
        prompt: str, 
        n: int = 1
    ) -> list[str]:
        """Generate with system prompt using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                n=n,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return [""] * n


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.call_count = 0
        self.last_prompt = ""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate mock responses."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Return a basic mock specification
        mock_response = '''```json
{
    "precondition": "in_basis(q0, |0⟩)",
    "postcondition": "superposition(q0)",
    "invariants": [],
    "reasoning": "Mock specification for testing"
}
```'''
        return [mock_response] * n
    
    def generate_with_system(
        self, 
        system: str, 
        prompt: str, 
        n: int = 1
    ) -> list[str]:
        """Generate mock responses with system prompt."""
        return self.generate(prompt, n)


def create_llm_interface(config: LLMConfig) -> BaseLLMInterface:
    """
    Factory function to create appropriate LLM interface.
    
    Args:
        config: LLM configuration
        
    Returns:
        Appropriate LLM interface instance
    """
    provider = config.provider.lower()
    
    if provider == "anthropic":
        return AnthropicInterface(config)
    elif provider == "openai":
        return OpenAIInterface(config)
    elif provider == "mock":
        return MockLLMInterface(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_default_interface() -> BaseLLMInterface:
    """Get default LLM interface based on available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return create_llm_interface(LLMConfig(provider="anthropic"))
    elif os.getenv("OPENAI_API_KEY"):
        return create_llm_interface(LLMConfig(provider="openai"))
    else:
        logger.warning("No API keys found. Using mock interface.")
        return MockLLMInterface()
