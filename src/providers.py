import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from openai import AsyncOpenAI
import google.generativeai as genai

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class ModelInfo:
    name: str
    provider: ProviderType
    description: str = ""
    supports_vision: bool = False
    supports_image_generation: bool = False


class BaseProvider(ABC):
    """Base class for configured AI providers."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate chat completion."""

    @abstractmethod
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate image from prompt."""

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""

    @abstractmethod
    def supports_image_generation(self) -> bool:
        """Check if provider supports image generation."""


class OpenAIProvider(BaseProvider):
    """Official OpenAI API provider."""

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        self.client = AsyncOpenAI(api_key=api_key)

    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        response = await self.client.images.generate(
            model=model or "dall-e-3",
            prompt=prompt,
            size=kwargs.get("size", "1024x1024"),
            quality=kwargs.get("quality", "standard"),
            n=1,
        )
        return response.data[0].url or ""

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("gpt-4o", ProviderType.OPENAI, "Most capable GPT-4o model", supports_vision=True),
            ModelInfo("gpt-4o-mini", ProviderType.OPENAI, "Affordable GPT-4o model", supports_vision=True),
            ModelInfo("dall-e-3", ProviderType.OPENAI, "DALL-E 3 image generation", supports_image_generation=True),
        ]

    def supports_image_generation(self) -> bool:
        return True


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible provider for vLLM or other /v1 chat endpoints."""

    def __init__(self, api_key: Optional[str], base_url: str):
        super().__init__(api_key=api_key, base_url=base_url)
        self.client = AsyncOpenAI(api_key=api_key or "local-dev-key", base_url=base_url)

    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model or os.getenv("LLM_MODEL", "vllm-local"),
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("OpenAI-compatible chat endpoints do not guarantee image generation.")

    def get_available_models(self) -> List[ModelInfo]:
        model = os.getenv("LLM_MODEL", "vllm-local")
        return [ModelInfo(model, ProviderType.OPENAI_COMPATIBLE, "Configured OpenAI-compatible model")]

    def supports_image_generation(self) -> bool:
        return False


class GeminiProvider(BaseProvider):
    """Official Google Gemini API provider."""

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        genai.configure(api_key=api_key)

    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        gemini_model = genai.GenerativeModel(model or os.getenv("LLM_MODEL", "gemini-1.5-flash"))
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return getattr(response, "text", "")

    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("Gemini image generation is not wired through the legacy provider manager.")

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("gemini-1.5-flash", ProviderType.GEMINI, "Fast Gemini model", supports_vision=True),
            ModelInfo("gemini-1.5-pro", ProviderType.GEMINI, "Advanced Gemini model", supports_vision=True),
        ]

    def supports_image_generation(self) -> bool:
        return False


class ProviderManager:
    """Manages the single configured legacy AI provider."""

    def __init__(self):
        self.providers: Dict[ProviderType, BaseProvider] = {}
        self.current_provider: Optional[ProviderType] = None
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        provider_name = os.getenv("LLM_PROVIDER")
        if not provider_name:
            raise RuntimeError("No LLM provider configured. Set LLM_PROVIDER=openai|gemini|openai_compatible.")

        try:
            provider_type = ProviderType(provider_name)
        except ValueError as exc:
            raise RuntimeError("No LLM provider configured. Set LLM_PROVIDER=openai|gemini|openai_compatible.") from exc

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if provider_type == ProviderType.OPENAI:
            key = api_key or os.getenv("OPENAI_KEY")
            if not key:
                raise RuntimeError("No LLM provider configured. Set LLM_API_KEY for LLM_PROVIDER=openai.")
            self.providers[provider_type] = OpenAIProvider(key)
        elif provider_type == ProviderType.GEMINI:
            key = api_key or os.getenv("GEMINI_KEY")
            if not key:
                raise RuntimeError("No LLM provider configured. Set LLM_API_KEY for LLM_PROVIDER=gemini.")
            self.providers[provider_type] = GeminiProvider(key)
        elif provider_type == ProviderType.OPENAI_COMPATIBLE:
            if not base_url:
                raise RuntimeError("No LLM provider configured. Set LLM_BASE_URL for openai_compatible.")
            self.providers[provider_type] = OpenAICompatibleProvider(api_key, base_url)

        self.current_provider = provider_type
        logger.info("Initialized %s provider", provider_type.value)

    def get_provider(self, provider_type: Optional[ProviderType] = None) -> BaseProvider:
        selected = provider_type or self.current_provider
        if selected not in self.providers:
            raise ValueError(f"Provider {selected.value if selected else selected} not available")
        return self.providers[selected]

    def set_current_provider(self, provider_type: ProviderType) -> None:
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type.value} not available")
        self.current_provider = provider_type

    def get_available_providers(self) -> List[ProviderType]:
        return list(self.providers.keys())

    def get_all_models(self) -> Dict[ProviderType, List[ModelInfo]]:
        return {
            provider_type: provider.get_available_models()
            for provider_type, provider in self.providers.items()
        }

    def get_provider_models(self, provider_type: ProviderType) -> List[ModelInfo]:
        if provider_type not in self.providers:
            return []
        return self.providers[provider_type].get_available_models()
