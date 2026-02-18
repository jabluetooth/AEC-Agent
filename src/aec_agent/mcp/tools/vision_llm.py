"""
Vision LLM Integration for Symbol Classification.

This module provides vision-capable LLM integration for detailed
symbol classification in AEC drawings. It supports multiple providers:

1. Google Gemini Pro Vision (recommended for cost)
2. OpenAI GPT-4o / GPT-4V
3. Anthropic Claude 3.5 Sonnet (vision)

Part of Phase C: Symbol Intelligence - enhances YOLOv8 detection
with detailed subtype classification using vision LLMs.

Usage:
    >>> classifier = VisionLLMClassifier()
    >>> result = await classifier.classify_symbol(image, "valve", "mechanical")
    >>> print(result.subtype)  # "gate_valve"
"""

import base64
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from aec_agent.config.settings import LLMProvider, get_settings

logger = structlog.get_logger(__name__)


class VisionProvider(str, Enum):
    """Supported vision LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AUTO = "auto"  # Auto-select based on available API keys


@dataclass
class VisionClassificationResult:
    """Result from Vision LLM symbol classification."""
    category: str
    type: str
    subtype: str
    direction: str = "none"
    size_hint: str | None = None
    system: str | None = None
    confidence: float = 0.0
    visual_features: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    error: str | None = None


class VisionLLMClassifier:
    """
    Vision LLM classifier for detailed symbol identification.

    Supports multiple vision LLM providers with automatic fallback.
    Optimized for AEC symbol classification with domain-specific prompts.

    Args:
        provider: Vision LLM provider to use (gemini, openai, anthropic, auto).
        fallback_providers: List of fallback providers if primary fails.
        temperature: LLM temperature for responses (lower = more deterministic).
        max_tokens: Maximum tokens in response.

    Example:
        >>> classifier = VisionLLMClassifier(provider="gemini")
        >>> result = await classifier.classify_symbol(
        ...     image=symbol_crop,
        ...     yolo_class="valve",
        ...     yolo_category="mechanical",
        ... )
        >>> print(f"{result.subtype} (confidence: {result.confidence:.2f})")
    """

    def __init__(
        self,
        provider: VisionProvider | str = VisionProvider.AUTO,
        fallback_providers: list[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ):
        self.settings = get_settings()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = VisionProvider(provider.lower())

        self.provider = provider
        self.fallback_providers = fallback_providers or ["gemini", "openai", "anthropic"]

        # Lazy-loaded clients
        self._gemini_model = None
        self._openai_client = None
        self._anthropic_client = None

    def _get_available_provider(self) -> VisionProvider | None:
        """Determine which provider is available based on API keys."""
        if self.provider != VisionProvider.AUTO:
            return self.provider

        # Check providers in order of preference (cost-effective first)
        if self.settings.gemini_api_key:
            return VisionProvider.GEMINI
        if self.settings.openai_api_key:
            return VisionProvider.OPENAI
        if self.settings.anthropic_api_key:
            return VisionProvider.ANTHROPIC

        return None

    def _get_gemini_client(self):
        """Lazy-load Gemini client (new SDK)."""
        if self._gemini_model is None:
            try:
                from google import genai
                self._gemini_model = genai.Client(api_key=self.settings.gemini_api_key)
                logger.debug("Gemini client initialized")
            except ImportError:
                logger.warning(
                    "google-genai not installed. "
                    "Install with: pip install google-genai"
                )
                raise
        return self._gemini_model

    async def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                logger.debug("OpenAI client initialized")
            except ImportError:
                logger.warning(
                    "openai not installed. Install with: pip install openai"
                )
                raise
        return self._openai_client

    async def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic

                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
                logger.debug("Anthropic client initialized")
            except ImportError:
                logger.warning(
                    "anthropic not installed. Install with: pip install anthropic"
                )
                raise
        return self._anthropic_client

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        try:
            import cv2
            from PIL import Image
        except ImportError:
            raise ImportError("OpenCV and Pillow required for image encoding")

        # Ensure image is in BGR format for encoding
        if len(image.shape) == 2:
            # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image and encode as PNG
        pil_image = Image.fromarray(image_rgb)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def _image_to_pil(self, image: np.ndarray):
        """Convert numpy image to PIL Image."""
        try:
            import cv2
            from PIL import Image
        except ImportError:
            raise ImportError("OpenCV and Pillow required for image encoding")

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        return Image.fromarray(image_rgb)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code block markers if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response", error=str(e), text=text[:200])
            return {}

    async def classify_symbol(
        self,
        image: np.ndarray,
        yolo_class: str,
        yolo_category: str,
        prompt: str,
        provider: VisionProvider | str | None = None,
    ) -> VisionClassificationResult:
        """
        Classify a symbol using Vision LLM.

        Args:
            image: Cropped symbol image (numpy array).
            yolo_class: Initial YOLO classification.
            yolo_category: Category from YOLO.
            prompt: Classification prompt (from symbol_classification.py).
            provider: Optional override for provider.

        Returns:
            VisionClassificationResult with detailed classification.
        """
        # Determine provider
        use_provider = provider
        if use_provider is None:
            use_provider = self._get_available_provider()
        elif isinstance(use_provider, str):
            use_provider = VisionProvider(use_provider.lower())

        if use_provider is None:
            return VisionClassificationResult(
                category=yolo_category,
                type=yolo_class,
                subtype=yolo_class,
                confidence=0.0,
                error="No vision LLM provider available (no API keys configured)",
            )

        # Try primary provider, then fallbacks
        providers_to_try = [use_provider]
        for fb in self.fallback_providers:
            fb_provider = VisionProvider(fb.lower())
            if fb_provider not in providers_to_try:
                providers_to_try.append(fb_provider)

        last_error = None
        for provider in providers_to_try:
            try:
                if provider == VisionProvider.GEMINI:
                    return await self._classify_with_gemini(image, prompt, yolo_class, yolo_category)
                elif provider == VisionProvider.OPENAI:
                    return await self._classify_with_openai(image, prompt, yolo_class, yolo_category)
                elif provider == VisionProvider.ANTHROPIC:
                    return await self._classify_with_anthropic(image, prompt, yolo_class, yolo_category)
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Vision LLM classification failed with {provider.value}",
                    error=last_error,
                )
                continue

        # All providers failed
        return VisionClassificationResult(
            category=yolo_category,
            type=yolo_class,
            subtype=yolo_class,
            confidence=0.0,
            error=f"All providers failed. Last error: {last_error}",
        )

    async def _classify_with_gemini(
        self,
        image: np.ndarray,
        prompt: str,
        yolo_class: str,
        yolo_category: str,
    ) -> VisionClassificationResult:
        """Classify using Google Gemini (new SDK)."""
        from google.genai import types
        import io

        client = self._get_gemini_client()
        pil_image = self._image_to_pil(image)

        # Convert PIL image to bytes for new SDK
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        image_bytes = buf.getvalue()

        # Build content with new SDK types
        contents = [types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type='image/png')
            ]
        )]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        )

        # Parse response
        text = response.text
        data = self._parse_json_response(text)

        if not data:
            return VisionClassificationResult(
                category=yolo_category,
                type=yolo_class,
                subtype=yolo_class,
                confidence=0.0,
                raw_response={"text": text},
                provider="gemini",
                error="Failed to parse JSON response",
            )

        return VisionClassificationResult(
            category=data.get("category", yolo_category),
            type=data.get("type", yolo_class),
            subtype=data.get("subtype", yolo_class),
            direction=data.get("direction", "none"),
            size_hint=data.get("size_hint"),
            system=data.get("system"),
            confidence=float(data.get("confidence", 0.5)),
            visual_features=data.get("visual_features", []),
            reasoning=data.get("reasoning", ""),
            raw_response=data,
            provider="gemini",
        )

    async def _classify_with_openai(
        self,
        image: np.ndarray,
        prompt: str,
        yolo_class: str,
        yolo_category: str,
    ) -> VisionClassificationResult:
        """Classify using OpenAI GPT-4o."""
        client = await self._get_openai_client()
        base64_image = self._image_to_base64(image)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        text = response.choices[0].message.content
        data = self._parse_json_response(text)

        if not data:
            return VisionClassificationResult(
                category=yolo_category,
                type=yolo_class,
                subtype=yolo_class,
                confidence=0.0,
                raw_response={"text": text},
                provider="openai",
                error="Failed to parse JSON response",
            )

        return VisionClassificationResult(
            category=data.get("category", yolo_category),
            type=data.get("type", yolo_class),
            subtype=data.get("subtype", yolo_class),
            direction=data.get("direction", "none"),
            size_hint=data.get("size_hint"),
            system=data.get("system"),
            confidence=float(data.get("confidence", 0.5)),
            visual_features=data.get("visual_features", []),
            reasoning=data.get("reasoning", ""),
            raw_response=data,
            provider="openai",
        )

    async def _classify_with_anthropic(
        self,
        image: np.ndarray,
        prompt: str,
        yolo_class: str,
        yolo_category: str,
    ) -> VisionClassificationResult:
        """Classify using Anthropic Claude."""
        client = await self._get_anthropic_client()
        base64_image = self._image_to_base64(image)

        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        text = message.content[0].text
        data = self._parse_json_response(text)

        if not data:
            return VisionClassificationResult(
                category=yolo_category,
                type=yolo_class,
                subtype=yolo_class,
                confidence=0.0,
                raw_response={"text": text},
                provider="anthropic",
                error="Failed to parse JSON response",
            )

        return VisionClassificationResult(
            category=data.get("category", yolo_category),
            type=data.get("type", yolo_class),
            subtype=data.get("subtype", yolo_class),
            direction=data.get("direction", "none"),
            size_hint=data.get("size_hint"),
            system=data.get("system"),
            confidence=float(data.get("confidence", 0.5)),
            visual_features=data.get("visual_features", []),
            reasoning=data.get("reasoning", ""),
            raw_response=data,
            provider="anthropic",
        )

    async def classify_batch(
        self,
        images: list[np.ndarray],
        yolo_classes: list[str],
        yolo_categories: list[str],
        prompts: list[str],
    ) -> list[VisionClassificationResult]:
        """
        Classify multiple symbols in parallel.

        Args:
            images: List of cropped symbol images.
            yolo_classes: List of initial YOLO classifications.
            yolo_categories: List of categories from YOLO.
            prompts: List of classification prompts.

        Returns:
            List of VisionClassificationResult.
        """
        import asyncio

        tasks = [
            self.classify_symbol(img, cls, cat, prompt)
            for img, cls, cat, prompt in zip(
                images, yolo_classes, yolo_categories, prompts
            )
        ]

        return await asyncio.gather(*tasks)

    async def identify_unknown_symbol(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> VisionClassificationResult:
        """
        Identify a completely unknown symbol.

        Used when YOLO doesn't recognize the symbol at all.

        Args:
            image: Symbol image.
            prompt: Unknown symbol prompt.

        Returns:
            VisionClassificationResult.
        """
        return await self.classify_symbol(
            image=image,
            yolo_class="unknown",
            yolo_category="unknown",
            prompt=prompt,
        )


# Global classifier instance (lazy initialization)
_classifier: VisionLLMClassifier | None = None


def get_vision_classifier(
    provider: str | None = None,
    **kwargs: Any,
) -> VisionLLMClassifier:
    """
    Get or create the global Vision LLM classifier instance.

    Args:
        provider: Vision provider to use (gemini, openai, anthropic, auto).
        **kwargs: Additional arguments for VisionLLMClassifier.

    Returns:
        VisionLLMClassifier instance.
    """
    global _classifier

    if _classifier is None:
        _classifier = VisionLLMClassifier(
            provider=provider or VisionProvider.AUTO,
            **kwargs,
        )

    return _classifier


def is_vision_llm_available() -> bool:
    """Check if any Vision LLM provider is available."""
    settings = get_settings()
    return bool(
        settings.gemini_api_key
        or settings.openai_api_key
        or settings.anthropic_api_key
    )
