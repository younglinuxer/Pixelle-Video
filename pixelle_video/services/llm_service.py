# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM (Large Language Model) Service - Direct OpenAI SDK implementation

Supports structured output via response_type parameter (Pydantic model).
"""

import json
import re
from typing import Any, Optional, Type, TypeVar, Union

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when an LLM cannot produce a usable structured response."""

    def __init__(
        self,
        reason: str,
        finish_reason: Optional[str] = None,
        diagnostic: Optional[str] = None,
    ):
        self.reason = reason
        self.finish_reason = finish_reason
        self.diagnostic = diagnostic
        details = f" (finish_reason={finish_reason})" if finish_reason else ""
        messages = {
            "empty": "The LLM returned an empty structured response after one retry",
            "truncated": "The LLM response reached its token limit before JSON was complete",
            "invalid": "The LLM returned JSON that does not match the required structure",
        }
        super().__init__(f"{messages.get(reason, messages['invalid'])}{details}")


class LLMService:
    """
    LLM (Large Language Model) service
    
    Direct implementation using OpenAI SDK. No capability layer needed.
    
    Supports all OpenAI SDK compatible providers:
    - OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
    - Alibaba Qwen (qwen-max, qwen-plus, qwen-turbo)
    - Anthropic Claude (claude-sonnet-4-5, claude-opus-4, claude-haiku-4)
    - DeepSeek (deepseek-chat)
    - Moonshot Kimi (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k)
    - Ollama (llama3.2, qwen2.5, mistral, codellama) - FREE & LOCAL!
    - Any custom provider with OpenAI-compatible API
    
    Usage:
        # Direct call
        answer = await pixelle_video.llm("Explain atomic habits")
        
        # With parameters
        answer = await pixelle_video.llm(
            prompt="Explain atomic habits in 3 sentences",
            temperature=0.7,
            max_tokens=2000
        )
    """
    
    def __init__(self, config: dict):
        """
        Initialize LLM service
        
        Args:
            config: Full application config dict (kept for backward compatibility)
        """
        # Note: We no longer cache config here to support hot reload
        # Config is read dynamically from config_manager in _get_config_value()
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_config_value(self, key: str, default=None):
        """
        Get config value dynamically from config_manager (supports hot reload)
        
        Args:
            key: Config key name
            default: Default value if not found
        
        Returns:
            Config value
        """
        from pixelle_video.config import config_manager
        return getattr(config_manager.config.llm, key, default)
    
    def _create_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> AsyncOpenAI:
        """
        Create OpenAI client
        
        Args:
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
        
        Returns:
            AsyncOpenAI client instance
        """
        # Get API key (priority: parameter > config)
        final_api_key = (
            api_key
            or self._get_config_value("api_key")
            or "dummy-key"  # Ollama doesn't need real key
        )
        
        # Get base URL (priority: parameter > config)
        final_base_url = (
            base_url
            or self._get_config_value("base_url")
        )
        
        # Create client
        client_kwargs = {"api_key": final_api_key}
        if final_base_url:
            client_kwargs["base_url"] = final_base_url
        
        return AsyncOpenAI(**client_kwargs)
    
    async def __call__(
        self,
        prompt: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        response_type: Optional[Type[T]] = None,
        **kwargs
    ) -> Union[str, T]:
        """
        Generate text using LLM
        
        Args:
            prompt: The prompt to generate from
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            max_tokens: Maximum tokens to generate
            response_type: Optional Pydantic model class for structured output.
                          If provided, returns parsed model instance instead of string.
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text (str) or parsed Pydantic model instance (if response_type provided)
        
        Examples:
            # Basic text generation
            answer = await pixelle_video.llm("Explain atomic habits")
            
            # Structured output with Pydantic model
            class MovieReview(BaseModel):
                title: str
                rating: int
                summary: str
            
            review = await pixelle_video.llm(
                prompt="Review the movie Inception",
                response_type=MovieReview
            )
            print(review.title)  # Structured access
        """
        # Create client (new instance each time to support parameter overrides)
        client = self._create_client(api_key=api_key, base_url=base_url)
        
        # Get model (priority: parameter > config)
        final_model = (
            model
            or self._get_config_value("model")
            or "gpt-3.5-turbo"  # Default fallback
        )
        
        logger.debug(f"LLM call: model={final_model}, base_url={client.base_url}, response_type={response_type}")
        
        try:
            if response_type is not None:
                # Structured output mode - try beta.chat.completions.parse first
                return await self._call_with_structured_output(
                    client=client,
                    model=final_model,
                    prompt=prompt,
                    response_type=response_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                # Standard text output mode
                request_kwargs = dict(kwargs)
                self._apply_thinking_preference(client, request_kwargs)
                response = await client.chat.completions.create(
                    model=final_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **request_kwargs
                )
                
                raw_content = response.choices[0].message.content
                result = raw_content if isinstance(raw_content, str) else ""
                logger.debug(f"LLM response length: {len(result)} chars")
                if not result or not result.strip():
                    logger.warning(
                        f"LLM returned empty text content (model={final_model}, base_url={client.base_url})"
                    )
                
                return result
        
        except Exception as e:
            logger.error(f"LLM call error (model={final_model}, base_url={client.base_url}): {e}")
            raise
    
    async def _call_with_structured_output(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        response_type: Type[T],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> T:
        """
        Call LLM with structured output support
        
        Uses JSON schema instruction appended to prompt for maximum compatibility
        across all OpenAI-compatible providers (Qwen, DeepSeek, etc.).
        
        Args:
            client: OpenAI client
            model: Model name
            prompt: The prompt
            response_type: Pydantic model class
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters
        
        Returns:
            Parsed Pydantic model instance
        """
        # Build JSON schema instruction and append to prompt.
        json_schema_instruction = self._get_json_schema_instruction(response_type)
        enhanced_prompt = f"{prompt}\n\n{json_schema_instruction}"

        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("response_format", {"type": "json_object"})
        self._apply_thinking_preference(client, request_kwargs)

        last_finish_reason: Optional[str] = None
        last_reason = "invalid"
        last_diagnostic: Optional[str] = None
        invalid_content: Optional[str] = None
        for attempt in range(2):
            attempt_prompt = enhanced_prompt
            attempt_max_tokens = max_tokens
            if attempt:
                if invalid_content and not self._looks_like_json_schema(invalid_content):
                    attempt_prompt += (
                        "\n\nThe previous JSON object did not match the required schema. Repair it while "
                        "preserving its useful content. Return one complete JSON object with the exact "
                        "field names and value types from the schema. Do not include reasoning, prose, "
                        f"or Markdown fences. Previous JSON:\n{invalid_content}"
                    )
                else:
                    attempt_prompt += (
                        "\n\nThe previous response was empty, truncated, or returned a JSON Schema "
                        "definition instead of actual data. Return one complete data object now. "
                        "Do not include schema keywords, reasoning, prose, or Markdown fences."
                    )
                attempt_max_tokens = min(max(max_tokens * 2, 4096), 16384)

            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": attempt_prompt}],
                "temperature": temperature,
                "max_tokens": attempt_max_tokens,
                **request_kwargs,
            }
            try:
                response = await client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                if not self._json_mode_is_unsupported(exc, call_kwargs):
                    raise
                logger.warning(
                    "Provider rejected response_format; retrying with schema prompt only "
                    "(model={}, base_url={})",
                    model,
                    client.base_url,
                )
                request_kwargs.pop("response_format", None)
                call_kwargs.pop("response_format", None)
                response = await client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]
            last_finish_reason = getattr(choice, "finish_reason", None)
            content = self._message_content_as_text(choice.message)
            logger.debug(
                "Structured output attempt={} length={} finish_reason={}",
                attempt + 1,
                len(content),
                last_finish_reason,
            )

            if not content.strip():
                last_reason = "truncated" if last_finish_reason == "length" else "empty"
                logger.warning(
                    "LLM returned empty structured content (model={}, base_url={}, "
                    "finish_reason={}, retry={})",
                    model,
                    client.base_url,
                    last_finish_reason,
                    attempt == 0,
                )
                continue

            try:
                return self._parse_response_as_model(content, response_type)
            except StructuredOutputError as exc:
                last_reason = "truncated" if last_finish_reason == "length" else "invalid"
                last_diagnostic = exc.diagnostic
                invalid_content = content
                logger.warning(
                    "LLM returned unusable structured content (model={}, finish_reason={}, "
                    "retry={}, diagnostic={})",
                    model,
                    last_finish_reason,
                    attempt == 0,
                    exc.diagnostic or "unavailable",
                )

        raise StructuredOutputError(last_reason, last_finish_reason, last_diagnostic)

    @staticmethod
    def _is_deepseek(client: AsyncOpenAI) -> bool:
        """Return whether the current OpenAI-compatible endpoint is DeepSeek."""
        return "deepseek.com" in str(client.base_url).lower()

    def _apply_thinking_preference(
        self,
        client: AsyncOpenAI,
        request_kwargs: dict[str, Any],
    ) -> None:
        """Apply the configured DeepSeek reasoning mode without affecting providers."""
        if not self._is_deepseek(client):
            return
        enabled = bool(self._get_config_value("thinking_enabled", False))
        extra_body = dict(request_kwargs.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "enabled" if enabled else "disabled"})
        request_kwargs["extra_body"] = extra_body

    @staticmethod
    def _json_mode_is_unsupported(exc: Exception, call_kwargs: dict[str, Any]) -> bool:
        """Detect providers that reject the OpenAI ``response_format`` option."""
        if "response_format" not in call_kwargs:
            return False
        message = str(exc).lower()
        mentions_json_mode = "response_format" in message or "json mode" in message
        unsupported = any(
            marker in message
            for marker in ("unsupported", "unknown", "unrecognized", "not support", "extra input")
        )
        return mentions_json_mode and unsupported

    @staticmethod
    def _message_content_as_text(message: Any) -> str:
        """Normalize provider-specific message content into plain text."""
        raw_content = getattr(message, "content", None)
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(getattr(item, "text", None), str):
                    parts.append(item.text)
            return "".join(parts)
        return ""

    @staticmethod
    def _looks_like_json_schema(content: str) -> bool:
        """Detect when a model copied the requested schema instead of producing data."""
        try:
            payload = json.loads(content.strip())
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(payload, dict) and payload.get("type") == "object" and any(
            key in payload for key in ("properties", "required", "$defs")
        )
    
    def _get_json_schema_instruction(self, response_type: Type[T]) -> str:
        """
        Generate JSON schema instruction for LLM fallback mode
        
        Args:
            response_type: Pydantic model class
        
        Returns:
            Formatted instruction string with JSON schema
        """
        try:
            # Give providers a concrete data shape. Some OpenAI-compatible models
            # copy a full JSON Schema verbatim instead of filling it with data.
            schema = response_type.model_json_schema()
            example = self._schema_example(schema, schema)
            example_str = json.dumps(example, indent=2, ensure_ascii=False)
            
            return f"""## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text).
Fill in actual data using exactly this JSON shape and these field names:

```json
{example_str}
```

The shown strings and numbers are type placeholders, not literal output values.
Do NOT return a JSON Schema definition or keys such as `type`, `properties`, `$defs`, or `required`.
Output ONLY the completed JSON data object, nothing else."""
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema: {e}")
            return """## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text)."""

    @classmethod
    def _schema_example(cls, node: dict[str, Any], root: dict[str, Any]) -> Any:
        """Convert a Pydantic JSON Schema node into a compact, fillable data example."""
        if "$ref" in node:
            resolved: Any = root
            for part in node["$ref"].removeprefix("#/").split("/"):
                resolved = resolved[part]
            return cls._schema_example(resolved, root)
        if "anyOf" in node:
            non_null = [item for item in node["anyOf"] if item.get("type") != "null"]
            return cls._schema_example(non_null[0] if non_null else node["anyOf"][0], root)
        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            return {
                key: cls._schema_example(value, root)
                for key, value in node.get("properties", {}).items()
            }
        if node_type == "array":
            return [cls._schema_example(node.get("items", {}), root)]
        if node_type == "integer":
            return 1
        if node_type == "number":
            return 1.0
        if node_type == "boolean":
            return True
        return "string"
    
    def _parse_response_as_model(self, content: str, response_type: Type[T]) -> T:
        """
        Parse LLM response content as Pydantic model
        
        Args:
            content: Raw LLM response text
            response_type: Target Pydantic model class
        
        Returns:
            Parsed model instance
        """
        candidates = [content.strip()]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
        )

        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                _, end = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            candidates.append(content[index:index + end])

        seen: set[str] = set()
        diagnostics: list[str] = []
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                payload = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

            try:
                return response_type.model_validate(payload)
            except ValidationError as exc:
                if isinstance(payload, dict):
                    keys = sorted(str(key) for key in payload)[:20]
                    errors = [
                        f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                        for item in exc.errors(include_input=False)[:10]
                    ]
                    diagnostics.append(f"keys={keys}; errors={errors}")
                else:
                    diagnostics.append(
                        f"top_level={type(payload).__name__}; errors="
                        f"{[item['type'] for item in exc.errors(include_input=False)[:10]]}"
                    )

        # Do not echo model output here: prompts and responses may contain user data.
        diagnostic = diagnostics[-1] if diagnostics else "no-decodable-json-object"
        raise StructuredOutputError("invalid", diagnostic=diagnostic)
    
    @property
    def active(self) -> str:
        """
        Get active model name
        
        Returns:
            Active model name
        
        Example:
            print(f"Using model: {pixelle_video.llm.active}")
        """
        return self._get_config_value("model", "gpt-3.5-turbo")
    
    def __repr__(self) -> str:
        """String representation"""
        model = self.active
        base_url = self._get_config_value("base_url", "default")
        return f"<LLMService model={model!r} base_url={base_url!r}>"
