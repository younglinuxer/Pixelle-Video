from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from pixelle_video.services.llm_service import LLMService, StructuredOutputError
from pixelle_video.services.project_service import OptimizedBrief, StoryboardPlan
from pixelle_video.utils.content_generators import (
    NarrationResponse,
    generate_narrations_from_topic,
)


class ExampleResponse(BaseModel):
    title: str
    items: list[int]


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, base_url: str, responses):
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def completion(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


@pytest.mark.asyncio
async def test_deepseek_structured_output_disables_thinking_and_retries_truncation(monkeypatch):
    client = FakeClient(
        "https://api.deepseek.com",
        [
            completion("", finish_reason="length"),
            completion('{"title":"ok","items":[1,2]}'),
        ],
    )
    service = LLMService({})
    monkeypatch.setattr(
        service,
        "_get_config_value",
        lambda key, default=None: False if key == "thinking_enabled" else default,
    )

    result = await service._call_with_structured_output(
        client=client,
        model="deepseek-v4-pro",
        prompt="Return the plan",
        response_type=ExampleResponse,
        temperature=0.4,
        max_tokens=2000,
    )

    assert result.title == "ok"
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert client.chat.completions.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert client.chat.completions.calls[1]["max_tokens"] == 4096


def test_deepseek_thinking_can_be_enabled_from_configuration(monkeypatch):
    service = LLMService({})
    monkeypatch.setattr(
        service,
        "_get_config_value",
        lambda key, default=None: True if key == "thinking_enabled" else default,
    )
    client = FakeClient("https://api.deepseek.com", [])
    request_kwargs = {}

    service._apply_thinking_preference(client, request_kwargs)

    assert request_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_thinking_preference_is_not_sent_to_other_providers(monkeypatch):
    service = LLMService({})
    monkeypatch.setattr(service, "_get_config_value", lambda key, default=None: True)
    client = FakeClient("https://api.openai.com/v1", [])
    request_kwargs = {}

    service._apply_thinking_preference(client, request_kwargs)

    assert request_kwargs == {}


@pytest.mark.asyncio
async def test_structured_retry_repairs_the_previous_invalid_json():
    invalid = '{"title":"draft","items":[{"value":1}]}'
    client = FakeClient(
        "https://api.deepseek.com",
        [completion(invalid), completion('{"title":"fixed","items":[1]}')],
    )
    service = LLMService({})

    result = await service._call_with_structured_output(
        client=client,
        model="deepseek-v4-pro",
        prompt="Return the plan",
        response_type=ExampleResponse,
        temperature=0.4,
        max_tokens=8192,
    )

    retry_prompt = client.chat.completions.calls[1]["messages"][0]["content"]
    assert result.title == "fixed"
    assert invalid in retry_prompt
    assert "Repair it" in retry_prompt


@pytest.mark.asyncio
async def test_structured_retry_does_not_repeat_a_schema_shaped_response():
    copied_schema = '{"type":"object","properties":{"title":{"type":"string"}}}'
    client = FakeClient(
        "https://api.deepseek.com",
        [completion(copied_schema), completion('{"title":"fixed","items":[1]}')],
    )
    service = LLMService({})

    result = await service._call_with_structured_output(
        client=client,
        model="deepseek-v4-pro",
        prompt="Return the plan",
        response_type=ExampleResponse,
        temperature=0.4,
        max_tokens=8192,
    )

    retry_prompt = client.chat.completions.calls[1]["messages"][0]["content"]
    assert result.title == "fixed"
    assert copied_schema not in retry_prompt
    assert "schema keywords" in retry_prompt


def test_schema_instruction_uses_a_fillable_shape_not_json_schema():
    instruction = LLMService({})._get_json_schema_instruction(StoryboardPlan)

    assert '"title": "string"' in instruction
    assert '"scenes": [' in instruction
    assert '"narration": "string"' in instruction
    assert '"$defs":' not in instruction


def test_structured_parser_accepts_json_inside_markdown_and_extra_text():
    service = LLMService({})

    result = service._parse_response_as_model(
        "Here is the result:\n```json\n{\"title\":\"ok\",\"items\":[3]}\n```\nDone.",
        ExampleResponse,
    )

    assert result == ExampleResponse(title="ok", items=[3])


def test_structured_parser_does_not_echo_invalid_provider_content():
    service = LLMService({})
    sensitive_text = "not json and must not appear in the exception"

    with pytest.raises(StructuredOutputError) as exc_info:
        service._parse_response_as_model(sensitive_text, ExampleResponse)

    assert exc_info.value.reason == "invalid"
    assert sensitive_text not in str(exc_info.value)
    assert exc_info.value.diagnostic == "no-decodable-json-object"


def test_structured_parser_reports_safe_schema_diagnostics_only():
    content = '{"secret_field":"do-not-log-this-value"}'

    with pytest.raises(StructuredOutputError) as exc_info:
        LLMService({})._parse_response_as_model(content, ExampleResponse)

    assert "secret_field" in exc_info.value.diagnostic
    assert "do-not-log-this-value" not in exc_info.value.diagnostic
    assert "title:missing" in exc_info.value.diagnostic


def test_optimized_brief_normalizes_provider_suggestion_objects():
    result = OptimizedBrief.model_validate(
        {
            "optimizedPrompt": "A sharper short-video brief",
            "suggestions": [
                {"area": "hook", "change": "Lead with the risk"},
                "Tighten the ending",
            ],
        }
    )

    assert result.optimized_prompt == "A sharper short-video brief"
    assert result.improvements == ["hook：Lead with the risk", "Tighten the ending"]


def test_storyboard_plan_normalizes_common_provider_aliases():
    result = StoryboardPlan.model_validate(
        {
            "video_title": "Security explainer",
            "storyboard": [
                {
                    "voiceover": "Start with the problem.",
                    "image_prompt": "A secure office entrance",
                    "duration": "4 seconds",
                    "media_type": "video",
                }
            ],
        }
    )

    assert result.title == "Security explainer"
    assert result.scenes[0].estimated_duration == 4
    assert result.scenes[0].media_mode == "video"


@pytest.mark.asyncio
async def test_provider_without_json_mode_falls_back_to_schema_prompt():
    client = FakeClient(
        "http://local-provider.test/v1",
        [
            ValueError("response_format is unsupported"),
            completion('{"title":"fallback","items":[9]}'),
        ],
    )
    service = LLMService({})

    result = await service._call_with_structured_output(
        client=client,
        model="local-model",
        prompt="Return the plan",
        response_type=ExampleResponse,
        temperature=0.4,
        max_tokens=8192,
    )

    assert result.title == "fallback"
    assert "response_format" in client.chat.completions.calls[0]
    assert "response_format" not in client.chat.completions.calls[1]


@pytest.mark.asyncio
async def test_legacy_topic_planner_no_longer_uses_2000_token_json_path():
    captured = {}

    async def fake_llm(*args, **kwargs):
        captured.update(kwargs)
        return NarrationResponse(narrations=["one", "two"])

    narrations = await generate_narrations_from_topic(fake_llm, "topic", n_scenes=2)

    assert narrations == ["one", "two"]
    assert captured["max_tokens"] == 8192
    assert captured["response_type"] is NarrationResponse
