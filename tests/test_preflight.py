import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from pixelle_video.config.schema import PixelleVideoConfig
from pixelle_video.pipelines.standard import StandardPipeline
from pixelle_video.services.preflight import (
    GenerationPreflightError,
    PreflightReport,
    PreflightService,
)


def _config() -> PixelleVideoConfig:
    return PixelleVideoConfig(
        llm={
            "api_key": "test-key",
            "base_url": "https://llm.test/v1",
            "model": "model-ok",
        },
        comfyui={
            "comfyui_url": "https://comfy.test",
            "comfyui_api_key": "comfy-key",
        },
    )


def _write_workflow(root: Path, node_type: str = "KnownNode") -> None:
    path = root / "workflows" / "selfhost" / "image_test.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "1": {
                    "inputs": {"value": "test"},
                    "class_type": node_type,
                    "_meta": {"title": "$prompt.value!"},
                }
            }
        ),
        encoding="utf-8",
    )


async def test_preflight_accepts_reachable_llm_and_compatible_selfhost_workflow(
    tmp_path: Path,
):
    _write_workflow(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "llm.test":
            assert request.headers["Authorization"] == "Bearer test-key"
            return httpx.Response(200, json={"data": [{"id": "model-ok"}]})
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.url.path == "/object_info":
            assert request.headers["Authorization"] == "Bearer comfy-key"
            return httpx.Response(200, json={"KnownNode": {}})
        raise AssertionError(f"Unexpected request: {request.url}")

    service = PreflightService(
        config=_config(),
        project_root=tmp_path,
        http_transport=httpx.MockTransport(handler),
    )
    report = await service.check_requirements(
        require_llm=True,
        workflows=["selfhost/image_test.json"],
    )

    assert report.ok
    assert "llm" in report.checks
    assert "workflow:selfhost/image_test.json" in report.checks


async def test_preflight_blocks_missing_comfyui_nodes(tmp_path: Path):
    _write_workflow(tmp_path, node_type="MissingNode")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.url.path == "/object_info":
            return httpx.Response(200, json={"KnownNode": {}})
        raise AssertionError(f"Unexpected request: {request.url}")

    service = PreflightService(
        config=_config(),
        project_root=tmp_path,
        http_transport=httpx.MockTransport(handler),
    )
    report = await service.check_requirements(workflows=["selfhost/image_test.json"])

    assert not report.ok
    assert [issue.code for issue in report.errors] == ["comfyui.nodes_missing"]


async def test_preflight_blocks_runninghub_without_credentials(tmp_path: Path):
    wrapper = tmp_path / "workflows" / "runninghub" / "video_test.json"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        json.dumps({"source": "runninghub", "workflow_id": "123456"}),
        encoding="utf-8",
    )

    service = PreflightService(config=_config(), project_root=tmp_path)
    report = await service.check_requirements(workflows=["runninghub/video_test.json"])

    assert not report.ok
    assert [issue.code for issue in report.errors] == ["runninghub.key_missing"]


async def test_preflight_fetches_runninghub_graph_without_creating_task(tmp_path: Path):
    wrapper = tmp_path / "workflows" / "runninghub" / "video_test.json"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        json.dumps({"source": "runninghub", "workflow_id": "123456"}),
        encoding="utf-8",
    )
    config = _config()
    config.comfyui.runninghub_api_key = "runninghub-key"
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert request.method == "POST"
        assert json.loads(request.content) == {
            "apiKey": "runninghub-key",
            "workflowId": "123456",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "prompt": json.dumps(
                        {"1": {"class_type": "KnownNode", "inputs": {}}}
                    )
                },
            },
        )

    service = PreflightService(
        config=config,
        project_root=tmp_path,
        http_transport=httpx.MockTransport(handler),
    )
    report = await service.check_requirements(workflows=["runninghub/video_test.json"])

    assert report.ok
    assert requested_paths == ["/api/openapi/getJsonApiFormat"]


async def test_standard_pipeline_stops_before_setup_when_preflight_fails(monkeypatch):
    class FailedPreflight:
        async def check_standard(self, text, params):
            report = PreflightReport()
            report.add_error("runtime.ffmpeg_missing", "ffmpeg is missing.")
            return report

    core = SimpleNamespace(
        preflight=FailedPreflight(),
        llm=None,
        tts=None,
        media=None,
        video=None,
    )
    pipeline = StandardPipeline(core)
    setup_called = False

    async def fail_if_setup_runs(context):
        nonlocal setup_called
        setup_called = True

    monkeypatch.setattr(pipeline, "setup_environment", fail_if_setup_runs)

    with pytest.raises(GenerationPreflightError):
        await pipeline(text="test", mode="fixed", title="test")

    assert setup_called is False
