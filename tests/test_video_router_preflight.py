from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from api.routers import video
from api.schemas.video import VideoGenerateRequest
from pixelle_video.services.preflight import PreflightReport


async def test_async_generation_rejects_preflight_before_task_creation(monkeypatch):
    class FailedPreflight:
        async def check_standard(self, text, params):
            report = PreflightReport()
            report.add_error("comfyui.unreachable", "ComfyUI is unreachable.")
            return report

    core = SimpleNamespace(preflight=FailedPreflight())
    request_body = VideoGenerateRequest(
        text="test",
        mode="fixed",
        title="test",
        frame_template="1080x1920/static_default.html",
    )
    request = Request({"type": "http", "scheme": "http", "server": ("test", 80)})

    def fail_if_task_is_created(*args, **kwargs):
        raise AssertionError("task must not be created before preflight passes")

    monkeypatch.setattr(video.task_manager, "create_task", fail_if_task_is_created)

    with pytest.raises(HTTPException) as exc_info:
        await video.generate_video_async(request_body, core, request)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["errors"][0]["code"] == "comfyui.unreachable"
