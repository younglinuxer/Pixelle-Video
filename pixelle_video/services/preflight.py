# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Read-only readiness checks that run before generation starts."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from pixelle_video.config import config_manager
from pixelle_video.config.schema import PixelleVideoConfig
from pixelle_video.utils.llm_util import _build_models_url
from pixelle_video.utils.template_util import get_template_type, resolve_template_path


@dataclass(frozen=True)
class PreflightIssue:
    """A single actionable preflight finding."""

    code: str
    message: str


@dataclass
class PreflightReport:
    """Aggregated readiness result with errors separated from advisory warnings."""

    checks: list[str] = field(default_factory=list)
    errors: list[PreflightIssue] = field(default_factory=list)
    warnings: list[PreflightIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_check(self, check: str) -> None:
        if check not in self.checks:
            self.checks.append(check)

    def add_error(self, code: str, message: str) -> None:
        issue = PreflightIssue(code=code, message=message)
        if issue not in self.errors:
            self.errors.append(issue)

    def add_warning(self, code: str, message: str) -> None:
        issue = PreflightIssue(code=code, message=message)
        if issue not in self.warnings:
            self.warnings.append(issue)

    def merge(self, other: "PreflightReport") -> None:
        for check in other.checks:
            self.add_check(check)
        for issue in other.errors:
            self.add_error(issue.code, issue.message)
        for issue in other.warnings:
            self.add_warning(issue.code, issue.message)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise GenerationPreflightError(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": list(self.checks),
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


class GenerationPreflightError(RuntimeError):
    """Raised before any generation work when required capabilities are unavailable."""

    def __init__(self, report: PreflightReport):
        self.report = report
        detail = "; ".join(issue.message for issue in report.errors)
        super().__init__(f"Generation preflight failed: {detail}")


class PreflightService:
    """Validate local/runtime/provider dependencies without submitting generation jobs."""

    CACHE_TTL_SECONDS = 30.0

    def __init__(
        self,
        core: Any = None,
        *,
        config: Optional[PixelleVideoConfig] = None,
        project_root: Optional[Path] = None,
        http_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.core = core
        self._config_override = config
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.http_transport = http_transport
        self._success_cache: dict[str, float] = {}
        self._comfyui_nodes_cache: dict[str, tuple[float, set[str]]] = {}

    @property
    def config(self) -> PixelleVideoConfig:
        return self._config_override or config_manager.config

    async def check_standard(self, text: str, params: dict[str, Any]) -> PreflightReport:
        """Check every dependency selected by the standard pipeline."""
        report = PreflightReport()
        if not text or not text.strip():
            report.add_error("input.empty", "Video source text is empty.")

        frame_template = (
            params.get("frame_template")
            or self.config.template.default_template
            or "1080x1920/image_default.html"
        )
        template_type = get_template_type(Path(frame_template).name)
        mode = params.get("mode", "generate")
        title = params.get("title")
        require_llm = mode == "generate" or not title or template_type in {"image", "video"}

        workflows: list[str] = []
        tts_mode = params.get("tts_inference_mode") or "local"
        if tts_mode == "comfyui":
            tts_workflow = (
                params.get("tts_workflow")
                or self.config.comfyui.tts.comfyui.default_workflow
            )
            if tts_workflow:
                workflows.append(tts_workflow)
            else:
                report.add_error(
                    "tts.workflow_missing",
                    "ComfyUI TTS mode requires a configured TTS workflow.",
                )

        if template_type in {"image", "video"}:
            configured_default = (
                self.config.comfyui.video.default_workflow
                if template_type == "video"
                else self.config.comfyui.image.default_workflow
            )
            media_workflow = params.get("media_workflow") or configured_default
            if media_workflow:
                workflows.append(media_workflow)
            else:
                report.add_error(
                    "media.workflow_missing",
                    f"The {template_type} template requires a {template_type} workflow or API model.",
                )

            api_video_params = params.get("api_video_params") or {}
            first_frame_workflow = api_video_params.get("first_frame_workflow")
            if first_frame_workflow:
                workflows.append(first_frame_workflow)

        ref_audio = params.get("ref_audio")
        input_files = [ref_audio] if ref_audio else []
        requirements = await self.check_requirements(
            require_llm=require_llm,
            require_local_tts=tts_mode == "local",
            workflows=workflows,
            frame_template=frame_template,
            require_browser=True,
            require_ffmpeg=True,
            input_files=input_files,
            bgm_path=params.get("bgm_path"),
            tts_voice=params.get("tts_voice") or params.get("voice_id"),
            tts_speed=params.get("tts_speed"),
        )
        report.merge(requirements)
        return report

    async def check_asset_based(self, params: dict[str, Any]) -> PreflightReport:
        """Check uploaded-asset pipeline inputs and selected providers."""
        assets = [str(path) for path in params.get("assets", []) if path]
        workflows: list[str] = []
        source = params.get("source", "runninghub")
        api_vlm_model = None

        if source == "api":
            api_vlm_model = params.get("analysis_vlm_model")
            if not api_vlm_model:
                report = PreflightReport()
                report.add_error(
                    "vlm.model_missing",
                    "API asset analysis requires an explicitly selected VLM model.",
                )
            else:
                report = PreflightReport()
        else:
            report = PreflightReport()
            asset_suffixes = {Path(path).suffix.lower() for path in assets}
            if asset_suffixes & {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                image_workflow = params.get("analysis_image_workflow")
                if image_workflow:
                    workflows.append(image_workflow)
                else:
                    report.add_error(
                        "analysis.image_workflow_missing",
                        "Image assets require an image-analysis workflow.",
                    )
            if asset_suffixes & {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                video_workflow = params.get("analysis_video_workflow")
                if video_workflow:
                    workflows.append(video_workflow)
                else:
                    report.add_error(
                        "analysis.video_workflow_missing",
                        "Video assets require a video-analysis workflow.",
                    )

        api_video_workflow = params.get("api_video_workflow")
        if api_video_workflow:
            workflows.append(api_video_workflow)

        requirements = await self.check_requirements(
            require_llm=True,
            require_local_tts=True,
            workflows=workflows,
            frame_template="1080x1920/asset_default.html",
            require_browser=True,
            require_ffmpeg=True,
            input_files=assets,
            bgm_path=params.get("bgm_path"),
            api_vlm_model=api_vlm_model,
            tts_voice=params.get("voice_id"),
            tts_speed=params.get("tts_speed"),
        )
        report.merge(requirements)
        return report

    async def check_requirements(
        self,
        *,
        require_llm: bool = False,
        require_local_tts: bool = False,
        workflows: Iterable[Optional[str]] = (),
        frame_template: Optional[str] = None,
        require_browser: bool = False,
        require_ffmpeg: bool = False,
        input_files: Iterable[Optional[str]] = (),
        bgm_path: Optional[str] = None,
        api_vlm_model: Optional[str] = None,
        tts_voice: Optional[str] = None,
        tts_speed: Optional[float] = None,
    ) -> PreflightReport:
        """Check an explicit set of capabilities used by a product flow."""
        report = PreflightReport()

        if require_llm:
            await self._check_llm(report)
        if require_local_tts:
            self._check_local_tts(report, tts_voice, tts_speed)
        if require_ffmpeg:
            self._check_binary(report, "ffmpeg")
            self._check_binary(report, "ffprobe")
        if frame_template:
            self._check_template(report, frame_template)
        if require_browser:
            await self._check_browser(report)

        for file_path in input_files:
            if file_path:
                self._check_input_file(report, file_path)
        if bgm_path:
            self._check_bgm(report, bgm_path)
        if api_vlm_model:
            self._check_api_vlm(report, api_vlm_model)

        seen_workflows: set[str] = set()
        for workflow in workflows:
            if not workflow:
                continue
            normalized = self._normalize_workflow_key(workflow)
            if normalized in seen_workflows:
                continue
            seen_workflows.add(normalized)
            await self._check_workflow(report, normalized)

        return report

    async def _check_llm(self, report: PreflightReport) -> None:
        llm = self.config.llm
        if not llm.api_key.strip() or not llm.base_url.strip() or not llm.model.strip():
            report.add_error(
                "llm.config_missing",
                "LLM api_key, base_url, and model must all be configured.",
            )
            return

        cache_key = f"llm:{llm.base_url}:{llm.model}:{hash(llm.api_key)}"
        if self._is_cached(cache_key):
            report.add_check("llm")
            return

        headers = {"Authorization": f"Bearer {llm.api_key}"}
        try:
            async with self._http_client(timeout=8.0) as client:
                response = await client.get(_build_models_url(llm.base_url), headers=headers)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("LLM models response is not an object")
            model_ids = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            if not model_ids:
                report.add_error("llm.models_empty", "The LLM models endpoint returned no models.")
                return
            if llm.model not in model_ids:
                report.add_error(
                    "llm.model_unavailable",
                    f"Configured LLM model '{llm.model}' is not returned by the provider.",
                )
                return
        except httpx.HTTPStatusError as exc:
            report.add_error(
                "llm.auth_or_endpoint_failed",
                f"LLM models endpoint returned HTTP {exc.response.status_code}.",
            )
            return
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            report.add_error(
                "llm.unreachable",
                f"LLM readiness check failed: {type(exc).__name__}.",
            )
            return

        self._mark_cached(cache_key)
        report.add_check("llm")

    def _check_local_tts(
        self,
        report: PreflightReport,
        voice: Optional[str],
        speed: Optional[float],
    ) -> None:
        if importlib.util.find_spec("edge_tts") is None:
            report.add_error("tts.edge_missing", "The edge-tts package is not installed.")
            return
        if voice is not None and not str(voice).strip():
            report.add_error("tts.voice_empty", "The selected Edge TTS voice is empty.")
            return
        if speed is not None:
            try:
                valid_speed = 0.5 <= float(speed) <= 2.0
            except (TypeError, ValueError):
                valid_speed = False
            if not valid_speed:
                report.add_error(
                    "tts.speed_invalid",
                    "Edge TTS speed must be between 0.5 and 2.0.",
                )
                return
        report.add_check("local_tts")

    def _check_binary(self, report: PreflightReport, binary: str) -> None:
        executable = shutil.which(binary)
        if not executable:
            report.add_error(f"runtime.{binary}_missing", f"{binary} is not installed or not in PATH.")
            return
        try:
            subprocess.run(
                [executable, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            report.add_error(f"runtime.{binary}_failed", f"{binary} cannot be executed.")
            return
        report.add_check(binary)

    def _check_template(self, report: PreflightReport, template: str) -> None:
        try:
            path = Path(resolve_template_path(template)).resolve()
            if not path.is_file():
                raise FileNotFoundError(template)
            size_dir = path.parent.name
            width, height = size_dir.lower().split("x", 1)
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError(size_dir)
        except (FileNotFoundError, ValueError, TypeError):
            report.add_error(
                "template.invalid",
                f"Template '{template}' is missing or not stored under a <WIDTH>x<HEIGHT> directory.",
            )
            return
        report.add_check("template")

    async def _check_browser(self, report: PreflightReport) -> None:
        cache_key = "runtime:chromium"
        if self._is_cached(cache_key):
            report.add_check("chromium")
            return
        playwright = None
        browser = None
        try:
            from playwright.async_api import async_playwright

            playwright = await asyncio.wait_for(async_playwright().start(), timeout=10)
            browser = await asyncio.wait_for(
                playwright.chromium.launch(
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
                ),
                timeout=15,
            )
        except Exception as exc:
            report.add_error(
                "runtime.chromium_failed",
                f"Playwright Chromium readiness check failed: {type(exc).__name__}.",
            )
            return
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception:
                    pass

        self._mark_cached(cache_key)
        report.add_check("chromium")

    async def _check_workflow(self, report: PreflightReport, workflow: str) -> None:
        if workflow.startswith("selfhost/"):
            await self._check_selfhost_workflow(report, workflow)
        elif workflow.startswith("runninghub/"):
            await self._check_runninghub_workflow(report, workflow)
        elif workflow.startswith("api/"):
            self._check_api_workflow(report, workflow)
        else:
            report.add_error(
                "workflow.namespace_invalid",
                f"Workflow '{workflow}' must use selfhost/, runninghub/, or api/ namespace.",
            )

    async def _check_selfhost_workflow(
        self,
        report: PreflightReport,
        workflow: str,
    ) -> None:
        path = self._resolve_workflow_file(report, workflow)
        if path is None:
            return
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report.add_error("workflow.json_invalid", f"Workflow '{workflow}' is not valid JSON.")
            return

        if not isinstance(graph, dict):
            report.add_error(
                "workflow.not_api_format",
                f"Workflow '{workflow}' is not a ComfyUI API-format graph.",
            )
            return
        required_nodes = {
            node.get("class_type")
            for node in graph.values()
            if isinstance(node, dict) and node.get("class_type")
        }
        if not required_nodes:
            report.add_error(
                "workflow.not_api_format",
                f"Workflow '{workflow}' is not a ComfyUI API-format graph.",
            )
            return

        comfyui = self.config.comfyui
        try:
            available_nodes = await self._fetch_comfyui_nodes(
                comfyui.comfyui_url,
                comfyui.comfyui_api_key,
            )
        except httpx.HTTPStatusError as exc:
            report.add_error(
                "comfyui.http_failed",
                f"ComfyUI readiness endpoint returned HTTP {exc.response.status_code}.",
            )
            return
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            report.add_error(
                "comfyui.unreachable",
                f"ComfyUI readiness check failed: {type(exc).__name__}.",
            )
            return

        missing_nodes = sorted(required_nodes - available_nodes)
        if missing_nodes:
            report.add_error(
                "comfyui.nodes_missing",
                f"ComfyUI is missing nodes required by '{workflow}': {', '.join(missing_nodes)}.",
            )
            return
        report.add_check(f"workflow:{workflow}")

    async def _fetch_comfyui_nodes(
        self,
        base_url: str,
        api_key: Optional[str],
    ) -> set[str]:
        cache_key = f"comfyui:{base_url}:{hash(api_key or '')}"
        cached = self._comfyui_nodes_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < self.CACHE_TTL_SECONDS:
            return cached[1]

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        base_url = base_url.rstrip("/")
        async with self._http_client(timeout=8.0) as client:
            stats_response, nodes_response = await asyncio.gather(
                client.get(f"{base_url}/system_stats", headers=headers),
                client.get(f"{base_url}/object_info", headers=headers),
            )
            stats_response.raise_for_status()
            nodes_response.raise_for_status()
            stats_response.json()
            nodes_payload = nodes_response.json()
        if not isinstance(nodes_payload, dict):
            raise ValueError("ComfyUI object_info response is not an object")
        nodes = set(nodes_payload)
        self._comfyui_nodes_cache[cache_key] = (time.monotonic(), nodes)
        return nodes

    async def _check_runninghub_workflow(
        self,
        report: PreflightReport,
        workflow: str,
    ) -> None:
        api_key = (self.config.comfyui.runninghub_api_key or "").strip()
        if not api_key:
            report.add_error(
                "runninghub.key_missing",
                "RunningHub workflow selected but runninghub_api_key is not configured.",
            )
            return
        path = self._resolve_workflow_file(report, workflow)
        if path is None:
            return
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report.add_error("workflow.json_invalid", f"Workflow '{workflow}' is not valid JSON.")
            return
        workflow_id = str(wrapper.get("workflow_id", ""))
        if wrapper.get("source") != "runninghub" or not workflow_id.isdigit():
            report.add_error(
                "runninghub.wrapper_invalid",
                f"Workflow '{workflow}' is not a valid RunningHub wrapper.",
            )
            return

        base_url = os.getenv("RUNNINGHUB_BASE_URL", "https://www.runninghub.ai").rstrip("/")
        cache_key = f"runninghub:{base_url}:{workflow_id}:{hash(api_key)}"
        if self._is_cached(cache_key):
            report.add_check(f"workflow:{workflow}")
            return

        try:
            async with self._http_client(timeout=8.0) as client:
                response = await client.post(
                    f"{base_url}/api/openapi/getJsonApiFormat",
                    json={"apiKey": api_key, "workflowId": workflow_id},
                )
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("RunningHub response is not an object")
            if payload.get("code") != 0:
                report.add_error(
                    "runninghub.auth_or_workflow_failed",
                    f"RunningHub rejected credentials or workflow '{workflow}'.",
                )
                return
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("RunningHub response data is not an object")
            prompt = data.get("prompt")
            remote_graph = json.loads(prompt) if isinstance(prompt, str) else prompt
            if not isinstance(remote_graph, dict) or not any(
                isinstance(node, dict) and node.get("class_type")
                for node in remote_graph.values()
            ):
                report.add_error(
                    "runninghub.graph_invalid",
                    f"RunningHub returned no valid API-format graph for '{workflow}'.",
                )
                return
        except httpx.HTTPStatusError as exc:
            report.add_error(
                "runninghub.http_failed",
                f"RunningHub readiness endpoint returned HTTP {exc.response.status_code}.",
            )
            return
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
            report.add_error(
                "runninghub.unreachable",
                f"RunningHub readiness check failed: {type(exc).__name__}.",
            )
            return

        self._mark_cached(cache_key)
        report.add_check(f"workflow:{workflow}")

    def _check_api_workflow(self, report: PreflightReport, workflow: str) -> None:
        api_media = getattr(self.core, "api_media", None)
        if api_media is None:
            report.add_error("api_media.unavailable", "The direct API media service is not initialized.")
            return
        try:
            info = api_media.resolve_workflow(workflow)
        except ValueError as exc:
            report.add_error("api_media.model_unknown", str(exc))
            return

        provider = info["provider"]
        provider_config_name = {
            "seedream": "ark",
            "seedance": "ark",
        }.get(provider, provider)
        providers = self.config.api_providers.model_dump()
        provider_config = providers.get(provider_config_name, {})
        if provider == "kling":
            configured = bool(
                str(provider_config.get("access_key") or "").strip()
                and str(provider_config.get("secret_key") or "").strip()
            )
        else:
            configured = bool(str(provider_config.get("api_key") or "").strip())
        if not configured:
            report.add_error(
                "api_media.credentials_missing",
                f"Credentials for API provider '{provider_config_name}' are not configured.",
            )
            return
        if info.get("media_type") == "video" and not info.get("api_contract_verified", False):
            report.add_error(
                "api_media.contract_unverified",
                f"API video model '{workflow}' does not have a verified adapter contract.",
            )
            return
        report.add_check(f"workflow:{workflow}")

    def _check_api_vlm(self, report: PreflightReport, model: str) -> None:
        from pixelle_video.services.api_asset_analysis import APIAssetAnalysisService

        supported = {
            item
            for models in APIAssetAnalysisService.VLM_MODELS.values()
            for item in models
        }
        if model not in supported:
            report.add_error("vlm.model_unknown", f"Unsupported API VLM model '{model}'.")
            return
        if not self.config.api_providers.dashscope.api_key.strip():
            report.add_error(
                "vlm.credentials_missing",
                "DashScope API key is required for the selected VLM model.",
            )
            return
        report.add_check("api_vlm")

    def _check_input_file(self, report: PreflightReport, file_path: str) -> None:
        path = Path(file_path)
        if not path.is_file() or path.stat().st_size <= 0:
            report.add_error("input.file_missing", f"Input file is missing or empty: {path.name}.")
            return
        report.add_check(f"input:{path.name}")

    def _check_bgm(self, report: PreflightReport, bgm_path: str) -> None:
        path = Path(bgm_path)
        candidates = [path]
        if not path.is_absolute():
            candidates = [
                self.project_root / "data" / "bgm" / path,
                self.project_root / "bgm" / path,
                self.project_root / path,
            ]
        if not any(candidate.resolve().is_file() for candidate in candidates):
            report.add_error("bgm.missing", f"Background music file was not found: {bgm_path}.")
            return
        report.add_check("bgm")

    def _resolve_workflow_file(
        self,
        report: PreflightReport,
        workflow: str,
    ) -> Optional[Path]:
        source, _, filename = workflow.partition("/")
        if source not in {"selfhost", "runninghub"} or not filename:
            report.add_error("workflow.path_invalid", f"Invalid workflow key '{workflow}'.")
            return None
        if Path(filename).name != filename or not filename.endswith(".json"):
            report.add_error("workflow.path_invalid", f"Invalid workflow key '{workflow}'.")
            return None

        candidates = [
            self.project_root / "data" / "workflows" / source / filename,
            self.project_root / "workflows" / source / filename,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            base = candidate.parent.resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        report.add_error("workflow.missing", f"Workflow file was not found: {workflow}.")
        return None

    @staticmethod
    def _normalize_workflow_key(workflow: str) -> str:
        normalized = str(workflow).replace("\\", "/").lstrip("./")
        if normalized.startswith("workflows/"):
            normalized = normalized[len("workflows/") :]
        if normalized.startswith("data/workflows/"):
            normalized = normalized[len("data/workflows/") :]
        return normalized

    def _http_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.http_transport)

    def _is_cached(self, key: str) -> bool:
        checked_at = self._success_cache.get(key)
        return bool(checked_at and time.monotonic() - checked_at < self.CACHE_TTL_SECONDS)

    def _mark_cached(self, key: str) -> None:
        self._success_cache[key] = time.monotonic()
