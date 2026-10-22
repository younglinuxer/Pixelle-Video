"""Exact render-combination hashing and runtime-readiness gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx

from pixelle_video.models.project import (
    CompatibilityRecord,
    CompatibilityStatus,
    ProjectDraft,
    RenderProfile,
    utc_now,
)
from pixelle_video.utils.os_util import get_resource_path


class CompatibilityGateError(RuntimeError):
    """Raised when the selected render combination is not runnable."""


class CompatibilityService:
    """Bind readiness evidence to style, template, workflow and runtime fingerprints."""

    def __init__(
        self,
        root: str | Path = "data/compatibility",
        project_root: str | Path | None = None,
        comfyui_url: str = "http://127.0.0.1:8188",
        api_key: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "approvals.json"
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.comfyui_url = comfyui_url.rstrip("/")
        self.api_key = api_key
        self.runninghub_api_key = runninghub_api_key

    async def inspect(self, project: ProjectDraft) -> tuple[RenderProfile, CompatibilityRecord]:
        """Read local contracts and the selected runtime without submitting work."""
        if project.style is None:
            raise CompatibilityGateError("A style must be selected before compatibility checks")
        render = project.render_profile.model_copy(deep=True)
        render.style_id = project.style.style_id
        render.style_version = project.style.version
        render.style_hash = project.style.validation_hash()
        render.template_hash = self._hash_contract("templates", render.frame_template)
        if render.media_workflow:
            render.workflow_hash = self._hash_contract("workflows", render.media_workflow)
            runtime_fingerprint, ready, details = await self._inspect_comfyui(render.media_workflow)
        else:
            render.workflow_hash = hashlib.sha256(b"no-media-workflow").hexdigest()
            runtime_fingerprint = hashlib.sha256(b"local-static-renderer-v1").hexdigest()
            ready = True
            details = {"mode": "static", "message": "No media workflow required"}
        render.environment_fingerprint = runtime_fingerprint
        combination_hash = render.combination_hash()
        record = self._records().get(combination_hash)
        if record is None:
            record = CompatibilityRecord(
                combination_hash=combination_hash,
                status=(
                    CompatibilityStatus.RUNTIME_READY
                    if ready
                    else CompatibilityStatus.UNVERIFIED
                ),
                style_id=render.style_id,
                style_version=render.style_version,
                style_hash=render.style_hash,
                template_hash=render.template_hash,
                workflow_hash=render.workflow_hash,
                environment_fingerprint=render.environment_fingerprint,
                technical_checks={"runtime": details},
            )
        else:
            record.technical_checks["runtime"] = details
            if record.status == CompatibilityStatus.UNVERIFIED and ready:
                record.status = CompatibilityStatus.RUNTIME_READY
            elif record.status == CompatibilityStatus.RUNTIME_READY and not ready:
                record.status = CompatibilityStatus.UNVERIFIED
            record.updated_at = utc_now()
        self._save_record(record)
        return render, record

    def get(self, combination_hash: str) -> Optional[CompatibilityRecord]:
        """Get persisted evidence for a combination hash."""
        return self._records().get(combination_hash)

    def list_records(self) -> list[CompatibilityRecord]:
        """List approval evidence, newest first."""
        return sorted(self._records().values(), key=lambda item: item.updated_at, reverse=True)

    async def inspect_workflow(self, workflow: str) -> dict[str, Any]:
        """Read-only workflow/runtime audit for the dedicated maintenance page."""
        fingerprint, ready, details = await self._inspect_comfyui(workflow)
        return {"ready": ready, "environment_fingerprint": fingerprint, **details}

    def record_technical_pass(
        self,
        combination_hash: str,
        *,
        smoke_evidence: dict[str, Any],
        canary_evidence: dict[str, Any],
        technical_checks: dict[str, Any],
    ) -> CompatibilityRecord:
        """Record externally executed smoke/canary results after validating all gates."""
        record = self._require_record(combination_hash)
        if record.status not in {
            CompatibilityStatus.RUNTIME_READY,
            CompatibilityStatus.TECHNICAL_PASS,
        }:
            raise CompatibilityGateError("Runtime readiness must pass before technical evidence")
        required = {
            "queue_history",
            "decode",
            "dimensions",
            "frame_rate",
            "streams",
            "duration",
            "black_frames",
            "frozen_frames",
            "audio_volume",
        }
        failed = sorted(key for key in required if technical_checks.get(key) is not True)
        if failed:
            raise CompatibilityGateError(
                f"Technical evidence is incomplete or failed: {', '.join(failed)}"
            )
        if not smoke_evidence or not canary_evidence:
            raise CompatibilityGateError("Smoke and three-scene canary evidence are both required")
        record.status = CompatibilityStatus.TECHNICAL_PASS
        record.smoke_evidence = smoke_evidence
        record.canary_evidence = canary_evidence
        record.technical_checks.update(technical_checks)
        record.updated_at = utc_now()
        self._save_record(record)
        return record

    def approve(self, combination_hash: str, approved_by: str) -> CompatibilityRecord:
        """Record explicit human approval after technical checks pass."""
        if not approved_by.strip():
            raise CompatibilityGateError("Human approver is required")
        record = self._require_record(combination_hash)
        if record.status != CompatibilityStatus.TECHNICAL_PASS:
            raise CompatibilityGateError("Technical pass is required before human approval")
        record.status = CompatibilityStatus.APPROVED
        record.human_approved_by = approved_by.strip()
        record.human_approved_at = utc_now()
        record.updated_at = utc_now()
        self._save_record(record)
        return record

    def require_approved(self, render_profile: RenderProfile) -> CompatibilityRecord:
        """Require legacy human approval evidence.

        Kept for callers that still consume historical approval records. The normal
        creation flow uses :meth:`require_runtime_ready` instead.
        """
        combination_hash = render_profile.combination_hash()
        record = self._records().get(combination_hash)
        if record is None or record.status != CompatibilityStatus.APPROVED:
            status = record.status.value if record else CompatibilityStatus.UNVERIFIED.value
            raise CompatibilityGateError(
                f"Render combination is not approved (status={status}, hash={combination_hash[:12]})"
            )
        return record

    def require_runtime_ready(self, render_profile: RenderProfile) -> CompatibilityRecord:
        """Block production only when the selected workflow cannot run in this environment."""
        combination_hash = render_profile.combination_hash()
        record = self._records().get(combination_hash)
        ready_statuses = {
            CompatibilityStatus.RUNTIME_READY,
            CompatibilityStatus.TECHNICAL_PASS,
            CompatibilityStatus.APPROVED,
        }
        if record is None or record.status not in ready_statuses:
            status = record.status.value if record else CompatibilityStatus.UNVERIFIED.value
            raise CompatibilityGateError(
                "Render environment is not ready "
                f"(status={status}, hash={combination_hash[:12]})"
            )
        return record

    def _hash_contract(self, base: str, relative_path: str) -> str:
        path = self._resolve_contract_path(base, relative_path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _resolve_contract_path(self, base: str, relative_path: str) -> Path:
        """Resolve the same custom-over-default resource that production will consume."""
        parts = Path(relative_path).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise CompatibilityGateError(f"Invalid contract path: {relative_path}")
        try:
            if base in {"templates", "workflows"}:
                path = Path(get_resource_path(base, *parts)).resolve()
            else:
                path = (self.project_root / base / relative_path).resolve()
        except FileNotFoundError as exc:
            raise CompatibilityGateError(
                f"Contract file does not exist: {relative_path}"
            ) from exc
        if not path.is_file():
            raise CompatibilityGateError(f"Contract file does not exist: {relative_path}")
        return path

    async def _inspect_comfyui(self, workflow: str) -> tuple[str, bool, dict[str, Any]]:
        workflow_path = self._resolve_contract_path("workflows", workflow)
        payload = json.loads(workflow_path.read_text(encoding="utf-8"))
        if payload.get("source") == "runninghub":
            return await self._inspect_runninghub(workflow, payload)
        required_nodes = sorted(
            {
                node.get("class_type")
                for node in payload.values()
                if isinstance(node, dict) and node.get("class_type")
            }
        )
        model_suffixes = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
        expected_models = sorted(
            {
                value
                for node in payload.values()
                if isinstance(node, dict)
                for value in node.get("inputs", {}).values()
                if isinstance(value, str) and value.lower().endswith(model_suffixes)
            }
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                stats_response, nodes_response, queue_response = await _gather_responses(
                    client,
                    self.comfyui_url,
                    headers,
                )
            stats_response.raise_for_status()
            nodes_response.raise_for_status()
            queue_response.raise_for_status()
            stats = stats_response.json()
            object_info = nodes_response.json()
            queue = queue_response.json()
            missing = sorted(set(required_nodes) - set(object_info))
            relevant_contracts = {
                node_type: object_info[node_type]
                for node_type in required_nodes
                if node_type in object_info
            }
            available_models = {
                _normalize_model_path(value)
                for value in _iter_strings(relevant_contracts)
                if value.lower().endswith(model_suffixes)
            }
            missing_models = [
                model
                for model in expected_models
                if _normalize_model_path(model) not in available_models
            ]
            fingerprint_payload = {
                "system": stats.get("system", {}),
                "devices": stats.get("devices", []),
                "required_node_contracts": relevant_contracts,
            }
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            ready = not missing and not missing_models
            details = {
                "workflow": workflow,
                "required_node_count": len(required_nodes),
                "available_node_count": len(object_info),
                "missing_nodes": missing,
                "expected_models": expected_models,
                "missing_models": missing_models,
                "queue_running": len(queue.get("queue_running", [])),
                "queue_pending": len(queue.get("queue_pending", [])),
            }
            return fingerprint, ready, details
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            fingerprint = hashlib.sha256(
                f"unavailable:{self.comfyui_url}".encode("utf-8")
            ).hexdigest()
            return fingerprint, False, {
                "workflow": workflow,
                "error": type(exc).__name__,
                "message": "ComfyUI readiness check failed",
            }

    async def _inspect_runninghub(
        self,
        workflow: str,
        wrapper: dict[str, Any],
    ) -> tuple[str, bool, dict[str, Any]]:
        workflow_id = str(wrapper.get("workflow_id", ""))
        if not self.runninghub_api_key or not workflow_id.isdigit():
            fingerprint = hashlib.sha256(
                f"runninghub-unavailable:{workflow_id}".encode("utf-8")
            ).hexdigest()
            return fingerprint, False, {
                "workflow": workflow,
                "provider": "runninghub",
                "message": "RunningHub credential or workflow ID is missing",
            }
        base_url = os.getenv("RUNNINGHUB_BASE_URL", "https://www.runninghub.ai").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{base_url}/api/openapi/getJsonApiFormat",
                    json={"apiKey": self.runninghub_api_key, "workflowId": workflow_id},
                )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            prompt = data.get("prompt") if isinstance(data, dict) else None
            graph = json.loads(prompt) if isinstance(prompt, str) else prompt
            if payload.get("code") != 0 or not isinstance(graph, dict):
                raise ValueError("RunningHub did not return an API-format graph")
            fingerprint = hashlib.sha256(
                json.dumps(graph, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            return fingerprint, True, {
                "workflow": workflow,
                "provider": "runninghub",
                "workflow_id": workflow_id,
                "remote_node_count": len(graph),
            }
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
            fingerprint = hashlib.sha256(
                f"runninghub-unavailable:{workflow_id}".encode("utf-8")
            ).hexdigest()
            return fingerprint, False, {
                "workflow": workflow,
                "provider": "runninghub",
                "error": type(exc).__name__,
                "message": "RunningHub readiness check failed",
            }

    def _require_record(self, combination_hash: str) -> CompatibilityRecord:
        record = self._records().get(combination_hash)
        if record is None:
            raise CompatibilityGateError("Compatibility record does not exist; inspect first")
        return record

    def _records(self) -> dict[str, CompatibilityRecord]:
        if not self.records_path.exists():
            return {}
        try:
            payload = json.loads(self.records_path.read_text(encoding="utf-8"))
            return {
                key: CompatibilityRecord.model_validate(value)
                for key, value in payload.items()
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _save_record(self, record: CompatibilityRecord) -> None:
        records = self._records()
        records[record.combination_hash] = record
        payload = {key: value.model_dump(mode="json") for key, value in records.items()}
        temp_path = self.records_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.records_path)


async def _gather_responses(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
) -> tuple[httpx.Response, httpx.Response, httpx.Response]:
    """Fetch readiness endpoints concurrently without submitting generation work."""
    import asyncio

    responses = await asyncio.gather(
        client.get(f"{base_url}/system_stats", headers=headers),
        client.get(f"{base_url}/object_info", headers=headers),
        client.get(f"{base_url}/queue", headers=headers),
    )
    return responses[0], responses[1], responses[2]


def _iter_strings(value: Any):
    """Yield nested string values from a ComfyUI node contract."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _normalize_model_path(value: str) -> str:
    """Normalize ComfyUI model choices across Windows and POSIX hosts."""
    return value.replace("\\", "/").casefold()
