"""Build explicit manifests from committed templates and ComfyUI API workflows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pixelle_video.models.catalog import TemplateManifest, WorkflowManifest


class RenderCatalogService:
    """Expose only supported canvases and inspect execution contracts without running them."""

    SUPPORTED_CANVASES = {(1080, 1920), (1920, 1080), (1080, 1080)}
    MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()

    def list_templates(self) -> list[TemplateManifest]:
        manifests = []
        for width, height in sorted(self.SUPPORTED_CANVASES):
            directory = self.project_root / "templates" / f"{width}x{height}"
            for path in sorted(directory.glob("*.html")):
                manifests.append(self._template_manifest(path, width, height))
        return manifests

    def list_workflows(self) -> list[WorkflowManifest]:
        manifests = []
        for path in sorted((self.project_root / "workflows").glob("*/*.json")):
            manifests.append(self._workflow_manifest(path))
        return manifests

    def _template_manifest(self, path: Path, width: int, height: int) -> TemplateManifest:
        text = path.read_text(encoding="utf-8")
        media_width = self._meta_int(text, "template:media-width")
        media_height = self._meta_int(text, "template:media-height")
        coverage = (
            (media_width * media_height) / (width * height)
            if media_width and media_height
            else 0
        )
        family = path.stem
        if "long_text" in family or "excerpt" in family:
            capacity = "long"
        elif "full" in family or family.startswith("video_"):
            capacity = "short"
        else:
            capacity = "medium"
        tags = [token for token in family.split("_") if token not in {"image", "video", "static", "asset"}]
        return TemplateManifest(
            key=str(path.relative_to(self.project_root / "templates")),
            family=family,
            canvas_width=width,
            canvas_height=height,
            media_width=media_width,
            media_height=media_height,
            media_slot_ratio=(
                f"{media_width}:{media_height}" if media_width and media_height else None
            ),
            text_capacity=capacity,
            style_tags=tags,
            fit_mode="cover" if coverage >= 0.8 else "contain",
            source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def _workflow_manifest(self, path: Path) -> WorkflowManifest:
        raw = path.read_bytes()
        graph = json.loads(raw)
        required_nodes: set[str] = set()
        output_nodes: set[str] = set()
        model_files: set[str] = set()
        bindings: dict[str, dict[str, Any]] = {}
        native_width = None
        native_height = None
        frame_count = None
        fps = None
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            if class_type:
                required_nodes.add(str(class_type))
                lowered = str(class_type).lower()
                if any(token in lowered for token in ("saveimage", "savevideo", "videocombine", "previewimage")):
                    output_nodes.add(str(class_type))
            inputs = node.get("inputs", {})
            title = str(node.get("_meta", {}).get("title", ""))
            if title.startswith("$"):
                bindings[title[1:]] = {"node_id": node_id, "inputs": sorted(inputs)}
            for key, value in inputs.items():
                if isinstance(value, str) and value.lower().endswith(self.MODEL_SUFFIXES):
                    model_files.add(value)
                normalized = title.lower().lstrip("$")
                if isinstance(value, (int, float)):
                    if normalized.startswith("width") or key == "width":
                        native_width = int(value)
                    elif normalized.startswith("height") or key == "height":
                        native_height = int(value)
                    elif "frame" in normalized and ("count" in normalized or "length" in normalized):
                        frame_count = int(value)
                    elif normalized.startswith("fps") or key == "fps":
                        fps = float(value)
        name = path.stem.lower()
        media_type = "unknown"
        for candidate in ("image", "video", "tts", "analysis"):
            if name.startswith(candidate) or (candidate == "analysis" and name.startswith("analyse")):
                media_type = candidate
                break
        return WorkflowManifest(
            key=str(path.relative_to(self.project_root / "workflows")),
            source=path.parent.name,
            provider_workflow_id=str(graph.get("workflow_id")) if graph.get("workflow_id") else None,
            media_type=media_type,
            dsl_bindings=bindings,
            native_width=native_width,
            native_height=native_height,
            frame_count=frame_count,
            fps=fps,
            required_nodes=sorted(required_nodes),
            model_files=sorted(model_files),
            output_node_types=sorted(output_nodes),
            graph_hash=hashlib.sha256(raw).hexdigest(),
        )

    @staticmethod
    def _meta_int(text: str, name: str) -> int | None:
        pattern = rf'<meta\s+name=["\']{re.escape(name)}["\']\s+content=["\'](\d+)["\']'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None
