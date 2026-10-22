"""Inspectable template and workflow execution contracts."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from pixelle_video.models.project import CompatibilityStatus


class TemplateManifest(BaseModel):
    key: str
    family: str
    canvas_width: int
    canvas_height: int
    media_width: Optional[int] = None
    media_height: Optional[int] = None
    media_slot_ratio: Optional[str] = None
    text_capacity: Literal["short", "medium", "long"] = "medium"
    style_tags: list[str] = Field(default_factory=list)
    fit_mode: Literal["cover", "contain"] = "contain"
    source_hash: str
    validation_status: CompatibilityStatus = CompatibilityStatus.UNVERIFIED


class WorkflowManifest(BaseModel):
    key: str
    source: str
    provider_workflow_id: Optional[str] = None
    media_type: Literal["image", "video", "tts", "analysis", "unknown"]
    dsl_bindings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    native_width: Optional[int] = None
    native_height: Optional[int] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    required_nodes: list[str] = Field(default_factory=list)
    model_files: list[str] = Field(default_factory=list)
    output_node_types: list[str] = Field(default_factory=list)
    graph_hash: str
    validation_status: CompatibilityStatus = CompatibilityStatus.UNVERIFIED
