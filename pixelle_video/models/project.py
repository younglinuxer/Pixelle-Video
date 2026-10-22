"""Persistent project, style, asset, and render-contract models."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp for durable records."""
    return datetime.now(timezone.utc)


class ProjectStatus(str, Enum):
    """Durable lifecycle for a staged creation project."""

    DRAFT = "draft"
    PLANNED = "planned"
    LOCKED = "locked"
    PRODUCING = "producing"
    COMPLETED = "completed"
    FAILED = "failed"


class CompatibilityStatus(str, Enum):
    """Validation lifecycle for an exact render combination."""

    UNVERIFIED = "unverified"
    RUNTIME_READY = "runtime_ready"
    TECHNICAL_PASS = "technical_pass"
    APPROVED = "approved"


class SceneDirtyState(BaseModel):
    """Tracks which scene products must be regenerated or reconfirmed."""

    visual_prompt: bool = True
    template_preview: bool = True
    crop: bool = True
    media: bool = True

    @property
    def any(self) -> bool:
        return any(self.model_dump().values())


class StyleProfile(BaseModel):
    """Versioned creative direction applied consistently to a whole project."""

    style_id: str
    version: str = "1.0.0"
    name: str
    description: str
    suitable_for: list[str] = Field(default_factory=list)
    script_tone: str
    narrative_pacing: str
    wording_rules: list[str] = Field(default_factory=list)
    visual_prompt_prefix: str
    negative_prompt: str = ""
    color_rules: list[str] = Field(default_factory=list)
    lighting_rules: list[str] = Field(default_factory=list)
    composition_rules: list[str] = Field(default_factory=list)
    material_rules: list[str] = Field(default_factory=list)
    character_rules: list[str] = Field(default_factory=list)
    camera_rules: list[str] = Field(default_factory=list)
    action_intensity: Literal["low", "medium", "high"] = "medium"
    shot_pacing: str = "medium"
    transition_suggestions: list[str] = Field(default_factory=list)
    recommended_template_families: list[str] = Field(default_factory=list)
    recommended_aspect_ratios: list[str] = Field(default_factory=lambda: ["9:16"])
    allowed_workflows: list[str] = Field(default_factory=list)
    recommended_voice_ids: list[str] = Field(default_factory=list)
    tts_speed_range: tuple[float, float] = (0.9, 1.2)
    bgm_tags: list[str] = Field(default_factory=list)
    bgm_volume_range: tuple[float, float] = (0.1, 0.3)
    preview_images: list[str] = Field(default_factory=list)
    preview_prompts: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    cost_tier: Literal["low", "medium", "high"] = "medium"
    builtin: bool = False
    source_description: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_ranges(self) -> "StyleProfile":
        if self.tts_speed_range[0] > self.tts_speed_range[1]:
            raise ValueError("tts_speed_range minimum cannot exceed maximum")
        if self.bgm_volume_range[0] > self.bgm_volume_range[1]:
            raise ValueError("bgm_volume_range minimum cannot exceed maximum")
        return self

    def validation_hash(self) -> str:
        """Return a stable hash of the creative contract, excluding previews and timestamps."""
        payload = self.model_dump(
            mode="json",
            exclude={"preview_images", "preview_prompts", "created_at"},
        )
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SceneDraft(BaseModel):
    """An editable scene with a stable identity across revisions."""

    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order: int = Field(ge=0)
    narration: str = Field(min_length=1)
    visual_prompt: str = ""
    asset_ids: list[str] = Field(default_factory=list)
    media_mode: Literal["static", "image", "video", "asset"] = "video"
    estimated_duration: float = Field(default=4.0, gt=0, le=300)
    template_name: Optional[str] = None
    dirty: SceneDirtyState = Field(default_factory=SceneDirtyState)
    last_render_task_id: Optional[str] = None
    last_audio_path: Optional[str] = None
    last_image_path: Optional[str] = None
    last_video_path: Optional[str] = None
    last_composed_image_path: Optional[str] = None
    last_segment_path: Optional[str] = None
    last_render_duration: Optional[float] = None


class AssetRecord(BaseModel):
    """Immutable original plus derived preview/proxy metadata."""

    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_name: str
    original_path: str
    sha256: str
    media_kind: Literal["image", "video", "audio"]
    size_bytes: int = Field(ge=0)
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    frame_rate: Optional[float] = None
    duration: Optional[float] = None
    has_alpha: bool = False
    orientation: Optional[str] = None
    thumbnail_path: Optional[str] = None
    proxy_path: Optional[str] = None
    fit_mode: Literal["cover", "contain"] = "contain"
    focal_x: float = Field(default=0.5, ge=0, le=1)
    focal_y: float = Field(default=0.5, ge=0, le=1)
    quality_warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class RenderProfile(BaseModel):
    """Frozen production configuration used to calculate compatibility."""

    canvas_width: int = Field(default=1080, gt=0)
    canvas_height: int = Field(default=1920, gt=0)
    frame_template: str = "1080x1920/image_default.html"
    media_workflow: Optional[str] = None
    media_width: int = Field(default=1024, gt=0)
    media_height: int = Field(default=1024, gt=0)
    video_fps: int = Field(default=30, gt=0, le=120)
    tts_inference_mode: Literal["local", "comfyui"] = "local"
    voice_id: Optional[str] = "zh-CN-YunjianNeural"
    tts_workflow: Optional[str] = None
    tts_speed: float = Field(default=1.1, ge=0.5, le=2.0)
    bgm_path: Optional[str] = None
    bgm_volume: float = Field(default=0.2, ge=0, le=1)
    bgm_mode: Literal["truncate", "loop"] = "truncate"
    crop_strategy: Literal["cover", "contain"] = "cover"
    style_id: Optional[str] = None
    style_version: Optional[str] = None
    style_hash: Optional[str] = None
    template_hash: Optional[str] = None
    workflow_hash: Optional[str] = None
    environment_fingerprint: Optional[str] = None

    def combination_hash(self) -> str:
        """Hash every field that may affect rendered output or compatibility."""
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProjectDraft(BaseModel):
    """Durable staged video project."""

    schema_version: str = "1.0"
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    revision: int = Field(default=0, ge=0)
    status: ProjectStatus = ProjectStatus.DRAFT
    original_prompt: str
    optimized_prompt: Optional[str] = None
    optimization_role: str = "short_video_director"
    title: Optional[str] = None
    scene_count: int = Field(default=5, ge=1, le=50)
    style: Optional[StyleProfile] = None
    scenes: list[SceneDraft] = Field(default_factory=list)
    assets: list[AssetRecord] = Field(default_factory=list)
    render_profile: RenderProfile = Field(default_factory=RenderProfile)
    locked_snapshot_hash: Optional[str] = None
    last_task_id: Optional[str] = None
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def snapshot_hash(self) -> str:
        """Hash the exact content/config consumed by production."""
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "title": self.title,
            "style": self.style.model_dump(mode="json") if self.style else None,
            "scenes": [scene.model_dump(mode="json") for scene in self.scenes],
            "assets": [asset.model_dump(mode="json") for asset in self.assets],
            "render_profile": self.render_profile.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CompatibilityRecord(BaseModel):
    """Evidence for one exact style/template/workflow/environment combination."""

    combination_hash: str
    status: CompatibilityStatus = CompatibilityStatus.UNVERIFIED
    style_id: str
    style_version: str
    style_hash: str
    template_hash: str
    workflow_hash: str
    environment_fingerprint: str
    smoke_evidence: dict[str, Any] = Field(default_factory=dict)
    canary_evidence: dict[str, Any] = Field(default_factory=dict)
    technical_checks: dict[str, Any] = Field(default_factory=dict)
    human_approved_by: Optional[str] = None
    human_approved_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=utc_now)
