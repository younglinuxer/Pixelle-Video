"""HTTP contracts for staged projects, styles, assets, and runtime evidence."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    original_prompt: str = Field(min_length=1)
    scene_count: int = Field(default=5, ge=1, le=50)


class RevisionRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class PromptOptimizeRequest(RevisionRequest):
    role: str = "short_video_director"
    custom_role: Optional[str] = None


class PromptApplyRequest(RevisionRequest):
    optimized_prompt: str = Field(min_length=1)
    role: str = "short_video_director"


class ApplyStyleRequest(RevisionRequest):
    style_id: str


class SceneUpdateRequest(RevisionRequest):
    narration: Optional[str] = None
    visual_prompt: Optional[str] = None
    estimated_duration: Optional[float] = Field(default=None, gt=0, le=300)
    media_mode: Optional[str] = None
    asset_ids: Optional[list[str]] = None


class SceneRegenerateRequest(RevisionRequest):
    instruction: str = ""


class SceneSplitRequest(RevisionRequest):
    split_at: int = Field(gt=0)


class TitleUpdateRequest(RevisionRequest):
    title: str = Field(min_length=1)


class SceneReorderRequest(RevisionRequest):
    scene_ids: list[str]


class RenderProfileUpdateRequest(RevisionRequest):
    updates: dict[str, Any]


class LocalAssetImportRequest(RevisionRequest):
    source_path: str


class AssetPresentationUpdateRequest(RevisionRequest):
    fit_mode: str
    focal_x: float = Field(ge=0, le=1)
    focal_y: float = Field(ge=0, le=1)


class CustomStyleRequest(BaseModel):
    description: str = Field(min_length=3)


class StylePreviewRequest(BaseModel):
    workflow: str


class CanaryValidationRequest(BaseModel):
    smoke_path: str
    canary_paths: list[str] = Field(min_length=3, max_length=3)
    expected_width: int = Field(gt=0)
    expected_height: int = Field(gt=0)
    queue_history: dict[str, Any]


class ApprovalRequest(BaseModel):
    approved_by: str = Field(min_length=1)
