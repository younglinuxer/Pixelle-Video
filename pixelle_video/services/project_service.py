"""Application service for staged creation, revisions, locking, and production."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from pixelle_video.models.project import (
    AssetRecord,
    ProjectDraft,
    ProjectStatus,
    RenderProfile,
    SceneDraft,
    StyleProfile,
)
from pixelle_video.services.compatibility import CompatibilityService
from pixelle_video.services.project_persistence import ProjectPersistenceService
from pixelle_video.services.style_service import StyleService


class OptimizedBrief(BaseModel):
    """Structured prompt optimization result."""

    optimized_prompt: str
    improvements: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_fields(cls, value: Any) -> Any:
        """Accept common JSON-object variants emitted by compatible LLM providers."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "optimized_prompt" not in normalized:
            for key in ("optimizedPrompt", "optimized_brief", "prompt", "content"):
                candidate = normalized.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    normalized["optimized_prompt"] = candidate
                    break
        if "improvements" not in normalized:
            for key in ("changes", "suggestions", "optimization_points"):
                if key in normalized:
                    normalized["improvements"] = normalized[key]
                    break
        return normalized

    @field_validator("improvements", mode="before")
    @classmethod
    def normalize_improvements(cls, value: Any) -> list[str]:
        """Turn provider-specific suggestion objects into concise display strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [line.strip(" -•\t") for line in value.splitlines() if line.strip()]
        if not isinstance(value, list):
            value = [value]
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = "：".join(str(part).strip() for part in item.values() if str(part).strip())
            else:
                text = str(item).strip()
            if text:
                result.append(text)
        return result


class PlannedScene(BaseModel):
    """Provider response for one editable scene."""

    narration: str
    visual_prompt: str
    estimated_duration: float = Field(default=4.0, gt=0, le=60)
    media_mode: str = "video"

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_fields(cls, value: Any) -> Any:
        """Normalize frequent scene-field aliases without weakening the public model."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        aliases = {
            "narration": ("voiceover", "script", "text"),
            "visual_prompt": ("visualPrompt", "image_prompt", "prompt"),
            "estimated_duration": ("estimatedDuration", "duration", "duration_seconds"),
            "media_mode": ("mediaMode", "media_type", "type"),
        }
        for target, source_keys in aliases.items():
            if target not in normalized:
                for source in source_keys:
                    if source in normalized:
                        normalized[target] = normalized[source]
                        break
        return normalized

    @field_validator("estimated_duration", mode="before")
    @classmethod
    def normalize_duration(cls, value: Any) -> Any:
        """Accept values such as ``4 seconds`` while retaining numeric validation."""
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            if match:
                return float(match.group())
        return value


class StoryboardPlan(BaseModel):
    """Provider response for script and visual planning only."""

    title: str
    scenes: list[PlannedScene]

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_fields(cls, value: Any) -> Any:
        """Accept common storyboard container names from OpenAI-compatible models."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "title" not in normalized:
            for key in ("video_title", "project_title", "name"):
                if key in normalized:
                    normalized["title"] = normalized[key]
                    break
        if "scenes" not in normalized:
            for key in ("storyboard", "shots", "scene_list"):
                if key in normalized:
                    candidate = normalized[key]
                    if isinstance(candidate, dict):
                        candidate = candidate.get("scenes", candidate.get("shots"))
                    normalized["scenes"] = candidate
                    break
        return normalized


class VisualPromptPlan(BaseModel):
    """Visual-only replan used after style or canvas changes."""

    prompts: list[str]


class RegeneratedNarration(BaseModel):
    """One locally revised narration."""

    narration: str = Field(min_length=1)


class ProjectStateError(RuntimeError):
    """Raised when an operation is invalid for the current project state."""


class ProjectService:
    """Coordinate planning services while keeping production snapshot-only."""

    CANVASES = {(1080, 1920), (1920, 1080), (1080, 1080)}

    def __init__(
        self,
        core: Any,
        persistence: Optional[ProjectPersistenceService] = None,
        styles: Optional[StyleService] = None,
        compatibility: Optional[CompatibilityService] = None,
    ):
        self.core = core
        self.persistence = persistence or ProjectPersistenceService()
        self.styles = styles or StyleService(core.llm)
        comfy = core.config.get("comfyui", {})
        self.compatibility = compatibility or CompatibilityService(
            comfyui_url=comfy.get("comfyui_url", "http://127.0.0.1:8188"),
            api_key=comfy.get("comfyui_api_key"),
        )
        self._recover_interrupted_projects()

    def create(self, original_prompt: str, scene_count: int = 5) -> ProjectDraft:
        """Create a durable draft before any provider call."""
        if not original_prompt.strip():
            raise ValueError("Creative brief cannot be empty")
        project = ProjectDraft(original_prompt=original_prompt.strip(), scene_count=scene_count)
        return self.persistence.create(project)

    def get(self, project_id: str) -> ProjectDraft:
        return self.persistence.load(project_id)

    def list(self, limit: int = 100) -> list[ProjectDraft]:
        return self.persistence.list(limit=limit)

    async def optimize_prompt(
        self,
        project_id: str,
        expected_revision: int,
        role: str = "short_video_director",
        custom_role: Optional[str] = None,
    ) -> OptimizedBrief:
        """Propose an optimized brief without mutating the project."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        role_instruction = {
            "short_video_director": "short-video director focused on hook, pacing, and feasible shots",
            "copywriter": "copywriter focused on clarity, audience benefit, and memorable phrasing",
            "storyboard_director": "storyboard director focused on visible actions and shot continuity",
            "custom": custom_role or "creative director",
        }.get(role, role)
        style_context = self._style_prompt(project.style) if project.style else "No style selected yet."
        result = await self.core.llm(
            f"""Act as a {role_instruction}. Improve the brief without inventing factual claims.
Preserve the user's intent, target audience, must-have details and constraints. The result will be
reviewed before it is applied. Project style contract: {style_context}
Original brief: {project.original_prompt}""",
            response_type=OptimizedBrief,
            temperature=0.4,
            max_tokens=8192,
        )
        return result

    def apply_optimized_prompt(
        self,
        project_id: str,
        expected_revision: int,
        optimized_prompt: str,
        role: str,
    ) -> ProjectDraft:
        """Apply a previously reviewed optimization proposal."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if not optimized_prompt.strip():
            raise ValueError("Optimized prompt cannot be empty")
        project.optimized_prompt = optimized_prompt.strip()
        project.optimization_role = role.strip() or "short_video_director"
        project.status = ProjectStatus.DRAFT
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def apply_style(
        self,
        project_id: str,
        style_id: str,
        expected_revision: int,
    ) -> ProjectDraft:
        """Apply an independent style copy and invalidate only visual-dependent outputs."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        style = self.styles.get_style(style_id)
        project.style = style
        project.render_profile.style_id = style.style_id
        project.render_profile.style_version = style.version
        project.render_profile.style_hash = style.validation_hash()
        self._apply_recommended_render_defaults(project.render_profile, style)
        project.render_profile.template_hash = None
        project.render_profile.workflow_hash = None
        project.render_profile.environment_fingerprint = None
        for scene in project.scenes:
            scene.visual_prompt = ""
            scene.dirty.visual_prompt = True
            scene.dirty.template_preview = True
            scene.dirty.crop = True
            scene.dirty.media = True
        project.status = ProjectStatus.PLANNED if project.scenes else ProjectStatus.DRAFT
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def plan(self, project_id: str, expected_revision: int) -> ProjectDraft:
        """Generate editable script and storyboard; never produce media."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if project.style is None:
            raise ProjectStateError("Select a style before planning")
        source = project.optimized_prompt or project.original_prompt
        style_prompt = self._style_prompt(project.style)
        plan = await self.core.llm(
            f"""Create exactly {project.scene_count} scenes for a short video. Return editable
narration and an English visual-generation prompt for each scene. Maintain subject, era, palette,
wardrobe and rendering continuity. Do not include camera instructions that contradict the style.
Every scene will be generated as AI video: describe visible motion and temporal continuity, and set
media_mode to video for every scene.
Style contract: {style_prompt}
Creative brief: {source}""",
            response_type=StoryboardPlan,
            temperature=0.5,
            max_tokens=8192,
        )
        if len(plan.scenes) != project.scene_count:
            raise ProjectStateError(
                f"Planner returned {len(plan.scenes)} scenes; expected {project.scene_count}"
            )
        project.title = plan.title.strip()
        project.scenes = [
            SceneDraft(
                order=index,
                narration=scene.narration.strip(),
                visual_prompt=self._compose_visual_prompt(project.style, scene.visual_prompt),
                estimated_duration=scene.estimated_duration,
                media_mode="video",
            )
            for index, scene in enumerate(plan.scenes)
        ]
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def regenerate_scene_visual(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
        instruction: str = "",
    ) -> ProjectDraft:
        """Regenerate one visual prompt without changing other scenes or narration."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if project.style is None:
            raise ProjectStateError("Select a style first")
        scene = self._scene(project, scene_id)
        prompt = await self.core.llm(
            f"""Write one English visual-generation prompt for this narration. Keep the project
style and continuity. Return only the prompt. Style: {self._style_prompt(project.style)}
Narration: {scene.narration}
Revision instruction: {instruction or 'Improve shot clarity and feasibility.'}""",
            temperature=0.4,
            max_tokens=700,
        )
        scene.visual_prompt = self._compose_visual_prompt(project.style, str(prompt).strip())
        scene.dirty.visual_prompt = False
        scene.dirty.template_preview = True
        scene.dirty.crop = True
        scene.dirty.media = True
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def regenerate_scene_narration(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
        instruction: str = "",
    ) -> ProjectDraft:
        """Rewrite one narration and require only that scene's visual plan to be reviewed."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if project.style is None:
            raise ProjectStateError("Select a style first")
        scene = self._scene(project, scene_id)
        response = await self.core.llm(
            f"""Rewrite only this short-video narration. Preserve factual meaning and project
continuity. Do not add unsupported claims. Style contract: {self._style_prompt(project.style)}
Current narration: {scene.narration}
Revision instruction: {instruction or 'Improve clarity, rhythm, and spoken delivery.'}""",
            response_type=RegeneratedNarration,
            temperature=0.4,
            max_tokens=700,
        )
        scene.narration = response.narration.strip()
        scene.visual_prompt = ""
        scene.dirty.visual_prompt = True
        scene.dirty.template_preview = True
        scene.dirty.media = True
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def replan_visuals(
        self,
        project_id: str,
        expected_revision: int,
    ) -> ProjectDraft:
        """Regenerate every visual prompt while preserving narration, IDs, and order."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if project.style is None or not project.scenes:
            raise ProjectStateError("Style and existing scenes are required")
        numbered = "\n".join(
            f"{index + 1}. {scene.narration}" for index, scene in enumerate(project.scenes)
        )
        response = await self.core.llm(
            f"""Write exactly {len(project.scenes)} English visual-generation prompts in order.
Preserve recurring subjects and continuity. Do not rewrite narration. Style contract:
{self._style_prompt(project.style)}
Scenes:
{numbered}""",
            response_type=VisualPromptPlan,
            temperature=0.4,
            max_tokens=4000,
        )
        if len(response.prompts) != len(project.scenes):
            raise ProjectStateError("Visual planner returned a different number of scenes")
        for scene, prompt in zip(project.scenes, response.prompts):
            scene.visual_prompt = self._compose_visual_prompt(project.style, prompt)
            scene.dirty.visual_prompt = False
            scene.dirty.template_preview = True
            scene.dirty.crop = True
            scene.dirty.media = True
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def update_scene(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
        *,
        narration: Optional[str] = None,
        visual_prompt: Optional[str] = None,
        estimated_duration: Optional[float] = None,
        media_mode: Optional[str] = None,
        asset_ids: Optional[list[str]] = None,
    ) -> ProjectDraft:
        """Edit one scene and invalidate only its dependent outputs."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        scene = self._scene(project, scene_id)
        if narration is not None and narration.strip() != scene.narration:
            scene.narration = narration.strip()
            scene.dirty.media = True
        if visual_prompt is not None and visual_prompt.strip() != scene.visual_prompt:
            scene.visual_prompt = visual_prompt.strip()
            scene.dirty.visual_prompt = False
            scene.dirty.media = True
        if estimated_duration is not None:
            if estimated_duration <= 0 or estimated_duration > 300:
                raise ValueError("Estimated duration must be greater than 0 and at most 300 seconds")
            scene.estimated_duration = estimated_duration
            scene.dirty.media = True
        if media_mode is not None:
            if media_mode not in {"static", "image", "video", "asset"}:
                raise ValueError(f"Unsupported scene media mode: {media_mode}")
            scene.media_mode = media_mode
            scene.dirty.template_preview = True
            scene.dirty.media = True
        if asset_ids is not None:
            known = {asset.asset_id for asset in project.assets}
            unknown = set(asset_ids) - known
            if unknown:
                raise ValueError(f"Unknown project assets: {', '.join(sorted(unknown))}")
            scene.asset_ids = asset_ids
            scene.dirty.crop = True
            scene.dirty.media = True
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def render_scene_preview(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
    ) -> str:
        """Render a local template preview without calling an LLM or media provider."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        scene = self._scene(project, scene_id)

        image_path: Optional[str] = None
        for candidate in (scene.last_image_path, scene.last_composed_image_path):
            if candidate and Path(candidate).is_file():
                image_path = candidate
                break
        if image_path is None:
            assets_by_id = {asset.asset_id: asset for asset in project.assets}
            for asset_id in scene.asset_ids:
                asset = assets_by_id.get(asset_id)
                if asset is None:
                    continue
                for candidate in (asset.thumbnail_path, asset.original_path):
                    if candidate and Path(candidate).is_file():
                        image_path = candidate
                        break
                if image_path:
                    break
        if image_path is None:
            image_path = str(Path("resources/example.png").resolve())

        from pixelle_video.services.frame_html import HTMLFrameGenerator
        from pixelle_video.utils.template_util import resolve_template_path

        template = scene.template_name or project.render_profile.frame_template
        generator = HTMLFrameGenerator(resolve_template_path(template))
        preview_dir = self.persistence.root / project.project_id / "previews"
        preview_path = preview_dir / f"scene-{scene.scene_id}-r{project.revision}.png"
        return await generator.generate_frame(
            title=project.title or project.original_prompt[:80],
            text=scene.narration,
            image=image_path,
            ext={"index": scene.order + 1},
            output_path=str(preview_path),
        )

    def update_title(
        self,
        project_id: str,
        expected_revision: int,
        title: str,
    ) -> ProjectDraft:
        """Edit the reviewed title without regenerating scenes."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if not title.strip():
            raise ValueError("Title cannot be empty")
        project.title = title.strip()
        for scene in project.scenes:
            scene.dirty.template_preview = True
            scene.dirty.media = True
        project.status = ProjectStatus.PLANNED if project.scenes else ProjectStatus.DRAFT
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def split_scene(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
        split_at: int,
    ) -> ProjectDraft:
        """Split narration into two stable scenes and invalidate only those visuals."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        scene = self._scene(project, scene_id)
        if split_at <= 0 or split_at >= len(scene.narration):
            raise ValueError("Split position must be inside the narration")
        left = scene.narration[:split_at].strip()
        right = scene.narration[split_at:].strip()
        if not left or not right:
            raise ValueError("Both split narrations must contain text")
        index = project.scenes.index(scene)
        scene.narration = left
        scene.visual_prompt = ""
        scene.dirty = scene.dirty.model_copy(
            update={"visual_prompt": True, "template_preview": True, "crop": True, "media": True}
        )
        project.scenes.insert(
            index + 1,
            SceneDraft(
                order=index + 1,
                narration=right,
                media_mode=scene.media_mode,
                estimated_duration=max(1.0, scene.estimated_duration / 2),
            ),
        )
        scene.estimated_duration = max(1.0, scene.estimated_duration / 2)
        self._normalize_scene_order(project)
        project.scene_count = len(project.scenes)
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def merge_scene_with_next(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
    ) -> ProjectDraft:
        """Merge one scene with its successor while preserving the first stable ID."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        scene = self._scene(project, scene_id)
        index = project.scenes.index(scene)
        if index >= len(project.scenes) - 1:
            raise ValueError("The last scene has no following scene to merge")
        following = project.scenes.pop(index + 1)
        scene.narration = f"{scene.narration}\n{following.narration}".strip()
        scene.visual_prompt = ""
        scene.asset_ids = list(dict.fromkeys(scene.asset_ids + following.asset_ids))
        scene.estimated_duration += following.estimated_duration
        scene.dirty = scene.dirty.model_copy(
            update={"visual_prompt": True, "template_preview": True, "crop": True, "media": True}
        )
        self._normalize_scene_order(project)
        project.scene_count = len(project.scenes)
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def delete_scene(
        self,
        project_id: str,
        scene_id: str,
        expected_revision: int,
    ) -> ProjectDraft:
        """Remove one scene while keeping at least one."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if len(project.scenes) <= 1:
            raise ValueError("A storyboard must keep at least one scene")
        scene = self._scene(project, scene_id)
        project.scenes.remove(scene)
        self._normalize_scene_order(project)
        project.scene_count = len(project.scenes)
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def reorder_scenes(
        self,
        project_id: str,
        scene_ids: list[str],
        expected_revision: int,
    ) -> ProjectDraft:
        """Reorder by stable IDs without replacing scene content."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if (
            len(scene_ids) != len(project.scenes)
            or set(scene_ids) != {scene.scene_id for scene in project.scenes}
        ):
            raise ValueError("Scene order must contain each current scene ID exactly once")
        by_id = {scene.scene_id: scene for scene in project.scenes}
        project.scenes = [by_id[scene_id] for scene_id in scene_ids]
        for index, scene in enumerate(project.scenes):
            scene.order = index
        project.status = ProjectStatus.PLANNED
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    def add_asset(
        self,
        project_id: str,
        expected_revision: int,
        asset: AssetRecord,
    ) -> ProjectDraft:
        """Attach an already-ingested immutable asset to a project revision."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if any(item.asset_id == asset.asset_id for item in project.assets):
            raise ValueError(f"Asset already attached: {asset.asset_id}")
        project.assets.append(
            AssetRecord.model_validate(
                {**asset.model_dump(), "fit_mode": project.render_profile.crop_strategy}
            )
        )
        project.locked_snapshot_hash = None
        project.status = ProjectStatus.PLANNED if project.scenes else ProjectStatus.DRAFT
        return self.persistence.save(project, expected_revision)

    def update_asset_presentation(
        self,
        project_id: str,
        asset_id: str,
        expected_revision: int,
        *,
        fit_mode: str,
        focal_x: float,
        focal_y: float,
    ) -> ProjectDraft:
        """Persist a reversible fit/focus decision and invalidate only dependent scenes."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        if fit_mode not in {"cover", "contain"}:
            raise ValueError("Asset fit mode must be cover or contain")
        if not 0 <= focal_x <= 1 or not 0 <= focal_y <= 1:
            raise ValueError("Asset focal coordinates must be between 0 and 1")
        for index, asset in enumerate(project.assets):
            if asset.asset_id != asset_id:
                continue
            updated = AssetRecord.model_validate(
                {
                    **asset.model_dump(),
                    "fit_mode": fit_mode,
                    "focal_x": focal_x,
                    "focal_y": focal_y,
                }
            )
            if updated == asset:
                return project
            project.assets[index] = updated
            for scene in project.scenes:
                if asset_id in scene.asset_ids:
                    scene.dirty.crop = True
                    scene.dirty.media = True
            project.status = ProjectStatus.PLANNED if project.scenes else ProjectStatus.DRAFT
            project.locked_snapshot_hash = None
            return self.persistence.save(project, expected_revision)
        raise KeyError(f"Unknown project asset: {asset_id}")

    def update_render_profile(
        self,
        project_id: str,
        expected_revision: int,
        updates: dict[str, Any],
    ) -> ProjectDraft:
        """Update production settings and apply explicit canvas invalidation rules."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        old_canvas = (
            project.render_profile.canvas_width,
            project.render_profile.canvas_height,
        )
        old_render = project.render_profile.model_copy(deep=True)
        payload = project.render_profile.model_dump()
        payload.update(updates)
        workflow_name = Path(str(payload.get("media_workflow") or "")).name.lower()
        template_name = Path(str(payload["frame_template"])).name.lower()
        if workflow_name.startswith("video_") and not template_name.startswith("video_"):
            canvas_dir = Path(f"{payload['canvas_width']}x{payload['canvas_height']}")
            compatible_template = canvas_dir / "video_default.html"
            if not (Path("templates") / compatible_template).is_file():
                raise ValueError(
                    f"No tested video template is available for {canvas_dir.name}"
                )
            payload["frame_template"] = str(compatible_template)
        elif workflow_name.startswith("image_") and template_name.startswith("video_"):
            canvas_dir = Path(f"{payload['canvas_width']}x{payload['canvas_height']}")
            compatible_template = canvas_dir / "image_full.html"
            if not (Path("templates") / compatible_template).is_file():
                compatible_template = canvas_dir / "image_default.html"
            if not (Path("templates") / compatible_template).is_file():
                raise ValueError(
                    f"No tested image template is available for {canvas_dir.name}"
                )
            payload["frame_template"] = str(compatible_template)
        if payload["frame_template"] != old_render.frame_template and not {
            "media_width",
            "media_height",
        }.intersection(updates):
            from pixelle_video.services.frame_html import HTMLFrameGenerator
            from pixelle_video.utils.template_util import resolve_template_path

            generator = HTMLFrameGenerator(resolve_template_path(str(payload["frame_template"])))
            payload["media_width"], payload["media_height"] = generator.get_media_size()
            coverage = (
                payload["media_width"] * payload["media_height"]
            ) / (payload["canvas_width"] * payload["canvas_height"])
            payload["crop_strategy"] = "cover" if coverage >= 0.8 else "contain"
        render = RenderProfile.model_validate(payload)
        render_changed = old_render.model_dump() != render.model_dump()
        if not render_changed:
            return project
        new_canvas = (render.canvas_width, render.canvas_height)
        if new_canvas not in self.CANVASES:
            raise ValueError("Only tested 1080x1920, 1920x1080, and 1080x1080 canvases are allowed")
        template_changed = old_render.frame_template != render.frame_template
        workflow_changed = old_render.media_workflow != render.media_workflow
        segment_contract_changed = any(
            getattr(old_render, field) != getattr(render, field)
            for field in (
                "video_fps",
                "tts_inference_mode",
                "voice_id",
                "tts_workflow",
                "tts_speed",
            )
        )
        if old_canvas != new_canvas or template_changed:
            for scene in project.scenes:
                scene.dirty.visual_prompt = True
                scene.dirty.template_preview = True
                scene.dirty.crop = True
                scene.dirty.media = True
        elif workflow_changed or segment_contract_changed:
            for scene in project.scenes:
                scene.dirty.media = True
        if project.style:
            project.style = self._derive_project_style(
                project,
                project.style,
                render,
                expected_revision + 1,
            )
            render.style_id = project.style.style_id
            render.style_version = project.style.version
            render.style_hash = project.style.validation_hash()
        render.template_hash = None
        render.workflow_hash = None
        render.environment_fingerprint = None
        project.render_profile = render
        project.status = ProjectStatus.PLANNED if project.scenes else ProjectStatus.DRAFT
        project.locked_snapshot_hash = None
        return self.persistence.save(project, expected_revision)

    async def inspect_compatibility(
        self,
        project_id: str,
        expected_revision: int,
    ) -> tuple[ProjectDraft, Any]:
        """Refresh exact hashes and read-only runtime readiness."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        render, record = await self.compatibility.inspect(project)
        project.render_profile = render
        project.locked_snapshot_hash = None
        if project.status == ProjectStatus.LOCKED:
            project.status = ProjectStatus.PLANNED
        return self.persistence.save(project, expected_revision), record

    async def prepare_for_production(
        self,
        project_id: str,
        expected_revision: int,
    ) -> tuple[ProjectDraft, Any]:
        """Inspect and lock with one atomic project revision on success."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        self._validate_lock_content(project)
        render, record = await self.compatibility.inspect(project)
        project.render_profile = render
        self.compatibility.require_runtime_ready(render)
        return self._finalize_lock(project, expected_revision), record

    def lock(self, project_id: str, expected_revision: int) -> ProjectDraft:
        """Freeze the reviewed script, storyboard, style, and render contract."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        self._ensure_editable(project)
        self._validate_lock_content(project)
        render = project.render_profile
        if not all(
            [render.style_hash, render.template_hash, render.workflow_hash, render.environment_fingerprint]
        ):
            raise ProjectStateError("Run compatibility inspection before locking")
        self.compatibility.require_runtime_ready(render)
        return self._finalize_lock(project, expected_revision)

    def _validate_lock_content(self, project: ProjectDraft) -> None:
        """Validate reviewed content without mutating the draft or checking the runtime."""
        if project.style is None or not project.title or not project.scenes:
            raise ProjectStateError("Style, title, and storyboard are required before locking")
        if not all(scene.narration.strip() and scene.visual_prompt.strip() for scene in project.scenes):
            raise ProjectStateError("Every scene needs narration and a visual prompt")
        if any(scene.dirty.visual_prompt for scene in project.scenes):
            raise ProjectStateError(
                "Visual planning changed and must be regenerated or explicitly edited before locking"
            )
        assets_by_id = {asset.asset_id: asset for asset in project.assets}
        for scene in project.scenes:
            if scene.media_mode != "asset":
                continue
            if not scene.asset_ids:
                raise ProjectStateError(
                    f"Scene {scene.order + 1} uses project assets but none are selected"
                )
            asset = assets_by_id.get(scene.asset_ids[0])
            if asset is None or asset.media_kind not in {"image", "video"}:
                raise ProjectStateError(
                    f"Scene {scene.order + 1} requires a valid image or video asset"
                )
            if not Path(asset.original_path).is_file():
                raise ProjectStateError(
                    f"Scene {scene.order + 1} asset original is missing"
                )

    def _finalize_lock(
        self,
        project: ProjectDraft,
        expected_revision: int,
    ) -> ProjectDraft:
        """Persist one immutable production snapshot after all gates have passed."""
        for scene in project.scenes:
            scene.dirty.visual_prompt = False
            scene.dirty.template_preview = False
            scene.dirty.crop = False
        project.status = ProjectStatus.LOCKED
        project.locked_snapshot_hash = project.snapshot_hash()
        return self.persistence.save(project, expected_revision)

    async def produce(self, project_id: str, expected_revision: int) -> dict[str, Any]:
        """Produce only from the locked snapshot; planning providers are never called here."""
        project = self.get(project_id)
        self._expect_revision(project, expected_revision)
        if project.status != ProjectStatus.LOCKED:
            raise ProjectStateError("Project must be locked before production")
        if not project.locked_snapshot_hash or project.locked_snapshot_hash != project.snapshot_hash():
            raise ProjectStateError("Locked project snapshot no longer matches current content")
        self.compatibility.require_runtime_ready(project.render_profile)
        snapshot_hash = project.locked_snapshot_hash
        render = project.render_profile
        assets_by_id = {asset.asset_id: asset for asset in project.assets}
        planned_prompts: list[Optional[str]] = []
        planned_media: list[dict[str, Any]] = []
        for scene in project.scenes:
            media: dict[str, Any] = {"source_asset_ids": list(scene.asset_ids)}
            if scene.media_mode == "asset":
                if not scene.asset_ids:
                    raise ProjectStateError(
                        f"Scene {scene.order + 1} uses project assets but none are selected"
                    )
                asset = assets_by_id.get(scene.asset_ids[0])
                if asset is None:
                    raise ProjectStateError(
                        f"Scene {scene.order + 1} references an unknown project asset"
                    )
                if asset.media_kind not in {"image", "video"}:
                    raise ProjectStateError(
                        f"Scene {scene.order + 1} requires an image or video asset"
                    )
                if not Path(asset.original_path).is_file():
                    raise ProjectStateError(
                        f"Scene {scene.order + 1} asset original is missing"
                    )
                media[f"{asset.media_kind}_path"] = self.core.assets.build_render_variant(
                    asset,
                    render.media_width,
                    render.media_height,
                )
                media["media_type"] = asset.media_kind
                planned_prompts.append(None)
            elif scene.media_mode == "static":
                planned_prompts.append(None)
            else:
                planned_prompts.append(scene.visual_prompt)
            planned_media.append(media)
        reusable_frames = {
            scene.scene_id: {
                "audio_path": scene.last_audio_path,
                "image_path": scene.last_image_path,
                "video_path": scene.last_video_path,
                "composed_image_path": scene.last_composed_image_path,
                "video_segment_path": scene.last_segment_path,
                "media_type": "video" if scene.last_video_path else "image",
                "duration": scene.last_render_duration or scene.estimated_duration,
            }
            for scene in project.scenes
            if not scene.dirty.media
            and scene.last_segment_path
            and Path(scene.last_segment_path).is_file()
        }
        project.status = ProjectStatus.PRODUCING
        producing = self.persistence.save(project, expected_revision)
        render = producing.render_profile
        try:
            result = await self.core.pipelines["standard"](
                text="\n".join(scene.narration for scene in producing.scenes),
                mode="fixed",
                title=producing.title,
                planned_narrations=[scene.narration for scene in producing.scenes],
                planned_image_prompts=planned_prompts,
                planned_media=planned_media,
                scene_ids=[scene.scene_id for scene in producing.scenes],
                reusable_frames=reusable_frames,
                prompt_prefix="",
                n_scenes=len(producing.scenes),
                frame_template=render.frame_template,
                media_workflow=render.media_workflow,
                media_width=render.media_width,
                media_height=render.media_height,
                video_fps=render.video_fps,
                tts_inference_mode=render.tts_inference_mode,
                voice_id=render.voice_id,
                tts_voice=render.voice_id,
                tts_workflow=render.tts_workflow,
                tts_speed=render.tts_speed,
                bgm_path=render.bgm_path,
                bgm_volume=render.bgm_volume,
                bgm_mode=render.bgm_mode,
                project_id=producing.project_id,
                project_revision=producing.revision,
                project_snapshot_hash=snapshot_hash,
                style_id=producing.style.style_id if producing.style else None,
                style_version=producing.style.version if producing.style else None,
                style_hash=render.style_hash,
                render_combination_hash=render.combination_hash(),
            )
        except asyncio.CancelledError:
            failed = self.get(project_id)
            failed.status = ProjectStatus.FAILED
            failed.last_error = "Production was cancelled"
            self.persistence.save(failed, failed.revision)
            raise
        except Exception as exc:
            failed = self.get(project_id)
            failed.status = ProjectStatus.FAILED
            failed.last_error = str(exc)
            self.persistence.save(failed, failed.revision)
            raise
        completed = self.get(project_id)
        completed.status = ProjectStatus.COMPLETED
        completed.last_task_id = result.storyboard.config.task_id
        completed.last_error = None
        result_frames = {
            frame.scene_id: frame for frame in result.storyboard.frames if frame.scene_id
        }
        for scene in completed.scenes:
            frame = result_frames.get(scene.scene_id)
            if frame is None:
                continue
            scene.last_render_task_id = result.storyboard.config.task_id
            scene.last_audio_path = frame.audio_path
            scene.last_image_path = frame.image_path
            scene.last_video_path = frame.video_path
            scene.last_composed_image_path = frame.composed_image_path
            scene.last_segment_path = frame.video_segment_path
            scene.last_render_duration = frame.duration
            scene.dirty.media = False
        saved = self.persistence.save(completed, completed.revision)
        return {
            "project": saved,
            "task_id": result.storyboard.config.task_id,
            "video_path": result.video_path,
            "duration": result.duration,
            "file_size": result.file_size,
        }

    def _recover_interrupted_projects(self) -> None:
        """Make process-restart interruption explicit instead of leaving a false running state."""
        for project in self.persistence.list(limit=10000):
            if project.status != ProjectStatus.PRODUCING:
                continue
            project.status = ProjectStatus.FAILED
            project.last_error = "Production was interrupted by an application restart"
            self.persistence.save(project, project.revision)

    @staticmethod
    def _expect_revision(project: ProjectDraft, expected_revision: int) -> None:
        if project.revision != expected_revision:
            from pixelle_video.services.project_persistence import RevisionConflictError

            raise RevisionConflictError(expected_revision, project.revision)

    @staticmethod
    def _ensure_editable(project: ProjectDraft) -> None:
        if project.status == ProjectStatus.PRODUCING:
            raise ProjectStateError("A project cannot be edited while production is running")

    @staticmethod
    def _scene(project: ProjectDraft, scene_id: str) -> SceneDraft:
        for scene in project.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"Unknown scene: {scene_id}")

    @staticmethod
    def _normalize_scene_order(project: ProjectDraft) -> None:
        for index, scene in enumerate(project.scenes):
            scene.order = index

    @staticmethod
    def _style_prompt(style: StyleProfile) -> str:
        return (
            f"tone={style.script_tone}; pacing={style.narrative_pacing}; "
            f"visual={style.visual_prompt_prefix}; colors={'; '.join(style.color_rules)}; "
            f"lighting={'; '.join(style.lighting_rules)}; camera={'; '.join(style.camera_rules)}; "
            f"characters={'; '.join(style.character_rules)}; negative={style.negative_prompt}"
        )

    @staticmethod
    def _compose_visual_prompt(style: StyleProfile, prompt: str) -> str:
        negative = f" Negative prompt: {style.negative_prompt}." if style.negative_prompt else ""
        return f"{style.visual_prompt_prefix}. {prompt.strip()}.{negative}".strip()

    @staticmethod
    def _apply_recommended_render_defaults(render: RenderProfile, style: StyleProfile) -> None:
        template_roots = {
            "9:16": Path("templates/1080x1920"),
            "16:9": Path("templates/1920x1080"),
            "1:1": Path("templates/1080x1080"),
        }
        aspect = style.recommended_aspect_ratios[0] if style.recommended_aspect_ratios else "9:16"
        canvas = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}.get(
            aspect,
            (1080, 1920),
        )
        root = template_roots.get(aspect, template_roots["9:16"])
        existing_workflows = [
            item for item in style.allowed_workflows if (Path("workflows") / item).is_file()
        ]
        render.media_workflow = existing_workflows[0] if existing_workflows else None
        is_video_workflow = bool(
            render.media_workflow
            and Path(render.media_workflow).name.lower().startswith("video_")
        )
        candidates = []
        for family in style.recommended_template_families:
            if is_video_workflow and not family.startswith("video_"):
                continue
            candidates.extend(root.glob(f"{family}.html"))
        if not candidates and is_video_workflow:
            candidates = list(root.glob("video_default.html")) or list(root.glob("video_*.html"))
        elif not candidates:
            candidates = list(root.glob("image_full.html")) or list(root.glob("image_*.html"))
        render.canvas_width, render.canvas_height = canvas
        if candidates:
            render.frame_template = str(candidates[0].relative_to("templates"))
            from pixelle_video.services.frame_html import HTMLFrameGenerator

            generator = HTMLFrameGenerator(candidates[0])
            render.media_width, render.media_height = generator.get_media_size()
            coverage = (render.media_width * render.media_height) / (
                render.canvas_width * render.canvas_height
            )
            render.crop_strategy = "cover" if coverage >= 0.8 else "contain"
        if style.recommended_voice_ids:
            render.voice_id = style.recommended_voice_ids[0]
        render.tts_speed = sum(style.tts_speed_range) / 2
        render.bgm_volume = sum(style.bgm_volume_range) / 2

    @staticmethod
    def _derive_project_style(
        project: ProjectDraft,
        style: StyleProfile,
        render: RenderProfile,
        revision: int,
    ) -> StyleProfile:
        """Save advanced overrides as a project-owned profile, never mutate a system preset."""
        derived = style.model_copy(deep=True)
        derived.style_id = f"project-{project.project_id}"
        derived.version = f"1.0.{revision}"
        derived.name = f"{style.name} · project override"
        derived.builtin = False
        derived.source_description = f"Derived from {style.style_id}@{style.version}"
        ratio = {
            (1080, 1920): "9:16",
            (1920, 1080): "16:9",
            (1080, 1080): "1:1",
        }[(render.canvas_width, render.canvas_height)]
        derived.recommended_aspect_ratios = [ratio]
        derived.recommended_template_families = [Path(render.frame_template).stem]
        derived.allowed_workflows = [render.media_workflow] if render.media_workflow else []
        derived.recommended_voice_ids = [render.voice_id] if render.voice_id else []
        derived.tts_speed_range = (render.tts_speed, render.tts_speed)
        derived.bgm_volume_range = (render.bgm_volume, render.bgm_volume)
        return derived
