"""Style catalog and AI-assisted custom style creation."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from pixelle_video.models.project import StyleProfile, utc_now
from pixelle_video.style_profiles import get_builtin_styles


class CustomStyleDefinition(BaseModel):
    """Provider-friendly schema for converting natural language into a style."""

    name: str
    description: str
    suitable_for: list[str] = Field(min_length=1, max_length=6)
    script_tone: str
    narrative_pacing: str
    wording_rules: list[str]
    visual_prompt_prefix: str
    negative_prompt: str
    color_rules: list[str]
    lighting_rules: list[str]
    composition_rules: list[str]
    material_rules: list[str]
    character_rules: list[str]
    camera_rules: list[str]
    action_intensity: str
    shot_pacing: str
    transition_suggestions: list[str]
    recommended_template_families: list[str]
    recommended_aspect_ratios: list[str]
    allowed_workflows: list[str]
    recommended_voice_ids: list[str]
    tts_speed_min: float
    tts_speed_max: float
    bgm_tags: list[str]
    bgm_volume_min: float
    bgm_volume_max: float
    palette: list[str]
    cost_tier: str
    preview_prompts: list[str] = Field(min_length=3, max_length=3)


class StyleService:
    """Load built-ins and persist custom profiles independently of projects."""

    def __init__(self, llm: Any = None, root: str | Path = "data/styles"):
        self.llm = llm
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_styles(self) -> list[StyleProfile]:
        """Return built-in and valid custom styles."""
        styles = get_builtin_styles()
        for path in sorted(self.root.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    styles.append(StyleProfile.model_validate(json.load(handle)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return styles

    def get_style(self, style_id: str) -> StyleProfile:
        """Get a style without exposing a mutable system preset."""
        for style in self.list_styles():
            if style.style_id == style_id:
                return style.model_copy(deep=True)
        raise KeyError(f"Unknown style profile: {style_id}")

    async def create_custom_style(self, description: str) -> tuple[StyleProfile, list[str]]:
        """Convert a natural-language brief to a structured style and three preview prompts."""
        if self.llm is None:
            raise RuntimeError("LLM service is required to create a custom style")
        prompt = f"""You are a senior video art director. Convert the user's description into a
cohesive project-level video style. Be specific enough to keep many scenes consistent. Prefer one
or more video workflows because planned scenes default to AI video. Only list workflow paths from
this set when appropriate: selfhost/video_minimax_h3_t2v.json,
selfhost/video_wan2.1_fusionx.json, runninghub/video_wan2.1_fusionx.json,
runninghub/video_wan2.2.json,
runninghub/video_Z_image_wan2.2.json, runninghub/video_qwen_wan2.2.json,
selfhost/image_flux.json, selfhost/image_qwen.json, selfhost/image_nano_banana.json,
runninghub/image_flux.json, runninghub/image_qwen.json,
runninghub/image_qwen_chinese_cartoon.json. Create exactly three English preview prompts showing
the same style with portrait, environment, and object/detail subjects. User description:
{description}"""
        generated = await self.llm(
            prompt,
            response_type=CustomStyleDefinition,
            temperature=0.4,
            max_tokens=8192,
        )
        slug = re.sub(r"[^a-z0-9]+", "-", generated.name.lower()).strip("-") or "custom"
        style_id = f"custom-{slug}-{int(utc_now().timestamp())}"
        style = StyleProfile(
            style_id=style_id,
            name=generated.name,
            description=generated.description,
            suitable_for=generated.suitable_for,
            script_tone=generated.script_tone,
            narrative_pacing=generated.narrative_pacing,
            wording_rules=generated.wording_rules,
            visual_prompt_prefix=generated.visual_prompt_prefix,
            negative_prompt=generated.negative_prompt,
            color_rules=generated.color_rules,
            lighting_rules=generated.lighting_rules,
            composition_rules=generated.composition_rules,
            material_rules=generated.material_rules,
            character_rules=generated.character_rules,
            camera_rules=generated.camera_rules,
            action_intensity=generated.action_intensity,
            shot_pacing=generated.shot_pacing,
            transition_suggestions=generated.transition_suggestions,
            recommended_template_families=generated.recommended_template_families,
            recommended_aspect_ratios=generated.recommended_aspect_ratios,
            allowed_workflows=generated.allowed_workflows,
            recommended_voice_ids=generated.recommended_voice_ids,
            tts_speed_range=(generated.tts_speed_min, generated.tts_speed_max),
            bgm_tags=generated.bgm_tags,
            bgm_volume_range=(generated.bgm_volume_min, generated.bgm_volume_max),
            palette=generated.palette,
            cost_tier=generated.cost_tier,
            builtin=False,
            source_description=description,
            preview_prompts=generated.preview_prompts,
        )
        self._save(style)
        return style, generated.preview_prompts

    async def generate_previews(
        self,
        style_id: str,
        media_service: Any,
        workflow: str,
    ) -> StyleProfile:
        """Generate exactly three explicit style previews after the user requests production."""
        style = self.get_style(style_id)
        if workflow not in style.allowed_workflows:
            raise ValueError("Preview workflow is not allowed by this style")
        if not style.preview_prompts:
            subjects = [
                "a person in a simple environment",
                "a wide environmental establishing shot",
                "a close-up object detail",
            ]
            style.preview_prompts = [
                f"{style.visual_prompt_prefix}, {subject}" for subject in subjects
            ]
        preview_dir = self.root / "previews" / style.style_id / style.version
        preview_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = []
        for index, prompt in enumerate(style.preview_prompts[:3], start=1):
            output_path = preview_dir / f"preview-{index}.png"
            result = await media_service(
                prompt=prompt,
                workflow=workflow,
                media_type="image",
                width=768,
                height=768,
                negative_prompt=style.negative_prompt,
                output_path=str(output_path),
                index=index,
            )
            if output_path.is_file():
                generated_paths.append(str(output_path))
                continue
            source = Path(result.url)
            if source.is_file():
                shutil.copy2(source, output_path)
            elif result.url.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.get(result.url)
                    response.raise_for_status()
                temp_path = output_path.with_suffix(".partial")
                temp_path.write_bytes(response.content)
                os.replace(temp_path, output_path)
            else:
                raise RuntimeError("Preview provider did not return a readable image")
            generated_paths.append(str(output_path))
        style.preview_images = generated_paths
        if not style.builtin:
            self._save(style)
        return style

    def _save(self, style: StyleProfile) -> None:
        path = self.root / f"{style.style_id}.json"
        temp_path = path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(style.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
