"""Staged creation workspace for style, script, storyboard, and production locking."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from pixelle_video.config import config_manager
from pixelle_video.config.schema import WebConfig
from pixelle_video.models.project import CompatibilityStatus, ProjectStatus
from pixelle_video.services.llm_service import StructuredOutputError
from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name
from web.components.content_input import (
    render_bgm_section,
    render_content_input,
    render_version_info,
)
from web.components.output_preview import render_output_preview
from web.components.style_config import get_template_preview_path, render_style_config
from web.i18n import get_language, tr
from web.pipelines.base import PipelineUI
from web.utils.async_helpers import run_async


def _scene_expander_label(scene: Any) -> str:
    """Build a compact shot summary that remains useful while the editor is collapsed."""
    narration = " ".join(scene.narration.split())
    if len(narration) > 48:
        narration = f"{narration[:47]}…"
    state = (
        tr("staged.scene_needs_review")
        if scene.dirty.visual_prompt or not scene.visual_prompt
        else tr("staged.scene_ready")
    )
    return (
        f"{tr('staged.scene')} {scene.order + 1} · "
        f"{scene.estimated_duration:g}s · {_media_mode_label(scene.media_mode)} · "
        f"{state} · {narration}"
    )


def _media_mode_label(media_mode: str) -> str:
    """Return a localized media-source label without exposing internal enum values."""
    return tr(f"staged.media_mode.{media_mode}", fallback=media_mode)


def _resource_name(value: str | None) -> str:
    """Turn template/workflow identifiers into concise creator-facing names."""
    if not value:
        return tr("staged.none")
    stem = Path(value).stem
    fallback = stem.replace("_", " ").strip().title()
    return tr(f"staged.resource.{stem}", fallback=fallback)


def _voice_name(voice_id: str | None) -> str:
    """Display a localized voice name while preserving the provider ID internally."""
    if not voice_id:
        return tr("staged.none")
    return get_voice_display_name(voice_id, tr, get_language())


def _style_name(name: str) -> str:
    """Hide persistence provenance suffixes from creator-facing style names."""
    return name.removesuffix(" · project override")


def _project_duration(project: Any) -> float:
    if project.status == ProjectStatus.COMPLETED:
        rendered = [scene.last_render_duration for scene in project.scenes]
        if rendered and all(value is not None and value > 0 for value in rendered):
            return round(sum(float(value) for value in rendered), 1)
    return sum(float(scene.estimated_duration) for scene in project.scenes)


def _runtime_ready(record: Any) -> bool:
    """Treat legacy technical/approval records as runtime-ready without requiring them."""
    return bool(
        record
        and record.status
        in {
            CompatibilityStatus.RUNTIME_READY,
            CompatibilityStatus.TECHNICAL_PASS,
            CompatibilityStatus.APPROVED,
        }
    )


class StagedCreationUI(PipelineUI):
    """Project-first creation flow; media production is the last gated step."""

    name = "quick_create"
    icon = "🎬"

    @property
    def display_name(self):
        return tr("pipeline.quick_create.name")

    @property
    def description(self):
        return tr("pipeline.quick_create.description")

    def render(self, pixelle_video: Any):
        web_config = getattr(config_manager.config, "web", WebConfig())
        if web_config.enable_legacy_quick_create and web_config.creation_flow == "legacy":
            self._render_legacy(pixelle_video)
            return

        st.caption(tr("staged.flow_caption"))
        project = self._project_header(pixelle_video)
        if project is None:
            return

        if project.scenes:
            st.caption(
                tr(
                    "staged.project_progress",
                    status=tr(f"staged.status.{project.status.value}"),
                    scenes=len(project.scenes),
                    seconds=f"{_project_duration(project):g}",
                )
            )
        else:
            st.caption(
                tr(
                    "staged.project_draft_progress",
                    status=tr(f"staged.status.{project.status.value}"),
                    scenes=project.scene_count,
                )
            )
        steps = [
            tr("staged.step.brief"),
            tr("staged.step.style"),
            tr("staged.step.script"),
            tr("staged.step.storyboard"),
            tr("staged.step.generate"),
        ]
        step_key = f"staged-step:{project.project_id}"
        requested_key = f"staged-step-request:{project.project_id}"
        requested_step = st.session_state.pop(requested_key, None)
        if requested_step is not None:
            st.session_state[step_key] = requested_step
        if step_key not in st.session_state:
            st.session_state[step_key] = self._default_step(project)
        active_step = st.segmented_control(
            tr("staged.current_step"),
            options=range(len(steps)),
            format_func=lambda index: steps[index],
            key=step_key,
            label_visibility="collapsed",
            selection_mode="single",
        )
        if active_step == 0:
            self._render_brief(pixelle_video, project)
        elif active_step == 1:
            self._render_styles(pixelle_video, project)
        elif active_step == 2:
            self._render_script(pixelle_video, project)
        elif active_step == 3:
            self._render_storyboard(pixelle_video, project)
        else:
            if self._render_production(pixelle_video, project):
                self._render_generate(pixelle_video, project)

    @staticmethod
    def _default_step(project: Any) -> int:
        if project.status in {
            ProjectStatus.LOCKED,
            ProjectStatus.PRODUCING,
            ProjectStatus.COMPLETED,
            ProjectStatus.FAILED,
        }:
            return 4
        if project.scenes:
            return 3
        if project.style:
            return 2
        return 0

    @staticmethod
    def _go_to_step(project_id: str, step: int) -> None:
        st.session_state[f"staged-step-request:{project_id}"] = step
        st.rerun()

    @staticmethod
    def _run_action(label: str, action: Callable[[], Any]) -> Any:
        try:
            with st.spinner(label):
                return action()
        except Exception as exc:
            if isinstance(exc, StructuredOutputError):
                st.error(tr(f"staged.llm_error.{exc.reason}"))
            else:
                st.error(str(exc))
            return None

    def _project_header(self, pixelle_video: Any):
        projects = pixelle_video.projects.list()
        left, right = st.columns([3, 1])
        with left:
            if projects:
                labels = {
                    item.project_id: (
                        f"{item.title or item.original_prompt[:36]} · "
                        f"{tr(f'staged.status.{item.status.value}')}"
                    )
                    for item in projects
                }
                ids = [item.project_id for item in projects]
                current_id = st.session_state.get("staged_project_id")
                index = ids.index(current_id) if current_id in ids else 0
                selected_id = st.selectbox(
                    tr("staged.project_select"),
                    ids,
                    index=index,
                    format_func=lambda value: labels[value],
                )
                st.session_state.staged_project_id = selected_id
            else:
                selected_id = None
                st.info(tr("staged.no_projects"))
        with right:
            if st.button(tr("staged.new_project"), use_container_width=True):
                st.session_state.staged_show_create = True
                st.rerun()
        if st.session_state.get("staged_show_create") or selected_id is None:
            self._render_create_project(pixelle_video, can_cancel=selected_id is not None)
            return None
        return pixelle_video.projects.get(selected_id)

    def _render_create_project(self, pixelle_video: Any, *, can_cancel: bool) -> None:
        heading, action = st.columns([4, 1])
        heading.subheader(tr("staged.create_title"))
        if can_cancel and action.button(
            tr("staged.cancel_create"),
            use_container_width=True,
        ):
            st.session_state.staged_show_create = False
            st.rerun()
        with st.form("staged_create_project"):
            brief = st.text_area(
                tr("staged.brief_label"),
                height=160,
                placeholder=tr("staged.brief_placeholder"),
            )
            scene_count = st.slider(tr("staged.scene_count"), 1, 20, 5)
            submitted = st.form_submit_button(
                tr("staged.create_button"),
                type="primary",
                use_container_width=True,
            )
        if submitted:
            project = self._run_action(
                tr("staged.creating"),
                lambda: pixelle_video.projects.create(brief, scene_count),
            )
            if project:
                st.session_state.staged_project_id = project.project_id
                st.session_state.staged_show_create = False
                st.rerun()

    def _render_brief(self, pixelle_video: Any, project: Any) -> None:
        heading, next_step = st.columns([3, 1])
        heading.subheader(tr("staged.brief_title"))
        if next_step.button(
            tr("staged.next_style"),
            type="primary",
            use_container_width=True,
        ):
            self._go_to_step(project.project_id, 1)
        st.text_area(
            tr("staged.original_prompt"),
            value=project.original_prompt,
            height=140,
            disabled=True,
        )
        roles = {
            "short_video_director": tr("staged.role.director"),
            "copywriter": tr("staged.role.copywriter"),
            "storyboard_director": tr("staged.role.storyboard"),
            "custom": tr("staged.role.custom"),
        }
        role = st.selectbox(
            tr("staged.optimize_role"),
            list(roles),
            format_func=lambda value: roles[value],
        )
        custom_role = st.text_input(tr("staged.custom_role")) if role == "custom" else None
        if st.button(tr("staged.optimize_button")):
            proposal = self._run_action(
                tr("staged.optimizing"),
                lambda: run_async(
                    pixelle_video.projects.optimize_prompt(
                        project.project_id,
                        project.revision,
                        role,
                        custom_role,
                    )
                ),
            )
            if proposal:
                st.session_state[f"proposal:{project.project_id}"] = proposal.model_dump()
                st.session_state[f"proposal_role:{project.project_id}"] = (
                    custom_role if role == "custom" else role
                )
        if project.style is None:
            st.caption(tr("staged.optimize_without_style"))
        proposal_data = st.session_state.get(f"proposal:{project.project_id}")
        if proposal_data:
            optimized = proposal_data["optimized_prompt"]
            st.markdown(tr("staged.diff_title"))
            diff = "\n".join(
                difflib.unified_diff(
                    project.original_prompt.splitlines(),
                    optimized.splitlines(),
                    fromfile=tr("staged.before"),
                    tofile=tr("staged.after"),
                    lineterm="",
                )
            )
            st.code(diff or optimized, language="diff")
            if proposal_data.get("improvements"):
                st.caption(" · ".join(proposal_data["improvements"]))
            left, right = st.columns(2)
            if left.button(tr("staged.apply_optimization"), type="primary"):
                updated = self._run_action(
                    tr("staged.saving"),
                    lambda: pixelle_video.projects.apply_optimized_prompt(
                        project.project_id,
                        project.revision,
                        optimized,
                        st.session_state.get(f"proposal_role:{project.project_id}", role),
                    ),
                )
                if updated:
                    st.session_state.pop(f"proposal:{project.project_id}", None)
                    st.rerun()
            if right.button(tr("staged.discard_optimization")):
                st.session_state.pop(f"proposal:{project.project_id}", None)
                st.rerun()
        if project.optimized_prompt:
            st.success(tr("staged.optimized_applied"))
            st.text_area(
                tr("staged.applied_prompt"),
                value=project.optimized_prompt,
                height=140,
                disabled=True,
            )
    def _render_styles(self, pixelle_video: Any, project: Any) -> None:
        heading, next_step = st.columns([3, 1])
        heading.subheader(tr("staged.style_title"))
        if project.style and next_step.button(
            tr("staged.next_script"),
            type="primary",
            use_container_width=True,
        ):
            self._go_to_step(project.project_id, 2)
        if project.style:
            st.success(tr("staged.current_style", style=_style_name(project.style.name)))
        styles = pixelle_video.styles.list_styles()
        for start in range(0, len(styles), 4):
            for column, style in zip(st.columns(4), styles[start:start + 4]):
                with column, st.container(border=True):
                    selected = bool(project.style and project.style.style_id == style.style_id)
                    preview_path = next(
                        (Path(item) for item in style.preview_images if Path(item).is_file()),
                        None,
                    )
                    if preview_path:
                        preview_column = st.columns([1, 2, 1])[1]
                        preview_column.image(str(preview_path), width="stretch")
                    st.markdown(f"### {'✓ ' if selected else ''}{_style_name(style.name)}")
                    st.caption(style.description)
                    st.caption(
                        f"{tr('staged.suitable_for')}: {', '.join(style.suitable_for)}"
                    )
                    st.caption(
                        f"{tr('staged.ratio')}: {', '.join(style.recommended_aspect_ratios)} · "
                        f"{tr('staged.cost')}: {tr(f'staged.cost.{style.cost_tier}')}"
                    )
                    workflows = [
                        item
                        for item in style.allowed_workflows
                        if (Path("workflows") / item).is_file()
                    ]
                    if not workflows:
                        st.error(tr("staged.style_missing_workflow"))
                    else:
                        st.caption(tr("staged.style_ready_for_creation"))
                    if st.button(
                        tr("staged.style_selected") if selected else tr("staged.select_style"),
                        key=f"style:{style.style_id}",
                        disabled=selected,
                        use_container_width=True,
                    ):
                        if self._run_action(
                            tr("staged.saving"),
                            lambda style_id=style.style_id: pixelle_video.projects.apply_style(
                                project.project_id,
                                style_id,
                                project.revision,
                            ),
                        ):
                            st.rerun()
        with st.expander(tr("staged.custom_style_title")):
            description = st.text_area(tr("staged.custom_style_description"))
            if st.button(tr("staged.custom_style_create"), disabled=not description.strip()):
                result = self._run_action(
                    tr("staged.custom_style_building"),
                    lambda: run_async(pixelle_video.styles.create_custom_style(description)),
                )
                if result:
                    style, preview_prompts = result
                    st.session_state[f"style_preview_prompts:{style.style_id}"] = preview_prompts
                    if pixelle_video.projects.apply_style(
                        project.project_id,
                        style.style_id,
                        project.revision,
                    ):
                        st.rerun()
        if project.style:
            prompts = project.style.preview_prompts or st.session_state.get(
                f"style_preview_prompts:{project.style.style_id}"
            )
            if prompts:
                st.caption(tr("staged.preview_prompts"))
                for prompt in prompts:
                    st.code(prompt)
                preview_workflows = [
                    item
                    for item in project.style.allowed_workflows
                    if (Path("workflows") / item).is_file() and "image_" in Path(item).name
                ]
                if preview_workflows and st.button(tr("staged.generate_style_previews")):
                    generated = self._run_action(
                        tr("staged.generating_style_previews"),
                        lambda: run_async(
                            pixelle_video.styles.generate_previews(
                                project.style.style_id,
                                pixelle_video.media,
                                preview_workflows[0],
                            )
                        ),
                    )
                    if generated:
                        for image_path in generated.preview_images:
                            st.image(image_path, use_container_width=True)
            if project.style.preview_images:
                for image_path in project.style.preview_images:
                    if Path(image_path).is_file():
                        st.image(image_path, use_container_width=True)
    def _render_script(self, pixelle_video: Any, project: Any) -> None:
        st.subheader(tr("staged.script_title"))
        if project.style is None:
            st.warning(tr("staged.select_style_first"))
            return
        if not project.scenes:
            st.write(tr("staged.plan_explanation"))
            if st.button(tr("staged.plan_button"), type="primary"):
                if self._run_action(
                    tr("staged.planning"),
                    lambda: run_async(
                        pixelle_video.projects.plan(project.project_id, project.revision)
                    ),
                ):
                    st.rerun()
            return
        summary, next_step = st.columns([3, 1])
        summary.info(
            tr(
                "staged.script_summary",
                scenes=len(project.scenes),
                seconds=f"{_project_duration(project):g}",
            )
        )
        if next_step.button(
            tr("staged.next_storyboard"),
            type="primary",
            use_container_width=True,
        ):
            self._go_to_step(project.project_id, 3)
        title = st.text_input(
            tr("staged.video_title"),
            value=project.title or "",
            key=f"title:{project.project_id}:{project.revision}",
        )
        if title != project.title and st.button(tr("staged.save_title")):
            if self._run_action(
                tr("staged.saving"),
                lambda: pixelle_video.projects.update_title(
                    project.project_id,
                    project.revision,
                    title,
                ),
            ):
                st.rerun()
        for scene in project.scenes:
            with st.expander(f"{scene.order + 1}. {scene.narration[:42]}"):
                narration = st.text_area(
                    tr("staged.narration"),
                    value=scene.narration,
                    key=f"narration:{scene.scene_id}:{project.revision}",
                )
                controls = st.columns(6)
                if controls[0].button(tr("staged.save"), key=f"save-script:{scene.scene_id}"):
                    if self._run_action(
                        tr("staged.saving"),
                        lambda scene=scene, narration=narration: pixelle_video.projects.update_scene(
                            project.project_id,
                            scene.scene_id,
                            project.revision,
                            narration=narration,
                        ),
                    ):
                        st.rerun()
                if controls[1].button("↑", key=f"up:{scene.scene_id}", disabled=scene.order == 0):
                    ids = [item.scene_id for item in project.scenes]
                    ids[scene.order - 1], ids[scene.order] = ids[scene.order], ids[scene.order - 1]
                    if pixelle_video.projects.reorder_scenes(project.project_id, ids, project.revision):
                        st.rerun()
                if controls[2].button(
                    "↓",
                    key=f"down:{scene.scene_id}",
                    disabled=scene.order == len(project.scenes) - 1,
                ):
                    ids = [item.scene_id for item in project.scenes]
                    ids[scene.order + 1], ids[scene.order] = ids[scene.order], ids[scene.order + 1]
                    if pixelle_video.projects.reorder_scenes(project.project_id, ids, project.revision):
                        st.rerun()
                if controls[3].button(tr("staged.split"), key=f"split:{scene.scene_id}"):
                    if self._run_action(
                        tr("staged.saving"),
                        lambda scene=scene: pixelle_video.projects.split_scene(
                            project.project_id,
                            scene.scene_id,
                            project.revision,
                            max(1, len(scene.narration) // 2),
                        ),
                    ):
                        st.rerun()
                if controls[4].button(
                    tr("staged.merge"),
                    key=f"merge:{scene.scene_id}",
                    disabled=scene.order == len(project.scenes) - 1,
                ):
                    if self._run_action(
                        tr("staged.saving"),
                        lambda scene=scene: pixelle_video.projects.merge_scene_with_next(
                            project.project_id,
                            scene.scene_id,
                            project.revision,
                        ),
                    ):
                        st.rerun()
                if controls[5].button(
                    tr("staged.delete"),
                    key=f"delete:{scene.scene_id}",
                    disabled=len(project.scenes) <= 1,
                ):
                    if self._run_action(
                        tr("staged.saving"),
                        lambda scene=scene: pixelle_video.projects.delete_scene(
                            project.project_id,
                            scene.scene_id,
                            project.revision,
                        ),
                    ):
                        st.rerun()
                if st.button(
                    tr("staged.regenerate_narration"),
                    key=f"regen-narration:{scene.scene_id}",
                ):
                    if self._run_action(
                        tr("staged.planning"),
                        lambda scene=scene: run_async(
                            pixelle_video.projects.regenerate_scene_narration(
                                project.project_id,
                                scene.scene_id,
                                project.revision,
                            )
                        ),
                    ):
                        st.rerun()
    def _render_storyboard(self, pixelle_video: Any, project: Any) -> None:
        st.subheader(tr("staged.storyboard_title"))
        if not project.scenes:
            st.info(tr("staged.plan_first"))
            return
        visuals_dirty = any(
            scene.dirty.visual_prompt or not scene.visual_prompt for scene in project.scenes
        )
        if visuals_dirty:
            st.warning(tr("staged.visuals_dirty"))
            if st.button(tr("staged.replan_visuals"), type="primary"):
                if self._run_action(
                    tr("staged.planning"),
                    lambda: run_async(
                        pixelle_video.projects.replan_visuals(
                            project.project_id,
                            project.revision,
                        )
                    ),
                ):
                    st.rerun()
        else:
            summary, next_step = st.columns([3, 1])
            summary.success(
                tr(
                    "staged.storyboard_summary",
                    scenes=len(project.scenes),
                    seconds=f"{_project_duration(project):g}",
                )
            )
            if next_step.button(
                tr("staged.next_generate"),
                type="primary",
                use_container_width=True,
            ):
                self._go_to_step(project.project_id, 4)
        for scene in project.scenes:
            with st.expander(_scene_expander_label(scene), expanded=False):
                st.caption(scene.narration)
                template = scene.template_name or project.render_profile.frame_template
                template_preview = get_template_preview_path(template, get_language())
                if template_preview and Path(template_preview).is_file():
                    preview_column, detail_column = st.columns([1, 3])
                    preview_column.image(template_preview, use_container_width=True)
                    detail_column.caption(
                        f"{tr('staged.template_preview')}: {template}"
                    )
                preview_key = (
                    f"scene-preview:{project.project_id}:{scene.scene_id}:{project.revision}"
                )
                if st.button(
                    tr("staged.generate_scene_preview"),
                    key=f"generate-preview:{scene.scene_id}:{project.revision}",
                    help=tr("staged.generate_scene_preview_help"),
                ):
                    preview_path = self._run_action(
                        tr("staged.generating_scene_preview"),
                        lambda scene=scene: run_async(
                            pixelle_video.projects.render_scene_preview(
                                project.project_id,
                                scene.scene_id,
                                project.revision,
                            )
                        ),
                    )
                    if preview_path:
                        st.session_state[preview_key] = preview_path
                generated_preview = st.session_state.get(preview_key)
                if generated_preview and Path(generated_preview).is_file():
                    st.image(
                        generated_preview,
                        caption=tr("staged.scene_preview_caption"),
                        use_container_width=True,
                    )
                visual = st.text_area(
                    tr("staged.visual_prompt"),
                    value=scene.visual_prompt,
                    key=f"visual:{scene.scene_id}:{project.revision}",
                    height=120,
                )
                left, middle, right = st.columns([1, 1, 2])
                duration = left.number_input(
                    tr("staged.duration"),
                    min_value=0.5,
                    max_value=300.0,
                    value=float(scene.estimated_duration),
                    step=0.5,
                    key=f"duration:{scene.scene_id}:{project.revision}",
                )
                media_mode = middle.selectbox(
                    tr("staged.media_mode"),
                    ["video", "image", "static", "asset"],
                    index=["video", "image", "static", "asset"].index(scene.media_mode),
                    format_func=_media_mode_label,
                    key=f"media-mode:{scene.scene_id}:{project.revision}",
                )
                visual_assets = [
                    asset for asset in project.assets if asset.media_kind in {"image", "video"}
                ]
                asset_ids = st.multiselect(
                    tr("staged.scene_assets"),
                    [asset.asset_id for asset in visual_assets],
                    default=[
                        asset_id
                        for asset_id in scene.asset_ids
                        if any(asset.asset_id == asset_id for asset in visual_assets)
                    ],
                    format_func=lambda value: next(
                        (
                            asset.original_name
                            for asset in visual_assets
                            if asset.asset_id == value
                        ),
                        value,
                    ),
                    disabled=media_mode != "asset",
                    key=f"scene-assets:{scene.scene_id}:{project.revision}",
                )
                right.caption(
                    tr("staged.scene_needs_review")
                    if scene.dirty.visual_prompt or not scene.visual_prompt
                    else tr("staged.scene_ready")
                )
                save, regenerate = st.columns(2)
                if save.button(tr("staged.save_scene"), key=f"save-scene:{scene.scene_id}"):
                    if self._run_action(
                        tr("staged.saving"),
                        lambda scene=scene, visual=visual, duration=duration, media_mode=media_mode: (
                            pixelle_video.projects.update_scene(
                                project.project_id,
                                scene.scene_id,
                                project.revision,
                                visual_prompt=visual,
                                estimated_duration=duration,
                                media_mode=media_mode,
                                asset_ids=asset_ids,
                            )
                        ),
                    ):
                        st.rerun()
                if regenerate.button(tr("staged.regenerate_scene"), key=f"regen:{scene.scene_id}"):
                    if self._run_action(
                        tr("staged.planning"),
                        lambda scene=scene: run_async(
                            pixelle_video.projects.regenerate_scene_visual(
                                project.project_id,
                                scene.scene_id,
                                project.revision,
                            )
                        ),
                    ):
                        st.rerun()
        self._render_assets(pixelle_video, project)

    def _render_assets(self, pixelle_video: Any, project: Any) -> None:
        with st.expander(tr("staged.assets_title"), expanded=bool(project.assets)):
            uploads = st.file_uploader(
                tr("staged.assets_upload"),
                accept_multiple_files=True,
                key=f"assets:{project.project_id}:{project.revision}",
            )
            if st.button(
                tr("staged.assets_import_uploads"),
                disabled=not uploads,
                key=f"ingest:{project.project_id}:{project.revision}",
            ):
                revision = project.revision
                for upload in uploads:
                    upload.seek(0)
                    asset = self._run_action(
                        tr("staged.assets_processing"),
                        lambda upload=upload: pixelle_video.assets.ingest_upload(
                            project.project_id,
                            upload.name,
                            upload,
                        ),
                    )
                    if not asset:
                        break
                    revision = pixelle_video.projects.add_asset(
                        project.project_id,
                        revision,
                        asset,
                    ).revision
                else:
                    st.rerun()
            local_path = st.text_input(
                tr("staged.assets_local_path"),
                key=f"local-asset:{project.project_id}",
            )
            if st.button(tr("staged.assets_import_local"), disabled=not local_path.strip()):
                asset = self._run_action(
                    tr("staged.assets_processing"),
                    lambda: pixelle_video.assets.import_local(project.project_id, local_path),
                )
                if asset and pixelle_video.projects.add_asset(
                    project.project_id,
                    project.revision,
                    asset,
                ):
                    st.rerun()
            for asset in project.assets:
                with st.container(border=True):
                    cols = st.columns([1, 3])
                    if asset.media_kind == "audio":
                        cols[0].audio(asset.original_path)
                        cols[1].markdown(f"**{asset.original_name}**")
                        cols[1].caption(
                            f"audio · {asset.codec or '-'} · {asset.duration or '-'} s · "
                            f"SHA-256 {asset.sha256[:12]}"
                        )
                        if asset.quality_warnings:
                            cols[1].warning(", ".join(asset.quality_warnings))
                        continue
                    fit_mode = cols[1].selectbox(
                        tr("staged.asset_fit"),
                        ["cover", "contain"],
                        index=["cover", "contain"].index(asset.fit_mode),
                        key=f"asset-fit:{asset.asset_id}:{project.revision}",
                    )
                    focal_x = cols[1].slider(
                        tr("staged.asset_focal_x"),
                        0.0,
                        1.0,
                        float(asset.focal_x),
                        0.05,
                        key=f"asset-x:{asset.asset_id}:{project.revision}",
                    )
                    focal_y = cols[1].slider(
                        tr("staged.asset_focal_y"),
                        0.0,
                        1.0,
                        float(asset.focal_y),
                        0.05,
                        key=f"asset-y:{asset.asset_id}:{project.revision}",
                    )
                    preview_asset = asset.model_copy(
                        update={
                            "fit_mode": fit_mode,
                            "focal_x": focal_x,
                            "focal_y": focal_y,
                        }
                    )
                    try:
                        preview_path = pixelle_video.assets.build_crop_preview(
                            preview_asset,
                            project.render_profile.media_width,
                            project.render_profile.media_height,
                        )
                    except Exception as exc:
                        cols[0].warning(str(exc))
                    else:
                        cols[0].image(preview_path, use_container_width=True)
                    cols[1].markdown(f"**{asset.original_name}**")
                    cols[1].caption(
                        f"{asset.media_kind} · {asset.width or '-'}×{asset.height or '-'} · "
                        f"{asset.codec or '-'} · {asset.frame_rate or '-'} fps · "
                        f"{asset.duration or '-'} s · SHA-256 {asset.sha256[:12]}"
                    )
                    if asset.quality_warnings:
                        cols[1].warning(", ".join(asset.quality_warnings))
                    changed = (
                        fit_mode != asset.fit_mode
                        or focal_x != asset.focal_x
                        or focal_y != asset.focal_y
                    )
                    if cols[1].button(
                        tr("staged.save_asset_crop"),
                        disabled=not changed,
                        key=f"save-asset-crop:{asset.asset_id}:{project.revision}",
                    ):
                        if self._run_action(
                            tr("staged.saving"),
                            lambda asset=asset, fit_mode=fit_mode, focal_x=focal_x, focal_y=focal_y: (
                                pixelle_video.projects.update_asset_presentation(
                                    project.project_id,
                                    asset.asset_id,
                                    project.revision,
                                    fit_mode=fit_mode,
                                    focal_x=focal_x,
                                    focal_y=focal_y,
                                )
                            ),
                        ):
                            st.rerun()

    def _render_production(self, pixelle_video: Any, project: Any) -> bool:
        settings_prepared = project.status in {
            ProjectStatus.LOCKED,
            ProjectStatus.PRODUCING,
            ProjectStatus.COMPLETED,
        }
        st.subheader(
            tr("staged.generate_title")
            if settings_prepared
            else tr("staged.production_title")
        )
        if project.style is None:
            st.warning(tr("staged.select_style_first"))
            if st.button(tr("staged.go_style"), type="primary", use_container_width=True):
                self._go_to_step(project.project_id, 1)
            return False
        if not project.scenes:
            st.info(tr("staged.plan_first"))
            if st.button(tr("staged.go_script"), type="primary", use_container_width=True):
                self._go_to_step(project.project_id, 2)
            return False
        render = project.render_profile
        st.info(
            tr(
                "staged.production_summary",
                canvas=f"{render.canvas_width}×{render.canvas_height}",
                scenes=len(project.scenes),
                seconds=f"{_project_duration(project):g}",
                template=_resource_name(render.frame_template),
                workflow=_resource_name(render.media_workflow),
                voice=_voice_name(render.voice_id),
            )
        )
        if settings_prepared:
            return True

        workflow = render.media_workflow
        with st.expander(tr("staged.advanced_settings"), expanded=False):
            st.caption(tr("staged.advanced_settings_help"))
            canvas_options = [(1080, 1920), (1920, 1080), (1080, 1080)]
            current_canvas = (render.canvas_width, render.canvas_height)
            canvas = st.selectbox(
                tr("staged.canvas"),
                canvas_options,
                index=(
                    canvas_options.index(current_canvas)
                    if current_canvas in canvas_options
                    else 0
                ),
                format_func=lambda item: f"{item[0]}×{item[1]}",
            )
            template_dir = Path("templates") / f"{canvas[0]}x{canvas[1]}"
            templates = sorted(
                str(path.relative_to("templates")) for path in template_dir.glob("*.html")
            )
            template_index = (
                templates.index(render.frame_template)
                if render.frame_template in templates
                else 0
            )
            template = st.selectbox(
                tr("staged.template"),
                templates,
                index=template_index,
                disabled=not templates,
                format_func=_resource_name,
            )
            allowed = project.style.allowed_workflows if project.style else []
            workflows = [item for item in allowed if (Path("workflows") / item).is_file()]
            if not workflows:
                st.error(tr("staged.no_approved_workflow"))
                workflow = None
            else:
                workflow_index = (
                    workflows.index(render.media_workflow)
                    if render.media_workflow in workflows
                    else 0
                )
                workflow = st.selectbox(
                    tr("staged.workflow"),
                    workflows,
                    index=workflow_index,
                    format_func=_resource_name,
                )
            voice_options = [item["id"] for item in EDGE_TTS_VOICES]
            if render.voice_id and render.voice_id not in voice_options:
                voice_options.insert(0, render.voice_id)
            voice_id = st.selectbox(
                tr("staged.voice"),
                voice_options,
                index=voice_options.index(render.voice_id) if render.voice_id else 0,
                format_func=_voice_name,
            )
            tts_speed = st.slider(
                tr("staged.tts_speed"), 0.5, 2.0, float(render.tts_speed), 0.05
            )
            bgm_volume = st.slider(
                tr("staged.bgm_volume"), 0.0, 1.0, float(render.bgm_volume), 0.01
            )
            if st.button(tr("staged.save_production")):
                if self._run_action(
                    tr("staged.saving"),
                    lambda: pixelle_video.projects.update_render_profile(
                        project.project_id,
                        project.revision,
                        {
                            "canvas_width": canvas[0],
                            "canvas_height": canvas[1],
                            "frame_template": template,
                            "media_workflow": workflow,
                            "voice_id": voice_id,
                            "tts_speed": tts_speed,
                            "bgm_volume": bgm_volume,
                        },
                    ),
                ):
                    st.rerun()
        refreshed = pixelle_video.projects.get(project.project_id)
        if any(scene.dirty.visual_prompt for scene in refreshed.scenes):
            st.warning(tr("staged.visuals_dirty_production"))
            if st.button(
                tr("staged.go_storyboard"),
                type="primary",
                use_container_width=True,
            ):
                self._go_to_step(project.project_id, 3)
            return False
        st.caption(tr("staged.prepare_explanation"))
        if st.button(
            tr("staged.prepare_and_lock"),
            type="primary",
            disabled=not workflow,
            use_container_width=True,
        ):
            if self._run_action(
                tr("staged.preparing"),
                lambda: run_async(
                    pixelle_video.projects.prepare_for_production(
                        refreshed.project_id,
                        refreshed.revision,
                    )
                ),
            ):
                st.rerun()
        return False

    def _render_generate(self, pixelle_video: Any, project: Any) -> None:
        project = pixelle_video.projects.get(project.project_id)
        record = pixelle_video.compatibility.get(project.render_profile.combination_hash())
        runtime_ready = _runtime_ready(record)
        if project.last_task_id:
            st.success(f"task: {project.last_task_id}")
            final_path = Path("output") / project.last_task_id / "final.mp4"
            if final_path.is_file():
                st.video(str(final_path))
        if project.last_error:
            st.error(project.last_error)
        if project.status == ProjectStatus.PRODUCING:
            st.info(tr("staged.production_in_progress"))
            return
        if project.status == ProjectStatus.COMPLETED:
            return
        if not runtime_ready:
            st.warning(tr("staged.runtime_changed"))
            return
        st.success(tr("staged.ready_to_generate"))
        if st.button(
            tr("staged.start_production"),
            type="primary",
            use_container_width=True,
        ):
            result = self._run_action(
                tr("staged.producing"),
                lambda: run_async(
                    pixelle_video.projects.produce(project.project_id, project.revision)
                ),
            )
            if result:
                st.success(tr("staged.production_complete"))
                st.video(result["video_path"])

    @staticmethod
    def _render_legacy(pixelle_video: Any) -> None:
        left_col, middle_col, right_col = st.columns([1, 1, 1])
        with left_col:
            content_params = render_content_input()
            bgm_params = render_bgm_section()
            render_version_info()
        with middle_col:
            style_params = render_style_config(pixelle_video)
        with right_col:
            render_output_preview(
                pixelle_video,
                {
                    "pipeline": "quick_create",
                    **content_params,
                    **bgm_params,
                    **style_params,
                },
            )
