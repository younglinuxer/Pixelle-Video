"""Staged project and style APIs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.dependencies import PixelleVideoDep
from api.schemas.projects import (
    ApplyStyleRequest,
    ApprovalRequest,
    AssetPresentationUpdateRequest,
    CanaryValidationRequest,
    CustomStyleRequest,
    LocalAssetImportRequest,
    ProjectCreateRequest,
    PromptApplyRequest,
    PromptOptimizeRequest,
    RenderProfileUpdateRequest,
    RevisionRequest,
    SceneRegenerateRequest,
    SceneReorderRequest,
    SceneSplitRequest,
    SceneUpdateRequest,
    StylePreviewRequest,
    TitleUpdateRequest,
)
from api.tasks import TaskType, task_manager
from pixelle_video.services.asset_service import AssetValidationError
from pixelle_video.services.compatibility import CompatibilityGateError
from pixelle_video.services.media_quality import MediaQualityError
from pixelle_video.services.project_persistence import (
    ProjectNotFoundError,
    RevisionConflictError,
)
from pixelle_video.services.project_service import ProjectStateError

router = APIRouter(tags=["Staged Creation"])


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, ProjectNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, RevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "current_revision": exc.actual},
        ) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(
        exc,
        (
            ValueError,
            ProjectStateError,
            CompatibilityGateError,
            AssetValidationError,
            MediaQualityError,
        ),
    ):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/styles")
async def list_styles(pixelle_video: PixelleVideoDep):
    return {"styles": pixelle_video.styles.list_styles()}


@router.get("/catalog/templates")
async def list_template_manifests(pixelle_video: PixelleVideoDep):
    return {"templates": pixelle_video.render_catalog.list_templates()}


@router.get("/catalog/workflows")
async def list_workflow_manifests(pixelle_video: PixelleVideoDep):
    return {"workflows": pixelle_video.render_catalog.list_workflows()}


@router.post("/styles/custom", status_code=status.HTTP_201_CREATED)
async def create_custom_style(body: CustomStyleRequest, pixelle_video: PixelleVideoDep):
    try:
        style, preview_prompts = await pixelle_video.styles.create_custom_style(body.description)
        return {"style": style, "preview_prompts": preview_prompts}
    except Exception as exc:
        _raise_http(exc)
@router.post("/styles/{style_id}/previews")
async def generate_style_previews(
    style_id: str,
    body: StylePreviewRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return await pixelle_video.styles.generate_previews(
            style_id,
            pixelle_video.media,
            body.workflow,
        )
    except Exception as exc:
        _raise_http(exc)
@router.post("/projects", status_code=status.HTTP_201_CREATED)
async def create_project(body: ProjectCreateRequest, pixelle_video: PixelleVideoDep):
    try:
        return pixelle_video.projects.create(body.original_prompt, body.scene_count)
    except Exception as exc:
        _raise_http(exc)


@router.get("/projects")
async def list_projects(pixelle_video: PixelleVideoDep, limit: int = 100):
    return {"projects": pixelle_video.projects.list(limit=min(max(limit, 1), 500))}


@router.get("/projects/{project_id}")
async def get_project(project_id: str, pixelle_video: PixelleVideoDep):
    try:
        return pixelle_video.projects.get(project_id)
    except Exception as exc:
        _raise_http(exc)


@router.get("/projects/{project_id}/revisions/{revision}")
async def get_project_revision(project_id: str, revision: int, pixelle_video: PixelleVideoDep):
    try:
        return pixelle_video.projects.persistence.load_revision(project_id, revision)
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/prompt/optimize")
async def optimize_prompt(
    project_id: str,
    body: PromptOptimizeRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        proposal = await pixelle_video.projects.optimize_prompt(
            project_id,
            body.expected_revision,
            body.role,
            body.custom_role,
        )
        return {"proposal": proposal}
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/prompt/apply")
async def apply_optimized_prompt(
    project_id: str,
    body: PromptApplyRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.apply_optimized_prompt(
            project_id,
            body.expected_revision,
            body.optimized_prompt,
            body.role,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/style")
async def apply_style(
    project_id: str,
    body: ApplyStyleRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.apply_style(
            project_id,
            body.style_id,
            body.expected_revision,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/plan")
async def plan_project(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return await pixelle_video.projects.plan(project_id, body.expected_revision)
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/plan/visuals")
async def replan_visuals(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return await pixelle_video.projects.replan_visuals(
            project_id,
            body.expected_revision,
        )
    except Exception as exc:
        _raise_http(exc)


@router.patch("/projects/{project_id}/title")
async def update_title(
    project_id: str,
    body: TitleUpdateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.update_title(
            project_id,
            body.expected_revision,
            body.title,
        )
    except Exception as exc:
        _raise_http(exc)


@router.patch("/projects/{project_id}/scenes/{scene_id}")
async def update_scene(
    project_id: str,
    scene_id: str,
    body: SceneUpdateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.update_scene(
            project_id,
            scene_id,
            body.expected_revision,
            narration=body.narration,
            visual_prompt=body.visual_prompt,
            estimated_duration=body.estimated_duration,
            media_mode=body.media_mode,
            asset_ids=body.asset_ids,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/{scene_id}/preview")
async def render_scene_preview(
    project_id: str,
    scene_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        preview_path = await pixelle_video.projects.render_scene_preview(
            project_id,
            scene_id,
            body.expected_revision,
        )
        return {"preview_path": preview_path}
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/{scene_id}/regenerate-visual")
async def regenerate_scene_visual(
    project_id: str,
    scene_id: str,
    body: SceneRegenerateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return await pixelle_video.projects.regenerate_scene_visual(
            project_id,
            scene_id,
            body.expected_revision,
            body.instruction,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/{scene_id}/regenerate-narration")
async def regenerate_scene_narration(
    project_id: str,
    scene_id: str,
    body: SceneRegenerateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return await pixelle_video.projects.regenerate_scene_narration(
            project_id,
            scene_id,
            body.expected_revision,
            body.instruction,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/{scene_id}/split")
async def split_scene(
    project_id: str,
    scene_id: str,
    body: SceneSplitRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.split_scene(
            project_id,
            scene_id,
            body.expected_revision,
            body.split_at,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/{scene_id}/merge-next")
async def merge_scene_with_next(
    project_id: str,
    scene_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.merge_scene_with_next(
            project_id,
            scene_id,
            body.expected_revision,
        )
    except Exception as exc:
        _raise_http(exc)


@router.delete("/projects/{project_id}/scenes/{scene_id}")
async def delete_scene(
    project_id: str,
    scene_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.delete_scene(
            project_id,
            scene_id,
            body.expected_revision,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/scenes/reorder")
async def reorder_scenes(
    project_id: str,
    body: SceneReorderRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.reorder_scenes(
            project_id,
            body.scene_ids,
            body.expected_revision,
        )
    except Exception as exc:
        _raise_http(exc)


@router.patch("/projects/{project_id}/render-profile")
async def update_render_profile(
    project_id: str,
    body: RenderProfileUpdateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.update_render_profile(
            project_id,
            body.expected_revision,
            body.updates,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/compatibility")
async def inspect_compatibility(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        project, record = await pixelle_video.projects.inspect_compatibility(
            project_id,
            body.expected_revision,
        )
        return {"project": project, "compatibility": record}
    except Exception as exc:
        _raise_http(exc)


@router.post("/compatibility/{combination_hash}/validate-canary", deprecated=True)
async def validate_canary(
    combination_hash: str,
    body: CanaryValidationRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        smoke, canary, checks = pixelle_video.media_quality.validate_canary(
            smoke_path=body.smoke_path,
            canary_paths=body.canary_paths,
            expected_width=body.expected_width,
            expected_height=body.expected_height,
            queue_history=body.queue_history,
        )
        return pixelle_video.compatibility.record_technical_pass(
            combination_hash,
            smoke_evidence=smoke,
            canary_evidence=canary,
            technical_checks=checks,
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/compatibility/{combination_hash}/approve", deprecated=True)
async def approve_compatibility(
    combination_hash: str,
    body: ApprovalRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.compatibility.approve(combination_hash, body.approved_by)
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/lock")
async def lock_project(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.lock(project_id, body.expected_revision)
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/prepare")
async def prepare_project(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    """Check runtime readiness and lock the reviewed snapshot in one request."""
    try:
        project, compatibility = await pixelle_video.projects.prepare_for_production(
            project_id,
            body.expected_revision,
        )
        return {"project": project, "compatibility": compatibility}
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/produce", status_code=status.HTTP_202_ACCEPTED)
async def produce_project(
    project_id: str,
    body: RevisionRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        project = pixelle_video.projects.get(project_id)
        if project.revision != body.expected_revision:
            raise RevisionConflictError(body.expected_revision, project.revision)
        if project.status.value != "locked":
            raise ProjectStateError("Project must be locked before production")
        if not project.locked_snapshot_hash or project.locked_snapshot_hash != project.snapshot_hash():
            raise ProjectStateError("Locked project snapshot no longer matches current content")
        pixelle_video.compatibility.require_runtime_ready(project.render_profile)
        task = task_manager.create_task(
            TaskType.PROJECT_PRODUCTION,
            request_params={
                "project_id": project_id,
                "project_revision": body.expected_revision,
                "snapshot_hash": project.locked_snapshot_hash,
            },
        )
        await task_manager.execute_task(
            task.task_id,
            pixelle_video.projects.produce,
            project_id,
            body.expected_revision,
        )
        return {"task_id": task.task_id, "project_id": project_id}
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/assets/upload")
async def upload_asset(
    project_id: str,
    pixelle_video: PixelleVideoDep,
    expected_revision: Annotated[int, Form(ge=0)],
    file: Annotated[UploadFile, File()],
):
    try:
        project = pixelle_video.projects.get(project_id)
        if project.revision != expected_revision:
            raise RevisionConflictError(expected_revision, project.revision)
        asset = pixelle_video.assets.ingest_upload(project_id, file.filename or "upload.bin", file.file)
        updated = pixelle_video.projects.add_asset(project_id, expected_revision, asset)
        return {"project": updated, "asset": asset}
    except Exception as exc:
        _raise_http(exc)


@router.post("/projects/{project_id}/assets/import")
async def import_local_asset(
    project_id: str,
    body: LocalAssetImportRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        project = pixelle_video.projects.get(project_id)
        if project.revision != body.expected_revision:
            raise RevisionConflictError(body.expected_revision, project.revision)
        asset = pixelle_video.assets.import_local(project_id, body.source_path)
        updated = pixelle_video.projects.add_asset(project_id, body.expected_revision, asset)
        return {"project": updated, "asset": asset}
    except Exception as exc:
        _raise_http(exc)


@router.patch("/projects/{project_id}/assets/{asset_id}/presentation")
async def update_asset_presentation(
    project_id: str,
    asset_id: str,
    body: AssetPresentationUpdateRequest,
    pixelle_video: PixelleVideoDep,
):
    try:
        return pixelle_video.projects.update_asset_presentation(
            project_id,
            asset_id,
            body.expected_revision,
            fit_mode=body.fit_mode,
            focal_x=body.focal_x,
            focal_y=body.focal_y,
        )
    except Exception as exc:
        _raise_http(exc)
