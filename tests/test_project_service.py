from pathlib import Path

import pytest

from pixelle_video.models.project import (
    AssetRecord,
    CompatibilityRecord,
    CompatibilityStatus,
    ProjectDraft,
    ProjectStatus,
    SceneDraft,
)
from pixelle_video.services.compatibility import (
    CompatibilityGateError,
    CompatibilityService,
)
from pixelle_video.services.project_persistence import ProjectPersistenceService
from pixelle_video.services.project_service import PlannedScene, ProjectService, StoryboardPlan
from pixelle_video.services.style_service import StyleService


class FakeLLM:
    async def __call__(self, *args, **kwargs):
        raise AssertionError("This test must not invoke the planner")


class FakeCore:
    def __init__(self):
        self.llm = FakeLLM()
        self.config = {"comfyui": {"comfyui_url": "http://127.0.0.1:9"}}
        self.pipelines = {}


@pytest.fixture
def project_service(tmp_path):
    core = FakeCore()
    compatibility = CompatibilityService(
        root=tmp_path / "compatibility",
        project_root=Path(__file__).resolve().parents[1],
        comfyui_url="http://127.0.0.1:9",
    )
    return ProjectService(
        core,
        persistence=ProjectPersistenceService(tmp_path / "projects"),
        styles=StyleService(core.llm, tmp_path / "styles"),
        compatibility=compatibility,
    )


def test_style_switch_keeps_narration_and_invalidates_only_visual_outputs(project_service):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    project.scenes = [
        SceneDraft(order=0, narration="keep me", visual_prompt="old", estimated_duration=3)
    ]
    project.title = "title"
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)

    changed = project_service.apply_style(
        project.project_id,
        "cinematic_narrative",
        project.revision,
    )

    assert changed.scenes[0].narration == "keep me"
    assert changed.scenes[0].visual_prompt == ""
    assert changed.scenes[0].dirty.visual_prompt is True
    assert changed.scenes[0].dirty.media is True


def test_style_switch_revokes_old_contract_hashes_and_updates_media_slot(project_service):
    project = project_service.create("brief", 1)
    project.render_profile.template_hash = "old-template"
    project.render_profile.workflow_hash = "old-workflow"
    project.render_profile.environment_fingerprint = "old-environment"
    project = project_service.persistence.save(project, project.revision)

    changed = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )

    assert changed.render_profile.template_hash is None
    assert changed.render_profile.workflow_hash is None
    assert changed.render_profile.environment_fingerprint is None
    assert changed.render_profile.media_width > 0
    assert changed.render_profile.media_height > 0
    assert changed.render_profile.media_workflow == "selfhost/video_minimax_h3_t2v.json"
    assert changed.render_profile.frame_template == "1080x1920/video_default.html"


def test_video_workflow_replaces_an_opaque_image_template(project_service):
    project = project_service.create("brief", 1)
    project.render_profile.frame_template = "1080x1920/image_neon.html"
    project.render_profile.media_width = 1024
    project.render_profile.media_height = 1024
    project = project_service.persistence.save(project, project.revision)

    changed = project_service.update_render_profile(
        project.project_id,
        project.revision,
        {"media_workflow": "selfhost/video_minimax_h3_t2v.json"},
    )

    assert changed.render_profile.frame_template == "1080x1920/video_default.html"
    assert changed.render_profile.media_width == 512
    assert changed.render_profile.media_height == 288


@pytest.mark.asyncio
async def test_planner_forces_every_scene_to_ai_video(project_service):
    project = project_service.create("brief", 2)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )

    async def fake_planner(prompt, **_kwargs):
        assert "Every scene will be generated as AI video" in prompt
        return StoryboardPlan(
            title="title",
            scenes=[
                PlannedScene(
                    narration="one",
                    visual_prompt="first motion",
                    media_mode="image",
                ),
                PlannedScene(
                    narration="two",
                    visual_prompt="second motion",
                    media_mode="static",
                ),
            ],
        )

    project_service.core.llm = fake_planner
    planned = await project_service.plan(project.project_id, project.revision)

    assert [scene.media_mode for scene in planned.scenes] == ["video", "video"]


def test_single_scene_edit_does_not_dirty_other_scene(project_service):
    project = project_service.create("brief", 2)
    project.scenes = [
        SceneDraft(order=0, narration="one", visual_prompt="v1"),
        SceneDraft(order=1, narration="two", visual_prompt="v2"),
    ]
    project.scenes[1].dirty.media = False
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)

    updated = project_service.update_scene(
        project.project_id,
        project.scenes[0].scene_id,
        project.revision,
        narration="one changed",
    )

    assert updated.scenes[0].dirty.media is True
    assert updated.scenes[1].narration == "two"
    assert updated.scenes[1].dirty.media is False


@pytest.mark.asyncio
async def test_scene_preview_uses_local_template_renderer_without_planning(
    project_service,
    monkeypatch,
):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    project.title = "Reviewed title"
    project.scenes = [SceneDraft(order=0, narration="Reviewed narration", visual_prompt="visual")]
    project = project_service.persistence.save(project, project.revision)
    captured = {}

    async def fake_generate(self, **kwargs):
        captured.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"preview")
        return str(output_path)

    monkeypatch.setattr(
        "pixelle_video.services.frame_html.HTMLFrameGenerator.generate_frame",
        fake_generate,
    )

    preview_path = await project_service.render_scene_preview(
        project.project_id,
        project.scenes[0].scene_id,
        project.revision,
    )

    assert Path(preview_path).read_bytes() == b"preview"
    assert captured["title"] == "Reviewed title"
    assert captured["text"] == "Reviewed narration"
    assert captured["ext"] == {"index": 1}


@pytest.mark.asyncio
async def test_reinspection_promotes_existing_runtime_record(project_service, monkeypatch):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    ready = False

    async def fake_inspect(_workflow):
        return "stable-environment", ready, {"missing_models": [] if ready else ["model.bin"]}

    monkeypatch.setattr(project_service.compatibility, "_inspect_comfyui", fake_inspect)
    project, first_record = await project_service.inspect_compatibility(
        project.project_id,
        project.revision,
    )
    assert first_record.status == CompatibilityStatus.UNVERIFIED

    ready = True
    _project, second_record = await project_service.inspect_compatibility(
        project.project_id,
        project.revision,
    )

    assert second_record.combination_hash == first_record.combination_hash
    assert second_record.status == CompatibilityStatus.RUNTIME_READY
    assert second_record.technical_checks["runtime"]["missing_models"] == []


def test_advanced_override_creates_project_style_without_mutating_preset(project_service):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    original_hash = project.style.validation_hash()

    updated = project_service.update_render_profile(
        project.project_id,
        project.revision,
        {"tts_speed": 1.35},
    )

    assert updated.style.style_id == f"project-{project.project_id}"
    assert updated.style.builtin is False
    assert updated.style.tts_speed_range == (1.35, 1.35)
    assert project_service.styles.get_style("documentary_realism").validation_hash() == original_hash


def test_asset_focus_invalidates_only_scenes_that_reference_it(project_service, tmp_path):
    original = tmp_path / "source.png"
    original.write_bytes(b"source")
    project = project_service.create("brief", 2)
    asset = AssetRecord(
        original_name="source.png",
        original_path=str(original),
        sha256="a" * 64,
        media_kind="image",
        size_bytes=6,
    )
    project = project_service.add_asset(project.project_id, project.revision, asset)
    project.scenes = [
        SceneDraft(order=0, narration="one", visual_prompt="v1", asset_ids=[asset.asset_id]),
        SceneDraft(order=1, narration="two", visual_prompt="v2"),
    ]
    project.scenes[0].dirty.media = False
    project.scenes[1].dirty.media = False
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)

    updated = project_service.update_asset_presentation(
        project.project_id,
        asset.asset_id,
        project.revision,
        fit_mode="cover",
        focal_x=0.75,
        focal_y=0.25,
    )

    assert updated.assets[0].fit_mode == "cover"
    assert updated.scenes[0].dirty.media is True
    assert updated.scenes[1].dirty.media is False


def test_interrupted_production_is_recovered_as_failed(tmp_path):
    persistence = ProjectPersistenceService(tmp_path / "projects")
    project = persistence.create(ProjectDraft(original_prompt="brief"))
    project.status = ProjectStatus.PRODUCING
    persistence.save(project, project.revision)
    core = FakeCore()

    service = ProjectService(
        core,
        persistence=persistence,
        styles=StyleService(core.llm, tmp_path / "styles"),
        compatibility=CompatibilityService(
            root=tmp_path / "compatibility",
            project_root=Path(__file__).resolve().parents[1],
            comfyui_url="http://127.0.0.1:9",
        ),
    )

    recovered = service.get(project.project_id)
    assert recovered.status == ProjectStatus.FAILED
    assert "restart" in recovered.last_error


def test_production_gate_requires_runtime_ready(project_service):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    project.title = "title"
    project.scenes = [SceneDraft(order=0, narration="one", visual_prompt="visual")]
    project.scenes[0].dirty.visual_prompt = False
    render = project.render_profile
    render.template_hash = "template"
    render.workflow_hash = "workflow"
    render.environment_fingerprint = "environment"
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)
    with pytest.raises(CompatibilityGateError):
        project_service.lock(project.project_id, project.revision)

    combination_hash = project.render_profile.combination_hash()
    project_service.compatibility._save_record(
        CompatibilityRecord(
            combination_hash=combination_hash,
            status=CompatibilityStatus.RUNTIME_READY,
            style_id=project.style.style_id,
            style_version=project.style.version,
            style_hash=project.render_profile.style_hash,
            template_hash="template",
            workflow_hash="workflow",
            environment_fingerprint="environment",
            technical_checks={"runtime": {"ready": True}},
        )
    )

    project = project_service.lock(project.project_id, project.revision)
    assert project.status == ProjectStatus.LOCKED
    assert (
        project_service.compatibility.require_runtime_ready(project.render_profile).status
        == CompatibilityStatus.RUNTIME_READY
    )


@pytest.mark.asyncio
async def test_prepare_for_production_checks_runtime_and_locks_in_one_action(
    project_service,
    monkeypatch,
):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    project.title = "title"
    project.scenes = [SceneDraft(order=0, narration="one", visual_prompt="visual")]
    project.scenes[0].dirty.visual_prompt = False
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)

    async def runtime_ready(_workflow):
        return "environment", True, {"missing_nodes": [], "missing_models": []}

    monkeypatch.setattr(project_service.compatibility, "_inspect_comfyui", runtime_ready)

    locked, record = await project_service.prepare_for_production(
        project.project_id,
        project.revision,
    )

    assert locked.status == ProjectStatus.LOCKED
    assert locked.locked_snapshot_hash
    assert locked.revision == project.revision + 1
    assert record.status == CompatibilityStatus.RUNTIME_READY


@pytest.mark.asyncio
async def test_prepare_failure_does_not_create_partial_project_revision(
    project_service,
    monkeypatch,
):
    project = project_service.create("brief", 1)
    project = project_service.apply_style(
        project.project_id,
        "documentary_realism",
        project.revision,
    )
    project.title = "title"
    project.scenes = [SceneDraft(order=0, narration="one", visual_prompt="visual")]
    project.scenes[0].dirty.visual_prompt = False
    project.status = ProjectStatus.PLANNED
    project = project_service.persistence.save(project, project.revision)

    async def runtime_unavailable(_workflow):
        return "environment", False, {"missing_nodes": ["ExampleNode"]}

    monkeypatch.setattr(
        project_service.compatibility,
        "_inspect_comfyui",
        runtime_unavailable,
    )

    with pytest.raises(CompatibilityGateError):
        await project_service.prepare_for_production(
            project.project_id,
            project.revision,
        )

    persisted = project_service.get(project.project_id)
    assert persisted.revision == project.revision
    assert persisted.status == ProjectStatus.PLANNED
