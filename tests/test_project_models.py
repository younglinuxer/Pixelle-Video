from pixelle_video.models.project import (
    ProjectDraft,
    RenderProfile,
    SceneDraft,
)
from pixelle_video.style_profiles import get_builtin_styles


def test_builtin_styles_are_complete_and_independent():
    styles = get_builtin_styles()

    assert len(styles) == 8
    assert len({style.style_id for style in styles}) == 8
    assert all(style.visual_prompt_prefix for style in styles)
    assert all(style.allowed_workflows for style in styles)
    assert all(style.validation_hash() for style in styles)

    styles[0].name = "changed"
    assert get_builtin_styles()[0].name != "changed"


def test_snapshot_hash_ignores_lifecycle_revision_but_tracks_content():
    project = ProjectDraft(
        original_prompt="brief",
        title="title",
        scenes=[SceneDraft(order=0, narration="hello", visual_prompt="world")],
    )
    initial = project.snapshot_hash()

    project.revision += 1
    assert project.snapshot_hash() == initial

    project.scenes[0].visual_prompt = "different"
    assert project.snapshot_hash() != initial


def test_new_scene_defaults_to_ai_video():
    scene = SceneDraft(order=0, narration="hello")

    assert scene.media_mode == "video"


def test_builtin_styles_offer_video_workflow_first():
    styles = get_builtin_styles()

    assert all(
        style.allowed_workflows[0] == "selfhost/video_minimax_h3_t2v.json"
        for style in styles
    )


def test_render_combination_hash_tracks_exact_contract():
    render = RenderProfile(style_hash="a", template_hash="b", workflow_hash="c")
    initial = render.combination_hash()

    render.video_fps = 24
    assert render.combination_hash() != initial
