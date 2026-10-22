from pathlib import Path
from types import SimpleNamespace


def test_staged_creation_module_imports_from_cold_start():
    from web.pipelines.staged import StagedCreationUI

    assert StagedCreationUI.name == "quick_create"


def test_template_preview_helper_is_available_at_module_scope():
    from web.components.style_config import get_template_preview_path

    preview = get_template_preview_path("1080x1920/image_default.html", "zh_CN")

    assert Path(preview).is_file()


def test_completed_project_duration_uses_rendered_segments():
    from pixelle_video.models.project import ProjectStatus
    from web.pipelines.staged import _project_duration

    project = SimpleNamespace(
        status=ProjectStatus.COMPLETED,
        scenes=[
            SimpleNamespace(estimated_duration=4, last_render_duration=5.041667),
            SimpleNamespace(estimated_duration=4, last_render_duration=6.125),
        ],
    )

    assert _project_duration(project) == 11.2
