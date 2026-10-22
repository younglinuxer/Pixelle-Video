import pytest

from pixelle_video.models.project import ProjectDraft
from pixelle_video.services.project_persistence import (
    ProjectPersistenceService,
    RevisionConflictError,
)


def test_project_survives_service_restart_and_keeps_revisions(tmp_path):
    store = ProjectPersistenceService(tmp_path / "projects")
    project = store.create(ProjectDraft(original_prompt="first"))
    project.title = "reviewed title"
    saved = store.save(project, expected_revision=0)

    restarted = ProjectPersistenceService(tmp_path / "projects")
    loaded = restarted.load(saved.project_id)

    assert loaded.revision == 1
    assert loaded.title == "reviewed title"
    assert restarted.load_revision(saved.project_id, 0).title is None


def test_revision_conflict_does_not_overwrite_newer_project(tmp_path):
    store = ProjectPersistenceService(tmp_path / "projects")
    project = store.create(ProjectDraft(original_prompt="first"))
    first_editor = store.load(project.project_id)
    second_editor = store.load(project.project_id)
    first_editor.title = "newer"
    store.save(first_editor, expected_revision=0)
    second_editor.title = "stale"

    with pytest.raises(RevisionConflictError):
        store.save(second_editor, expected_revision=0)

    assert store.load(project.project_id).title == "newer"
