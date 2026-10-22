"""Atomic filesystem persistence for staged projects and revision snapshots."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pixelle_video.models.project import ProjectDraft, utc_now


class ProjectNotFoundError(FileNotFoundError):
    """Raised when a requested project does not exist."""


class RevisionConflictError(RuntimeError):
    """Raised when a caller tries to overwrite a newer project revision."""

    def __init__(self, expected: int, actual: int):
        super().__init__(f"Project revision conflict: expected {expected}, current is {actual}")
        self.expected = expected
        self.actual = actual


class ProjectPersistenceService:
    """Store projects under ``data/projects`` with optimistic concurrency."""

    def __init__(self, root: str | Path = "data/projects"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _project_dir(self, project_id: str) -> Path:
        if not project_id or any(char not in "0123456789abcdef-" for char in project_id.lower()):
            raise ValueError("Invalid project ID")
        return self.root / project_id

    def _project_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def create(self, project: ProjectDraft) -> ProjectDraft:
        """Create a new project without overwriting an existing ID."""
        with self._lock:
            path = self._project_file(project.project_id)
            if path.exists():
                raise FileExistsError(f"Project already exists: {project.project_id}")
            project.updated_at = utc_now()
            self._atomic_json_write(path, project.model_dump(mode="json"))
            self._write_revision(project)
            return project.model_copy(deep=True)

    def load(self, project_id: str) -> ProjectDraft:
        """Load the latest revision."""
        path = self._project_file(project_id)
        if not path.exists():
            raise ProjectNotFoundError(f"Project not found: {project_id}")
        with path.open("r", encoding="utf-8") as handle:
            return ProjectDraft.model_validate(json.load(handle))

    def save(self, project: ProjectDraft, expected_revision: int) -> ProjectDraft:
        """Save a mutation if the caller still owns the latest revision."""
        with self._lock:
            current = self.load(project.project_id)
            if current.revision != expected_revision:
                raise RevisionConflictError(expected_revision, current.revision)
            updated = project.model_copy(deep=True)
            updated.revision = current.revision + 1
            updated.updated_at = utc_now()
            self._atomic_json_write(
                self._project_file(updated.project_id),
                updated.model_dump(mode="json"),
            )
            self._write_revision(updated)
            return updated

    def list(self, limit: int = 100) -> list[ProjectDraft]:
        """List latest projects, newest first, ignoring malformed runtime files."""
        projects: list[ProjectDraft] = []
        for path in self.root.glob("*/project.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    projects.append(ProjectDraft.model_validate(json.load(handle)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        projects.sort(key=lambda item: item.updated_at, reverse=True)
        return projects[:limit]

    def load_revision(self, project_id: str, revision: int) -> ProjectDraft:
        """Load an immutable revision snapshot."""
        path = self._project_dir(project_id) / "revisions" / f"{revision:06d}.json"
        if not path.exists():
            raise ProjectNotFoundError(
                f"Project revision not found: {project_id}@{revision}"
            )
        with path.open("r", encoding="utf-8") as handle:
            return ProjectDraft.model_validate(json.load(handle))

    def _write_revision(self, project: ProjectDraft) -> None:
        path = self._project_dir(project.project_id) / "revisions" / f"{project.revision:06d}.json"
        self._atomic_json_write(path, project.model_dump(mode="json"))
