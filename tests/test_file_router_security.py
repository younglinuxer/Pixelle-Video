from pathlib import Path

import pytest
from fastapi import HTTPException

from api.routers.files import _resolve_allowed_file


def test_resolves_prefixed_and_legacy_output_paths(tmp_path: Path):
    video = tmp_path / "output" / "task-1" / "final.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    assert _resolve_allowed_file("output/task-1/final.mp4", tmp_path) == video.resolve()
    assert _resolve_allowed_file("task-1/final.mp4", tmp_path) == video.resolve()


@pytest.mark.parametrize(
    "requested_path",
    [
        "output/../config.yaml",
        "templates/../../config.yaml",
        "/etc/passwd",
        "C:/Windows/win.ini",
    ],
)
def test_rejects_traversal_and_absolute_paths(tmp_path: Path, requested_path: str):
    (tmp_path / "output").mkdir()
    (tmp_path / "templates").mkdir()
    (tmp_path / "config.yaml").write_text("secret", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        _resolve_allowed_file(requested_path, tmp_path)

    assert exc_info.value.status_code == 403


def test_rejects_symlink_escape(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    secret = tmp_path / "config.yaml"
    secret.write_text("secret", encoding="utf-8")
    (output_dir / "leak").symlink_to(secret)

    with pytest.raises(HTTPException) as exc_info:
        _resolve_allowed_file("output/leak", tmp_path)

    assert exc_info.value.status_code == 403
