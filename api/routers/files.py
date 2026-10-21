# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File service endpoints

Provides access to generated files (videos, images, audio) and resource files.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

router = APIRouter(prefix="/files", tags=["Files"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIRECTORIES = (
    "output",
    "workflows",
    "templates",
    "bgm",
    "data/bgm",
    "data/templates",
    "resources",
)


def _resolve_allowed_file(file_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a user path and reject traversal and symlink escapes."""
    raw_path = file_path.replace("\\", "/")
    first_component = raw_path.split("/", 1)[0]
    if raw_path.startswith("/") or Path(raw_path).is_absolute() or ":" in first_component:
        raise HTTPException(status_code=403, detail="Access denied")

    normalized = raw_path.lstrip("/")
    has_allowed_prefix = any(
        normalized == directory or normalized.startswith(f"{directory}/")
        for directory in ALLOWED_DIRECTORIES
    )
    relative_path = normalized if has_allowed_prefix else f"output/{normalized}"

    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    allowed_roots = [(root / directory).resolve() for directory in ALLOWED_DIRECTORIES]

    for allowed_root in allowed_roots:
        try:
            candidate.relative_to(allowed_root)
            return candidate
        except ValueError:
            continue

    raise HTTPException(status_code=403, detail="Access denied")


@router.get("/{file_path:path}")
async def get_file(file_path: str):
    """
    Get file by path
    
    Serves files from allowed directories:
    - output/ - Generated files (videos, images, audio)
    - workflows/ - ComfyUI workflow files
    - templates/ - HTML templates
    - bgm/ - Background music
    - data/bgm/ - Custom background music
    - data/templates/ - Custom templates
    - resources/ - Other resources (images, fonts, etc.)
    
    - **file_path**: File path relative to allowed directories
    
    Examples:
    - "abc123.mp4" → output/abc123.mp4
    - "workflows/runninghub/image_flux.json" → workflows/runninghub/image_flux.json
    - "templates/1080x1920/default.html" → templates/1080x1920/default.html
    - "bgm/default.mp3" → bgm/default.mp3
    - "resources/example.png" → resources/example.png
    
    Returns file for download or preview.
    """
    try:
        abs_path = _resolve_allowed_file(file_path)
        
        if not abs_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if not abs_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
        
        # Determine media type
        suffix = abs_path.suffix.lower()
        media_types = {
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.html': 'text/html',
            '.json': 'application/json',
        }
        media_type = media_types.get(suffix, 'application/octet-stream')
        
        # Use inline disposition for browser preview
        return FileResponse(
            path=str(abs_path),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{abs_path.name}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File access error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

