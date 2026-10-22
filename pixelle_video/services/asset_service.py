"""Lossless project asset ingestion with real decoder inspection."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from pixelle_video.models.project import AssetRecord


class AssetValidationError(ValueError):
    """Raised when an asset cannot be decoded or violates an upload policy."""


class AssetService:
    """Preserve originals and create traceable previews without extension allowlists."""

    def __init__(
        self,
        root: str | Path = "data/assets",
        max_file_size_mb: int = 512,
        allowed_import_roots: Iterable[str | Path] = (),
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.allowed_import_roots = [Path(item).expanduser().resolve() for item in allowed_import_roots]

    def ingest_upload(self, project_id: str, filename: str, source: BinaryIO) -> AssetRecord:
        """Stream a browser upload to an immutable project asset directory."""
        asset_id = str(uuid.uuid4())
        original_dir = self.root / project_id / asset_id / "original"
        original_dir.mkdir(parents=True, exist_ok=False)
        safe_name = Path(filename).name or "upload.bin"
        destination = original_dir / safe_name
        partial = destination.with_suffix(f"{destination.suffix}.partial")
        digest = hashlib.sha256()
        total = 0
        try:
            with partial.open("xb") as handle:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_file_size:
                        raise AssetValidationError(
                            f"Asset exceeds configured limit of {self.max_file_size // 1024 // 1024} MB"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, destination)
            return self._inspect_and_derive(
                asset_id=asset_id,
                original_name=safe_name,
                path=destination,
                sha256=digest.hexdigest(),
                size_bytes=total,
            )
        except Exception:
            partial.unlink(missing_ok=True)
            if destination.exists():
                destination.unlink()
            shutil.rmtree(original_dir.parent, ignore_errors=True)
            raise

    def import_local(self, project_id: str, source_path: str | Path) -> AssetRecord:
        """Import from an explicitly allowed local directory without mutating the source."""
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise AssetValidationError("Local asset path is not a file")
        if not any(source.is_relative_to(root) for root in self.allowed_import_roots):
            raise AssetValidationError("Local asset is outside configured allowed import roots")
        if source.stat().st_size > self.max_file_size:
            raise AssetValidationError(
                f"Asset exceeds configured limit of {self.max_file_size // 1024 // 1024} MB"
            )
        with source.open("rb") as handle:
            return self.ingest_upload(project_id, source.name, handle)

    def build_crop_preview(
        self,
        asset: AssetRecord,
        canvas_width: int,
        canvas_height: int,
    ) -> str:
        """Create a small focus/fit preview without changing the original asset."""
        source = Path(asset.thumbnail_path or asset.original_path)
        if not source.is_file():
            raise AssetValidationError("Asset preview source is missing")
        width = min(480, canvas_width)
        height = max(1, round(width * canvas_height / canvas_width))
        if height > 480:
            height = 480
            width = max(1, round(height * canvas_width / canvas_height))
        target = source.parent / (
            f"crop-preview-{width}x{height}-{asset.fit_mode}-"
            f"{round(asset.focal_x, 3)}-{round(asset.focal_y, 3)}.png"
        )
        if not target.is_file():
            self._render_image_variant(source, target, width, height, asset)
        return str(target)

    def build_render_variant(
        self,
        asset: AssetRecord,
        width: int,
        height: int,
    ) -> str:
        """Create a traceable canvas-ready derivative while preserving the original."""
        source = Path(asset.original_path)
        if not source.is_file():
            raise AssetValidationError("Asset original is missing")
        derived_dir = source.parent.parent / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            f"{asset.sha256}:{width}x{height}:{asset.fit_mode}:"
            f"{asset.focal_x:.4f}:{asset.focal_y:.4f}".encode("utf-8")
        ).hexdigest()[:16]
        if asset.media_kind == "image":
            target = derived_dir / f"render-{width}x{height}-{key}.png"
            if not target.is_file():
                self._render_image_variant(source, target, width, height, asset)
            return str(target)
        if asset.media_kind == "video":
            target = derived_dir / f"render-{width}x{height}-{key}.mp4"
            if not target.is_file():
                self._render_video_variant(source, target, width, height, asset)
            return str(target)
        raise AssetValidationError("Only image and video assets can be used as scene visuals")

    @staticmethod
    def _render_image_variant(
        source: Path,
        target: Path,
        width: int,
        height: int,
        asset: AssetRecord,
    ) -> None:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            source_width, source_height = image.size
            if asset.fit_mode == "cover":
                scale = max(width / source_width, height / source_height)
                resized = image.resize(
                    (round(source_width * scale), round(source_height * scale)),
                    Image.Resampling.LANCZOS,
                )
                left = round((resized.width - width) * asset.focal_x)
                top = round((resized.height - height) * asset.focal_y)
                result = resized.crop((left, top, left + width, top + height))
            else:
                scale = min(width / source_width, height / source_height)
                resized = image.resize(
                    (round(source_width * scale), round(source_height * scale)),
                    Image.Resampling.LANCZOS,
                )
                result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                left = round((width - resized.width) * asset.focal_x)
                top = round((height - resized.height) * asset.focal_y)
                result.alpha_composite(resized, (left, top))
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f"{target.stem}.partial{target.suffix}")
            result.save(partial, "PNG", optimize=True)
            os.replace(partial, target)

    @staticmethod
    def _render_video_variant(
        source: Path,
        target: Path,
        width: int,
        height: int,
        asset: AssetRecord,
    ) -> None:
        if asset.fit_mode == "cover":
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:(iw-ow)*{asset.focal_x}:"
                f"(ih-oh)*{asset.focal_y}"
            )
        else:
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)*{asset.focal_x}:"
                f"(oh-ih)*{asset.focal_y}:black"
            )
        partial = target.with_name(f"{target.stem}.partial{target.suffix}")
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?", "-vf", video_filter,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-movflags", "+faststart", str(partial),
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=1800)
            os.replace(partial, target)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            partial.unlink(missing_ok=True)
            raise AssetValidationError("Failed to build the traceable video crop variant") from exc

    def _inspect_and_derive(
        self,
        *,
        asset_id: str,
        original_name: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> AssetRecord:
        try:
            return self._inspect_image(asset_id, original_name, path, sha256, size_bytes)
        except (UnidentifiedImageError, OSError, ValueError):
            pass
        try:
            return self._inspect_media(asset_id, original_name, path, sha256, size_bytes)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise AssetValidationError(
                "Asset is not decodable by Pillow or the installed FFprobe"
            ) from exc

    def _inspect_image(
        self,
        asset_id: str,
        original_name: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> AssetRecord:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            warnings = self._dimension_warnings(width, height)
            thumbnail_path = path.parent.parent / "derived" / "thumbnail.webp"
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            preview = image.copy()
            if preview.mode not in {"RGB", "RGBA"}:
                preview = preview.convert("RGBA" if has_alpha else "RGB")
            preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
            preview.save(thumbnail_path, "WEBP", quality=88, method=6)
            return AssetRecord(
                asset_id=asset_id,
                original_name=original_name,
                original_path=str(path),
                sha256=sha256,
                media_kind="image",
                size_bytes=size_bytes,
                mime_type=Image.MIME.get(image.format or "") or mimetypes.guess_type(path.name)[0],
                width=width,
                height=height,
                codec=image.format,
                has_alpha=has_alpha,
                orientation=self._orientation(width, height),
                thumbnail_path=str(thumbnail_path),
                quality_warnings=warnings,
            )

    def _inspect_media(
        self,
        asset_id: str,
        original_name: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> AssetRecord:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
        if not video and not audio:
            raise ValueError("No audio or video stream")
        media_kind = "video" if video else "audio"
        width = int(video["width"]) if video and video.get("width") else None
        height = int(video["height"]) if video and video.get("height") else None
        duration_value = (video or audio or {}).get("duration") or payload.get("format", {}).get("duration")
        duration = float(duration_value) if duration_value else None
        frame_rate = self._parse_rate(video.get("avg_frame_rate")) if video else None
        warnings = self._dimension_warnings(width, height) if width and height else []
        proxy_path = None
        thumbnail_path = None
        if video:
            derived_dir = path.parent.parent / "derived"
            derived_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_path = derived_dir / "thumbnail.jpg"
            thumb_command = [
                "ffmpeg", "-y", "-v", "error", "-ss", "0", "-i", str(path),
                "-frames:v", "1", "-vf", "scale=640:640:force_original_aspect_ratio=decrease",
                str(thumbnail_path),
            ]
            try:
                subprocess.run(thumb_command, capture_output=True, check=True, timeout=60)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                thumbnail_path = None
                warnings.append("thumbnail_generation_failed")
            proxy_path = derived_dir / "proxy.mp4"
            proxy_command = [
                "ffmpeg", "-y", "-v", "error", "-i", str(path),
                "-map", "0:v:0", "-map", "0:a?", "-vf",
                "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "23",
                "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart",
                str(proxy_path),
            ]
            try:
                subprocess.run(proxy_command, capture_output=True, check=True, timeout=1800)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                proxy_path = None
                warnings.append("proxy_generation_failed")
        stream = video or audio
        return AssetRecord(
            asset_id=asset_id,
            original_name=original_name,
            original_path=str(path),
            sha256=sha256,
            media_kind=media_kind,
            size_bytes=size_bytes,
            mime_type=mimetypes.guess_type(path.name)[0],
            width=width,
            height=height,
            codec=stream.get("codec_name"),
            frame_rate=frame_rate,
            duration=duration,
            has_alpha=bool(video and "a" in str(video.get("pix_fmt", ""))),
            orientation=self._orientation(width, height) if width and height else None,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            proxy_path=str(proxy_path) if proxy_path else None,
            quality_warnings=warnings,
        )

    @staticmethod
    def _parse_rate(value: str | None) -> float | None:
        if not value or value in {"0/0", "N/A"}:
            return None
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)

    @staticmethod
    def _orientation(width: int, height: int) -> str:
        if width == height:
            return "square"
        return "landscape" if width > height else "portrait"

    @staticmethod
    def _dimension_warnings(width: int, height: int) -> list[str]:
        warnings = []
        if min(width, height) < 720:
            warnings.append("low_resolution")
        if max(width, height) >= 3840:
            warnings.append("high_resolution_source_preserved")
        return warnings
