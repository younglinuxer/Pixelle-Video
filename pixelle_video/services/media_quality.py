"""FFprobe/FFmpeg technical checks used by smoke and canary approval gates."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


class MediaQualityError(RuntimeError):
    """Raised when media cannot be inspected safely."""


class MediaQualityService:
    """Run deterministic media checks without modifying the generated originals."""

    def __init__(self, output_root: str | Path = "output"):
        self.output_root = Path(output_root).resolve()

    def inspect_video(
        self,
        path: str | Path,
        *,
        expected_width: int,
        expected_height: int,
        max_black_duration: float = 0.5,
        max_freeze_duration: float = 10.0,
        min_mean_volume_db: float = -45.0,
    ) -> dict[str, Any]:
        """Inspect one output and return raw evidence plus normalized pass/fail gates."""
        video_path = Path(path).resolve(strict=True)
        if not video_path.is_relative_to(self.output_root):
            raise MediaQualityError("QC input must be inside the configured output directory")
        probe = self._run_json(
            [
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                str(video_path),
            ]
        )
        streams = probe.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
        duration_value = probe.get("format", {}).get("duration") or (video or {}).get("duration")
        duration = float(duration_value) if duration_value else 0.0
        frame_rate = self._rate((video or {}).get("avg_frame_rate"))
        black_log = self._run_filter(
            video_path,
            f"blackdetect=d={max_black_duration}:pix_th=0.10",
        )
        freeze_log = self._run_filter(
            video_path,
            "freezedetect=n=-60dB:d=1",
        )
        volume_log = self._run_filter(video_path, "volumedetect", audio_only=True) if audio else ""
        black_durations = [float(value) for value in re.findall(r"black_duration:([0-9.]+)", black_log)]
        freeze_durations = [float(value) for value in re.findall(r"freeze_duration: ([0-9.]+)", freeze_log)]
        mean_match = re.search(r"mean_volume:\s*(-?inf|-?[0-9.]+) dB", volume_log)
        mean_volume = None
        if mean_match and mean_match.group(1) != "-inf":
            mean_volume = float(mean_match.group(1))
        checks = {
            "decode": video is not None and duration > 0,
            "dimensions": bool(
                video
                and int(video.get("width", 0)) == expected_width
                and int(video.get("height", 0)) == expected_height
            ),
            "frame_rate": frame_rate is not None and frame_rate > 0,
            "streams": video is not None and audio is not None,
            "duration": duration > 0,
            "black_frames": not black_durations or max(black_durations) <= max_black_duration,
            "frozen_frames": not freeze_durations or max(freeze_durations) <= max_freeze_duration,
            "audio_volume": mean_volume is not None and mean_volume >= min_mean_volume_db,
        }
        return {
            "path": str(video_path),
            "duration": duration,
            "width": int(video.get("width", 0)) if video else None,
            "height": int(video.get("height", 0)) if video else None,
            "frame_rate": frame_rate,
            "video_codec": video.get("codec_name") if video else None,
            "audio_codec": audio.get("codec_name") if audio else None,
            "max_black_duration": max(black_durations, default=0.0),
            "max_freeze_duration": max(freeze_durations, default=0.0),
            "mean_volume_db": mean_volume,
            "checks": checks,
        }

    def validate_canary(
        self,
        *,
        smoke_path: str,
        canary_paths: list[str],
        expected_width: int,
        expected_height: int,
        queue_history: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
        """Validate one smoke output and exactly three representative scene outputs."""
        if len(canary_paths) != 3:
            raise MediaQualityError("Exactly three canary scene outputs are required")
        all_paths = [
            str(Path(smoke_path).resolve()),
            *(str(Path(path).resolve()) for path in canary_paths),
        ]
        if len(set(all_paths)) != 4:
            raise MediaQualityError("Smoke and three canary outputs must be four distinct files")
        smoke = self.inspect_video(
            smoke_path,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        canaries = [
            self.inspect_video(
                path,
                expected_width=expected_width,
                expected_height=expected_height,
            )
            for path in canary_paths
        ]
        technical = {
            key: smoke["checks"][key] and all(item["checks"][key] for item in canaries)
            for key in smoke["checks"]
        }
        provider_output_ids = queue_history.get("provider_output_ids") or []
        technical["queue_history"] = bool(
            queue_history.get("queue_confirmed")
            and queue_history.get("history_confirmed")
            and isinstance(provider_output_ids, list)
            and len(provider_output_ids) >= 3
        )
        return smoke, {"scenes": canaries, "queue_history": queue_history}, technical

    @staticmethod
    def _run_json(command: list[str]) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return json.loads(completed.stdout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            raise MediaQualityError(f"Media probe failed: {type(exc).__name__}") from exc

    @staticmethod
    def _run_filter(path: Path, filter_value: str, audio_only: bool = False) -> str:
        command = ["ffmpeg", "-v", "info", "-i", str(path)]
        if audio_only:
            command.extend(["-af", filter_value, "-f", "null", "-"])
        else:
            command.extend(["-vf", filter_value, "-an", "-f", "null", "-"])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            return completed.stderr
        except subprocess.TimeoutExpired as exc:
            raise MediaQualityError("FFmpeg quality filter timed out") from exc

    @staticmethod
    def _rate(value: str | None) -> float | None:
        if not value or value in {"0/0", "N/A"}:
            return None
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
