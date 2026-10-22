import shutil
import subprocess

import pytest

from pixelle_video.services.media_quality import MediaQualityError, MediaQualityService


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_canary_qc_requires_three_decodable_av_outputs(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    paths = []
    for index in range(4):
        path = output_root / f"sample-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2:r=24",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                str(path),
            ],
            check=True,
        )
        paths.append(str(path))
    service = MediaQualityService(output_root)

    smoke, canary, checks = service.validate_canary(
        smoke_path=paths[0],
        canary_paths=paths[1:],
        expected_width=320,
        expected_height=240,
        queue_history={
            "queue_confirmed": True,
            "history_confirmed": True,
            "provider_output_ids": ["one", "two", "three"],
        },
    )

    assert smoke["video_codec"] == "h264"
    assert len(canary["scenes"]) == 3
    assert all(checks.values())


def test_qc_refuses_paths_outside_output_root(tmp_path):
    service = MediaQualityService(tmp_path / "output")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not-video")

    with pytest.raises(MediaQualityError):
        service.inspect_video(outside, expected_width=1, expected_height=1)


def test_canary_qc_requires_four_distinct_outputs(tmp_path):
    service = MediaQualityService(tmp_path / "output")

    with pytest.raises(MediaQualityError, match="four distinct files"):
        service.validate_canary(
            smoke_path="output/same.mp4",
            canary_paths=["output/same.mp4", "output/two.mp4", "output/three.mp4"],
            expected_width=1080,
            expected_height=1920,
            queue_history={},
        )
