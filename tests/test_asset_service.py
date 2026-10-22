import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from pixelle_video.services.asset_service import AssetService


def _image_bytes(format_name: str, size=(1200, 800)) -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", size, "navy").save(stream, format=format_name)
    stream.seek(0)
    return stream


def test_image_is_accepted_by_decoder_not_filename_extension(tmp_path):
    service = AssetService(tmp_path / "assets")

    asset = service.ingest_upload("project", "quality-source.unknown", _image_bytes("TIFF"))

    assert asset.media_kind == "image"
    assert asset.codec == "TIFF"
    assert asset.width == 1200
    assert asset.height == 800
    assert asset.sha256
    assert asset.thumbnail_path
    assert (tmp_path / "assets" / "project" / asset.asset_id / "original").is_dir()


def test_high_resolution_original_is_preserved(tmp_path):
    service = AssetService(tmp_path / "assets")
    source = _image_bytes("PNG", size=(4000, 2200))
    original_bytes = source.getvalue()

    asset = service.ingest_upload("project", "4k.png", source)

    assert "high_resolution_source_preserved" in asset.quality_warnings
    assert Path(asset.original_path).read_bytes() == original_bytes


def test_crop_preview_and_render_variant_are_traceable_derivatives(tmp_path):
    service = AssetService(tmp_path / "assets")
    source = _image_bytes("PNG", size=(1600, 900))
    original_bytes = source.getvalue()
    asset = service.ingest_upload("project", "wide.png", source)
    asset = asset.model_copy(update={"fit_mode": "cover", "focal_x": 0.8, "focal_y": 0.4})

    preview = Path(service.build_crop_preview(asset, 1080, 1920))
    rendered = Path(service.build_render_variant(asset, 540, 960))

    assert preview.is_file()
    assert rendered.is_file()
    with Image.open(rendered) as image:
        assert image.size == (540, 960)
    assert Path(asset.original_path).read_bytes() == original_bytes


@pytest.mark.parametrize("format_name", ["JPEG", "PNG", "WEBP", "TIFF"])
def test_common_image_decoders(format_name, tmp_path):
    service = AssetService(tmp_path / "assets")

    asset = service.ingest_upload("project", "no-extension", _image_bytes(format_name))

    assert asset.media_kind == "image"
    assert asset.width == 1200


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
@pytest.mark.parametrize(
    ("suffix", "codec"),
    [("mp4", "libx264"), ("mov", "prores_ks"), ("mkv", "libx265"), ("webm", "libvpx-vp9")],
)
def test_video_containers_and_environment_codecs_are_inspected(suffix, codec, tmp_path):
    source = tmp_path / f"source.{suffix}"
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=320x240:rate=12:duration=1", "-c:v", codec,
    ]
    if codec == "libx265":
        command.extend(["-x265-params", "log-level=error"])
    command.append(str(source))
    subprocess.run(command, check=True)
    service = AssetService(tmp_path / "assets")

    with source.open("rb") as handle:
        asset = service.ingest_upload("project", source.name, handle)

    assert asset.media_kind == "video"
    assert asset.width == 320
    assert asset.height == 240
    assert asset.frame_rate == 12
    assert asset.proxy_path and Path(asset.proxy_path).is_file()
    assert Path(asset.original_path).read_bytes() == source.read_bytes()
