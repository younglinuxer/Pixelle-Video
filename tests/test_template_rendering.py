from pathlib import Path

import pytest
from PIL import Image

from pixelle_video.services.frame_html import HTMLFrameGenerator


@pytest.mark.asyncio
async def test_supported_canvas_templates_render(tmp_path):
    root = Path(__file__).resolve().parents[1]
    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (640, 640), (0, 0, 0, 0)).save(transparent)
    cases = [
        ("templates/1080x1920/image_long_text.html", (1080, 1920), "长文本测试。" * 35),
        ("templates/1920x1080/image_full.html", (1920, 1080), "Short landscape copy"),
        ("templates/1080x1080/image_minimal_framed.html", (1080, 1080), "Square copy"),
    ]
    try:
        for template, expected_size, text in cases:
            output = tmp_path / f"{expected_size[0]}x{expected_size[1]}.png"
            generator = HTMLFrameGenerator(str(root / template))
            await generator.generate_frame(
                title="Template validation",
                text=text,
                image=str(transparent),
                output_path=str(output),
            )
            with Image.open(output) as rendered:
                assert rendered.size == expected_size
                assert rendered.getbbox() is not None
    finally:
        await HTMLFrameGenerator.close_browser()
