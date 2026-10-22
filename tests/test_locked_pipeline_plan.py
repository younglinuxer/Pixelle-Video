import pytest

from pixelle_video.pipelines.linear import PipelineContext
from pixelle_video.pipelines.standard import StandardPipeline


class NoLLM:
    async def __call__(self, *args, **kwargs):
        raise AssertionError("Locked production must not call the LLM")


class NoFrameProcessor:
    async def __call__(self, *args, **kwargs):
        raise AssertionError("An unchanged scene must reuse its segment")


@pytest.mark.asyncio
async def test_locked_narrations_and_prompts_bypass_planning_calls():
    core = type(
        "Core",
        (),
        {"llm": NoLLM(), "tts": None, "media": None, "video": None},
    )()
    pipeline = StandardPipeline(core)
    context = PipelineContext(
        input_text="ignored",
        params={
            "planned_narrations": ["n1", "n2"],
            "planned_image_prompts": ["p1", "p2"],
        },
    )

    await pipeline.generate_content(context)
    await pipeline.plan_visuals(context)

    assert context.narrations == ["n1", "n2"]
    assert context.image_prompts == ["p1", "p2"]


@pytest.mark.asyncio
async def test_unchanged_scene_reuses_previous_segment(tmp_path):
    segment = tmp_path / "scene.mp4"
    segment.write_bytes(b"existing-segment")
    core = type(
        "Core",
        (),
        {
            "llm": NoLLM(),
            "tts": None,
            "media": None,
            "video": None,
            "frame_processor": NoFrameProcessor(),
        },
    )()
    pipeline = StandardPipeline(core)
    context = PipelineContext(
        input_text="ignored",
        params={
            "planned_narrations": ["n1"],
            "planned_image_prompts": ["p1"],
            "scene_ids": ["scene-1"],
            "reusable_frames": {
                "scene-1": {
                    "video_segment_path": str(segment),
                    "duration": 2.5,
                    "media_type": "image",
                }
            },
            "title": "title",
            "media_width": 1024,
            "media_height": 1024,
            "media_workflow": "selfhost/image_flux.json",
            "frame_template": "1080x1920/image_default.html",
        },
    )
    context.task_id = "task"
    context.title = "title"

    await pipeline.generate_content(context)
    await pipeline.plan_visuals(context)
    await pipeline.initialize_storyboard(context)
    await pipeline.produce_assets(context)

    assert context.storyboard.frames[0].video_segment_path == str(segment)
    assert context.storyboard.total_duration == 2.5


@pytest.mark.asyncio
async def test_locked_project_asset_is_attached_without_media_generation(tmp_path):
    image = tmp_path / "high-quality-source.tiff"
    image.write_bytes(b"immutable-original")
    core = type(
        "Core",
        (),
        {"llm": NoLLM(), "tts": None, "media": None, "video": None},
    )()
    pipeline = StandardPipeline(core)
    context = PipelineContext(
        input_text="ignored",
        params={
            "planned_narrations": ["n1"],
            "planned_image_prompts": [None],
            "planned_media": [
                {
                    "source_asset_ids": ["asset-1"],
                    "media_type": "image",
                    "image_path": str(image),
                }
            ],
            "title": "title",
            "media_width": 1024,
            "media_height": 1024,
            "frame_template": "1080x1920/image_default.html",
        },
    )
    context.task_id = "task"
    context.title = "title"

    await pipeline.generate_content(context)
    await pipeline.plan_visuals(context)
    await pipeline.initialize_storyboard(context)

    frame = context.storyboard.frames[0]
    assert frame.image_prompt is None
    assert frame.image_path == str(image)
    assert frame.source_asset_ids == ["asset-1"]
