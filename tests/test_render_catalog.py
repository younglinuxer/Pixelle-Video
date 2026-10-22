from pathlib import Path

from pixelle_video.services.render_catalog import RenderCatalogService


def test_catalog_exposes_only_supported_canvas_templates_and_workflow_contracts():
    root = Path(__file__).resolve().parents[1]
    catalog = RenderCatalogService(root)

    templates = catalog.list_templates()
    workflows = catalog.list_workflows()

    assert templates
    assert {(item.canvas_width, item.canvas_height) for item in templates} <= {
        (1080, 1920),
        (1920, 1080),
        (1080, 1080),
    }
    assert all(item.source_hash for item in templates)
    assert workflows
    assert all(item.graph_hash for item in workflows)
    assert all(item.required_nodes for item in workflows if item.source == "selfhost")
    assert all(item.provider_workflow_id for item in workflows if item.source == "runninghub")
    flux = next(item for item in workflows if item.key == "selfhost/image_flux.json")
    assert "width.value" in flux.dsl_bindings
    assert flux.native_width == 1024

    h3 = next(
        item for item in workflows if item.key == "selfhost/video_minimax_h3_t2v.json"
    )
    assert h3.media_type == "video"
    assert {"prompt.prompt!", "duration.value", "width.value", "height.value"} <= set(
        h3.dsl_bindings
    )
    assert "MiniMaxH3ImageToVideo" in h3.required_nodes
    assert "MiniMaxH3SigmaShift" in h3.required_nodes
    assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in h3.model_files
    assert "SaveVideo" in h3.output_node_types
