import hashlib
import json
from pathlib import Path

from pixelle_video.services.compatibility import _iter_strings, _normalize_model_path


def test_model_paths_match_across_windows_and_posix_separators():
    assert _normalize_model_path(
        r"wan-fusionx\WanT2V_MasterModel.safetensors"
    ) == _normalize_model_path("wan-fusionx/WanT2V_MasterModel.safetensors")


def test_iter_strings_reads_nested_comfyui_choices():
    contract = {
        "input": {
            "required": {
                "unet_name": [[r"wan-fusionx\WanT2V_MasterModel.safetensors"]]
            }
        }
    }

    assert r"wan-fusionx\WanT2V_MasterModel.safetensors" in set(_iter_strings(contract))


def test_contract_hash_prefers_the_runtime_data_override(tmp_path, monkeypatch):
    from pixelle_video.services.compatibility import CompatibilityService

    monkeypatch.setenv("PIXELLE_VIDEO_ROOT", str(tmp_path))
    default = tmp_path / "workflows" / "selfhost" / "video.json"
    override = tmp_path / "data" / "workflows" / "selfhost" / "video.json"
    default.parent.mkdir(parents=True)
    override.parent.mkdir(parents=True)
    default.write_bytes(b"default")
    override.write_bytes(b"override")
    service = CompatibilityService(
        root=tmp_path / "compatibility",
        project_root=tmp_path,
    )

    assert service._hash_contract("workflows", "selfhost/video.json") == hashlib.sha256(
        b"override"
    ).hexdigest()


def test_wan_workflow_binds_duration_to_a_dynamic_frame_count():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "workflows"
        / "selfhost"
        / "video_wan2.1_fusionx.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    assert workflow["52"]["_meta"]["title"] == "$duration.value"
    assert workflow["53"]["class_type"] == "ComfyMathExpression"
    assert workflow["53"]["inputs"]["values.a"] == ["52", 0]
    assert workflow["40"]["inputs"]["length"] == ["53", 1]
    assert "round(a * 16)" in workflow["53"]["inputs"]["expression"]
