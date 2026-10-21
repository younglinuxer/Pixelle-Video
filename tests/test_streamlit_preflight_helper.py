from pixelle_video.services.preflight import PreflightReport
from web.utils import streamlit_helpers


def _stub_preflight(monkeypatch, report: PreflightReport) -> None:
    def fake_run_async(coroutine):
        coroutine.close()
        return report

    monkeypatch.setattr(streamlit_helpers, "run_async", fake_run_async)
    monkeypatch.setattr(streamlit_helpers.st, "session_state", {})


def test_successful_selfhost_preflight_does_not_show_alert(monkeypatch):
    report = PreflightReport(checks=["workflow:selfhost/image_flux.json"])
    _stub_preflight(monkeypatch, report)
    shown_alerts = []
    monkeypatch.setattr(
        streamlit_helpers,
        "_show_js_alert",
        lambda *args, **kwargs: shown_alerts.append((args, kwargs)),
    )

    allowed = streamlit_helpers.check_and_warn_selfhost_workflow(
        "selfhost/image_flux.json"
    )

    assert allowed is True
    assert shown_alerts == []


def test_failed_selfhost_preflight_shows_one_deduplicated_alert(monkeypatch):
    report = PreflightReport()
    report.add_error("comfyui.unreachable", "ComfyUI is unreachable.")
    _stub_preflight(monkeypatch, report)
    shown_alerts = []
    monkeypatch.setattr(
        streamlit_helpers,
        "_show_js_alert",
        lambda *args, **kwargs: shown_alerts.append((args, kwargs)),
    )

    first_allowed = streamlit_helpers.check_and_warn_selfhost_workflow(
        "selfhost/image_flux.json"
    )
    second_allowed = streamlit_helpers.check_and_warn_selfhost_workflow(
        "selfhost/image_flux.json"
    )

    assert first_allowed is False
    assert second_allowed is False
    assert len(shown_alerts) == 1


def test_selfhost_alert_escapes_script_terminators(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        streamlit_helpers.config_manager,
        "get_comfyui_config",
        lambda: {"comfyui_url": "http://test/</script><script>evil()"},
    )
    monkeypatch.setattr(
        streamlit_helpers.components,
        "html",
        lambda content, **kwargs: rendered.append(content),
    )

    streamlit_helpers._show_js_alert("selfhost/image_flux.json")

    assert len(rendered) == 1
    assert "http://test/</script><script>evil()" not in rendered[0]
    assert "\\u003c/script\\u003e\\u003cscript\\u003eevil()" in rendered[0]
