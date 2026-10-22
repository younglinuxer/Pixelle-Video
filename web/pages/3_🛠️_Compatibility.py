"""Optional operator-facing ComfyUI runtime diagnostics."""

import streamlit as st

from web.i18n import tr
from web.state.session import get_pixelle_video, init_i18n, init_session_state
from web.utils.async_helpers import run_async

init_session_state()
init_i18n()
core = get_pixelle_video()


def status_label(status) -> str:
    return tr(f"compat.status.{status.value}")


def show_runtime_findings(record) -> None:
    runtime = record.technical_checks.get("runtime", {})
    missing_nodes = runtime.get("missing_nodes", [])
    missing_models = runtime.get("missing_models", [])
    if missing_nodes:
        st.error(f"{tr('compat.missing_nodes')}: {', '.join(missing_nodes)}")
    if missing_models:
        st.error(f"{tr('compat.missing_models')}: {', '.join(missing_models)}")
    if runtime.get("message"):
        st.error(runtime["message"])


st.title(tr("compat.title"))
st.caption(tr("compat.caption"))

st.subheader(tr("compat.audit_title"))
selfhost_workflows = [
    item.key for item in core.render_catalog.list_workflows() if item.source == "selfhost"
]
workflow = st.selectbox(tr("compat.workflow"), selfhost_workflows)
if st.button(tr("compat.audit_button"), type="primary"):
    try:
        with st.spinner(tr("compat.auditing")):
            st.session_state.compatibility_audit = run_async(
                core.compatibility.inspect_workflow(workflow)
            )
    except Exception as exc:
        st.error(str(exc))
audit = st.session_state.get("compatibility_audit")
if audit:
    st.success(tr("compat.ready") if audit.get("ready") else tr("compat.not_ready"))
    st.json(audit)

st.caption(tr("compat.repair_unavailable"))

st.subheader(tr("compat.records_title"))
records = core.compatibility.list_records()
if not records:
    st.info(tr("compat.inspect_first"))
else:
    rows = []
    for record in records[:50]:
        runtime = record.technical_checks.get("runtime", {})
        rows.append(
            {
                tr("compat.combination"): record.combination_hash[:16],
                tr("compat.style"): record.style_id,
                tr("compat.current_status"): status_label(record.status),
                tr("compat.missing_nodes"): ", ".join(runtime.get("missing_nodes", [])),
                tr("compat.missing_models"): ", ".join(runtime.get("missing_models", [])),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
