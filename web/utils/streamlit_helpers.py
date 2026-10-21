# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Streamlit helper functions
"""

import hashlib
import json
import time

import streamlit as st
import streamlit.components.v1 as components

from pixelle_video.config import config_manager
from pixelle_video.services.preflight import PreflightIssue, PreflightReport, PreflightService
from web.i18n import tr
from web.utils.async_helpers import run_async


def safe_rerun():
    """Safe rerun that works with both old and new Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


# ============================================================================
# SelfHost Workflow Preflight - Alert Only When Blocking
# ============================================================================
# The compatibility check is silent on success. A native browser alert is used
# only for blocking findings so the user sees the failure before generation.

def check_and_warn_selfhost_workflow(workflow_path: str) -> bool:
    """Preflight a self-hosted workflow and alert only on blocking findings."""
    if not workflow_path:
        return True
    
    is_selfhost = workflow_path.startswith("selfhost/")

    if not is_selfhost:
        return True

    comfyui = config_manager.config.comfyui
    cache_key = "|".join(
        [
            workflow_path,
            comfyui.comfyui_url,
            hashlib.sha256((comfyui.comfyui_api_key or "").encode()).hexdigest(),
        ]
    )
    cache = st.session_state.setdefault("selfhost_preflight_cache", {})
    cached = cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < PreflightService.CACHE_TTL_SECONDS:
        report = cached[1]
    else:
        report = run_async(
            PreflightService().check_requirements(workflows=[workflow_path])
        )
        cache[cache_key] = (time.monotonic(), report)

    # A healthy backend and compatible graph require no acknowledgement.
    if report.ok:
        st.session_state.pop("last_selfhost_preflight_warning", None)
        return True

    # Avoid reopening the same native alert on every Streamlit rerun.
    warning_key = (cache_key, tuple(issue.code for issue in report.errors))
    if st.session_state.get("last_selfhost_preflight_warning") != warning_key:
        details = "\n".join(
            f"- {_translate_preflight_issue(issue)}" for issue in report.errors
        )
        _show_js_alert(workflow_path, details=details)
        st.session_state["last_selfhost_preflight_warning"] = warning_key
    return False


def _show_js_alert(workflow_path: str, details: str = ""):
    """
    Show a native JavaScript alert with selfhost workflow warning.
    
    Args:
        workflow_path: The workflow path to display in the alert
    """
    # Get ComfyUI URL from config
    comfyui_config = config_manager.get_comfyui_config()
    comfyui_url = comfyui_config.get("comfyui_url", "http://localhost:8188")
    
    # Build alert message
    title = tr("selfhost.warning.title")
    message = tr("selfhost.warning.message", 
                 comfyui_url=comfyui_url, 
                 workflow_path=f"workflows/{workflow_path}")
    hint = tr("selfhost.warning.hint")
    
    # Clean up markdown formatting for plain text alert
    # Remove ** (bold markers) and other markdown
    message = message.replace("**", "").replace("*", "")
    hint = hint.replace("**", "").replace("*", "")
    
    # Combine into a JSON-encoded JavaScript string to prevent HTML/JS injection.
    full_message = f"{title}\n\n{message}"
    if details:
        full_message += f"\n\n{tr('preflight.failed.title')}:\n{details}"
    full_message += f"\n\n{hint}"
    encoded_message = (
        json.dumps(full_message)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    
    # Inject JavaScript alert
    js_code = f"""
    <script>
        alert({encoded_message});
    </script>
    """
    
    components.html(js_code, height=0, width=0)


def render_preflight_report(report: PreflightReport) -> bool:
    """Render blocking findings only; successful preflight stays silent."""
    if report.ok:
        return True

    st.error(tr("preflight.failed.title"))
    for issue in report.errors:
        st.error(f"[{issue.code}] {_translate_preflight_issue(issue)}")
    return False


def _translate_preflight_issue(issue: PreflightIssue) -> str:
    """Translate known Web-facing preflight findings and preserve API detail as fallback."""
    return tr(f"preflight.issue.{issue.code}", fallback=issue.message)

