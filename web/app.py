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
Pixelle-Video Web UI - Main Entry Point

This is the entry point for the Streamlit multi-page application.
Uses st.navigation to define pages and set the default page to Home.
"""

# ruff: noqa: E402 -- Streamlit script mode adds the repository root before imports.

import sys
from pathlib import Path

# Add project root to sys.path for module imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

# Setup page config (must be first Streamlit command)
st.set_page_config(
    page_title="Pixelle-Video - AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _apply_responsive_styles() -> None:
    """Keep creator controls usable at narrow widths and browser zoom up to 200%."""
    st.markdown(
        """
        <style>
        @media (max-width: 900px) {
            div[data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.75rem;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }
            div[data-testid="stButtonGroup"] {
                flex-wrap: wrap;
            }
            div[data-testid="stTabs"] [role="tablist"] {
                overflow-x: auto;
                scrollbar-width: thin;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main entry point with navigation"""
    _apply_responsive_styles()
    # Define pages using st.Page
    home_page = st.Page(
        "pages/1_🎬_Home.py",
        title="Home",
        icon="🎬",
        default=True
    )
    
    history_page = st.Page(
        "pages/2_📚_History.py",
        title="History",
        icon="📚"
    )

    # Runtime diagnostics remain available as a standalone operator page, but
    # ordinary creators should only see creation and history in the main app.
    pg = st.navigation([home_page, history_page])
    pg.run()


if __name__ == "__main__":
    main()
