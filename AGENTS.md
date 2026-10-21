# Repository Guidelines

## Scope and Instruction Priority

This file applies to the entire repository. Follow more specific `AGENTS.md` files if they are added below a subdirectory. Preserve existing behavior unless the task explicitly requests a breaking change. Prefer small, reviewable changes and avoid unrelated cleanup. Never modify generated output, local credentials, or user data as part of a source change.

## Project Structure & Module Organization

- `pixelle_video/` is the core Python package.
  - `pipelines/` coordinates complete video-generation flows. `LinearVideoPipeline` defines the standard lifecycle: setup, content, title, visual planning, storyboard initialization, asset production, post-production, and finalization.
  - `services/` contains LLM, TTS, image/video, HTML-frame, history, and persistence integrations.
  - `models/` contains Pydantic/domain models; `config/` owns configuration loading and validation; `utils/` contains shared helpers.
- `api/` is the FastAPI service. Add HTTP contracts in `schemas/`, endpoints in `routers/`, and background-task behavior in `tasks/`.
- `web/` is the Streamlit UI. Page-level flows belong in `pages/` or `pipelines/`; reusable UI belongs in `components/`; session state belongs in `state/`.
- `templates/<WIDTH>x<HEIGHT>/` stores HTML frame templates. Directory names are parsed as output dimensions.
- `workflows/selfhost/` and `workflows/runninghub/` store provider-specific workflow JSON. They are separate execution contracts, not interchangeable assets.
- `docs/en/` and `docs/zh/` contain paired documentation. `resources/` and `bgm/` contain committed static assets.
- `data/`, `output/`, `config.yaml`, logs, and media outputs are local runtime artifacts and are ignored by Git.

## Architecture and Data Flow

The Streamlit UI and FastAPI routers call the same core service and pipeline layer. A normal linear run creates a `PipelineContext`, builds narrations and visual prompts, creates a storyboard, processes each frame, concatenates segments, adds audio/BGM, and persists results. Keep orchestration in pipelines and provider details in services; do not call Streamlit APIs from core code or place business logic in routers.

`api.tasks.TaskManager` tracks active API jobs in memory, so task state does not survive a process restart. Durable generation artifacts are stored under `output/<task_id>/`, typically as `metadata.json`, `storyboard.json`, `frames/`, and `final.mp4`. Do not describe an in-memory API task as durable merely because its media files persist.

## Setup, Build, and Development Commands

Python 3.11+, `uv`, FFmpeg/FFprobe, and a Chromium installation usable by Playwright are expected.

```bash
uv sync --extra dev
uv run playwright install chromium
cp config.example.yaml config.yaml
```

Use these commands from the repository root:

- `uv run streamlit run web/app.py` — start the Web UI on port 8501.
- `uv run uvicorn api.app:app --reload --port 8000` — start the REST API; Swagger is at `/docs`.
- `uv run python api/app.py --host 0.0.0.0 --port 8000` — run the API through its script entry point.
- `uv run pytest` — run tests configured under `tests/`.
- `uv run ruff check .` — enforce import ordering and the configured `E`, `F`, and `I` rules.
- `uv build` — build the source distribution and wheel.
- `docker compose up --build` — build and run the API and Web UI together.
- `mkdocs serve` — preview documentation after installing `requirements-docs.txt`.

Do not claim provider-backed generation works solely because the UI or API starts. A real end-to-end check requires valid local configuration and an available ComfyUI, RunningHub, or direct API provider.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8, English identifiers/comments, and UTF-8 files. Ruff targets Python 3.11 with a 100-character line length; `E501` is intentionally ignored. Keep imports sorted, remove unused imports, and add type hints to public functions and cross-layer interfaces.

Use `snake_case` for modules, functions, variables, and JSON-facing field names; `PascalCase` for classes and Pydantic models; and `UPPER_SNAKE_CASE` for constants. Async I/O functions should remain `async` through the call chain rather than hiding event-loop calls. Use `pathlib.Path` for filesystem work and Loguru for operational logging. User-facing Web text must use the existing i18n mechanism in `web/i18n/locales/`; update both `en_US.json` and `zh_CN.json` when adding labels.

Keep request/response validation in Pydantic schemas. Avoid broad exception swallowing: log actionable context, preserve the original exception where possible, and update task/persistence status on failure.

## Templates, Workflows, and Media Changes

Template paths must follow `<WIDTH>x<HEIGHT>/<name>.html`, such as `1080x1920/image_default.html`. Preserve the filename prefixes used for behavior detection: `static_` skips generated media, `image_` requests images, `video_` requests video, and `asset_` is used by uploaded-asset flows. Validate template rendering at its declared dimensions and check text overflow, missing variables, and transparent/blank frames.

Workflow paths use `<source>/<service>.json`, for example `selfhost/image_flux.json`. When editing workflow JSON:

1. Verify it is exported in ComfyUI API format.
2. Confirm required node types exist on the target backend.
3. Check every DSL-bound node title/input against the actual graph.
4. Run a representative request and inspect queue/history or provider results.
5. Probe produced media with FFprobe; execution duration is not media duration.

Keep self-hosted, RunningHub, and direct API behavior explicit. A successful test of one does not validate the others.

## Testing and Verification

Pytest and `pytest-asyncio` are the supported frameworks; async mode is `auto`. The configured location is `tests/`, although this checkout currently has no committed test suite. Add files as `tests/test_<module>.py` and functions as `test_<behavior>`. Note that the current `.gitignore` pattern `test_*.py` also ignores conventionally named tests; update that rule when introducing a real test suite instead of silently force-adding files.

Mock paid or remote LLM, ComfyUI, RunningHub, TTS, and media APIs in unit tests. Cover success, validation failure, timeout, cancellation, retry, empty provider response, and partial-output cleanup. For a changed area, use the narrowest adequate verification:

- Core/service change: targeted Pytest plus Ruff.
- API change: targeted tests and a `/health` or endpoint smoke test.
- Streamlit change: launch the page and capture before/after screenshots.
- Template change: render representative short and long text at the target resolution.
- Media/pipeline change: inspect files with FFprobe and verify audio/video duration, stream presence, and final playback.
- Documentation-only change: check links, commands, and `mkdocs build` when practical.

Separate source-level checks from external validation. If credentials, models, GPU nodes, or provider access are unavailable, report the remaining check as `PENDING_EXTERNAL`; do not fabricate a successful render.

## Security & Configuration

Create local settings from `config.example.yaml`; never commit `config.yaml`, `.env`, API keys, access/secret keys, tokens, signed URLs, or provider response bodies containing credentials. Do not log secrets or include them in screenshots and fixtures. Keep example values inert. Treat `data/` and `output/` as potentially sensitive because they may contain uploads, narration, prompts, task metadata, and generated media.

Avoid changing default provider URLs or proxy behavior without documenting the compatibility and security impact. Do not delete or overwrite existing runtime output during tests; use a temporary directory or a unique task ID.

## Commit and Pull Request Guidelines

Recent history uses short imperative subjects and commonly prefixes fixes with `fix:`. Examples include `fix: handle videos without audio` and `fix: lazy load API clients`. Keep commits focused, describe the observable change, and avoid mixing formatting with behavior changes.

Pull requests should include:

- the problem and implemented solution;
- affected UI/API/pipeline/provider paths;
- exact verification commands and results;
- linked issues when applicable;
- screenshots for Streamlit or template changes;
- sample request/response shapes for API changes, with secrets removed;
- compatibility notes for configuration or workflow changes;
- explicit `PENDING_EXTERNAL` items and rollback considerations.

Before handing off a change, inspect `git diff`, run `git diff --check`, confirm that no runtime artifacts or secrets were added, and summarize both completed verification and untested external dependencies.

## Agent-Specific Working Rules

Start with read-only inspection of the relevant entry point, call chain, configuration schema, and existing tests. Preserve user changes in a dirty worktree and do not reformat unrelated files. Prefer `rg`/`rg --files` for discovery and make edits with focused patches. Do not start paid generation, change credentials, delete outputs, or restart shared external services unless the task explicitly requires it. When modifying a pipeline, trace the full lifecycle and confirm that progress, cancellation, persistence, and error paths remain coherent.
