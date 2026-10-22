# Staged creation and video styles

The default creation flow is:

`Creative brief → Style → Script → Storyboard → Generate video`

Planning does not generate media. The final step shows the recommended generation settings and
offers a direct **Review the storyboard** action when visual planning still needs confirmation.
Confirming the settings automatically checks the exact style/template/workflow/runtime combination
and prepares generation; Canary QC and human approval are no longer required.

## Project revisions

Projects are stored under `data/projects/<project_id>/`. Every write carries an
`expected_revision`; stale writes return HTTP 409 instead of overwriting a newer edit. Locking
creates a content hash. Production consumes the locked narration and visual prompts and never
calls the planner to rewrite them.

Editing one scene marks only that scene's media dirty. On the next production run, unchanged
scene segments are reused and the final video is assembled in the current scene order.

## Styles

A `StyleProfile` controls writing tone, pacing, visual and negative prompts, color, lighting,
composition, materials, character continuity, camera, transitions, templates, workflows, voice,
speech speed, and BGM guidance. System presets are immutable. Advanced overrides create a
project-owned derived profile.

Custom styles are structured from natural language. Three representative image previews are
generated only after the user explicitly requests them because they consume provider/GPU capacity.

## Assets

Browser uploads default to 512 MB per file. Local imports are denied unless their absolute parent
directory is listed in `assets.allowed_import_roots`. Pillow and FFprobe decide whether the file is
decodable; the extension is not trusted. Originals and SHA-256 hashes are preserved while
thumbnails and H.264 proxies are stored as derived files.

Each scene can reference project assets by stable asset ID. `cover`/`contain` and horizontal or
vertical focal points are previewed before locking. Production creates a traceable canvas-ready
derivative from those settings; it never crops, recompresses, or overwrites the immutable original.

## Automatic runtime check

Runtime checks bind the style version and hash, template hash, workflow hash, and relevant runtime
node and model contracts. A changed contract produces a different combination hash and is checked
again before locking.

Normal creation automatically verifies the template, workflow, ComfyUI nodes, and models. It does
not require a separate Canary run or human approval. Canvas, template, workflow, voice, speech
speed, and BGM use the style recommendations by default and remain in optional advanced settings.

Operators can still launch the standalone **Compatibility** diagnostics page for a read-only
workflow audit and historical records; it is absent from the normal creation navigation. Normal
generation never installs models/nodes or restarts ComfyUI; remote repair requires a separate
management transport.
