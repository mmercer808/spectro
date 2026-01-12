Prompt for next chat (copy/paste)

We have an engine seed project at:

moderngl_engine_seed_v2.zip (contains app_mglw.py, optional app_qt.py, and engine/ modules)

Architecture: per-panel ViewportArea with its own EntityGraph, async extraction to CommandLists (worker thread), GL-thread executes commandlists into render targets, then compositor draws viewport textures into arbitrary panel rects. Multi-camera per viewport is supported via sub-rect viewports/scissors. Picking skeleton writes entity IDs to attachment 1 and reads pixel on click.

Your job: continue the engine with robust async + scalable rendering.
Do not remove portability: a viewport must remain “render-to-texture + composite anywhere”.

Goals

Make async extraction truly safe + scalable (no race conditions with graph mutation).

Add UploadQueue + resource versioning so dynamic meshes/instance buffers/textures can be updated without hitches.

Introduce a real RenderWorld batching layer (material buckets + instancing) so commandlists stay small.

Improve picking so it returns (camera_id, entity_id) reliably in multi-camera viewports.

Constraints / Rules

Worker threads must never call ModernGL/OpenGL.

GL-thread only for executing GL calls and consuming upload jobs.

Viewport outputs textures; compositor is separate and draws those textures in panel rects.

Multi-camera sub-viewports inside a single viewport target are required.

TODO list (ordered)
A) Graph snapshotting (RCU / double buffer) — must do first

Implement a snapshot model so workers always traverse a stable graph:

Option A: GraphSnapshot (immutable list of RenderItems built on sim thread), workers only operate on RenderItem[].

Option B: double-buffered component arrays (transforms/materials) + swap at frame boundary.

Make ViewportArea.kick_extract_if_needed() consume a snapshot handle, not the live graph.

Add “dirty reasons” properly (graph/camera/resize/material) and only rebuild commandlists when needed.

Acceptance test: spam graph mutations while moving camera; no crashes, no inconsistent partial frames.

B) UploadQueue + resource versioning

Add engine/render/uploader.py:

UploadJob(kind, key, payload, nbytes, priority)

queue is thread-safe; workers push CPU blobs

Add ResourceRegistry methods:

request_mesh_update(key, cpu_mesh_blob, version)

request_texture_update(key, cpu_tex_blob, version)

GL thread drains queue with budget + updates GPU resources

Add version guards:

if GPU has version 10 and job is version 9 → drop

prevent “late” async jobs from clobbering newer resources

Acceptance test: push frequent instance-buffer updates; frame rate remains stable; no hitches.

C) RenderWorld batching + instancing

Replace per-item draw emission with:

RenderWorld.buckets[(pipeline_id, mesh_id, material_key)] -> list[instances]

one draw command per bucket:

bind pipeline/mesh once

upload instance buffer (or reference cached GPU instance buffer)

draw instanced

Add CmdDrawMeshInstanced to command set:

mesh_id, pipeline_id, instance_buffer_key, instance_count, etc.

Build instance transforms in snapshot/extract stage (CPU), upload via UploadQueue, reference by key in CommandList.

Acceptance test: 1000 cubes in scene should still render smoothly.

D) Multi-camera picking correctness

Current: pick reads one attachment pixel but doesn’t tell which camera region.

Add camera sub-viewport metadata:

ViewportArea.camera_viewports should include camera_index + rect

On click:

determine which camera rect contains the click

read pick ID and return (camera_index, entity_id)

Optionally add “camera index into alpha” or separate small metadata texture later.

Acceptance test: quad view picks correct entity and prints the camera index.

E) RenderGraph scaffolding (lightweight)

Keep current “merged commandlist” approach, but introduce:

Pass(name, target, cmdlist)

ViewportRenderPlan = [Pass...]

This is not a full framegraph yet; just enough structure to grow.

Acceptance test: viewport can add an optional post pass without rewriting everything.

F) Quality-of-life & robustness

Add logging + debug overlays:

show per-area: dirty flags, last extract time, command count, upload bytes/frame

Add unit-test-ish checks:

commandlists must not hold GL objects

upload jobs must be pure CPU data

Add graceful handling for resize storms:

only reallocate targets after a small debounce (optional)

Design decisions to defend (don’t let next chat “simplify” these away)

Viewport = render-to-texture, always
This is the key that lets you “extract a viewport and draw it anywhere.” If you let viewports draw directly to the screen, you lose compositing flexibility and future window/layout systems become painful.

Two-phase architecture: Extract (async) → Execute (GL thread)
This is the only sane path for:

multi-viewports

large scenes

background updates

zero GL thread stalls
It also makes the engine testable (commandlists can be inspected/replayed).

CommandLists contain no GL objects
Workers should only record stable IDs/keys + numbers/arrays. GL objects are context-bound and not thread-safe. Keeping CommandLists “pure data” prevents heisenbugs.

EntityGraph is for authoring; RenderWorld is for rendering
Graph traversal for every draw is fine at small scale but becomes slow and unbatchable.
RenderWorld is the render-friendly snapshot: bucketed, instanced, sorted. This is how you scale.

Versioning on uploads is non-negotiable
Async jobs complete out of order. Without versions, older jobs can overwrite newer GPU resources. Version checks eliminate this entire class of bugs.

Multi-camera is implemented with viewports + scissors into one target
This keeps:

one viewport texture for the panel

consistent compositing

simple readback for picking
It’s also the cheapest GPU approach compared to separate targets per camera (unless you choose that later intentionally).