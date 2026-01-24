# SPECTRO 2D Rendering Pipeline

> A detailed walkthrough of the procedural UI rendering system

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Stage 1: Scene Graph & Data Model](#2-stage-1-scene-graph--data-model)
3. [Stage 2: Style Tokens](#3-stage-2-style-tokens)
4. [Stage 3: Animation Update](#4-stage-3-animation-update)
5. [Stage 4: Collect & Batch](#5-stage-4-collect--batch)
6. [Stage 5: Instance Packing](#6-stage-5-instance-packing)
7. [Stage 6: GPU Upload](#7-stage-6-gpu-upload)
8. [Stage 7: Vertex Shader](#8-stage-7-vertex-shader)
9. [Stage 8: Fragment Shader](#9-stage-8-fragment-shader)
10. [Stage 9: Compositing](#10-stage-9-compositing)
11. [Complete Frame Example](#11-complete-frame-example)
12. [Performance Considerations](#12-performance-considerations)

---

## 1. Architecture Overview

The rendering pipeline transforms high-level UI descriptions into GPU-rendered pixels through a series of stages. The key insight is that **every UI component is a screen-aligned quad** whose appearance is computed procedurally by fragment shaders.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│                                                                             │
│   GraphSpace          StyleTokens         Transport/TimeCamera              │
│   └── GraphObjects    └── Colors          └── beat, bpm, phase              │
│       └── DrawNodes       └── Glow params                                   │
│           └── Properties  └── Radii                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UPDATE PHASE (CPU)                                 │
│                                                                             │
│   1. Animations evaluate → property changes                                 │
│   2. Dirty flags propagate                                                  │
│   3. Transform matrices recompute (if dirty)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COLLECT PHASE (CPU)                                 │
│                                                                             │
│   1. Spatial culling (QuadTree query)                                       │
│   2. Visible objects → Instance batches (grouped by ComponentKind)          │
│   3. Pack instances into numpy array (20 floats per instance)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UPLOAD PHASE (CPU → GPU)                            │
│                                                                             │
│   1. Write packed array to SSBO (Shader Storage Buffer Object)              │
│   2. Bind SSBO to binding point 0                                           │
│   3. Set uniforms (resolution, time, etc.)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RENDER PHASE (GPU)                                  │
│                                                                             │
│   For each ComponentKind with instances:                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  VERTEX SHADER                                                       │  │
│   │  • Read instance rect from SSBO[gl_InstanceID * 5 + 0]              │  │
│   │  • Expand unit quad [0,1]² to pixel rect                            │  │
│   │  • Transform to NDC                                                  │  │
│   │  • Pass UV coords to fragment shader                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  RASTERIZER                                                          │  │
│   │  • Generates fragments for each pixel in the quad                    │  │
│   │  • Interpolates UV coords across triangle                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  FRAGMENT SHADER (procedural)                                        │  │
│   │  • Read instance data from SSBO                                      │  │
│   │  • UV → local coords [-1,+1]                                         │  │
│   │  • Compute SDF (signed distance function)                            │  │
│   │  • Apply smoothstep for anti-aliasing                                │  │
│   │  • Combine layers: fill, stroke, glow                                │  │
│   │  • Output: vec4(color, alpha)                                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  BLEND & OUTPUT                                                      │  │
│   │  • Alpha blending with existing framebuffer                          │  │
│   │  • Write to render target (screen or FBO)                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1: Scene Graph & Data Model

### What happens here

The scene graph holds the logical structure of the UI. Each `GraphObject` has:
- A `Transform2D` (position, rotation, scale, anchor)
- Zero or more `DrawNode` children (visual primitives)
- Optional hierarchy (parent/children)

### Data structures

```python
# GraphObject - a node in the scene
class GraphObject:
    id: str                      # Unique identifier
    name: str                    # Human-readable name
    transform: Transform2D       # Local transform
    parent: GraphObject | None   # Parent in hierarchy
    children: List[GraphObject]  # Child objects
    draw_nodes: List[DrawNode]   # Attached visuals
    visible: bool                # Visibility flag
    opacity: float               # Alpha multiplier
    z_index: int                 # Draw order

# Transform2D - 2D affine transform
@dataclass
class Transform2D:
    position: Vec2 = Vec2(0, 0)
    rotation: float = 0.0        # Radians
    scale: Vec2 = Vec2(1, 1)
    anchor: Vec2 = Vec2(0, 0)    # Pivot point

# DrawNode - visual primitive (abstract base)
class DrawNode:
    owner: GraphObject | None
    visible: bool
    opacity: float
    z_offset: int
```

### Example: Creating a beat circle

```python
# Create a GraphObject for beat 1
beat1 = GraphObject(name="beat1_circle")
beat1.set_position(100, 100)

# Attach a circle visual
beat1.add_draw_node(CircleNode(
    cx=0, cy=0,           # Local coords (relative to object)
    radius=45,
    fill=None,            # No fill (transparent center)
    stroke=StrokeStyle(
        color=Color.from_hex("#ff6b6b"),
        width=3.0
    )
))

# Attach a phase dot (will be animated)
phase_dot = CircleNode(cx=0, cy=-40, radius=6, fill=Color.from_hex("#ff6b6b"))
beat1.add_draw_node(phase_dot)

# Add to space
space.add(beat1, layer="ui")
```

### Snapshot: After this stage

```
GraphSpace "main"
├── Layer "ui" (z_order=0)
│   └── Objects:
│       ├── beat1_circle @ (100, 100)
│       │   ├── CircleNode(radius=45, stroke)
│       │   └── CircleNode(radius=6, fill)  ← phase dot
│       ├── beat2_circle @ (210, 100)
│       │   └── ...
│       └── ...
└── Layer "overlay" (z_order=100)
    └── ...
```

---

## 3. Stage 2: Style Tokens

### What happens here

Style tokens define the visual vocabulary: colors, glow strengths, radii, spacing. Components reference tokens semantically (e.g., `accent_beat1`) rather than hardcoding colors.

### Data structure

```python
@dataclass(frozen=True)
class StyleTokens:
    # Backgrounds
    bg_app: Vec4 = (0.04, 0.04, 0.06, 1.0)
    bg_panel: Vec4 = (0.08, 0.08, 0.10, 1.0)

    # Beat colors
    accent_beat1: Vec4 = (1.0, 0.42, 0.42, 1.0)   # Red
    accent_beat2: Vec4 = (1.0, 0.85, 0.24, 1.0)   # Yellow
    accent_beat3: Vec4 = (0.42, 0.80, 0.47, 1.0)  # Green
    accent_beat4: Vec4 = (0.30, 0.59, 1.0, 1.0)   # Blue

    # Glow
    glow_core: float = 0.55
    glow_halo: float = 0.18

    # Geometry
    ring_thickness: float = 0.06
```

### GPU representation

Tokens are packed into a Uniform Buffer Object (UBO) for GPU access:

```
StyleUBO Layout (std140, binding=1):
┌────────────────────────────────────────────┐
│ offset 0:   vec4 bg_app                    │
│ offset 16:  vec4 bg_panel                  │
│ offset 32:  vec4 bg_well                   │
│ offset 48:  vec4 stroke_primary            │
│ offset 64:  vec4 stroke_secondary          │
│ offset 80:  vec4 text_primary              │
│ offset 96:  vec4 text_secondary            │
│ offset 112: vec4 accent_beat1              │
│ offset 128: vec4 accent_beat2              │
│ offset 144: vec4 accent_beat3              │
│ offset 160: vec4 accent_beat4              │
│ offset 176: vec4 accent_teal               │
│ offset 192: vec4 accent_gold               │
│ offset 208: vec4 params (glow_core, ...)   │
└────────────────────────────────────────────┘
Total: 224 bytes
```

### Snapshot: Theme switching

```python
# Switch from dark to light theme
renderer.set_style(LIGHT_THEME)

# Or create custom theme
custom = DARK_THEME.with_overrides(
    accent_beat1=(1.0, 0.2, 0.5, 1.0),  # Pink instead of red
    glow_core=0.8                        # Stronger glow
)
renderer.set_style(custom)
```

---

## 4. Stage 3: Animation Update

### What happens here

Animations modify properties over time. Each frame:
1. Delta time is added to animation elapsed time
2. Progress `t` is computed (0 to 1)
3. Easing function transforms `t`
4. Value is interpolated and applied via compiled setter
5. If transform changed, dirty flag propagates to children

### Animation data structure

```python
@dataclass
class PropertyAnimation:
    target_obj: GraphObject
    property_path: str           # e.g., "transform.position.x"
    target_value: Any
    duration: float
    easing: Callable[[float], float]
    delay: float = 0.0
    on_complete: Callable = None

    # Runtime state
    _elapsed: float = 0.0
    _start_value: Any = None
    _started: bool = False
```

### Update flow

```
┌─────────────────────────────────────────────────────────────────┐
│  animation.update(dt=0.016)   # 60 FPS frame                    │
│                                                                 │
│  1. _elapsed += dt            # 0.0 → 0.016                     │
│  2. if _elapsed < delay:      # Handle delay                    │
│        return False                                             │
│  3. if not _started:          # First frame past delay          │
│        _start_value = get_property()                            │
│        _started = True                                          │
│  4. active_time = _elapsed - delay                              │
│  5. t = clamp(active_time / duration, 0, 1)                     │
│  6. eased_t = easing(t)       # e.g., ease_out_cubic(0.5)=0.875│
│  7. value = lerp(start, target, eased_t)                        │
│  8. set_property(value)                                         │
│  9. if property_path.startswith("transform"):                   │
│        target_obj._mark_transform_dirty()                       │
│  10. return (t >= 1.0)        # True if complete                │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Animating phase dot rotation

```python
# Animate phase dot around the circle
# Position at angle θ: (cos(θ)*r, sin(θ)*r)

def update_phase_dot(beat_phase: float):
    """beat_phase is 0-1 within current beat"""
    angle = -math.pi/2 + beat_phase * 2 * math.pi  # Start at top
    radius = 40

    # Update dot position
    phase_dot.cx = math.cos(angle) * radius
    phase_dot.cy = math.sin(angle) * radius
```

### Snapshot: After animation update

```
Before: beat1.transform.position = Vec2(100, 100)
        phase_dot.cx = 0, cy = -40  (top of circle)

Animation: beat_phase = 0.25 (quarter through beat)

After:  phase_dot.cx = 40, cy = 0   (right side of circle)
        dirty_flag = True → world_transform will recompute
```

---

## 5. Stage 4: Collect & Batch

### What happens here

The collect phase gathers visible `DrawNodes` into batches organized by `ComponentKind`. This enables efficient instanced rendering (one draw call per component type).

### Process

```
┌─────────────────────────────────────────────────────────────────┐
│  GraphSpace.collect(batch, viewport)                            │
│                                                                 │
│  For each Layer (sorted by z_order):                            │
│  ├── if not layer.visible: skip                                 │
│  ├── visible_ids = layer.query_visible(viewport)  # QuadTree    │
│  └── For each object_id in visible_ids:                         │
│      ├── obj = objects[object_id]                               │
│      ├── if not obj.visible: skip                               │
│      ├── world_transform = obj.world_transform   # Cached       │
│      ├── world_opacity = obj.world_opacity * layer.opacity      │
│      └── For each draw_node in obj.draw_nodes:                  │
│          ├── if not node.visible: skip                          │
│          ├── effective_z = obj.z_index + node.z_offset          │
│          │                 + layer.z_order * 1000               │
│          └── batch.add(node, world_transform, opacity, z)       │
└─────────────────────────────────────────────────────────────────┘
```

### Spatial culling with QuadTree

```
Viewport: Rect(0, 0, 1200, 700)

QuadTree query:
┌─────────────────────────────────────────┐
│  ┌─────────┬─────────┐                  │
│  │ beat1 ● │ beat2 ● │ ← In viewport    │
│  │ beat3 ● │ beat4 ● │                  │
│  ├─────────┼─────────┤                  │
│  │ dial1 ● │ dial2 ● │                  │
│  │         │         │                  │
│  └─────────┴─────────┘                  │
│                           offscreen → ✗ │
└─────────────────────────────────────────┘

Result: {beat1, beat2, beat3, beat4, dial1, dial2}
        (offscreen objects excluded)
```

### Batching by ComponentKind

```python
# ProceduralRenderer maintains batches
self._batches = {
    ComponentKind.RECT: [],
    ComponentKind.CIRCLE: [],
    ComponentKind.DIAL: [],
    ComponentKind.ARC: [],
    ComponentKind.LINE: [],
}

# During collect, DrawNodes are converted to UIInstances
# and added to appropriate batch

# Example: CircleNode → ComponentKind.CIRCLE batch
renderer.add_circle(
    cx=100, cy=100, radius=45,
    fill=None,
    stroke=(1.0, 0.42, 0.42, 1.0),
    stroke_width=0.06,
    glow=0.4
)
# → Adds UIInstance to _batches[ComponentKind.CIRCLE]
```

### Snapshot: After collect

```
_batches = {
    RECT: [
        UIInstance(rect=(40,40,520,200), fill=bg_panel, ...),   # Panel 1
        UIInstance(rect=(580,40,580,200), fill=bg_panel, ...),  # Panel 2
    ],
    CIRCLE: [
        UIInstance(rect=(55,55,90,90), stroke=beat1_color, ...), # Beat 1 ring
        UIInstance(rect=(94,54,12,12), fill=beat1_color, ...),   # Beat 1 dot
        UIInstance(rect=(165,55,90,90), stroke=beat2_color, ...), # Beat 2 ring
        ...
    ],
    DIAL: [
        UIInstance(rect=(120,320,160,160), accent=teal, needle=0.3, ...),
        UIInstance(rect=(320,320,160,160), accent=gold, needle=1.2, ...),
    ],
    ARC: [
        UIInstance(rect=(1050,340,120,120), start=−π/2, end=π/4, ...),
    ],
    LINE: [],
}
```

---

## 6. Stage 5: Instance Packing

### What happens here

Each `UIInstance` is packed into a contiguous array of floats. The layout is 5 × vec4 = 20 floats per instance, aligned for GPU access.

### Instance memory layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Instance N (20 floats = 80 bytes)                              │
├─────────────────────────────────────────────────────────────────┤
│  Offset 0-3:   vec4 rect     [x, y, width, height]  pixels      │
│  Offset 4-7:   vec4 color1   [r, g, b, a]           accent/fill │
│  Offset 8-11:  vec4 color2   [r, g, b, a]           stroke      │
│  Offset 12-15: vec4 params0  [varies by kind]                   │
│  Offset 16-19: vec4 params1  [varies by kind]                   │
└─────────────────────────────────────────────────────────────────┘
```

### params0/params1 by ComponentKind

| Kind | params0 | params1 |
|------|---------|---------|
| RECT | corner_radius, stroke_width, opacity, _ | _ |
| CIRCLE | stroke_width, opacity, glow, _ | _ |
| DIAL | needle_angle, opacity, glow, ring_radius | arc0, arc1, arc2, arc3 |
| ARC | start_angle, end_angle, thickness, opacity | _ |
| LINE | line_width, opacity, _, _ | _ |

### Packing code

```python
def pack_instances(instances: List[UIInstance]) -> np.ndarray:
    """Pack instances into GPU-ready array."""
    out = np.zeros((len(instances), 20), dtype=np.float32)

    for i, inst in enumerate(instances):
        out[i, 0:4]   = inst.rect      # x, y, w, h
        out[i, 4:8]   = inst.color1    # accent/fill RGBA
        out[i, 8:12]  = inst.color2    # stroke RGBA
        out[i, 12:16] = inst.params0   # kind-specific
        out[i, 16:20] = inst.params1   # kind-specific

    return out.ravel()  # Flatten to 1D
```

### Snapshot: Packed buffer for 2 circles

```
Instance 0 (beat1 ring):
  rect:   [55.0, 55.0, 90.0, 90.0]      # Position and size
  color1: [0.0, 0.0, 0.0, 0.0]          # No fill (transparent)
  color2: [1.0, 0.42, 0.42, 1.0]        # Red stroke
  params0: [0.06, 1.0, 0.4, 0.0]        # stroke_width, opacity, glow
  params1: [0.0, 0.0, 0.0, 0.0]         # unused

Instance 1 (beat1 phase dot):
  rect:   [94.0, 54.0, 12.0, 12.0]      # Dot position
  color1: [1.0, 0.42, 0.42, 1.0]        # Red fill
  color2: [0.0, 0.0, 0.0, 0.0]          # No stroke
  params0: [0.0, 1.0, 0.8, 0.0]         # no stroke, full opacity, strong glow
  params1: [0.0, 0.0, 0.0, 0.0]

Packed array (40 floats):
[55, 55, 90, 90, 0, 0, 0, 0, 1, 0.42, 0.42, 1, 0.06, 1, 0.4, 0, 0, 0, 0, 0,
 94, 54, 12, 12, 1, 0.42, 0.42, 1, 0, 0, 0, 0, 0, 1, 0.8, 0, 0, 0, 0, 0]
```

---

## 7. Stage 6: GPU Upload

### What happens here

The packed numpy array is uploaded to a Shader Storage Buffer Object (SSBO). SSBOs allow shaders to read arbitrary data indexed by `gl_InstanceID`.

### SSBO setup

```python
# Create SSBO with initial capacity
self._instance_buffer = ctx.buffer(reserve=20 * 4 * 256)  # 256 instances

# Each frame: write packed data
packed = pack_instances(batch)
self._instance_buffer.write(packed.tobytes())

# Bind to shader binding point 0
self._instance_buffer.bind_to_storage_buffer(binding=0)
```

### Memory diagram

```
CPU Memory                          GPU Memory (SSBO binding=0)
┌─────────────────┐                ┌─────────────────────────────┐
│ numpy array     │  ═══════════>  │ Instance 0: 80 bytes        │
│ (float32)       │    glBufferData│ Instance 1: 80 bytes        │
│                 │                │ Instance 2: 80 bytes        │
│ [55,55,90,90,..]│                │ ...                         │
│                 │                │ Instance N-1: 80 bytes      │
└─────────────────┘                └─────────────────────────────┘
                                          ↓
                                   Shader reads via:
                                   data[gl_InstanceID * 5 + offset]
```

### Uniforms

```python
# Set per-frame uniforms
prog['u_resolution'].value = (width, height)  # Screen size
prog['u_time'].value = time                   # For animated effects
```

---

## 8. Stage 7: Vertex Shader

### What happens here

The vertex shader runs once per vertex (6 vertices × N instances). It:
1. Reads the instance rectangle from the SSBO
2. Expands the unit quad `[0,1]²` to the pixel rectangle
3. Converts to Normalized Device Coordinates (NDC)
4. Passes UV coordinates to the fragment shader

### Vertex shader code

```glsl
#version 330

// Input: unit quad corners
in vec2 in_pos;   // (0,0), (1,0), (1,1), (0,0), (1,1), (0,1)
in vec2 in_uv;    // Same as in_pos for unit quad

uniform vec2 u_resolution;  // Screen size in pixels

// SSBO containing all instance data
layout(std430, binding = 0) buffer Instances {
    vec4 data[];  // 5 vec4 per instance
};

out vec2 v_uv;           // Interpolated UV for fragment
flat out int v_instance; // Instance ID (not interpolated)

void main() {
    // Pass through
    v_uv = in_uv;
    v_instance = gl_InstanceID;

    // Read instance rect: [x, y, width, height] in pixels
    vec4 rect = data[gl_InstanceID * 5 + 0];

    // Expand unit quad to instance rectangle
    // in_uv is (0,0) at top-left, (1,1) at bottom-right
    vec2 pixel_pos = vec2(rect.x, rect.y) + in_uv * vec2(rect.z, rect.w);

    // Convert pixel coords to NDC [-1, +1]
    vec2 ndc = (pixel_pos / u_resolution) * 2.0 - 1.0;

    // Flip Y (OpenGL has Y-up, we want Y-down for screen coords)
    ndc.y *= -1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
```

### Visual: Quad expansion

```
Unit Quad [0,1]²              Instance Rect (pixels)           NDC [-1,+1]²
                              rect = (100, 50, 80, 80)

(0,0)───(1,0)                 (100,50)───(180,50)              (-0.83,+0.86)───(-0.70,+0.86)
  │       │      ═══════>       │           │       ═══════>        │               │
  │       │      expand         │   80x80   │       to NDC          │               │
  │       │                     │   pixels  │                       │               │
(0,1)───(1,1)                 (100,130)──(180,130)             (-0.83,+0.63)───(-0.70,+0.63)

For 1200x700 screen:
  pixel (100, 50) → NDC (100/1200*2-1, -(50/700*2-1)) = (-0.833, 0.857)
```

---

## 9. Stage 8: Fragment Shader

### What happens here

The fragment shader runs once per pixel within the rasterized quad. It:
1. Reads instance data from SSBO
2. Converts UV to local coordinates
3. Computes Signed Distance Function (SDF) for the shape
4. Applies smoothstep for anti-aliased edges
5. Combines layers (fill, stroke, glow)
6. Outputs final color and alpha

### Circle fragment shader (annotated)

```glsl
#version 330

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;            // 0-1 within quad
flat in int v_instance;  // Which instance
out vec4 fragColor;

void main() {
    // ─────────────────────────────────────────────────────────────
    // 1. READ INSTANCE DATA
    // ─────────────────────────────────────────────────────────────
    int base = v_instance * 5;
    vec4 rect   = data[base + 0];  // x, y, w, h (not used in frag)
    vec4 fill   = data[base + 1];  // Fill color RGBA
    vec4 stroke = data[base + 2];  // Stroke color RGBA
    vec4 params = data[base + 3];  // stroke_width, opacity, glow, _

    float stroke_width = params.x;
    float opacity = params.y;
    float glow = params.z;

    // ─────────────────────────────────────────────────────────────
    // 2. CONVERT TO LOCAL COORDINATES [-1, +1]
    // ─────────────────────────────────────────────────────────────
    // v_uv is [0,1], convert to [-1,+1] centered on quad
    vec2 p = v_uv * 2.0 - 1.0;

    // Distance from center (for circle, this is radius)
    float r = length(p);

    // ─────────────────────────────────────────────────────────────
    // 3. SIGNED DISTANCE FUNCTION (SDF)
    // ─────────────────────────────────────────────────────────────
    // Circle SDF: distance to edge (negative inside, positive outside)
    // For unit circle (radius 1.0), SDF = length(p) - 1.0
    float d = r - 1.0;

    //  p=(0,0): d = 0 - 1 = -1.0  (center, inside)
    //  p=(0.5,0): d = 0.5 - 1 = -0.5  (inside)
    //  p=(1,0): d = 1 - 1 = 0  (on edge)
    //  p=(1.2,0): d = 1.2 - 1 = 0.2  (outside)

    // ─────────────────────────────────────────────────────────────
    // 4. ANTI-ALIASED MASKS (smoothstep)
    // ─────────────────────────────────────────────────────────────

    // Fill mask: 1.0 inside circle, 0.0 outside, smooth transition
    // smoothstep(edge0, edge1, x) returns 0 if x<edge0, 1 if x>edge1
    float fill_mask = 1.0 - smoothstep(-0.02, 0.0, d);
    //  d = -0.5: smoothstep(-0.02, 0, -0.5) = 0 → fill_mask = 1
    //  d = -0.01: smoothstep = 0.5 → fill_mask = 0.5
    //  d = 0.1: smoothstep = 1 → fill_mask = 0

    // Stroke mask: ring at the edge
    float ring_inner = 1.0 - stroke_width;  // e.g., 0.94 for 6% stroke
    float ring_mask = smoothstep(ring_inner - 0.02, ring_inner, r)
                    * (1.0 - smoothstep(1.0, 1.0 + 0.02, r));
    //  Combines two smoothsteps: inner edge fade-in, outer edge fade-out

    // Glow mask: soft falloff outside circle
    float glow_mask = (1.0 - smoothstep(1.0, 1.0 + 0.15, r)) * glow;

    // ─────────────────────────────────────────────────────────────
    // 5. COMBINE LAYERS
    // ─────────────────────────────────────────────────────────────
    vec3 col = vec3(0.0);

    // Layer 1: Fill (if any)
    col += fill.rgb * fill.a * fill_mask;

    // Layer 2: Stroke ring
    col += stroke.rgb * stroke.a * ring_mask;

    // Layer 3: Glow halo (additive)
    col += fill.rgb * glow_mask * 0.5;

    // Compute final alpha
    float alpha = max(fill_mask * fill.a,
                  max(ring_mask * stroke.a,
                      glow_mask * 0.3));

    // ─────────────────────────────────────────────────────────────
    // 6. OUTPUT
    // ─────────────────────────────────────────────────────────────
    fragColor = vec4(col, alpha * opacity);
}
```

### Visual: SDF for circle

```
UV Space [0,1]²                 Local Space [-1,+1]²              SDF Values

(0,0)─────(1,0)                 (-1,-1)─────(+1,-1)              ┌─────────────┐
  │         │                      │           │                 │ +0.4  +0.1  │
  │    ●    │    ═══════>          │     ●     │    ═══════>    │      ╭─╮    │
  │  (0.5,  │    v_uv*2-1          │   (0,0)   │    length(p)-1 │ +0.1 │-1│+0.1│
  │   0.5)  │                      │           │                 │      ╰─╯    │
(0,1)─────(1,1)                 (-1,+1)─────(+1,+1)              │ +0.4  +0.1  │
                                                                 └─────────────┘
                                                                 Negative = inside
                                                                 Zero = on edge
                                                                 Positive = outside
```

### Dial fragment shader (key parts)

```glsl
// Angle calculation for polar effects
float ang = atan(p.y, p.x);  // [-π, +π]
if (ang < 0.0) ang += 6.28318;  // [0, 2π]

// Ring mask (donut shape)
float ring_r = 0.78;  // Ring radius
float ring_t = 0.06;  // Ring thickness
float ring_mask = smoothstep(ring_r - ring_t - 0.01, ring_r - ring_t, r)
                * (1.0 - smoothstep(ring_r + ring_t, ring_r + ring_t + 0.01, r));

// Quarter arcs (each π/2 radians)
// arcs.x = length of arc 0 (right), arcs.y = arc 1 (top), etc.
float q = 1.5707963;  // π/2
float arc_mask = 0.0;
for (int k = 0; k < 4; k++) {
    float a0 = float(k) * q;           // Start angle
    float a1 = a0 + q * arcs[k];       // End angle (scaled by arc length)
    float in_arc = step(a0, ang) * step(ang, a1);
    arc_mask = max(arc_mask, in_arc);
}

// Needle: distance to a ray from center
vec2 needle_dir = vec2(cos(needle_angle), sin(needle_angle));
float proj = dot(p, needle_dir);        // How far along the needle
vec2 perp = p - needle_dir * proj;      // Perpendicular distance
float d_needle = length(perp);
float needle_mask = smoothstep(0.025, 0.0, d_needle)  // Width
                  * smoothstep(-0.15, 0.3, proj)       // Start (near center)
                  * (1.0 - smoothstep(0.85, 0.9, proj)); // End (before edge)
```

---

## 10. Stage 9: Compositing

### What happens here

After all component batches are rendered, the results are composited:
1. Alpha blending combines overlapping elements
2. (Optional) UI rendered to offscreen FBO, then composited onto 3D scene

### Blend setup

```python
# Enable alpha blending
ctx.enable(moderngl.BLEND)
ctx.blend_func = (
    moderngl.SRC_ALPHA,          # Source factor
    moderngl.ONE_MINUS_SRC_ALPHA # Dest factor
)

# Blend equation: out = src.rgb * src.a + dst.rgb * (1 - src.a)
```

### Render order

```
1. Clear framebuffer to bg_app color

2. Render RECT instances (backgrounds, panels)
   └── Panels are drawn first (lowest z)

3. Render CIRCLE instances (beat rings, phase dots)
   └── Rings drawn, then dots on top (higher z_offset)

4. Render DIAL instances (complex dials with arcs)
   └── Single draw call renders all 4 dials

5. Render ARC instances (phase indicators)
   └── Layered on top

6. Render LINE instances (if any)
   └── Grid lines, connections

7. (Future) Render TEXT via font atlas
```

### FBO compositing (optional)

```python
# Render UI to offscreen texture
ui_fbo.use()
ctx.clear()
render_ui_batches()

# Composite onto main framebuffer
ctx.screen.use()
render_3d_scene()
blit_texture(ui_fbo.color_attachments[0], blend=True)
```

---

## 11. Complete Frame Example

### Frame timeline for 60 FPS

```
Frame N (16.67ms budget)
════════════════════════════════════════════════════════════════════════════

T+0.0ms: BEGIN FRAME
├── Read input events
├── Update transport: beat = (time * bpm / 60) % 4
│   └── beat = 2.35 (35% through beat 3)
│
T+0.5ms: ANIMATION UPDATE
├── For each active animation:
│   ├── phase_dot.animate("cx", target_x, 0.1s)
│   │   └── cx: 0 → 38.27 (interpolated)
│   └── dial.animate("needle", target_angle, 0.2s)
│       └── needle: 0.3 → 0.87 (eased)
├── Mark dirty transforms
│
T+1.0ms: COLLECT PHASE
├── Query visible objects (QuadTree): 24 objects
├── Convert to UIInstances:
│   ├── RECT: 3 instances (panels)
│   ├── CIRCLE: 12 instances (rings + dots)
│   ├── DIAL: 4 instances
│   └── ARC: 2 instances
│
T+2.0ms: PACK & UPLOAD
├── Pack 21 instances → 420 floats → 1680 bytes
├── Upload to SSBO
├── Bind SSBO to binding point 0
│
T+3.0ms: GPU RENDER
├── Set viewport, enable blend
├── Render RECT batch (3 instances, 18 vertices)
│   └── GPU: vertex shader × 18, fragment shader × ~300K pixels
├── Render CIRCLE batch (12 instances, 72 vertices)
│   └── GPU: fragment shader × ~150K pixels (many small quads)
├── Render DIAL batch (4 instances, 24 vertices)
│   └── GPU: fragment shader × ~200K pixels (complex shader)
├── Render ARC batch (2 instances, 12 vertices)
│
T+5.0ms: GPU FINISHES
├── All fragments written to framebuffer
├── Blend operations complete
│
T+6.0ms: SWAP BUFFERS
├── Present to screen
│
T+16.67ms: END FRAME (10.67ms idle for vsync)
════════════════════════════════════════════════════════════════════════════
```

### Data sizes

| Stage | Data | Size |
|-------|------|------|
| Scene graph | 24 GraphObjects | ~10 KB |
| Style tokens | 1 StyleTokens | 224 bytes |
| Instance batch | 21 UIInstances | 1,680 bytes |
| SSBO upload | 1 write | 1,680 bytes |
| GPU memory | SSBO + VBO + programs | ~50 KB |

---

## 12. Performance Considerations

### Why this architecture is fast

1. **Instanced rendering**: One draw call renders N instances
   - 100 circles = 1 draw call, not 100

2. **SSBO for instance data**: Read-only, indexed access
   - No uniform upload overhead per instance

3. **Procedural shaders**: No texture sampling for shapes
   - SDF computed mathematically, not fetched

4. **Spatial culling**: QuadTree excludes off-screen objects
   - Large scenes remain fast

5. **Dirty caching**: Skip unchanged frames
   - Static UI = no work

### Optimization strategies

```python
# 1. Batch by component kind (already done)
#    Minimizes shader program switches

# 2. Sort by z-index within batch
#    Reduces overdraw (front-to-back for opaque)

# 3. Frustum culling
#    Already using QuadTree

# 4. Instance buffer reuse
#    Don't reallocate every frame
if len(instances) > self._instance_capacity:
    # Only resize when necessary
    self._instance_buffer = ctx.buffer(reserve=new_size)

# 5. Dirty tracking
if not space._dirty:
    # Reuse last frame's buffer
    return

# 6. LOD for complex components
#    Simplify shader when zoomed out
if component_screen_size < 20:
    use_simple_shader()
```

### Memory bandwidth

```
Per frame (21 instances):
  CPU → GPU: 1,680 bytes (SSBO upload)
  GPU reads: 1,680 bytes × ~3 (vertex + fragment passes)

  Total: ~6.7 KB/frame = 400 KB/sec at 60 FPS

  Modern GPU bandwidth: 200+ GB/sec
  This is negligible (<0.001% utilization)
```

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `engine/graph/object.py` | GraphObject, Transform2D, PropertyAnimation |
| `engine/graph/nodes.py` | DrawNode subclasses (CircleNode, etc.) |
| `engine/graph/space.py` | GraphSpace, Layer, QuadTree |
| `engine/graph/style.py` | StyleTokens, themes |
| `engine/graph/renderer.py` | ProceduralRenderer, shaders, instance packing |
| `engine/graph/coordinates.py` | Cartesian2D, Polar, BeatSpace |
| `examples/graph_demo.py` | Animated demo |

---

*Document version: 2026-01-17*
