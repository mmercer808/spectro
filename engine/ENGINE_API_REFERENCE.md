# Engine API Reference

Complete list of all classes and functions in the `/engine` directory.

---

## engine/__init__.py

No classes or functions defined (only imports and `__all__`).

---

## engine/buffer.py

### Classes

- `InstanceFlags` - Flag bits for instance data
- `GPUBuffer` - Persistent GPU buffer with CPU staging and dirty tracking
- `BufferRegistry` - Central registry for GPU buffers

### GPUBuffer Methods

- `is_dirty` (property)
- `dirty_range` (property)
- `positions` (property)
- `scales` (property)
- `rotations` (property)
- `colors` (property)
- `values` (property)
- `flags` (property)
- `__getitem__`
- `__len__`
- `__setitem__`
- `write`
- `write_field`
- `set_slot`
- `set_range`
- `_mark_dirty`
- `mark_all_dirty`
- `clear_dirty`
- `sync`
- `upload_all`
- `init_defaults`
- `get_instance_format`
- `get_attribute_names`
- `get_stats`
- `release`

### BufferRegistry Methods

- `create`
- `get`
- `has`
- `sync_all`
- `release_all`
- `get_all_stats`

---

## engine/pointcloud.py

### Classes

- `PointCloud` - Collection of instances owning a range in GPUBuffer
- `PointCloudManager` - Manages multiple PointClouds sharing a GPUBuffer

### PointCloud Methods

- `end` (property)
- `slice` (property)
- `positions` (property/setter)
- `scales` (property/setter)
- `rotations` (property/setter)
- `colors` (property/setter)
- `values` (property/setter)
- `flags` (property/setter)
- `get_point`
- `set_point`
- `get_position`
- `set_position`
- `set_all_positions`
- `set_all_colors`
- `set_uniform_scale`
- `set_all_visible`
- `init_defaults`
- `init_grid`
- `init_random`
- `update_subset`
- `animate_positions`
- `animate_colors`
- `__repr__`

### PointCloudManager Methods

- `create`
- `get`
- `has`
- `all`
- `total_instances` (property)
- `allocated_slots` (property)
- `free_slots` (property)
- `get_stats`

---

## engine/test_pointcloud.py

### Classes

- `PointCloudDemo` - Demo showing persistent buffer + point cloud system

### Methods

- `__init__`
- `_create_instanced_vao`
- `on_render`
- `_animate_grid`
- `_animate_scatter`
- `_compute_view_matrix`
- `_compute_proj_matrix`
- `_look_at` (static)
- `_perspective` (static)
- `on_mouse_drag`
- `on_mouse_scroll`
- `on_key_press`
- `on_resize`

---

## engine/core/__init__.py

No classes or functions defined (only imports and `__all__`).

---

## engine/core/frame.py

### Classes

- `FrameState` - Immutable frame information (dataclass)

### Properties

- `fps`

---

## engine/core/manager.py

### Classes

- `SceneManagerConfig` - Configuration dataclass
- `SceneManager` - Top-level coordinator for SPECTRO engine

### SceneManager Methods

- `__init__`
- `_connect_signals`
- `_on_transport_changed`
- `update`
- `resize`
- `play`
- `pause`
- `stop`
- `toggle_playback`
- `seek`
- `set_bpm`
- `is_playing` (property)
- `current_beat` (property)
- `bpm` (property)
- `scroll_to_beat`
- `zoom_in`
- `zoom_out`
- `zoom_to_fit_selection`
- `set_follow_mode`
- `to_dict`
- `from_dict`
- `save`
- `load`
- `frame_id` (property)
- `get_stats`

---

## engine/core/math3d.py

### Classes

- `Vec2` - 2D vector for screen/panel coordinates
- `Vec3` - 3D vector for simulation space
- `Vec4` - 4D vector for homogeneous coordinates
- `Mat3` - 3x3 matrix for 2D transforms
- `Mat4` - 4x4 matrix for 3D transforms
- `Quat` - Quaternion for rotations
- `Transform` - Combined position, rotation, scale

### Vec2 Methods

- `__add__`, `__sub__`, `__mul__`, `__rmul__`, `__neg__`
- `dot`, `length`, `length_squared`, `normalized`, `lerp`, `to_tuple`
- `from_tuple` (static)

### Vec3 Methods

- `__add__`, `__sub__`, `__mul__`, `__rmul__`, `__neg__`
- `dot`, `cross`, `length`, `length_squared`, `normalized`, `lerp`, `to_tuple`, `xy`
- `from_tuple`, `unit_x`, `unit_y`, `unit_z` (static)

### Vec4 Methods

- `__add__`, `__sub__`, `__mul__`
- `dot`, `xyz`, `to_vec3`, `to_tuple`
- `from_vec3`, `point`, `direction` (static)

### Mat3 Methods

- `__getitem__`, `__matmul__`, `_mul_mat`, `_mul_vec`
- `transpose`, `determinant`, `to_tuple`, `to_list_column_major`
- `identity`, `scale`, `translate`, `rotate` (static)

### Mat4 Methods

- `__getitem__`, `__matmul__`, `_mul_mat`, `_mul_vec4`
- `transpose`, `to_tuple`, `to_list_column_major`, `to_mat3`, `inverse`
- `identity`, `scale`, `translate`, `translate_vec`, `rotate_x`, `rotate_y`, `rotate_z`, `rotate_axis`, `look_at`, `ortho`, `perspective` (static)

### Quat Methods

- `__mul__`, `conjugate`, `length`, `normalized`, `rotate_vec`, `to_mat4`, `slerp`
- `identity`, `from_axis_angle`, `from_euler` (static)

### Transform Methods

- `__post_init__`, `to_mat4`, `lerp`

### Standalone Functions

- `ease_linear`
- `ease_in_quad`
- `ease_out_quad`
- `ease_in_out_quad`
- `ease_in_cubic`
- `ease_out_cubic`
- `ease_in_out_cubic`
- `ease_in_elastic`
- `ease_out_elastic`
- `ease_out_bounce`
- `clamp`
- `lerp`
- `inverse_lerp`
- `remap`
- `smoothstep`
- `deg_to_rad`
- `rad_to_deg`

---

## engine/core/scene.py

### Classes

- `EntityType` (Enum) - Types of entities
- `Entity` - Base entity in unified space
- `AudioClipEntity` - Audio clip entity
- `MidiEventEntity` - MIDI event entity
- `MarkerEntity` - Marker entity
- `SceneBounds` - Scene boundary dataclass
- `Scene` - Unified 3D coordinate space

### Entity Properties

- `beat` (property/setter)
- `end_beat` (property)
- `duration_beats` (property/setter)
- `frequency` (property/setter)
- `intensity` (property/setter)

### Entity Methods

- `contains_beat`, `overlaps_range`, `to_dict`
- `from_dict` (static)

### Scene Methods

- `__init__`, `bind_bridge`, `add`, `remove`, `get`, `update`, `clear`
- `all`, `by_type`, `in_time_range`, `at_beat`, `in_box`, `visible`
- `count` (property)
- `select`, `deselect`, `deselect_all`, `selected`
- `selection_count` (property)
- `to_dict`, `from_dict`, `save`, `load`

---

## engine/core/signal.py

### Classes

- `Connection` - Handle to a signal connection
- `SignalBridge` - Central hub for signal routing
- `SignalDebugger` - Debug wrapper for logging signals
- `SignalEmitter` - Mixin class for emitting signals
- `SignalReceiver` - Mixin class for receiving signals

### Signal Constants

- `SIGNAL_DT`
- `SIGNAL_TRANSPORT_CHANGED`
- `SIGNAL_SEEK`
- `SIGNAL_PLAY`
- `SIGNAL_PAUSE`
- `SIGNAL_STOP`
- `SIGNAL_BPM_CHANGED`
- `SIGNAL_TIME_SIG_CHANGED`
- `SIGNAL_VIEW_CHANGED`
- `SIGNAL_ZOOM_CHANGED`
- `SIGNAL_SCROLL`
- `SIGNAL_POINTER_DOWN`
- `SIGNAL_POINTER_MOVE`
- `SIGNAL_POINTER_UP`
- `SIGNAL_KEY_DOWN`
- `SIGNAL_KEY_UP`
- `SIGNAL_WHEEL`
- `SIGNAL_ENTITY_ADDED`
- `SIGNAL_ENTITY_REMOVED`
- `SIGNAL_ENTITY_CHANGED`
- `SIGNAL_SELECTION_CHANGED`
- `SIGNAL_DIRTY`
- `SIGNAL_RESIZE`

### SignalBridge Methods

- `__init__`
- `connect`
- `connect_weak`
- `disconnect_all`
- `emit`
- `emit_later`
- `block`
- `unblock`
- `is_connected`
- `_remove_connection`
- `_do_remove`

### Standalone Functions

- `on_signal` - Decorator to connect function to signal

---

## engine/core/snapshot.py

### Classes

- `DirtyFlags` (Flag) - Reasons for re-extraction
- `SnapshotRenderItem` - Immutable render item (frozen dataclass)
- `SnapshotCamera` - Immutable camera state (frozen dataclass)
- `CameraViewport` - Camera's sub-region (frozen dataclass)
- `GraphSnapshot` - Complete immutable snapshot (frozen dataclass)
- `SnapshotBuilder` - Builds GraphSnapshot objects from live scene

### SnapshotRenderItem Methods

- `get_world_matrix`

### SnapshotCamera Methods

- `get_eye`, `get_target`, `get_up`

### CameraViewport Methods

- `contains`

### SnapshotBuilder Methods

- `_get_next_id` (classmethod)
- `from_graph` (static)
- `_extract_items` (static)
- `_snapshot_cameras` (static)
- `_compute_camera_viewports` (static)

---

## engine/render/commands.py

### Classes

- `CmdSetViewport` - Set GL viewport rectangle
- `CmdSetScissor` - Set scissor test rectangle
- `CmdClear` - Clear color/depth buffers
- `CmdDrawMesh` - Draw single mesh with uniforms
- `CmdDrawMeshInstanced` - Draw multiple instances
- `CmdBeginPass` - Begin named render pass
- `CmdEndPass` - End current render pass
- `CmdSetRenderTarget` - Switch render target
- `CmdBlit` - Blit texture to another
- `CommandList` - Ordered list of rendering commands

### CommandList Methods

- `add`
- `extend`
- `__len__`
- `__iter__`
- `validate`
- `get_stats`
- `get_draw_count`
- `get_instance_count`

### Standalone Functions

- `_looks_like_gl_object` - Heuristic check for GL objects
- `merge_command_lists` - Merge multiple command lists

---

## engine/render/renderer.py

### Classes

- `Renderer` - GL thread command executor

### Standalone Functions

- `encode_id_rgba` - Encode entity ID as RGBA color
- `decode_id_rgba` - Decode entity ID from RGBA bytes

### Renderer Methods

- `__init__`
- `execute_to_target`
- `_execute_command`
- `_draw_mesh`
- `_draw_mesh_instanced`
- `_get_or_create_instance_buffer`
- `_create_instanced_vao`
- `drain_uploads`
- `_process_upload`
- `read_pick_id`
- `composite_panels`
- `get_frame_stats`
- `cleanup`

---

## engine/render/resources.py

### Classes

- `MeshGPU` - GPU-side mesh data
- `PipelineGPU` - GPU-side shader pipeline
- `ResourceRegistry` - Central registry for GPU resources

### ResourceRegistry Methods

- `__init__`
- `bootstrap_defaults`
- `_create_default_pipelines`
- `_create_default_meshes`
- `_create_cube_mesh`
- `_create_blitter`
- `get_mesh`
- `get_pipeline`
- `has_mesh`
- `has_pipeline`
- `cleanup`

---

## engine/render/targets.py

### Classes

- `RenderTargetSpec` - Specification for render target
- `RenderTarget` - Render target with color, depth, pick attachments
- `RenderTargetPool` - Pool of render targets for reuse

### RenderTarget Methods

- `release`

### RenderTargetPool Methods

- `__init__`
- `_make_key`
- `acquire`
- `release`
- `_create_target`
- `cleanup`
- `get_stats`

---

## engine/render/uploader.py

### Classes

- `UploadKind` (Enum) - Types of GPU resources
- `UploadPriority` (Enum) - Upload priority levels
- `UploadJob` - Pending GPU upload request
- `ResourceVersion` - Tracks version of GPU resource
- `UploadQueue` - Thread-safe queue for GPU uploads
- `DrainResult` - Result of draining upload queue

### UploadJob Properties

- `nbytes`
- `__lt__`

### UploadQueue Methods

- `__init__`
- `push`
- `get_stats`
- `clear`

### Standalone Functions

- `create_mesh_upload`
- `create_instance_buffer_upload`
- `create_texture_upload`

---

## engine/render/world.py

### Classes

- `MaterialKey` - Unique identifier for material configuration
- `InstanceData` - Per-instance data for batched rendering
- `RenderBucket` - Bucket of instances sharing same material
- `RenderWorld` - Render-friendly representation of scene snapshot
- `RenderWorldStats` - Statistics dataclass

### RenderBucket Methods

- `instance_count` (property)
- `build_instance_buffer`
- `add`

### RenderWorld Methods

- `__init__`
- `_bucket_items`
- `get_bucket_count`
- `get_instance_count`
- `emit_commands_for_camera`
- `_emit_instanced`
- `_look_at` (static)
- `_perspective` (static)

### Standalone Functions

- `build_render_commands`
- `get_render_world_stats`

---

## engine/scene/graph.py

### Classes

- `Transform` - Transform component (dataclass)
- `MeshRenderer` - Mesh renderer component (dataclass)
- `RenderItem` - Extracted render data (frozen dataclass)
- `EntityNode` - Node in scene graph hierarchy

### Standalone Functions

- `next_entity_id` - Generate unique entity ID (thread-safe)
- `compose_matrix` - Compose 4x4 transformation matrix
- `decompose_matrix` - Decompose matrix to Transform

### EntityNode Methods

- `__init__`
- `add_child`
- `remove_child`
- `get_root`
- `find_by_name`
- `find_by_id`
- `traverse`
- `extract_render_items`
- `apply_demo_spin`
- `__repr__`
- `print_tree`

---

## engine/time/transport.py

### Classes

- `TimeSignature` - Time signature (frozen dataclass)
- `TransportState` - Immutable transport state snapshot (frozen dataclass)
- `Transport` - Mutable transport control

### TimeSignature Properties

- `beats_per_bar`
- `__str__`

### TransportState Properties

- `phase_in_beat`
- `phase_in_bar`
- `current_bar`
- `current_beat_in_bar`
- `seconds_per_beat`
- `beats_per_second`
- `bar_duration_beats`
- `bar_duration_seconds`

### Transport Methods

- `__init__`
- `play`
- `pause`
- `stop`
- `toggle`
- `seek_to_beat`
- `seek_to_time`
- `seek_by_bars`
- `seek_to_bar`
- `seek_to_next_bar`
- `seek_to_previous_bar`
- `set_bpm`
- `set_time_signature`
- `tap_tempo`
- `set_loop`
- `clear_loop`
- `toggle_loop`
- `update`
- `_snapshot`
- `on_beat`
- `on_bar`
- `on_loop`
- `beat_to_time`
- `time_to_beat`
- `beat_to_bar`
- `bar_to_beat`
- `format_time`
- `format_bar_beat`

### Type Aliases

- `BeatCallback`
- `BarCallback`
- `LoopCallback`
- `StateCallback`

---

## engine/time/camera.py

### Classes

- `TimeCameraMode` (Enum) - Camera modes
- `TimeCameraConfig` - Configuration dataclass
- `TimeCamera` - View transformation for time axis (extends SignalEmitter)

### TimeCamera Methods

- `_px_per_beat` (property)
- `beat_to_px`
- `px_to_beat`
- `beat_to_screen`
- `screen_to_beat`
- `is_beat_visible`
- `is_range_visible`
- `get_visible_range`
- `right_beat` (property)
- `center_beat` (property)
- `snap_to_grid`
- `nearest_beat`
- `nearest_bar`
- `snap_px_to_grid`
- `iter_bar_beats`
- `iter_beat_positions`
- `iter_subdivision_beats`
- `get_visible_bar_lines_px`
- `get_visible_beat_lines_px`
- `get_time_matrix`
- `get_inverse_time_matrix`
- `get_view_projection`
- `begin_drag`
- `update_drag`
- `end_drag`
- `cancel_drag`
- `zoom`
- `zoom_to_fit`
- `set_zoom_level`
- `animate_to_beat`
- `animate_to_center_on`
- `jump_to_beat`
- `update`
- `_update_animation`
- `_update_follow_mode`
- `_update_snap_mode`
- `bind`
- `_on_dt`
- `_emit_view_changed`
- `set_panel_size`

---

## engine/time/old_camera.py

### Classes

- `TimeCameraMode` (Enum)
- `TimeCameraConfig` (dataclass)
- `TimeCamera` (dataclass)

### TimeCamera Methods

- `_px_per_beat` (property)
- `beat_to_px`
- `px_to_beat`
- `is_beat_visible`
- `get_visible_range`
- `snap_to_grid`
- `begin_drag`
- `update_drag`
- `end_drag`

---

## engine/viewport/viewport.py

### Classes

- `PanelRect` - Rectangle for panel in screen space
- `AreaLayout` - Manages panel layout within window
- `Camera` - Mutable camera state
- `PickResult` (NamedTuple) - Result of picking operation
- `ViewportArea` - Viewport area rendering to its own texture
- `ExtractResult` - Result from async command extraction

### Standalone Functions

- `_worker_build_commands` - Worker function for building CommandLists

### PanelRect Methods

- `contains`

### AreaLayout Methods

- `set_window_size`
- `compute_panels`

### ViewportArea Methods

- `__init__`
- `mark_dirty`
- `is_dirty`
- `clear_dirty`
- `ensure_surface`
- `kick_extract_if_needed`
- `render_if_ready`
- `pick_at`
- `_find_camera_at`
- `handle_pointer`
- `handle_wheel`
- `_get_active_camera`
- `_orbit`
- `_zoom`
- `get_debug_info`

---

## engine/ui/__init__.py

No classes or functions defined (only imports and `__all__`).

---

## engine/ui/draw.py

### Classes

- `DrawQuad` - Single quad to draw
- `DrawLine` - Line segment
- `DrawText` - Text to draw
- `DrawBatch` - Collection of draw commands
- `DrawContext` - Context for drawing UI elements
- `UIRenderer` - Renders DrawBatch to GPU

### DrawBatch Methods

- `add_quad`
- `add_line`
- `add_text`
- `finalize`
- `clear`
- `quad_count` (property)
- `total_vertices` (property)

### DrawContext Methods

- `__init__`
- `push_offset`
- `pop_offset`
- `_transform`
- `push_clip`
- `pop_clip`
- `_intersect_rects`
- `_is_clipped`
- `draw_rect`
- `draw_rect_outline`
- `draw_line`
- `draw_text`
- `draw_text_in_rect`
- `finalize`
- `clear`

### UIRenderer Methods

- `__init__`
- `_ensure_initialized`
- `_create_ui_pipeline`
- `_ensure_quad_buffer`
- `_ensure_line_buffer`
- `render`
- `_render_quads`
- `_render_lines`

---

## engine/ui/layout.py

### Classes

- `LayoutDirection` (Enum)
- `Justify` (Enum) - Main axis distribution
- `Align` (Enum) - Cross axis alignment
- `Rect` - Rectangle with position and size
- `Constraints` - Size constraints for layout
- `FlexLayout` - Flexbox-style layout configuration

### Rect Methods

- `contains`
- `inset`
- `inset_by`
- `right` (property)
- `bottom` (property)
- `center_x` (property)
- `center_y` (property)
- `copy`

### Constraints Methods

- `constrain`
- `constrain_width`
- `constrain_height`
- `loosen`
- `tight` (static)
- `loose` (static)

### FlexLayout Methods

- `is_row`
- `is_column`

### Standalone Functions

- `layout_flex` - Perform flexbox layout
- `measure_flex` - Measure size needed for flex container

---

## engine/ui/style.py

### Classes

- `EdgeInsets` - Insets for padding/margin (frozen dataclass)
- `Border` - Border style (frozen dataclass)
- `Shadow` - Box shadow (frozen dataclass)
- `SizeValue` - Size value (frozen dataclass)
- `Style` - Complete style for widget (frozen dataclass)
- `Theme` - Collection of named styles

### Standalone Functions

- `color_rgba` - Normalize color to RGBA tuple
- `color_to_array` - Convert color to numpy array
- `hex_to_color` - Convert hex string to color

### EdgeInsets Static Methods

- `all`
- `symmetric`
- `only`

### SizeValue Methods

- `px` (static)
- `pct` (static)
- `auto` (static)
- `fill` (static)
- `resolve`
- `is_fixed`
- `is_flexible`

### Style Methods

- `with_size`
- `with_padding`
- `with_margin`
- `with_background`
- `with_border`
- `with_text`
- `with_flex`

### Theme Methods

- `panel_style`
- `button_style`
- `label_style`
- `title_bar_style`

---

## engine/ui/widget.py

### Classes

- `LayoutResult` - Result of layout pass
- `EventType` (Enum)
- `Event` - UI event
- `WidgetState` (Enum) - Interactive state
- `Widget` - Base class for all UI widgets
- `RootWidget` - Special root widget filling window

### Event Methods

- `stop_propagation`
- `prevent_default`
- `stopped` (property)
- `prevented` (property)

### Widget Methods

- `__init__`
- `children` (property)
- `add_child`
- `remove_child`
- `clear_children`
- `get_root`
- `rect` (property)
- `content_rect` (property)
- `measure`
- `layout`
- `draw`
- `hit_test`
- `on`
- `off`
- `emit`
- `handle_event`
- `state` (property)
- `set_hovered`
- `set_pressed`
- `set_focused`
- `set_enabled`
- `set_visible`
- `enabled` (property)
- `visible` (property)
- `__repr__`
- `print_tree`

### RootWidget Methods

- `__init__`
- `set_window_size`
- `do_layout`
- `dispatch_pointer_event`
- `dispatch_scroll_event`
- `dispatch_key_event`
- `set_focus`
- `_to_local`

---

## engine/ui/window_manager.py

### Classes

- `DockSide` (Enum)
- `SplitDirection` (Enum)
- `DockNode` - Node in dock tree (base)
- `DockLeaf` - Leaf node with single panel
- `DockSplit` - Split node with children
- `DockTabs` - Tab container with multiple panels
- `Splitter` - Draggable splitter widget
- `DockContainer` - Widget rendering DockNode tree
- `TabBar` - Tab bar for selecting panels
- `TabContainer` - Container with tab bar and content
- `WindowManager` - Manages entire UI layout

### Splitter Methods

- `__init__`
- `_on_pointer_down`
- `_on_pointer_move`
- `_on_pointer_up`
- `_on_enter`
- `_on_leave`
- `draw`

### DockContainer Methods

- `__init__`
- `dock_root` (property/setter)
- `_rebuild_widgets`
- `_build_node`
- `_handle_splitter_drag`

### TabBar Methods

- `__init__`
- `_rebuild_tabs`
- `_create_tab`
- `draw`

### TabContainer Methods

- `__init__`
- `_on_tab_select`
- `_update_content`

### WindowManager Methods

- `__init__`
- `root` (property)
- `set_size`
- `do_layout`
- `set_dock_root`
- `dock`
- `undock`
- `add_floating`
- `remove_floating`
- `bring_to_front`
- `handle_pointer_event`
- `handle_scroll_event`
- `handle_key_event`
- `draw`
- `save_layout`
- `_serialize_node`
- `load_layout`
- `_deserialize_node`

---

## engine/ui/widgets/__init__.py

No classes or functions defined (only imports and `__all__`).

---

## engine/ui/widgets/button.py

### Classes

- `Button` - Clickable button
- `IconButton` - Button with icon
- `ToggleButton` - Toggle button with on/off state

### Button Methods

- `__init__`
- `_make_hover_style`
- `_make_pressed_style`
- `_make_disabled_style`
- `text` (property/setter)
- `current_style` (property)
- `measure`
- `draw`

### ToggleButton Methods

- `toggled` (property/setter)
- `_handle_toggle`
- `current_style` (property)

---

## engine/ui/widgets/container.py

### Classes

- `Container` - Generic flex container
- `Row` - Horizontal flex container
- `Column` - Vertical flex container
- `Spacer` - Flexible empty space
- `Divider` - Visual separator line

### Spacer Methods

- `__init__`
- `measure`
- `draw`

### Divider Methods

- `__init__`
- `measure`
- `draw`

---

## engine/ui/widgets/label.py

### Classes

- `Label` - Text label
- `Heading` - Large heading text
- `Caption` - Small caption text

### Label Methods

- `__init__`
- `text` (property/setter)
- `measure`
- `draw`

---

## engine/ui/widgets/panel.py

### Classes

- `TitleBar` - Panel title bar
- `Panel` - Titled panel with chrome
- `FloatingPanel` - Panel floating at absolute position

### TitleBar Methods

- `__init__`
- `title` (property/setter)
- `_on_pointer_down`
- `_on_pointer_move`
- `_on_pointer_up`
- `measure`
- `draw`

### Panel Methods

- `__init__`
- `title` (property/setter)
- `content` (property)
- `add_content`
- `_handle_close`
- `_handle_drag`
- `draw`

### FloatingPanel Methods

- `__init__`
- `position` (property)
- `set_position`
- `layout`

---

## engine/ui/widgets/viewport.py

### Classes

- `Viewport3D` - 3D viewport display widget
- `ViewportToolbar` - Toolbar overlay for viewport

### Viewport3D Methods

- `__init__`
- `area` (property/setter)
- `_forward_pointer`
- `_forward_scroll`
- `measure`
- `draw`
- `get_viewport_rect`

---

## Summary

| Directory | Classes | Functions/Methods |
|-----------|---------|-------------------|
| `/core` | ~20 | ~150 |
| `/render` | ~20 | ~100 |
| `/scene` | ~5 | ~25 |
| `/time` | ~8 | ~80 |
| `/ui` | ~30 | ~200 |
| `/viewport` | ~6 | ~30 |
| **Total** | **~90** | **~600+** |
