# Main App: Accepted WebSocket Server for Saving Project Edits

The main app **must** have an **accepted WebSocket (WS) server** used to save **additional edits** for a project. This applies to **whichever project** is open: the app connects to an accepted WS endpoint and sends project edits (e.g. sequencer events, transport state, timeline changes) so they can be persisted or synced.

---

## 1. Requirement

- **Accepted WS server:** The app accepts (or is configured with) a WebSocket server URL. Connection to that server is the **accepted** channel for saving edits.
- **Saving additional edits:** Edits made in the app (e.g. new events, deleted events, BPM change, loop change) are sent over the WS connection so they can be saved for the current project.
- **Whichever project:** The same mechanism applies to any open project; the project identifier (or session id) can be sent with each edit so the server can associate edits with the correct project.

---

## 2. Behavior (Expected)

- **Connect:** On startup or when a project is opened, the app connects to the accepted WS server (if configured).
- **Send edits:** When the user (or the system) makes an edit, the app enqueues an "edit" payload (e.g. `{ "type": "edit", "project_id": "...", "payload": { ... } }`) and sends it over the WS connection. Sending can be done in the background (**dark scheduling**) so the main thread is not blocked.
- **Responses:** The server may send responses (e.g. `ok`, `error`, or updated state). The app **holds a text space** (e.g. a log panel) and **appends** each WS response to it so the user can see what happens.
- **Reconnect:** If the connection drops, the app may attempt to reconnect; pending edits may be queued and sent when reconnected.

---

## 3. Dark Scheduling

- **Dark scheduling** = background processing of "save edit" work so the main loop and UI are not blocked.
- Edits are pushed to a **queue**; a background thread (or a scheduled task) consumes the queue and sends payloads over the WS connection.
- WS **responses** are appended to the **response log** (text space) on the main thread (or via a thread-safe buffer that the main thread drains each frame) so the user sees what happens.

---

## 4. Response Text Space

- The app reserves a **text space** (e.g. a panel or log area) for WS responses.
- **Append-only:** New responses are **appended** to this space; the content is not cleared automatically so the user can scroll and see the full history of what happened.
- Optionally: max lines or max length to avoid unbounded growth; older lines can be trimmed while keeping the append-only semantics for new data.

---

## 5. Summary

| Item | Description |
|------|-------------|
| **Accepted WS server** | Required for the main app; used to save additional edits for a project (whichever project). |
| **Edits** | Sent over WS (e.g. project_id + edit payload); can be queued and sent in the background (dark scheduling). |
| **Responses** | Shown in a dedicated text space; append-only so the user can see what happens. |

---

---

## 6. Implementation

- **engine/ws/** — `ResponseLog` (append-only), `WSClient` (connect, send, push responses to log queue), `DarkScheduler` (background queue of edits, sends via client). Optional dependency: `websocket-client` for real WS.
- **demo/run_demo.py** — Uses `ResponseLog`, `WSClient`, `DarkScheduler`; reserves a right-side **text space** panel (WS_LOG_PANEL_WIDTH); each frame drains the log (append-only) and prints new lines to console so the user can see what happens. Connect via env `SPECTRO_WS_URL` if set.
- **Enqueue edits:** Call `dark_scheduler.enqueue_edit(project_id, payload)` to queue an edit for sending over WS.

*See also: required_standards (sourcebook); improvising_standards. Last updated: Feb 2025.*
