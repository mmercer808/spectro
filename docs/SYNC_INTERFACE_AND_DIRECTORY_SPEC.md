# Sync Interface (Input ↔ Main GUI ↔ Vercel) and Directory Convention

The **API** provides a **sync interface** between **input** (outside), the **main GUI thread**, and the **Vercel thread**. This document defines that interface and uses it as the **convention for directory structure**. It also specifies: observer dropbox-directory (stream into stereo-space), separate stream (convolution × mixing layers beside main), and loopback channel (input → queue; periodic thread to play audio or leave a message).

---

## 1. Sync Interface: Input ↔ Main GUI ↔ Vercel

### 1.1 Three domains

| Domain | Role |
|--------|------|
| **Input** | Outside world: MIDI, audio input, dropbox-directory events, loopback capture. |
| **Main GUI thread** | UI, timeline, mix channel, rendering; owns the primary stereo output (main stream). |
| **Vercel thread** | Server/frontend (Vercel): API, persistence, project edits, optional real-time bridge. |

The **sync interface** is the API boundary that keeps these three in sync: input events and external data flow into the main thread (or into a queue that the main thread drains); the main thread drives the GUI and main stream; the Vercel thread can read/write project state and send messages (e.g. via WS) that the main thread observes.

### 1.2 Sync API (conceptual)

- **Input → Main:** Input events (MIDI, new files in dropbox, loopback chunks) are **stuffed into a queue** (or written to a buffer). The **main GUI thread** (or a dedicated audio/input thread) **drains** the queue and applies updates (e.g. add to timeline, feed stereo-space, update mix).
- **Main → Vercel:** Main thread (or a bridge) can **emit** state changes (playhead, mix channel, timeline edits) to the Vercel thread via API/WS so the server/frontend stays in sync.
- **Vercel → Main:** Vercel thread sends **messages** (e.g. load project, set tempo, add object) that are queued and consumed by the main thread so the GUI and timeline stay in sync with the server.

So the **sync interface** = queues (or message channels) at the boundaries: **input queue** (input → main), **main→vercel** channel (e.g. WS/API), **vercel→main** queue (messages to main thread).

### 1.3 Directory structure convention (from sync interface)

Use the sync boundaries as the convention for layout:

```
sync/                    # Sync interface (API surface)
  input/                 # Input side: adapters, queues, observers
  main/                  # Main GUI thread: app, timeline, mix, main stream
  vercel/                # Vercel thread: API routes, WS, project persistence
  api/                   # Shared API types, message schemas (sync contract)
```

Or flattened:

```
input/                   # Input adapters, dropbox observer, loopback → queue
main/                    # Main GUI app, timeline, main stereo stream
vercel/                  # Vercel app (web/, api/)
api/                     # Sync API: message types, queue contracts
```

So: **input/** = everything that feeds from outside into the sync (queues, observers). **main/** = main-thread app and main stream. **vercel/** = Vercel app and server. **api/** (or **sync/api/**) = the shared sync contract (message types, queue semantics). This is the **convention**: structure follows the sync boundaries (input, main, vercel, api).

---

## 2. Observer Dropbox-Directory → Stream into Stereo-Space

### 2.1 Dropbox-directory (observer)

- The **creative** can **drop a directory** into an **observer dropbox-directory** (a watched folder).
- When new files (or a new directory) appear in that folder, the observer **notifies** the sync layer (e.g. pushes a message into the input queue or a dedicated “dropbox” queue).
- The main thread (or a worker) **consumes** those events and **streams** the content (e.g. audio files, stems) **into the stereo-space** — i.e. into the main mix or into a designated bus so they become part of the output.

### 2.2 Flow

- **Dropbox-directory** = watched path (e.g. `observer/dropbox/` or a user-configured path).
- **Observer** = file-system watcher (e.g. watchdog, chokidar, or OS API). On new/updated file or directory, enqueue **input** event: e.g. `{ type: "dropbox", path: "...", kind: "file" | "dir" }`.
- **Input queue** (or a dedicated consumer) passes events to **main**; main (or a DSP stage) **loads/streams** the content and **adds it to the stereo-space** (main mix or a side stream).

So: **observer dropbox-directory** → events into sync **input** → main thread (or DSP) → **stream into stereo-space**. Directory structure: **input/** can contain `observer/` or `dropbox/` (watcher + queue adapter).

---

## 3. Separate Stream: Convolution Matrix × Mixing Layers (Beside Main)

### 3.1 DSP: convolution matrix × mixing layers

- You are **multiplying** a few **DSP** (e.g. a **convolution matrix**) with a **mixing layers** algorithm, and **adding** them to a **separate stream** — one **beside the main** (main = main stereo output).
- So there are at least **two streams**:
  - **Main stream** — primary stereo output (main GUI thread, main mix).
  - **Side stream** — convolution matrix × mixing layers; output is added to this stream (or to a bus that can be mixed back into main or monitored separately).

### 3.2 Placement in directory convention

- **Main stream** lives in **main/** (or engine/main stream pipeline).
- **Side stream** (convolution × mixing layers) can live under **main/dsp/** or **main/streams/side/** or a dedicated **dsp/** that the main thread feeds. The sync interface does not change: input and Vercel still sync with **main**; main owns both the main stream and the side stream (or a separate process/thread that owns the side stream and receives from main).

So: **main/** (or **main/streams/**) = main stream + side stream (convolution matrix × mixing layers). Directory structure can be:

```
main/
  stream_main/     # Primary stereo output
  stream_side/     # Convolution × mixing layers (separate stream beside main)
  dsp/             # Convolution matrix, mixing layers algorithm
```

---

## 4. Loopback Interface Channel: Input → Queue; Periodic Thread to Play or Message

### 4.1 Loopback as input channel

- **Loopback** = capture of system output (e.g. WASAPI loopback) as an **input** to the app. So loopback is treated as an **input channel** (like a mic): it is **stuffed into a queue** (e.g. `AudioRingBuffer` or a lock-free queue).
- The **main thread** (or an **audio thread**) can:
  - **Periodically launch a thread** to **play** audio from that queue (e.g. re-inject into the main stream or the side stream), or
  - **Leave a message** (e.g. “loopback chunk ready”) so that another component (e.g. main GUI or DSP) can pull from the queue and process/play.

So: **loopback → queue**; then either **periodic thread** that plays from the queue, or **message** (“chunk ready”) so a consumer plays or processes. Existing code: `engine/buffers_v2.py` has `LoopbackAudioBuffer` (captures loopback into an `AudioRingBuffer`). That buffer can be the **queue**; the “periodic thread” or “message” is the consumer that reads from the buffer and plays or forwards to the main/side stream.

### 4.2 Placement in sync / directory convention

- **Loopback** is part of **input**: it feeds into the sync layer (queue). So **input/loopback/** or **input/audio/** can hold the loopback capture and the queue; the **periodic thread** that plays (or the component that leaves a message) can live in **input/loopback/** or in **main/** (main thread drains loopback queue and feeds main/side stream).
- Convention: **input/** = loopback capture + queue; **main/** (or a dedicated audio worker) = periodic thread or message handler that reads from the queue and plays or adds to a stream.

So: **input/loopback/** — loopback capture, stuff into queue; **main/** (or **main/audio/**) — periodically launch thread to play from queue, or leave message for consumer.

---

## 5. Directory Structure Convention (Summary)

Using the **sync interface** (input ↔ main ↔ Vercel) as the convention:

```
api/                     # Sync API: message types, queue contracts (input ↔ main ↔ vercel)
input/                   # Input side: outside → queue
  observer/              # Dropbox-directory watcher; events → queue
  dropbox/               # (or) watched path config; stream into stereo-space
  loopback/              # Loopback capture → queue; periodic play or message
main/                    # Main GUI thread
  stream_main/           # Main stereo stream
  stream_side/           # Separate stream: convolution × mixing layers
  dsp/                   # Convolution matrix, mixing layers algorithm
vercel/                  # Vercel app (or web/, app/)
  # API routes, WS, project persistence; sync with main via api/
```

Or with **sync** as explicit top-level:

```
sync/
  api/                   # Sync contract (messages, queues)
  input/                 # Input adapters, observer, loopback
  main/                  # Main app, main + side streams, DSP
  vercel/                # Vercel app
```

- **Creative drops a directory** into **input/observer/dropbox/** (or a configured path) → observer enqueues → main consumes and **streams into stereo-space**.
- **Loopback** → **input/loopback/** (queue) → **periodic thread** in main (or input) **plays** from queue or **leaves a message** for main to consume.
- **Convolution × mixing layers** → **main/stream_side/** (or main/dsp/) → **separate stream beside main**.

---

## 6. Relation to Existing Code

- **Loopback:** `engine/buffers_v2.py` — `LoopbackAudioBuffer`; can be used as the loopback input queue; add a **periodic thread** or **message** in the consumer (e.g. in main or in a small `input/loopback` adapter) that reads from the buffer and plays or forwards.
- **Sync / queues:** `engine/ws/` (dark scheduler, response log) and project edit queue; can be extended so that **input** and **vercel** use the same sync API (queues + message types).
- **Directory structure:** This spec is the **convention**; existing `engine/`, `demo/`, `docs/` can stay, and new **api/**, **input/**, **main/**, **vercel/** (or **sync/**) can be added to match the sync interface.

---

*Last updated: Feb 2025.*
