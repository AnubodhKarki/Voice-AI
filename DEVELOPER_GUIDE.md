# Developer Guide: AssemblyAI Explorer

This guide explains how the app is built, how data flows through it, and how to safely edit and extend it.

## 1) Project overview

The repository includes two practical surfaces:

- Notebook exploration in `notebooks/` for API learning and experimentation.
- A refactored Streamlit app launched by `assemblyai/app.py`.

The app itself is implemented in the `src/assemblyai_explorer/` package with modular separation of concerns.

## 2) Architecture and separation of concerns

### Entrypoint

- `assemblyai/app.py`
  - Minimal launcher.
  - Backward-compatible launcher for `streamlit run assemblyai/app.py`.
  - Adds `src/` to `sys.path`, then imports and runs `run_app`.

### Core modules

- `src/assemblyai_explorer/config.py`
  - Loads `.env` via `python-dotenv`.
  - Defines API key, endpoint constants, model/language options.
  - Exposes `auth_headers()` to centralize auth header creation.

- `src/assemblyai_explorer/api.py`
  - Contains HTTP calls to AssemblyAI (`upload`, `submit`, `poll`, `list`, `delete`, `get`, `sentences`, `paragraphs`, `health`).
  - Returns structured debug tuples (JSON, status, latency) for UI transparency.

- `src/assemblyai_explorer/payloads.py`
  - Pure functions for request payload construction and params snapshots.
  - No Streamlit dependencies.
  - Safe and easy to unit test.

- `src/assemblyai_explorer/rendering.py`
  - UI rendering logic for transcript outputs and feature-specific sections.
  - Includes helper to normalize IAB category score shapes.

- `src/assemblyai_explorer/state.py`
  - Initializes Streamlit session state defaults in one place.
  - Prevents scattered state setup across UI code.

- `src/assemblyai_explorer/streaming.py`
  - Encapsulates streaming SDK import checks and live-session lifecycle.
  - Runs the streaming session in a `multiprocessing.Process` (not a thread) so a PyAudio segfault (SIGSEGV) kills only the child process — Streamlit keeps running.
  - Communicates back to the UI via `multiprocessing.Queue` using typed events: `session_id`, `transcript_line`, `audio_duration`, `stream_ended`, `error`, `log`, `pid`.
  - Emits timestamped log events throughout the session lifecycle for the debug event log.
  - Uses a `multiprocessing.Event` as the stop signal so the parent can gracefully interrupt the child without calling cross-process object methods.

- `src/assemblyai_explorer/ui.py`
  - App composition layer.
  - Defines sidebar/history and all tabs (`Pre-recorded`, `Live Streaming`, `API Debug`).
  - Orchestrates calls to API, payload, rendering, and state modules.

## 3) Runtime flow (Pre-recorded tab)

1. User chooses source (default URL / custom URL / file upload).
2. UI collects model/language and optional features.
3. `build_transcript_payload(...)` generates request JSON.
4. `submit_transcript_debug(...)` sends request.
5. UI polls with `poll_transcript_debug(...)` until complete/error.
6. Output is rendered via `render_results(...)`.
7. History snapshot is appended to `st.session_state.history`.

## 4) Runtime flow (Live Streaming tab)

1. UI verifies streaming SDK availability.
2. On Start:
   - session state is reset (transcript, event log, PID, exit code)
   - `start_streaming_thread(...)` creates a `multiprocessing.Queue` and `multiprocessing.Event`, then spawns a child `Process` running `run_streaming_session(...)`
3. Child process lifecycle (isolated from Streamlit):
   - emits `pid` event immediately so the UI can display the process PID
   - opens `PyAudioMicrophoneStream` (checks `stop_event` on each `__next__`)
   - connects `StreamingClient` to `streaming.assemblyai.com`
   - emits `log` events at each stage for the session event log
   - emits `session_id`, `transcript_line`, `audio_duration`, `stream_ended` via the queue
   - if a segfault or fatal error kills the process, the queue and Streamlit are unaffected
4. `drain_stream_events(...)` is called on every UI rerun:
   - drains the queue and updates session state
   - checks `proc.exitcode`: if non-zero and streaming is still flagged active, sets `streaming = False` and surfaces a crash message
5. On Stop:
   - `stop_streaming(...)` sets the `multiprocessing.Event`
   - `PyAudioMicrophoneStream.__next__` sees the event and raises `StopIteration`
   - `client.stream(mic)` returns, `finally` block disconnects and closes mic
   - parent joins the process (3s timeout) then terminates if still alive

## 5) Setup and run instructions

## Prerequisites

- Python 3.10+ recommended
- AssemblyAI API key
- Optional for live streaming:
  - `pyaudio`
  - PortAudio system library on macOS (`brew install portaudio`)

## Install (Poetry)

```bash
poetry install
cp .env.example .env
```

Edit `.env` and set:

```env
ASSEMBLYAI_API_KEY=your_key_here
```

## Run app

Recommended:

```bash
poetry run explorer
```

Backward-compatible:

```bash
streamlit run assemblyai/app.py
```

## Run tests

```bash
poetry run pytest -q
```

## Optional sanity compile

```bash
poetry run python -m compileall -q src/assemblyai_explorer assemblyai tests
```

## 6) How to edit safely (without breaking behavior)

Use this checklist when making changes:

1. Keep `assemblyai/app.py` minimal and unchanged unless entry behavior must change.
2. Put business/request logic in pure helpers first (`payloads.py`, utility helpers).
3. Keep Streamlit widget orchestration in `ui.py` only.
4. Keep network logic in `api.py`; avoid HTTP calls directly in UI code.
5. Add or update tests for every logic change that can be tested without UI.
6. Run `pytest` before and after edits.

## 7) Where to add new features

### Add a new transcription feature flag

1. Add a checkbox/select in `render_prerecorded_tab()` (`ui.py`).
2. Add payload mapping in `build_transcript_payload()` (`payloads.py`).
3. Add output section in `render_results()` (`rendering.py`) if response data is displayed.
4. Add/adjust tests in `tests/test_payloads.py` and/or rendering helper tests.

### Add a new API debug action

1. Add function in `api.py` — return `(body, status_code, elapsed_ms)` to stay consistent with existing helpers.
2. Add UI controls and response display in `render_debug_tab()` (`ui.py`).
3. Optionally add a `_curl_get` / `_curl_delete` call to show the equivalent cURL command.

### Add a new streaming event type

1. Emit it in `run_streaming_session(...)` via `_emit(events_queue, "your_event", payload)`.
2. Handle it in the `drain_stream_events(...)` `while` loop in `streaming.py`.
3. Add the corresponding session state key and default in `state.py`.

### Add another tab

1. Create `render_<new_tab>_tab()` in `ui.py` (or separate module if large).
2. Wire it in `run_app()` tab layout.

## 8) Testing strategy

Current tests focus on deterministic pure logic:

- `tests/test_payloads.py`
  - validates payload composition rules and edge handling.
- `tests/test_rendering.py`
  - validates IAB score normalization helper behavior.

Recommended future additions:

- API client tests with request mocking for status/shape handling.
- Lightweight state initialization tests.
- Optional smoke test for importability of `run_app`.

## 9) Common troubleshooting

- `ASSEMBLYAI_API_KEY not found`
  - Check `.env` exists and contains valid key.
  - Restart shell/session after editing env variables if needed.
  - Use the **API Health Check** in the Debug tab — it will report auth failure immediately.

- Live streaming import error
  - Install PyAudio: `pip install pyaudio`
  - macOS may require `brew install portaudio` first.

- Empty transcript during streaming
  - Confirm microphone permission/system input source.
  - Speak clearly for a few seconds; only formatted turns are appended.
  - Check the **session event log** in the streaming tab — it records every turn received with word count. If turns show 0 words, the mic is active but speech detection isn't triggering.

- Streaming subprocess shows crashed (exit -11)
  - Exit code `-11` is SIGSEGV — a segfault from PyAudio/PortAudio's native C layer.
  - This is isolated to the child process; Streamlit is unaffected.
  - Try a different input device from the dropdown (some devices trigger PortAudio bugs on macOS).
  - The event log will show how far the session got before the crash.

## 10) Quick command reference

```bash
# install dependencies
poetry install

# run app (recommended)
poetry run explorer

# run app (legacy)
streamlit run assemblyai/app.py

# run tests
poetry run pytest -q

# compile check
poetry run python -m compileall -q src/assemblyai_explorer assemblyai tests
```

## 11) GitHub deployment and CI

This project is straightforward to publish on GitHub:

1. Push repository.
2. Use GitHub Actions for CI (lint/tests/build checks).
3. Deploy app from GitHub via Streamlit Community Cloud:
   - repo: this project
   - entrypoint: `assemblyai/app.py`
   - secrets: set `ASSEMBLYAI_API_KEY`

With this setup, deployment is typically smooth, not painful.
