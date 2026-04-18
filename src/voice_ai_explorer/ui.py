import time
from datetime import datetime

import streamlit as st

from .api import (
    check_api_health,
    delete_transcript,
    get_transcript,
    get_transcript_paragraphs,
    get_transcript_sentences,
    list_transcripts,
    poll_transcript_debug,
    probe_audio_url,
    submit_transcript_debug,
    upload_file,
)
from .audio_quality import (
    analyze_uploaded_audio,
    analyze_url_metadata,
    make_waveform_chart,
    make_spectrogram_chart,
    run_deep_analysis,
)
from .config import (
    DEFAULT_AUDIO_URL,
    DEEPGRAM_ENGLISH_ONLY_MODELS,
    DEEPGRAM_LANGUAGE_OPTIONS,
    DEEPGRAM_MODEL_OPTIONS,
    DEEPGRAM_STREAMING_MODEL_OPTIONS,
    LANGUAGE_OPTIONS,
    MODEL_OPTIONS,
    STREAMING_MODEL_OPTIONS,
    get_assemblyai_key,
    get_deepgram_key,
)
from .payloads import build_params_snapshot, build_transcript_payload
from .rendering import render_results
from .state import init_session_state
from .streaming import (
    WebRTCAudioQueue,
    convert_audio_frame,
    drain_stream_events,
    format_input_device_label,
    list_input_devices,
    start_deepgram_streaming_thread,
    start_streaming_thread,
    start_streaming_thread_browser,
    stop_streaming,
    streaming_sdk_import,
)
from .providers.deepgram_api import (
    build_options as dg_build_options,
    check_api_health as dg_check_api_health,
    extract_confidence,
    extract_transcript as dg_extract_transcript,
    get_projects as dg_get_projects,
    get_request as dg_get_request,
    list_requests as dg_list_requests,
    transcribe_file as dg_transcribe_file,
    transcribe_url as dg_transcribe_url,
)

try:
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False


def render_sidebar():
    with st.sidebar:
        st.header("Voice AI Explorer")

        with st.expander("API Keys", expanded=not (get_assemblyai_key() or get_deepgram_key())):
            aai_env = get_assemblyai_key()
            dg_env = get_deepgram_key()

            aai_placeholder = "Set" if aai_env else "Paste AssemblyAI key..."
            dg_placeholder = "Set" if dg_env else "Paste Deepgram key..."

            aai_input = st.text_input(
                "AssemblyAI API Key",
                value=st.session_state.aai_api_key,
                type="password",
                placeholder=aai_placeholder,
                key="_sidebar_aai_key",
            )
            dg_input = st.text_input(
                "Deepgram API Key",
                value=st.session_state.dg_api_key,
                type="password",
                placeholder=dg_placeholder,
                key="_sidebar_dg_key",
            )
            if aai_input != st.session_state.aai_api_key:
                st.session_state.aai_api_key = aai_input
            if dg_input != st.session_state.dg_api_key:
                st.session_state.dg_api_key = dg_input

            aai_resolved = get_assemblyai_key(st.session_state.aai_api_key)
            dg_resolved = get_deepgram_key(st.session_state.dg_api_key)
            st.caption(
                f"AssemblyAI: {'✅ Ready' if aai_resolved else '❌ Not set'}  \n"
                f"Deepgram: {'✅ Ready' if dg_resolved else '❌ Not set'}"
            )

        st.divider()
        st.header("History")
        if not st.session_state.history:
            st.caption("No transcriptions yet.")
        for item in reversed(st.session_state.history):
            with st.expander(f"{item['timestamp']} — {item.get('provider','AAI')} · {item['model']}"):
                st.write(f"**Source:** {item['audio_source']}")
                if item.get("id"):
                    st.write(f"**ID:** `{item['id']}`")
                st.write(item["snippet"])
                if item.get("result"):
                    render_results(
                        item["result"], item["params"], allow_expanders=False
                    )


def render_audio_quality_report(report: dict):
    st.markdown("### Audio Quality Check")
    col_score, col_label = st.columns(2)
    col_score.metric("Quality score", f"{report['score']}/100")
    col_label.metric("Quality label", report["label"].upper())

    warnings = report.get("warnings") or []
    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))
    else:
        st.success("No obvious quality concerns detected from available metadata.")

    with st.expander("Quality metrics", expanded=False):
        st.json(report.get("metrics", {}))


def render_deep_audio_analysis(analysis: dict):
    """
    Render deep audio analysis results in three sections:
      1. Readiness summary + metric cards
      2. Waveform and spectrogram
      3. Issue table with timestamps
    """
    readiness = analysis["readiness"]
    silence = analysis["silence"]
    clipping = analysis["clipping"]
    loudness = analysis["loudness"]
    metrics = analysis["metrics"]
    issues = analysis["issues"]

    st.markdown("---")
    st.markdown("### Audio Quality Analysis")

    # ── Section 1: Readiness summary + cards ──────────────────────────────────
    color = readiness["color"]
    label = readiness["label"]
    score = readiness["score"]

    # Readiness badge using colored markdown
    badge_colors = {"green": "#1e8c45", "orange": "#c97a10", "red": "#c0392b"}
    hex_color = badge_colors.get(color, "#555")
    st.markdown(
        f"<div style='background:{hex_color};color:white;padding:10px 16px;"
        f"border-radius:6px;font-weight:600;font-size:1.05rem;margin-bottom:12px'>"
        f"{'✅' if color == 'green' else '⚠️' if color == 'orange' else '🚫'} "
        f"{label} &nbsp;·&nbsp; Score: {score}/100</div>",
        unsafe_allow_html=True,
    )

    # Metric cards row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{metrics.get('duration_seconds', '—')} s")
    c2.metric("Sample Rate", f"{metrics.get('sample_rate_hz', '—')} Hz")
    c3.metric(
        "Silence",
        f"{silence['silence_pct']}%",
        delta=f"{len(silence['segments'])} segment(s)" if silence["segments"] else None,
        delta_color="inverse",
    )
    clipping_val = "Yes" if clipping["clipping_present"] else "None"
    c4.metric("Clipping", clipping_val)

    # Flags list
    flags = readiness.get("flags", [])
    if flags:
        with st.expander("Readiness factors", expanded=True):
            for flag in flags:
                st.markdown(f"- {flag}")
    else:
        st.success("No significant issues detected.")

    # Loudness row
    loudness_label = loudness.get("label", "unknown")
    loudness_color = {"consistent": "normal", "uneven": "off", "poor": "inverse"}.get(
        loudness_label, "normal"
    )
    st.metric(
        "Loudness consistency",
        loudness_label.capitalize(),
        delta=f"std dev ±{loudness.get('std_db')} dB"
        if loudness.get("std_db") is not None
        else None,
        delta_color=loudness_color,
    )

    # ── Section 2: Waveform and spectrogram ───────────────────────────────────
    st.markdown("#### Signal Visualizations")
    tab_wave, tab_spec = st.tabs(["Waveform", "Spectrogram"])

    y = analysis["y"]
    sr = analysis["sr"]

    with tab_wave:
        st.plotly_chart(
            make_waveform_chart(y, sr),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Amplitude over time — look for flat lines (silence) or square-wave tops (clipping)."
        )

    with tab_spec:
        st.plotly_chart(
            make_spectrogram_chart(y, sr),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Mel spectrogram (dBFS). Speech energy is typically 300 Hz – 3 kHz. "
            "Broadband noise shows as a uniform haze; dropouts appear as dark vertical bands."
        )

    # ── Section 3: Issue table ─────────────────────────────────────────────────
    st.markdown("#### Detected Issues")
    if not issues:
        st.success("No timestamped issues found.")
    else:
        # Severity colour highlight via pandas Styler
        try:
            import pandas as pd

            df = pd.DataFrame(issues)

            def _highlight_severity(row):
                if row.get("Severity") == "High":
                    return ["background-color: rgba(192,57,43,0.15)"] * len(row)
                if row.get("Severity") == "Medium":
                    return ["background-color: rgba(201,122,16,0.12)"] * len(row)
                return [""] * len(row)

            styled = df.style.apply(_highlight_severity, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(issues, use_container_width=True)

        st.caption(
            f"{len(issues)} issue(s) detected. High = likely to degrade transcription; Medium = worth noting."
        )


def render_prerecorded_tab():
    aai_key = get_assemblyai_key(st.session_state.aai_api_key)
    dg_key = get_deepgram_key(st.session_state.dg_api_key)

    provider = st.radio(
        "Provider",
        ["AssemblyAI", "Deepgram"],
        horizontal=True,
        key="prerecorded_provider",
    )

    if provider == "AssemblyAI" and not aai_key:
        st.warning("AssemblyAI API key not set. Add it in the sidebar or .env file.")
    if provider == "Deepgram" and not dg_key:
        st.warning("Deepgram API key not set. Add it in the sidebar or .env file.")

    st.subheader("Audio Source")
    source_mode = st.radio(
        "Input type",
        ["Default sample URL", "Paste a URL", "Upload a file"],
        horizontal=True,
    )

    audio_url = None
    uploaded_file = None

    if source_mode == "Default sample URL":
        st.code(DEFAULT_AUDIO_URL)
        audio_url = DEFAULT_AUDIO_URL
    elif source_mode == "Paste a URL":
        audio_url = st.text_input("Audio URL", placeholder="https://...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file", type=["mp3", "wav", "m4a", "ogg", "mp4"]
        )

    source_signature = "default"
    if source_mode == "Paste a URL":
        source_signature = f"url::{(audio_url or '').strip()}"
    elif source_mode == "Upload a file":
        if uploaded_file is None:
            source_signature = "upload::none"
        else:
            source_signature = f"upload::{uploaded_file.name}::{uploaded_file.size}"

    if st.session_state.audio_quality_signature != source_signature:
        st.session_state.audio_quality_signature = source_signature
        st.session_state.audio_quality_report = None
        st.session_state.audio_quality_probe_info = None
        st.session_state.audio_deep_analysis = None

    analyze_clicked = st.button("Analyze audio quality", key="analyze_audio_quality")
    if analyze_clicked:
        if source_mode == "Upload a file":
            if not uploaded_file:
                st.warning("Please upload a file before running quality analysis.")
            else:
                file_bytes = uploaded_file.getvalue()
                report = analyze_uploaded_audio(
                    file_name=uploaded_file.name,
                    file_type=uploaded_file.type,
                    file_bytes=file_bytes,
                )
                st.session_state.audio_quality_report = report
                st.session_state.audio_quality_probe_info = None

                # Deep waveform / signal analysis (requires librosa + soundfile)
                try:
                    with st.spinner("Running deep audio analysis…"):
                        st.session_state.audio_deep_analysis = run_deep_analysis(
                            file_bytes=file_bytes,
                            file_name=uploaded_file.name,
                        )
                except ImportError:
                    st.session_state.audio_deep_analysis = None
                    st.info(
                        "Install `librosa` and `soundfile` to enable waveform, "
                        "spectrogram, and deep signal analysis."
                    )
                except Exception as exc:
                    st.session_state.audio_deep_analysis = None
                    st.warning(f"Deep analysis failed: {exc}")
        else:
            target_url = (
                DEFAULT_AUDIO_URL
                if source_mode == "Default sample URL"
                else (audio_url or "").strip()
            )
            if not target_url:
                st.warning("Please enter a URL before running quality analysis.")
            else:
                probe_body, probe_status, probe_ms = probe_audio_url(target_url)
                report = analyze_url_metadata(
                    url=target_url, probe={**probe_body, "status_code": probe_status}
                )
                st.session_state.audio_quality_report = report
                st.session_state.audio_quality_probe_info = {
                    "status": probe_status,
                    "latency_ms": probe_ms,
                    "method": probe_body.get("method"),
                }

    if st.session_state.audio_quality_report:
        probe_info = st.session_state.audio_quality_probe_info
        if probe_info:
            st.caption(
                f"URL probe — HTTP {probe_info['status']} · {probe_info['latency_ms']} ms · {probe_info['method']}"
            )
        if st.session_state.audio_deep_analysis:
            # Deep analysis replaces the basic report for uploaded files
            render_deep_audio_analysis(st.session_state.audio_deep_analysis)
        else:
            render_audio_quality_report(st.session_state.audio_quality_report)
        st.divider()

    st.subheader("Model & Language")
    col1, col2 = st.columns(2)
    with col1:
        if provider == "AssemblyAI":
            model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
            model = MODEL_OPTIONS[model_label]
        else:
            dg_model_label = st.selectbox("Deepgram Model", list(DEEPGRAM_MODEL_OPTIONS.keys()), key="dg_model_label_prerecorded")
            model_label = dg_model_label
            model = DEEPGRAM_MODEL_OPTIONS[dg_model_label]
    with col2:
        if provider == "Deepgram":
            dg_model = DEEPGRAM_MODEL_OPTIONS.get(st.session_state.get("dg_model_label_prerecorded", "Nova-3 · General (Latest)"), "nova-3")
            base_model = dg_model.split("-")[0] if "-" in dg_model else dg_model  # "nova" from "nova-3-medical"
            # enhanced/base are English only; whisper and nova-2/3 support multiple languages
            is_english_only = any(dg_model.startswith(m) for m in DEEPGRAM_ENGLISH_ONLY_MODELS)
            if is_english_only:
                lang_opts = {k: v for k, v in DEEPGRAM_LANGUAGE_OPTIONS.items() if v is None or v.startswith("en")}
            else:
                lang_opts = DEEPGRAM_LANGUAGE_OPTIONS
            lang_label = st.selectbox("Language", list(lang_opts.keys()), key="dg_lang")
            language_code = lang_opts[lang_label]
        else:
            lang_label = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), key="aai_lang")
            language_code = LANGUAGE_OPTIONS[lang_label]

    st.subheader("Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        speaker_labels = st.checkbox("Speaker Labels")
        sentiment_analysis = st.checkbox("Sentiment Analysis")
        entity_detection = st.checkbox("Entity Detection")
    with c2:
        auto_highlights = st.checkbox("Key Phrases")
        iab_categories = st.checkbox("Topic Detection")
        filter_profanity = st.checkbox("Filter Profanity")
    with c3:
        punctuate = st.checkbox("Punctuation", value=True)
        format_text = st.checkbox("Format Text", value=True)
        disfluencies = st.checkbox("Include Filler Words")

    with st.expander("Advanced"):
        speakers_expected = st.number_input(
            "Expected speakers (0 = auto)", min_value=0, max_value=20, value=0
        )
        keyterms_input = st.text_area(
            "Keyterms (comma-separated)", placeholder="opal, Oprah Winfrey, ..."
        )
        prompt_input = st.text_area(
            "Context prompt (up to 1500 words)",
            placeholder="This is an interview about...",
        )

    if not st.button("Transcribe", type="primary"):
        return

    audio_bytes = None
    audio_content_type = "audio/wav"
    resolved_url = None
    audio_source_label = ""

    if source_mode == "Upload a file":
        if not uploaded_file:
            st.warning("Please upload a file.")
            return
        audio_bytes = uploaded_file.getvalue()
        audio_content_type = uploaded_file.type or "audio/wav"
        audio_source_label = uploaded_file.name
        if provider == "AssemblyAI":
            with st.spinner("Uploading file..."):
                resolved_url = upload_file(audio_bytes, aai_key)
    elif source_mode == "Paste a URL":
        if not audio_url:
            st.warning("Please enter a URL.")
            return
        resolved_url = audio_url
        audio_source_label = audio_url
    else:
        resolved_url = DEFAULT_AUDIO_URL
        audio_source_label = "default sample"

    params_snapshot = build_params_snapshot(
        speaker_labels=speaker_labels,
        sentiment_analysis=sentiment_analysis,
        entity_detection=entity_detection,
        auto_highlights=auto_highlights,
        iab_categories=iab_categories,
    )

    # ── AssemblyAI path ───────────────────────────────────────────────────────
    if provider == "AssemblyAI":
        payload = build_transcript_payload(
            audio_url=resolved_url,
            model=model,
            language_code=language_code,
            punctuate=punctuate,
            format_text=format_text,
            speaker_labels=speaker_labels,
            speakers_expected=speakers_expected,
            sentiment_analysis=sentiment_analysis,
            entity_detection=entity_detection,
            auto_highlights=auto_highlights,
            iab_categories=iab_categories,
            filter_profanity=filter_profanity,
            disfluencies=disfluencies,
            keyterms_input=keyterms_input,
            prompt_input=prompt_input,
        )

        with st.expander("Request payload (JSON)", expanded=False):
            st.json(payload)

        with st.spinner("Submitting..."):
            submit_json, submit_status, submit_ms = submit_transcript_debug(payload, aai_key)

        st.caption(f"Submit — HTTP {submit_status} · {submit_ms} ms")

        if submit_status >= 400:
            st.error(f"Submission failed (HTTP {submit_status}):")
            st.json(submit_json)
            return

        transcript_id = submit_json["id"]
        st.info(f"Transcript ID: `{transcript_id}` — polling for results...")

        with st.spinner("Transcribing..."):
            result, poll_status, poll_ms = poll_transcript_debug(transcript_id, aai_key)

        st.caption(f"Final poll — HTTP {poll_status} · {poll_ms} ms")

        with st.expander("Raw JSON response", expanded=False):
            st.json(result)

        text = result.get("text") or ""
        snippet = text[:120] + ("..." if len(text) > 120 else "")

        st.session_state.history.append(
            {
                "id": transcript_id,
                "audio_source": audio_source_label,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "model": model_label,
                "provider": "AssemblyAI",
                "snippet": snippet,
                "result": result,
                "params": params_snapshot,
            }
        )
        render_results(result, params_snapshot)

    # ── Deepgram path ─────────────────────────────────────────────────────────
    else:
        dg_model_label = st.session_state.get("dg_model_label_prerecorded", list(DEEPGRAM_MODEL_OPTIONS.keys())[0])
        dg_model = DEEPGRAM_MODEL_OPTIONS.get(dg_model_label, "nova-3")
        dg_opts = dg_build_options(
            model=dg_model,
            smart_format=format_text,
            punctuate=punctuate,
            diarize=speaker_labels,
            sentiment=sentiment_analysis,
            detect_entities=entity_detection,
            language=language_code,
        )

        with st.expander("Request options (JSON)", expanded=False):
            st.json(dg_opts)

        with st.spinner("Transcribing with Deepgram..."):
            if audio_bytes and source_mode == "Upload a file":
                dg_result, dg_status, dg_ms = dg_transcribe_file(
                    audio_bytes, audio_content_type, dg_opts, dg_key
                )
            else:
                dg_result, dg_status, dg_ms = dg_transcribe_url(resolved_url, dg_opts, dg_key)

        st.caption(f"Deepgram — HTTP {dg_status} · {dg_ms} ms")

        if dg_status >= 400:
            st.error(f"Deepgram transcription failed (HTTP {dg_status}):")
            st.json(dg_result)
            return

        with st.expander("Raw JSON response", expanded=False):
            st.json(dg_result)

        text = dg_extract_transcript(dg_result)
        confidence = extract_confidence(dg_result)
        snippet = text[:120] + ("..." if len(text) > 120 else "")

        st.session_state.history.append(
            {
                "id": None,
                "audio_source": audio_source_label,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "model": dg_model,
                "provider": "Deepgram",
                "snippet": snippet,
                "result": {"text": text, "status": "completed"},
                "params": params_snapshot,
            }
        )

        st.subheader("Transcript")
        st.write(text or "(empty)")
        if confidence is not None:
            st.caption(f"Confidence: {round(confidence * 100, 1)}%")


def render_streaming_tab():
    drain_stream_events(st.session_state)

    aai_key = get_assemblyai_key(st.session_state.aai_api_key)
    dg_key = get_deepgram_key(st.session_state.dg_api_key)

    st.subheader("Live Streaming Transcription")

    stream_provider = st.radio(
        "Provider",
        ["AssemblyAI", "Deepgram"],
        horizontal=True,
        key="streaming_provider",
    )

    if stream_provider == "AssemblyAI":
        sdk_available = streaming_sdk_import() is not None
        if not sdk_available:
            st.warning(
                "Streaming SDK components not available. "
                "Install PyAudio with: `pip install pyaudio`"
            )
            return
        stream_model_label = st.selectbox(
            "Streaming model", list(STREAMING_MODEL_OPTIONS.keys())
        )
        stream_model = STREAMING_MODEL_OPTIONS[stream_model_label]
        active_key = aai_key
        if not aai_key:
            st.warning("AssemblyAI API key not set.")
    else:
        stream_model_label = st.selectbox(
            "Deepgram model", list(DEEPGRAM_STREAMING_MODEL_OPTIONS.keys())
        )
        stream_model = DEEPGRAM_STREAMING_MODEL_OPTIONS[stream_model_label]
        active_key = dg_key
        if not dg_key:
            st.warning("Deepgram API key not set.")

    # ------------------------------------------------------------------ #
    # Mode detection: prefer browser mode when no local devices found,    #
    # but allow the user to override manually.                            #
    # ------------------------------------------------------------------ #
    if st.session_state.input_devices_cache is None:
        st.session_state.input_devices_cache = list_input_devices()
    input_devices = st.session_state.input_devices_cache
    has_local_devices = bool(input_devices)

    if has_local_devices and _WEBRTC_AVAILABLE:
        # Both modes available — let the user choose.
        mode_options = [
            "Browser microphone (works on deployment)",
            "Local microphone (PyAudio)",
        ]
        default_mode_idx = (
            0
            if st.session_state.stream_mode == "browser"
            else (1 if st.session_state.stream_mode == "local" else 0)
        )
        mode_label = st.radio(
            "Audio source", mode_options, index=default_mode_idx, horizontal=True
        )
        chosen_mode = "browser" if mode_label == mode_options[0] else "local"
    elif _WEBRTC_AVAILABLE:
        # No local mic detected — browser mode only.
        chosen_mode = "browser"
        st.caption("No local microphone devices detected; using browser audio capture.")
    else:
        # streamlit-webrtc not installed — fall back to PyAudio only.
        chosen_mode = "local"

    st.session_state.stream_mode = chosen_mode

    # ------------------------------------------------------------------ #
    # Local (PyAudio) mode                                                #
    # ------------------------------------------------------------------ #
    if chosen_mode == "local":
        st.caption(
            "Audio is captured by the local Python process (PyAudio), not by the browser tab."
        )

        selected_device_index = st.session_state.stream_device_index
        if input_devices:
            labels = [format_input_device_label(device) for device in input_devices]
            default_pos = 0
            for idx, device in enumerate(input_devices):
                if (
                    selected_device_index is not None
                    and device["index"] == selected_device_index
                ):
                    default_pos = idx
                    break
                if selected_device_index is None and device["is_default"]:
                    default_pos = idx
            selected_label = st.selectbox("Input device", labels, index=default_pos)
            selected_device_index = input_devices[labels.index(selected_label)]["index"]
            st.session_state.stream_device_index = selected_device_index
        else:
            st.warning("No input microphone devices were detected by PyAudio.")

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button(
                "Start",
                type="primary",
                disabled=st.session_state.streaming or not input_devices,
            ):
                st.session_state.streaming = True
                st.session_state.live_transcript = ""
                st.session_state.stream_session_id = None
                st.session_state.stream_word_count = None
                st.session_state.stream_audio_duration = None
                st.session_state.stream_error = None
                if stream_provider == "AssemblyAI":
                    start_streaming_thread(
                        st.session_state, stream_model, active_key, selected_device_index
                    )
                else:
                    start_deepgram_streaming_thread(st.session_state, stream_model, active_key)
                st.rerun()

        with col_stop:
            if st.button("Stop", disabled=not st.session_state.streaming):
                stop_streaming(st.session_state)
                st.rerun()

    # ------------------------------------------------------------------ #
    # Browser (streamlit-webrtc) mode                                     #
    # ------------------------------------------------------------------ #
    else:
        st.caption(
            f"Audio is captured from your **browser microphone** and streamed to {stream_provider}."
        )

        if not _WEBRTC_AVAILABLE:
            st.error(
                "`streamlit-webrtc` is not installed. "
                "Add it to your requirements: `pip install streamlit-webrtc av`"
            )
            return

        # _queue_holder is a persistent mutable dict stored in session state.
        # The AudioForwarder.recv() callback runs in streamlit-webrtc's thread
        # (not the Streamlit script thread), so it can't access st.session_state
        # directly. Mutating queue_holder["q"] in the main thread is visible to
        # the callback thread because both hold a reference to the same dict object.
        queue_holder = st.session_state._queue_holder

        class _AudioForwarder(AudioProcessorBase):
            def recv(self, frame):
                q = queue_holder["q"]
                if q is not None:
                    try:
                        q.push(convert_audio_frame(frame))
                    except Exception:
                        pass
                return frame

        # Always render webrtc_streamer so the peer connection is established while
        # the page is stable (no 0.5s polling loop). The Start button below is
        # disabled until the mic is actually streaming.
        webrtc_ctx = webrtc_streamer(
            key="browser_mic",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=_AudioForwarder,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
        )

        try:
            mic_active = webrtc_ctx is not None and webrtc_ctx.state.playing
        except Exception:
            mic_active = False

        if not mic_active:
            st.caption("Click **START** above to allow microphone access, then press **Start Transcription**.")

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button(
                "Start Transcription",
                type="primary",
                disabled=st.session_state.streaming or not mic_active,
                key="start_browser",
            ):
                st.session_state.streaming = True
                st.session_state.live_transcript = ""
                st.session_state.stream_session_id = None
                st.session_state.stream_word_count = None
                st.session_state.stream_audio_duration = None
                st.session_state.stream_error = None
                if stream_provider == "AssemblyAI":
                    aq = start_streaming_thread_browser(st.session_state, stream_model, active_key)
                else:
                    aq = start_deepgram_streaming_thread(st.session_state, stream_model, active_key)
                queue_holder["q"] = aq
                st.rerun()

        with col_stop:
            if st.button(
                "Stop", disabled=not st.session_state.streaming, key="stop_browser"
            ):
                queue_holder["q"] = None
                stop_streaming(st.session_state)
                st.rerun()

    # ------------------------------------------------------------------ #
    # Common status / output section (shared by both modes)               #
    # ------------------------------------------------------------------ #
    proc = st.session_state._stream_thread
    if proc is not None:
        pid = st.session_state._stream_proc_pid
        exitcode = st.session_state._stream_proc_exitcode
        alive = hasattr(proc, "is_alive") and proc.is_alive()
        if alive:
            label = (
                f"PID {pid or '...'}"
                if hasattr(proc, "pid")
                else f"TID {proc.ident or '...'}"
            )
            st.caption(f"Process: alive · {label}")
        elif exitcode is not None:
            if exitcode == 0:
                st.caption(f"Process: exited cleanly (PID {pid}, exit 0)")
            else:
                st.caption(f"Process: crashed (PID {pid}, exit {exitcode})")

    if st.session_state.stream_session_id:
        st.caption(f"Session ID: `{st.session_state.stream_session_id}`")

    if st.session_state.stream_error:
        stream_error = st.session_state.stream_error
        if "pyaudio" in stream_error.lower() or "portaudio" in stream_error.lower():
            st.error(
                f"PyAudio error: {stream_error}\n\n"
                "Install PyAudio: `pip install pyaudio` (may need PortAudio: `brew install portaudio`)"
            )
        elif "input overflowed" in stream_error.lower() or "-9981" in stream_error:
            st.error(
                f"Microphone overflow error: {stream_error}\n\n"
                "Try selecting a different input device and close apps that are actively using the same microphone."
            )
        else:
            st.error(f"Streaming error: {stream_error}")

    st.text_area(
        "Live transcript",
        value=st.session_state.live_transcript or "(waiting for speech...)"
        if st.session_state.streaming
        else st.session_state.live_transcript,
        height=300,
        disabled=True,
    )

    # Live metrics during streaming (must be before rerun)
    if st.session_state.streaming and st.session_state.stream_start_time:
        words_so_far = (
            len(st.session_state.live_transcript.split())
            if st.session_state.live_transcript.strip()
            else 0
        )
        elapsed_min = max(
            (datetime.now() - st.session_state.stream_start_time).total_seconds() / 60,
            0.01,
        )
        wpm = round(words_so_far / elapsed_min)
        col_a, col_b = st.columns(2)
        col_a.metric("Words so far", words_so_far)
        col_b.metric("WPM (est.)", wpm)

    if st.session_state.streaming:
        # Streaming callbacks update session state asynchronously; rerun to refresh UI.
        time.sleep(0.5)
        st.rerun()

    if (
        not st.session_state.streaming
        and st.session_state.stream_audio_duration is not None
    ):
        words = (
            len(st.session_state.live_transcript.split())
            if st.session_state.live_transcript.strip()
            else 0
        )
        elapsed_s = st.session_state.stream_audio_duration
        wpm = round(words / max(elapsed_s / 60, 0.01))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Word count", words)
        col_b.metric("Audio duration (s)", round(elapsed_s, 1))
        col_c.metric("Avg WPM", wpm)

    # Event log
    if st.session_state.stream_event_log:
        with st.expander(
            f"Session event log ({len(st.session_state.stream_event_log)} entries)",
            expanded=False,
        ):
            st.text("\n".join(st.session_state.stream_event_log))


def _curl_get(path: str, params: dict | None = None) -> str:
    from .config import API_KEY, BASE_URL

    qs = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
    return (
        f'curl -X GET "{BASE_URL}{path}{qs}" \\\n  -H "Authorization: {API_KEY[:8]}..."'
    )


def _curl_delete(path: str) -> str:
    from .config import API_KEY, BASE_URL

    return (
        f'curl -X DELETE "{BASE_URL}{path}" \\\n  -H "Authorization: {API_KEY[:8]}..."'
    )


def render_debug_tab():
    st.subheader("API Debug / Inspector")

    debug_provider = st.radio(
        "Provider", ["AssemblyAI", "Deepgram"], horizontal=True, key="debug_provider"
    )

    aai_key = get_assemblyai_key(st.session_state.aai_api_key)
    dg_key = get_deepgram_key(st.session_state.dg_api_key)

    # ── Health Check ──────────────────────────────────────────────────────────
    st.markdown("### API Health Check")

    if st.button("Run Health Check", key="debug_health"):
        if debug_provider == "AssemblyAI":
            if not aai_key:
                st.warning("AssemblyAI API key not set.")
            else:
                with st.spinner("Pinging AssemblyAI..."):
                    status, ms, rate_headers = check_api_health(aai_key)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("HTTP Status", status)
                col_b.metric("Latency (ms)", ms)
                col_c.metric("Auth", "OK" if status < 400 else "FAILED")
                if status == 401:
                    st.error("Authentication failed — check your ASSEMBLYAI_API_KEY.")
                elif status < 400:
                    st.success("AssemblyAI key valid and endpoint reachable.")
                if rate_headers:
                    with st.expander("Rate limit & request headers"):
                        st.json(rate_headers)
                with st.expander("Equivalent cURL"):
                    st.code(_curl_get("/v2/transcript", {"limit": 1}), language="bash")
        else:
            if not dg_key:
                st.warning("Deepgram API key not set.")
            else:
                with st.spinner("Pinging Deepgram..."):
                    dg_status, dg_ms, dg_headers = dg_check_api_health(dg_key)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("HTTP Status", dg_status)
                col_b.metric("Latency (ms)", dg_ms)
                col_c.metric("Auth", "OK" if dg_status < 400 else "FAILED")
                if dg_status == 401:
                    st.error("Authentication failed — check your DEEPGRAM_API_KEY.")
                elif dg_status < 400:
                    st.success("Deepgram key valid and endpoint reachable.")
                with st.expander("Equivalent cURL"):
                    st.code(
                        "curl https://api.deepgram.com/v1/projects \\\n"
                        '  -H "Authorization: Token $DEEPGRAM_API_KEY"',
                        language="bash",
                    )

    st.divider()

    if debug_provider == "Deepgram":
        # Fetch & cache the first project ID (needed for request lookups).
        def _get_dg_project_id():
            if not dg_key:
                return None, "Deepgram API key not set."
            proj_body, proj_status, _ = dg_get_projects(dg_key)
            projects = proj_body.get("projects", [])
            if proj_status >= 400 or not projects:
                return None, f"Could not fetch projects (HTTP {proj_status})."
            return projects[0]["project_id"], None

        # ── Request Inspector ─────────────────────────────────────────────────
        st.markdown("### Request Inspector")
        st.caption(
            "When using the Deepgram `callback` parameter, a `request_id` is returned "
            "immediately. Paste it here to inspect the request."
        )
        lookup_id = st.text_input("Request ID", key="debug_lookup_id")
        if st.button("Fetch", key="debug_fetch"):
            if not lookup_id.strip():
                st.warning("Enter a request ID.")
            else:
                project_id, err = _get_dg_project_id()
                if err:
                    st.error(err)
                else:
                    with st.spinner("Fetching..."):
                        body, status, ms = dg_get_request(project_id, lookup_id.strip(), dg_key)
                    st.caption(f"HTTP {status} · {ms} ms")
                    if status < 400:
                        st.success("Request found.")
                        tab_raw, tab_export = st.tabs(["Raw JSON", "Export"])
                        with tab_raw:
                            st.json(body)
                        with tab_export:
                            import json as _json
                            st.download_button(
                                "Download .json",
                                data=_json.dumps(body, indent=2),
                                file_name=f"{lookup_id.strip()}.json",
                                mime="application/json",
                            )
                    else:
                        st.error(f"Lookup failed (HTTP {status})")
                        st.json(body)

        st.divider()

        # ── Recent Requests ───────────────────────────────────────────────────
        st.markdown("### Recent Requests")
        limit = st.number_input("Limit", min_value=1, max_value=100, value=10, key="debug_limit")
        if st.button("List", key="debug_list"):
            project_id, err = _get_dg_project_id()
            if err:
                st.error(err)
            else:
                with st.spinner("Fetching..."):
                    body, status, ms = dg_list_requests(project_id, dg_key, limit=int(limit))
                st.caption(f"HTTP {status} · {ms} ms")
                reqs = body.get("requests", [])
                if reqs:
                    rows = [
                        {
                            "request_id": r.get("request_id"),
                            "created": r.get("created"),
                            "path": r.get("path"),
                            "response_code": r.get("response", {}).get("code") if isinstance(r.get("response"), dict) else r.get("response"),
                        }
                        for r in reqs
                    ]
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info("No requests found.")
                with st.expander("Raw JSON"):
                    st.json(body)
        return

    st.divider()

    # ── Transcript Lookup + Deep-Dive ─────────────────────────────────────────
    st.markdown("### Transcript Inspector")
    lookup_id = st.text_input("Transcript ID", key="debug_lookup_id")

    if st.button("Fetch", key="debug_fetch"):
        if not lookup_id.strip():
            st.warning("Enter a transcript ID.")
        else:
            tid = lookup_id.strip()
            with st.spinner("Fetching..."):
                body, status, ms, resp_headers = get_transcript(tid)
            st.caption(f"HTTP {status} · {ms} ms")

            transcript_status = body.get("status")
            transcript_error = body.get("error")
            if transcript_status:
                status_colors = {
                    "completed": "green",
                    "error": "red",
                    "processing": "orange",
                    "queued": "blue",
                }
                color = status_colors.get(transcript_status, "gray")
                st.markdown(f"**Status:** :{color}[{transcript_status}]")
            if transcript_error:
                st.error(f"Transcript error: {transcript_error}")

            tab_raw, tab_sentences, tab_paragraphs, tab_export, tab_headers = st.tabs(
                ["Raw JSON", "Sentences", "Paragraphs", "Export", "Response Headers"]
            )

            with tab_raw:
                st.json(body)

            with tab_sentences:
                if status < 400 and body.get("status") == "completed":
                    with st.spinner("Fetching sentences..."):
                        sent_body, sent_status, sent_ms = get_transcript_sentences(tid)
                    st.caption(f"HTTP {sent_status} · {sent_ms} ms")
                    sentences = sent_body.get("sentences", [])
                    if sentences:
                        rows = [
                            {
                                "start_ms": s.get("start"),
                                "end_ms": s.get("end"),
                                "duration_ms": (s.get("end", 0) - s.get("start", 0)),
                                "confidence": round(s.get("confidence", 0), 3),
                                "text": s.get("text", ""),
                            }
                            for s in sentences
                        ]
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("No sentences returned.")
                elif body.get("status") != "completed":
                    st.info(
                        f"Transcript status is '{body.get('status')}' — sentences only available when completed."
                    )
                else:
                    st.error(f"Fetch failed (HTTP {status})")

            with tab_paragraphs:
                if status < 400 and body.get("status") == "completed":
                    with st.spinner("Fetching paragraphs..."):
                        para_body, para_status, para_ms = get_transcript_paragraphs(tid)
                    st.caption(f"HTTP {para_status} · {para_ms} ms")
                    paragraphs = para_body.get("paragraphs", [])
                    if paragraphs:
                        for i, p in enumerate(paragraphs, 1):
                            start_s = p.get("start", 0) / 1000
                            end_s = p.get("end", 0) / 1000
                            st.markdown(
                                f"**¶{i}** `{start_s:.1f}s – {end_s:.1f}s` · "
                                f"confidence {round(p.get('confidence', 0), 3)}"
                            )
                            st.write(p.get("text", ""))
                    else:
                        st.info("No paragraphs returned.")
                elif body.get("status") != "completed":
                    st.info(
                        f"Transcript status is '{body.get('status')}' — paragraphs only available when completed."
                    )
                else:
                    st.error(f"Fetch failed (HTTP {status})")

            with tab_export:
                if status < 400 and body.get("status") == "completed":
                    plain_text = body.get("text", "")
                    st.download_button(
                        "Download .txt",
                        data=plain_text,
                        file_name=f"{tid}.txt",
                        mime="text/plain",
                    )
                    st.text_area(
                        "Text preview", value=plain_text, height=300, disabled=True
                    )
                elif body.get("status") != "completed":
                    st.info(
                        f"Transcript status is '{body.get('status')}' — export only available when completed."
                    )
                else:
                    st.error(f"Fetch failed (HTTP {status})")

            with tab_headers:
                st.json(resp_headers)

    st.divider()

    # ── Recent Transcripts ────────────────────────────────────────────────────
    st.markdown("### Recent Transcripts")
    limit = st.number_input(
        "Limit", min_value=1, max_value=100, value=10, key="debug_limit"
    )
    if st.button("List", key="debug_list"):
        with st.spinner("Fetching..."):
            body, status, ms = list_transcripts(int(limit))
        st.caption(f"HTTP {status} · {ms} ms")
        transcripts = body.get("transcripts", [])
        if transcripts:
            rows = [
                {
                    "id": transcript.get("id"),
                    "status": transcript.get("status"),
                    "created_at": transcript.get("created"),
                    "audio_duration": transcript.get("audio_duration"),
                }
                for transcript in transcripts
            ]
            st.dataframe(rows, use_container_width=True)
        with st.expander("Raw JSON response"):
            st.json(body)
        with st.expander("Equivalent cURL"):
            st.code(_curl_get("/v2/transcript", {"limit": int(limit)}), language="bash")

    st.divider()

    # ── Delete Transcript ─────────────────────────────────────────────────────
    st.markdown("### Delete Transcript")
    delete_id = st.text_input("Transcript ID to delete", key="debug_delete_id")
    confirm_delete = st.checkbox(
        "I confirm I want to delete this transcript", key="debug_confirm_delete"
    )
    if st.button(
        "Delete", type="primary", key="debug_delete", disabled=not confirm_delete
    ):
        if not delete_id.strip():
            st.warning("Enter a transcript ID.")
        else:
            with st.spinner("Deleting..."):
                body, status, ms = delete_transcript(delete_id.strip())
            st.caption(f"HTTP {status} · {ms} ms")
            if status < 400:
                st.success("Transcript deleted.")
            else:
                st.error(f"Delete failed (HTTP {status})")
            st.json(body)
            with st.expander("Equivalent cURL"):
                st.code(
                    _curl_delete(f"/v2/transcript/{delete_id.strip()}"), language="bash"
                )


def _word_diff_html(text_a: str, text_b: str) -> tuple[str, str]:
    """Return (html_a, html_b) with differing words colour-highlighted.

    Common words render as plain text. Words only in text_a get a red background;
    words only in text_b get a blue background.
    """
    import difflib

    words_a = text_a.split()
    words_b = text_b.split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)

    html_a: list[str] = []
    html_b: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in words_a[i1:i2]:
                html_a.append(w)
                html_b.append(w)
        elif tag == "replace":
            for w in words_a[i1:i2]:
                html_a.append(f'<mark style="background:#ffcccc;padding:1px 2px">{w}</mark>')
            for w in words_b[j1:j2]:
                html_b.append(f'<mark style="background:#cce5ff;padding:1px 2px">{w}</mark>')
        elif tag == "delete":
            for w in words_a[i1:i2]:
                html_a.append(f'<mark style="background:#ffcccc;padding:1px 2px">{w}</mark>')
        elif tag == "insert":
            for w in words_b[j1:j2]:
                html_b.append(f'<mark style="background:#cce5ff;padding:1px 2px">{w}</mark>')

    return " ".join(html_a), " ".join(html_b)


def render_compare_tab():
    import concurrent.futures
    import difflib
    import threading

    aai_key = get_assemblyai_key(st.session_state.aai_api_key)
    dg_key = get_deepgram_key(st.session_state.dg_api_key)

    st.subheader("Side-by-Side Provider Comparison")
    st.caption(
        "Both providers receive identical audio and settings. "
        "API calls are dispatched simultaneously via a thread barrier, neither gets a head start."
    )

    if not aai_key:
        st.warning("AssemblyAI API key not set. Add it in the sidebar.")
    if not dg_key:
        st.warning("Deepgram API key not set. Add it in the sidebar.")

    source_mode = st.radio(
        "Audio source",
        ["Default sample URL", "Paste a URL", "Upload a file"],
        horizontal=True,
        key="compare_source_mode",
    )

    audio_url = None
    uploaded_file = None

    if source_mode == "Default sample URL":
        st.code(DEFAULT_AUDIO_URL)
        audio_url = DEFAULT_AUDIO_URL
    elif source_mode == "Paste a URL":
        audio_url = st.text_input("Audio URL", placeholder="https://...", key="compare_url")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio", type=["mp3", "wav", "m4a", "ogg", "mp4"], key="compare_upload"
        )

    # Shared settings applied identically to both providers
    with st.expander("Shared transcription settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        cmp_punctuate = sc1.checkbox("Punctuation", value=True, key="cmp_punctuate")
        cmp_smart_format = sc2.checkbox("Smart format", value=True, key="cmp_smart_format")
        cmp_diarize = sc3.checkbox("Speaker diarization", value=False, key="cmp_diarize")

    col_aai_model, col_dg_model = st.columns(2)
    with col_aai_model:
        aai_model_label = st.selectbox("AssemblyAI model", list(MODEL_OPTIONS.keys()), key="compare_aai_model")
        aai_model = MODEL_OPTIONS[aai_model_label]
    with col_dg_model:
        dg_model_label = st.selectbox("Deepgram model", list(DEEPGRAM_MODEL_OPTIONS.keys()), key="compare_dg_model")
        dg_model = DEEPGRAM_MODEL_OPTIONS[dg_model_label]

    if not st.button("Transcribe with Both", type="primary", key="compare_run"):
        return

    if not aai_key or not dg_key:
        st.error("Both API keys are required for comparison.")
        return

    audio_bytes = None
    audio_content_type = "audio/wav"
    resolved_url = None

    if source_mode == "Upload a file":
        if not uploaded_file:
            st.warning("Please upload a file.")
            return
        audio_bytes = uploaded_file.getvalue()
        audio_content_type = uploaded_file.type or "audio/wav"
    elif source_mode == "Paste a URL":
        if not audio_url:
            st.warning("Please enter a URL.")
            return
        resolved_url = audio_url
    else:
        resolved_url = DEFAULT_AUDIO_URL

    # Barrier synchronises the moment both threads call their provider's API,
    # ensuring wall-clock latency is measured from the same instant.
    barrier = threading.Barrier(2)

    def run_aai():
        # Pre-barrier: AAI requires uploading file bytes to get a hosted URL first.
        if audio_bytes:
            url = upload_file(audio_bytes, aai_key)
        else:
            url = resolved_url
        payload = build_transcript_payload(
            audio_url=url, model=aai_model, language_code=None,
            punctuate=cmp_punctuate, format_text=cmp_smart_format,
            speaker_labels=cmp_diarize, speakers_expected=0,
            sentiment_analysis=False, entity_detection=False,
            auto_highlights=False, iab_categories=False,
            filter_profanity=False, disfluencies=False,
            keyterms_input="", prompt_input="",
        )
        barrier.wait()
        t0 = time.time()
        submit_json, submit_status, _ = submit_transcript_debug(payload, aai_key)
        if submit_status >= 400:
            return None, submit_status, 0, "Submission failed"
        tid = submit_json["id"]
        result, _, _ = poll_transcript_debug(tid, aai_key)
        return result, 200, round((time.time() - t0) * 1000), None

    def run_dg():
        dg_opts = dg_build_options(
            model=dg_model,
            smart_format=cmp_smart_format,
            punctuate=cmp_punctuate,
            diarize=cmp_diarize,
        )
        barrier.wait()
        t0 = time.time()
        if audio_bytes:
            dg_result, dg_status, _ = dg_transcribe_file(audio_bytes, audio_content_type, dg_opts, dg_key)
        else:
            dg_result, dg_status, _ = dg_transcribe_url(resolved_url, dg_opts, dg_key)
        return dg_result, dg_status, round((time.time() - t0) * 1000), None

    with st.spinner("Transcribing with both providers simultaneously..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            aai_future = executor.submit(run_aai)
            dg_future = executor.submit(run_dg)
            aai_result, aai_status, aai_ms, aai_error = aai_future.result()
            dg_result, dg_status, dg_ms, dg_error = dg_future.result()

    if aai_error or aai_result is None:
        st.error(f"AssemblyAI failed: {aai_error}")
        return
    if dg_error or dg_status >= 400:
        st.error(f"Deepgram failed (HTTP {dg_status}): {dg_error}")
        return

    # ── Extract and compute metrics ───────────────────────────────────────────
    aai_text = aai_result.get("text") or ""
    dg_text = dg_extract_transcript(dg_result)

    aai_word_list = aai_text.split()
    dg_word_list = dg_text.split()

    def _vocab(t: str) -> set[str]:
        return {w.lower().strip(".,!?\"'") for w in t.split() if w.strip(".,!?\"'")}

    def _sentences(t: str) -> int:
        return len([s for s in t.split(".") if s.strip()])

    aai_vocab = _vocab(aai_text)
    dg_vocab = _vocab(dg_text)
    aai_conf = aai_result.get("confidence")
    dg_conf = extract_confidence(dg_result)

    similarity = difflib.SequenceMatcher(
        None,
        [w.lower() for w in aai_word_list],
        [w.lower() for w in dg_word_list],
        autojunk=False,
    ).ratio()

    only_aai = sorted(aai_vocab - dg_vocab)
    only_dg = sorted(dg_vocab - aai_vocab)

    # ── Similarity headline ───────────────────────────────────────────────────
    st.markdown(f"### Results  ·  Word-level similarity: **{similarity * 100:.1f}%**")
    st.caption(
        "Latency = wall-clock time from the barrier (both threads start the API call at the same instant)."
    )

    # ── Metrics table (manual columns, no pandas dependency) ─────────────────
    def _conf(c: float | None) -> str:
        return f"{round(c * 100, 1)}%" if c is not None else "N/A"

    faster = "AssemblyAI" if aai_ms < dg_ms else "Deepgram"
    rows = [
        ("Latency (ms)", f"{aai_ms:,}", f"{dg_ms:,}", f"{abs(aai_ms - dg_ms):,} ms — {faster} faster"),
        ("Word count", len(aai_word_list), len(dg_word_list), f"Δ {abs(len(aai_word_list) - len(dg_word_list))}"),
        ("Characters (no spaces)", len(aai_text.replace(" ", "")), len(dg_text.replace(" ", "")),
         f"Δ {abs(len(aai_text.replace(' ', '')) - len(dg_text.replace(' ', '')))}"),
        ("Confidence", _conf(aai_conf), _conf(dg_conf), ""),
        ("Sentences", _sentences(aai_text), _sentences(dg_text),
         f"Δ {abs(_sentences(aai_text) - _sentences(dg_text))}"),
        ("Unique vocabulary", len(aai_vocab), len(dg_vocab), f"Δ {abs(len(aai_vocab) - len(dg_vocab))}"),
        ("Exclusive words", len(only_aai), len(only_dg), "words not shared between transcripts"),
    ]

    hcols = st.columns([2, 1.5, 1.5, 2.5])
    for col, hdr in zip(hcols, ["Metric", "AssemblyAI", "Deepgram", "Delta"]):
        col.markdown(f"**{hdr}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)
    for metric, aai_val, dg_val, delta in rows:
        rc = st.columns([2, 1.5, 1.5, 2.5])
        rc[0].write(metric)
        rc[1].write(str(aai_val))
        rc[2].write(str(dg_val))
        rc[3].write(delta or "—")

    st.divider()

    # ── Diff-highlighted transcript view ──────────────────────────────────────
    st.markdown("#### Transcript comparison")
    st.caption(":red[Red highlight] = only in AssemblyAI  ·  :blue[Blue highlight] = only in Deepgram")

    aai_html, dg_html = _word_diff_html(aai_text, dg_text)
    _panel = (
        "border:1px solid #e0e0e0;border-radius:6px;padding:12px;"
        "height:280px;overflow-y:auto;font-size:0.88rem;line-height:1.75"
    )
    col_aai, col_dg = st.columns(2)
    with col_aai:
        st.markdown("**AssemblyAI**")
        st.markdown(f'<div style="{_panel}">{aai_html}</div>', unsafe_allow_html=True)
        with st.expander("Raw JSON"):
            st.json(aai_result)
    with col_dg:
        st.markdown("**Deepgram**")
        st.markdown(f'<div style="{_panel}">{dg_html}</div>', unsafe_allow_html=True)
        with st.expander("Raw JSON"):
            st.json(dg_result)

    # ── Vocabulary differences ────────────────────────────────────────────────
    if only_aai or only_dg:
        st.markdown("#### Vocabulary differences")
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown(f"**Only in AssemblyAI** — {len(only_aai)} word(s)")
            st.write(", ".join(only_aai[:60]) if only_aai else "_none_")
        with vc2:
            st.markdown(f"**Only in Deepgram** — {len(only_dg)} word(s)")
            st.write(", ".join(only_dg[:60]) if only_dg else "_none_")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("#### Visual comparison")
    import plotly.graph_objects as go

    providers = ["AssemblyAI", "Deepgram"]
    colors = ["#1f77b4", "#ff7f0e"]
    chart_layout = dict(height=260, margin=dict(l=10, r=10, t=36, b=20))

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure([go.Bar(x=providers, y=[aai_ms, dg_ms], marker_color=colors)])
        fig.update_layout(title="Latency (ms)", **chart_layout)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure([go.Bar(x=providers, y=[len(aai_word_list), len(dg_word_list)], marker_color=colors)])
        fig.update_layout(title="Word count", **chart_layout)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = go.Figure([go.Bar(
            x=providers,
            y=[round(aai_conf * 100, 1) if aai_conf else 0, round(dg_conf * 100, 1) if dg_conf else 0],
            marker_color=colors,
        )])
        fig.update_layout(title="Confidence (%)", yaxis_range=[0, 100], **chart_layout)
        st.plotly_chart(fig, use_container_width=True)


def run_app():
    st.set_page_config(page_title="Voice AI Explorer", page_icon="🎙️", layout="wide")
    init_session_state(st.session_state)
    render_sidebar()

    st.title("Voice AI Explorer")
    st.caption("Multi-provider transcription playground — AssemblyAI + Deepgram")

    aai_key = get_assemblyai_key(st.session_state.aai_api_key)
    dg_key = get_deepgram_key(st.session_state.dg_api_key)

    if not aai_key and not dg_key:
        st.warning("No API keys found. Add your AssemblyAI and/or Deepgram keys in the sidebar or .env file.")

    tab_prerecorded, tab_streaming, tab_compare, tab_debug = st.tabs(
        ["Pre-recorded", "Live Streaming", "Compare", "API Debug"]
    )
    with tab_prerecorded:
        render_prerecorded_tab()
    with tab_streaming:
        render_streaming_tab()
    with tab_compare:
        render_compare_tab()
    with tab_debug:
        render_debug_tab()
