def init_session_state(session_state):
    if "history" not in session_state:
        session_state.history = []

    for key, default in [
        ("aai_api_key", ""),
        ("dg_api_key", ""),
        ("streaming", False),
        ("input_devices_cache", None),
        ("live_transcript", ""),
        ("stream_session_id", None),
        ("stream_word_count", None),
        ("stream_audio_duration", None),
        ("stream_error", None),
        ("stream_device_index", None),
        ("_stream_thread", None),
        ("_stream_client", None),
        ("_stream_microphone", None),
        ("_stream_events", None),
        ("_stream_stop_event", None),
        ("_stream_audio_queue", None),
        ("stream_mode", None),
        ("stream_event_log", []),
        ("stream_start_time", None),
        ("_stream_proc_pid", None),
        ("_stream_proc_exitcode", None),
        ("audio_quality_report", None),
        ("audio_quality_signature", None),
        ("audio_quality_probe_info", None),
        ("audio_deep_analysis", None),
    ]:
        if key not in session_state:
            session_state[key] = default
