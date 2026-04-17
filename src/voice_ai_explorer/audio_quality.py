import io
import os
import wave

import numpy as np

# ─── Tunable thresholds ───────────────────────────────────────────────────────
SILENCE_THRESHOLD_DBFS = -50.0  # frames below this are considered silent
SILENCE_MIN_DURATION_S = 0.75  # min silence run to report (seconds)
CLIPPING_AMPLITUDE = 0.98  # amplitude at or above which samples are clipped
CLIPPING_MIN_DURATION_S = 0.02  # min clipping run to report (seconds)
LOUDNESS_WINDOW_S = 1.0  # window size for per-window RMS (seconds)
LOUDNESS_STD_POOR_DB = 12.0  # std dev above which loudness is "poor"
LOUDNESS_STD_UNEVEN_DB = 6.0  # std dev above which loudness is "uneven"
LOUDNESS_QUIET_THRESHOLD_DBFS = -35.0  # windows below this are flagged quiet
WAVEFORM_MAX_POINTS = 8000  # downsample target for waveform chart


# ─── Existing metadata-based analysis (unchanged) ─────────────────────────────


def _quality_label(score: int) -> str:
    if score >= 80:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"


def evaluate_quality(*, metrics: dict, warnings: list[str]) -> dict:
    score = 100
    for warning in warnings:
        if "short audio" in warning.lower():
            score -= 20
        elif "sample rate" in warning.lower():
            score -= 25
        elif "content-type" in warning.lower():
            score -= 30
        elif "very small" in warning.lower():
            score -= 10
        else:
            score -= 8
    score = max(score, 0)
    return {
        "score": score,
        "label": _quality_label(score),
        "warnings": warnings,
        "metrics": metrics,
    }


def analyze_uploaded_audio(
    *, file_name: str, file_type: str | None, file_bytes: bytes
) -> dict:
    size_bytes = len(file_bytes)
    extension = os.path.splitext(file_name)[1].lower().lstrip(".")
    metrics = {
        "source_type": "upload",
        "file_name": file_name,
        "format": extension or "unknown",
        "mime_type": file_type or "unknown",
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "duration_seconds": None,
        "sample_rate_hz": None,
        "channels": None,
        "bitrate_kbps": None,
    }
    warnings: list[str] = []

    if extension == "wav":
        try:
            with wave.open(io.BytesIO(file_bytes), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                duration = frame_count / frame_rate if frame_rate else 0
                bitrate_kbps = (
                    round((frame_rate * channels * sample_width * 8) / 1000, 1)
                    if frame_rate
                    else None
                )

            metrics["duration_seconds"] = round(duration, 2)
            metrics["sample_rate_hz"] = frame_rate
            metrics["channels"] = channels
            metrics["bitrate_kbps"] = bitrate_kbps

            if duration and duration < 5:
                warnings.append("Short audio (<5s) can reduce transcription context.")
            if frame_rate and frame_rate < 16000:
                warnings.append("Low sample rate (<16kHz) may reduce accuracy.")
            if channels and channels > 2:
                warnings.append(
                    "More than 2 channels detected; verify channel mapping."
                )
        except (wave.Error, EOFError):
            warnings.append(
                "WAV header could not be parsed; verify that the file is a valid WAV."
            )
    else:
        warnings.append(
            "Detailed duration/sample-rate checks are currently available for WAV uploads only."
        )
        if extension not in {"mp3", "m4a", "ogg", "mp4"}:
            warnings.append(
                "Unrecognized upload format; verify codec/container compatibility."
            )

    if size_bytes < 100 * 1024:
        warnings.append("Very small file size; audio may be too short or silent.")

    return evaluate_quality(metrics=metrics, warnings=warnings)


def analyze_url_metadata(*, url: str, probe: dict) -> dict:
    headers = probe.get("headers") or {}
    content_type = headers.get("content_type") or "unknown"
    content_length = headers.get("content_length_bytes")
    metrics = {
        "source_type": "url",
        "url": url,
        "http_status": probe.get("status_code"),
        "content_type": content_type,
        "content_length_bytes": content_length,
        "content_length_mb": round(content_length / (1024 * 1024), 2)
        if isinstance(content_length, int)
        else None,
        "accept_ranges": headers.get("accept_ranges"),
    }

    warnings: list[str] = []
    if not probe.get("reachable"):
        warnings.append("URL probe failed; verify accessibility before transcription.")
    if content_type != "unknown" and not (
        content_type.startswith("audio/") or content_type.startswith("video/")
    ):
        warnings.append("Content-Type does not look like audio/video.")
    if isinstance(content_length, int) and content_length < 100 * 1024:
        warnings.append("Very small content-length; source may be too short.")
    if content_length is None:
        warnings.append(
            "Content-Length unavailable; duration/bitrate cannot be estimated from headers."
        )

    return evaluate_quality(metrics=metrics, warnings=warnings)


# ─── Deep audio analysis (requires librosa, soundfile, numpy) ─────────────────


def _rms_to_dbfs(rms: float) -> float:
    """Convert linear RMS amplitude to dBFS."""
    return 20.0 * np.log10(max(float(rms), 1e-9))


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.ss for the issue table."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def load_audio_array(file_bytes: bytes, file_name: str) -> tuple[np.ndarray, int]:
    """
    Decode audio bytes into a mono float32 numpy array.
    Tries soundfile first (fast, WAV/FLAC/OGG), falls back to librosa (MP3, M4A, …).
    Returns (y, sample_rate).
    """
    import soundfile as sf

    buf = io.BytesIO(file_bytes)
    y, sr = sf.read(buf, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mix down to mono
    return y.astype(np.float32), int(sr)


def make_waveform_chart(y: np.ndarray, sr: int):
    """Plotly waveform of the full audio, downsampled for browser performance."""
    import plotly.graph_objects as go

    step = max(1, len(y) // WAVEFORM_MAX_POINTS)
    y_ds = y[::step]
    t = np.arange(len(y_ds)) * step / sr

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_ds,
            mode="lines",
            line=dict(color="#4A90D9", width=0.6),
            name="Amplitude",
            hovertemplate="t=%{x:.2f}s<br>amp=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=180,
        margin=dict(l=50, r=10, t=10, b=35),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#2a2a2a"),
        xaxis=dict(gridcolor="#2a2a2a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_spectrogram_chart(y: np.ndarray, sr: int):
    """
    Plotly STFT spectrogram (dBFS heatmap) built with scipy — no numba/librosa required.
    Shows noise, hum, hiss, dropouts, and speech energy distribution at a glance.
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    import plotly.graph_objects as go

    nperseg = 1024
    noverlap = 768  # 75 % overlap — good time/frequency resolution

    freqs, times, Sxx = scipy_spectrogram(
        y, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="spectrum"
    )

    # Convert power to dBFS; clip at -80 dB floor
    with np.errstate(divide="ignore"):
        S_db = 10.0 * np.log10(np.maximum(Sxx, 1e-10))
    S_db = np.clip(S_db, -80, 0)

    # Limit to 8 kHz — speech lives below that; reduces chart noise above
    freq_cap = min(8000, sr // 2)
    mask = freqs <= freq_cap
    freqs_plot = freqs[mask] / 1000.0  # → kHz
    S_plot = S_db[mask, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=S_plot,
            x=times,
            y=freqs_plot,
            colorscale="Viridis",
            colorbar=dict(title="dBFS", thickness=10, len=0.9),
            zmin=-80,
            zmax=0,
            hovertemplate="t=%{x:.2f}s<br>freq=%{y:.2f}kHz<br>%{z:.1f} dBFS<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=55, r=10, t=10, b=35),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (kHz)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def detect_silence_segments(
    y: np.ndarray,
    sr: int,
    threshold_dbfs: float = SILENCE_THRESHOLD_DBFS,
    min_duration_s: float = SILENCE_MIN_DURATION_S,
) -> dict:
    """
    Detect prolonged silence / dropout regions using frame-level RMS.
    Returns silence_pct (float) and segments list [{start, end}] in seconds.
    """
    frame_len = 2048
    hop = 512

    # Vectorised RMS per frame
    frames = np.array(
        [
            np.sqrt(np.mean(y[i : i + frame_len] ** 2))
            for i in range(0, len(y) - frame_len + 1, hop)
        ]
    )
    dbfs = np.array([_rms_to_dbfs(r) for r in frames])
    silent = dbfs < threshold_dbfs

    silence_pct = round(float(silent.mean() * 100), 1)

    # Merge contiguous silent frames into segments
    min_frames = max(1, int(min_duration_s * sr / hop))
    segments: list[dict] = []
    in_silence = False
    seg_start = 0

    for i, is_sil in enumerate(silent):
        if is_sil and not in_silence:
            in_silence = True
            seg_start = i
        elif not is_sil and in_silence:
            in_silence = False
            if i - seg_start >= min_frames:
                segments.append(
                    {
                        "start": round(seg_start * hop / sr, 2),
                        "end": round(i * hop / sr, 2),
                    }
                )

    # Handle trailing silence
    if in_silence and len(silent) - seg_start >= min_frames:
        segments.append(
            {
                "start": round(seg_start * hop / sr, 2),
                "end": round(len(y) / sr, 2),
            }
        )

    return {"silence_pct": silence_pct, "segments": segments}


def detect_clipping_segments(
    y: np.ndarray,
    sr: int,
    threshold: float = CLIPPING_AMPLITUDE,
    min_duration_s: float = CLIPPING_MIN_DURATION_S,
) -> dict:
    """
    Detect amplitude saturation (clipping/distortion).
    Returns clipping_present (bool), clipped_sample_pct (float),
    and segments list [{start, end}] in seconds.
    """
    clipped = np.abs(y) >= threshold
    min_samples = max(1, int(min_duration_s * sr))

    segments: list[dict] = []
    in_clip = False
    seg_start = 0

    for i, c in enumerate(clipped):
        if c and not in_clip:
            in_clip = True
            seg_start = i
        elif not c and in_clip:
            in_clip = False
            if i - seg_start >= min_samples:
                segments.append(
                    {"start": round(seg_start / sr, 3), "end": round(i / sr, 3)}
                )

    if in_clip and len(clipped) - seg_start >= min_samples:
        segments.append(
            {"start": round(seg_start / sr, 3), "end": round(len(y) / sr, 3)}
        )

    return {
        "clipping_present": len(segments) > 0,
        "clipped_sample_pct": round(float(clipped.mean() * 100), 2),
        "segments": segments,
    }


def analyze_loudness_consistency(
    y: np.ndarray,
    sr: int,
    window_s: float = LOUDNESS_WINDOW_S,
    std_poor_db: float = LOUDNESS_STD_POOR_DB,
    std_uneven_db: float = LOUDNESS_STD_UNEVEN_DB,
    quiet_threshold_dbfs: float = LOUDNESS_QUIET_THRESHOLD_DBFS,
) -> dict:
    """
    Measure loudness variation across non-overlapping windows.
    Returns label (consistent / uneven / poor), std_db, mean_dbfs,
    and quiet_windows list [{start, end}].
    """
    win_samples = int(window_s * sr)
    dbfs_vals: list[float] = []
    timestamps: list[float] = []

    for i in range(0, len(y) - win_samples + 1, win_samples):
        chunk = y[i : i + win_samples]
        rms = np.sqrt(np.mean(chunk**2))
        dbfs_vals.append(_rms_to_dbfs(rms))
        timestamps.append(round(i / sr, 2))

    if not dbfs_vals:
        return {
            "label": "unknown",
            "std_db": None,
            "mean_dbfs": None,
            "quiet_windows": [],
            "dbfs_values": [],
            "timestamps": [],
        }

    arr = np.array(dbfs_vals)
    std_db = round(float(arr.std()), 2)
    mean_dbfs = round(float(arr.mean()), 2)

    quiet_windows = [
        {"start": timestamps[i], "end": round(timestamps[i] + window_s, 2)}
        for i, v in enumerate(dbfs_vals)
        if v < quiet_threshold_dbfs
    ]

    if std_db >= std_poor_db:
        label = "poor"
    elif std_db >= std_uneven_db:
        label = "uneven"
    else:
        label = "consistent"

    return {
        "label": label,
        "std_db": std_db,
        "mean_dbfs": mean_dbfs,
        "quiet_windows": quiet_windows,
        "dbfs_values": [round(v, 2) for v in dbfs_vals],
        "timestamps": timestamps,
    }


def aggregate_issues(
    silence_result: dict,
    clipping_result: dict,
    loudness_result: dict,
) -> list[dict]:
    """
    Flatten all detected issues into a sorted list of timestamped rows
    for the UI issue table.
    """
    issues: list[dict] = []

    for seg in silence_result.get("segments", []):
        dur = seg["end"] - seg["start"]
        issues.append(
            {
                "Start": _fmt_time(seg["start"]),
                "End": _fmt_time(seg["end"]),
                "Issue": "Silence / Dropout",
                "Severity": "High" if dur > 5 else "Medium",
                "Note": f"{dur:.1f}s of silence",
            }
        )

    for seg in clipping_result.get("segments", []):
        dur = seg["end"] - seg["start"]
        issues.append(
            {
                "Start": _fmt_time(seg["start"]),
                "End": _fmt_time(seg["end"]),
                "Issue": "Clipping",
                "Severity": "High",
                "Note": f"Amplitude saturation over {dur:.2f}s",
            }
        )

    for win in loudness_result.get("quiet_windows", []):
        issues.append(
            {
                "Start": _fmt_time(win["start"]),
                "End": _fmt_time(win["end"]),
                "Issue": "Low Volume",
                "Severity": "Medium",
                "Note": f"Below {LOUDNESS_QUIET_THRESHOLD_DBFS} dBFS",
            }
        )

    # Sort by raw start seconds (parse MM:SS.ss back to float for sort key)
    issues.sort(key=lambda r: _parse_time(r["Start"]))
    return issues


def _parse_time(t: str) -> float:
    """Parse MM:SS.ss back to float seconds (used only for sorting)."""
    try:
        parts = t.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return 0.0


def score_readiness(
    silence_result: dict,
    clipping_result: dict,
    loudness_result: dict,
    metrics: dict,
) -> dict:
    """
    Estimate transcription readiness based on AssemblyAI-relevant factors.
    Returns label, color, score (0–100), and a list of flag strings.
    """
    deductions: list[int] = []
    flags: list[str] = []

    # Clipping
    if clipping_result.get("clipping_present"):
        pct = clipping_result.get("clipped_sample_pct", 0)
        if pct > 1.0:
            deductions.append(30)
            flags.append(f"Significant clipping ({pct}% of samples saturated)")
        else:
            deductions.append(15)
            flags.append("Minor clipping detected")

    # Silence / dropouts
    silence_pct = silence_result.get("silence_pct", 0)
    n_segments = len(silence_result.get("segments", []))
    if silence_pct > 50:
        deductions.append(30)
        flags.append(f"Audio is mostly silent ({silence_pct}%)")
    elif n_segments >= 3:
        deductions.append(15)
        flags.append(f"{n_segments} prolonged silence/dropout segments")
    elif n_segments >= 1:
        deductions.append(5)
        flags.append(f"{n_segments} silence segment(s) detected")

    # Loudness consistency
    loudness_label = loudness_result.get("label", "unknown")
    std_db = loudness_result.get("std_db")
    if loudness_label == "poor":
        deductions.append(20)
        flags.append(f"Poor loudness consistency (±{std_db} dB std dev)")
    elif loudness_label == "uneven":
        deductions.append(10)
        flags.append(f"Uneven loudness (±{std_db} dB std dev)")

    # Sample rate
    sr = metrics.get("sample_rate_hz")
    if sr and sr < 16000:
        deductions.append(15)
        flags.append(
            f"Low sample rate ({sr} Hz) — 16 kHz+ recommended for best accuracy"
        )

    # Duration
    duration = metrics.get("duration_seconds")
    if duration and duration < 5:
        deductions.append(10)
        flags.append("Very short audio (<5s) — limited transcription context")

    score = max(0, 100 - sum(deductions))

    if score >= 75:
        label = "Ready for transcription"
        color = "green"
    elif score >= 50:
        label = "Usable with minor issues"
        color = "orange"
    else:
        label = "High risk for degraded transcription"
        color = "red"

    return {"score": score, "label": label, "color": color, "flags": flags}


def run_deep_analysis(file_bytes: bytes, file_name: str) -> dict:
    """
    Entry point for full deep audio analysis of an uploaded file.
    Loads audio, runs all checks, and returns a structured result dict.
    Requires: librosa, soundfile, numpy.
    """
    y, sr = load_audio_array(file_bytes, file_name)
    duration = round(len(y) / sr, 2)

    metrics = {
        "sample_rate_hz": sr,
        "duration_seconds": duration,
        "channels": 1,  # always mono after loading
        "num_samples": len(y),
    }

    silence = detect_silence_segments(y, sr)
    clipping = detect_clipping_segments(y, sr)
    loudness = analyze_loudness_consistency(y, sr)
    issues = aggregate_issues(silence, clipping, loudness)
    readiness = score_readiness(silence, clipping, loudness, metrics)

    return {
        "y": y,
        "sr": sr,
        "metrics": metrics,
        "silence": silence,
        "clipping": clipping,
        "loudness": loudness,
        "issues": issues,
        "readiness": readiness,
    }
