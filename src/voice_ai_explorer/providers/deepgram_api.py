import time

import requests

DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1"


def _dg_headers(api_key: str) -> dict:
    return {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}


def check_api_health(api_key: str):
    """Validate Deepgram key and measure latency. Returns (status_code, elapsed_ms, headers)."""
    t0 = time.perf_counter()
    resp = requests.get(
        f"{DEEPGRAM_BASE_URL}/projects",
        headers={"Authorization": f"Token {api_key}"},
        timeout=10,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    rate_headers = {
        k: v for k, v in resp.headers.items() if "x-request-id" in k.lower()
    }
    return resp.status_code, elapsed, rate_headers


def transcribe_url(url: str, options: dict, api_key: str):
    """Submit a URL for transcription. Returns (json, status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{DEEPGRAM_BASE_URL}/listen",
        headers=_dg_headers(api_key),
        json={"url": url},
        params=options,
        timeout=120,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def transcribe_file(file_bytes: bytes, content_type: str, options: dict, api_key: str):
    """Upload raw audio bytes and transcribe. Returns (json, status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type or "audio/wav",
    }
    resp = requests.post(
        f"{DEEPGRAM_BASE_URL}/listen",
        headers=headers,
        data=file_bytes,
        params=options,
        timeout=180,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def extract_transcript(dg_response: dict) -> str:
    """Pull the transcript text out of a Deepgram response."""
    try:
        return dg_response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        return ""


def extract_confidence(dg_response: dict) -> float | None:
    """Pull the overall confidence score from a Deepgram response."""
    try:
        return dg_response["results"]["channels"][0]["alternatives"][0]["confidence"]
    except (KeyError, IndexError):
        return None


def extract_words(dg_response: dict) -> list[dict]:
    """Return word-level details from a Deepgram response."""
    try:
        return dg_response["results"]["channels"][0]["alternatives"][0].get("words", [])
    except (KeyError, IndexError):
        return []


def extract_utterances(dg_response: dict) -> list[dict]:
    """Return speaker-diarized utterances from a Deepgram response (requires diarize=true)."""
    return dg_response.get("results", {}).get("utterances", [])


def get_projects(api_key: str):
    """List all projects. Returns (json, status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = requests.get(
        f"{DEEPGRAM_BASE_URL}/projects",
        headers={"Authorization": f"Token {api_key}"},
        timeout=10,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def get_request(project_id: str, request_id: str, api_key: str):
    """Look up a single Deepgram request by project + request ID. Returns (json, status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = requests.get(
        f"{DEEPGRAM_BASE_URL}/projects/{project_id}/requests/{request_id}",
        headers={"Authorization": f"Token {api_key}"},
        timeout=10,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def list_requests(project_id: str, api_key: str, limit: int = 10):
    """List recent requests for a project. Returns (json, status_code, elapsed_ms)."""
    t0 = time.perf_counter()
    resp = requests.get(
        f"{DEEPGRAM_BASE_URL}/projects/{project_id}/requests",
        headers={"Authorization": f"Token {api_key}"},
        params={"limit": limit},
        timeout=10,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def transcribe_url_async(url: str, options: dict, callback_url: str, api_key: str):
    """Submit a URL for async transcription with callback. Returns (json, status_code, elapsed_ms).
    Deepgram responds immediately with a request_id; result is POSTed to callback_url."""
    params = {**options, "callback": callback_url}
    t0 = time.perf_counter()
    resp = requests.post(
        f"{DEEPGRAM_BASE_URL}/listen",
        headers=_dg_headers(api_key),
        json={"url": url},
        params=params,
        timeout=30,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return data, resp.status_code, elapsed


def build_options(
    *,
    model: str = "nova-3",
    smart_format: bool = True,
    punctuate: bool = True,
    diarize: bool = False,
    sentiment: bool = False,
    detect_entities: bool = False,
    language: str | None = None,
) -> dict:
    """Build Deepgram query-string options dict."""
    opts: dict = {
        "model": model,
        "smart_format": "true" if smart_format else "false",
        "punctuate": "true" if punctuate else "false",
    }
    if diarize:
        opts["diarize"] = "true"
    if sentiment:
        opts["sentiment"] = "true"
    if detect_entities:
        opts["detect_entities"] = "true"
    if language:
        opts["language"] = language
    return opts
