import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.assemblyai.com"
DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1"

DEFAULT_AUDIO_URL = (
    "https://storage.googleapis.com/aai-docs-samples/sports_injuries.mp3"
)

# Legacy module-level constant kept for existing api.py callers.
API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")


def get_assemblyai_key(ui_key: str = "") -> str:
    return ui_key or os.getenv("ASSEMBLYAI_API_KEY", "")


def get_deepgram_key(ui_key: str = "") -> str:
    return ui_key or os.getenv("DEEPGRAM_API_KEY", "")


def auth_headers(api_key: str = "") -> dict:
    key = api_key or API_KEY
    return {"authorization": key}


LANGUAGE_OPTIONS = {
    "English (US)": "en_us",
    "English (UK)": "en_uk",
    "English (AU)": "en_au",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Hindi": "hi",
    "Auto-detect": None,
}

MODEL_OPTIONS = {
    "Best (Universal-3 Pro)": "universal-3-pro",
    "Nano (Universal-2)": "universal-2",
}

STREAMING_MODEL_OPTIONS = {
    "Universal-3 Pro Streaming (u3-rt-pro)": "u3-rt-pro",
    "Universal Streaming Multilingual": "universal-streaming-multilingual",
    "Universal Streaming English": "universal-streaming-english",
    "Whisper Streaming (99+ langs)": "whisper-rt",
}

DEEPGRAM_MODEL_OPTIONS = {
    "Nova-3 (Latest)": "nova-3",
    "Nova-2": "nova-2",
    "Enhanced": "enhanced",
    "Base": "base",
}

DEEPGRAM_STREAMING_MODEL_OPTIONS = {
    "Nova-3 Streaming": "nova-3",
    "Nova-2 Streaming": "nova-2",
}
