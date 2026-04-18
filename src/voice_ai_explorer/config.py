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
    # Nova-3
    "Nova-3 · General (Latest)": "nova-3",
    "Nova-3 · Medical": "nova-3-medical",
    # Nova-2
    "Nova-2 · General": "nova-2",
    "Nova-2 · Meeting": "nova-2-meeting",
    "Nova-2 · Phone Call": "nova-2-phonecall",
    "Nova-2 · Finance": "nova-2-finance",
    "Nova-2 · Medical": "nova-2-medical",
    "Nova-2 · Video": "nova-2-video",
    # Legacy
    "Enhanced": "enhanced",
    "Base": "base",
    # Whisper
    "Whisper · Medium": "whisper-medium",
    "Whisper · Large": "whisper-large",
}

DEEPGRAM_STREAMING_MODEL_OPTIONS = {
    "Nova-3 Streaming": "nova-3",
    "Nova-2 Streaming": "nova-2",
}

# Deepgram uses BCP-47 codes (hyphens, not underscores).
# nova-3 supports 50+ languages per https://developers.deepgram.com/docs/models-languages-overview
DEEPGRAM_LANGUAGE_OPTIONS = {
    "Auto-detect": None,
    # English variants
    "English": "en",
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "English (AU)": "en-AU",
    "English (India)": "en-IN",
    "English (NZ)": "en-NZ",
    # Major languages
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Mandarin, Simplified)": "zh-CN",
    "Chinese (Mandarin, Traditional)": "zh-TW",
    "Chinese (Cantonese)": "zh-HK",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Finnish": "fi",
    "French": "fr",
    "French (Canada)": "fr-CA",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malay": "ms",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Portuguese (Brazil)": "pt-BR",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Spanish (Latin America)": "es-419",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

# Models that only support English — used to filter the language selector.
# nova-2 has decent multilingual support; enhanced/base are English-only.
DEEPGRAM_ENGLISH_ONLY_MODELS = {"enhanced", "base"}
