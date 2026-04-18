---
title: Voice AI Explorer
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Voice AI Explorer

A multi-provider Voice AI orchestration playground built with Streamlit.  
Supports **AssemblyAI** and **Deepgram** side-by-side: transcription, live streaming, audio quality analysis, and provider comparison.

## Features

| Feature | AssemblyAI | Deepgram |
|---------|-----------|---------|
| Pre-recorded transcription | ✅ | ✅ |
| Speaker diarization | ✅ | ✅ |
| Sentiment analysis | ✅ | ✅ |
| Entity detection | ✅ | ✅ |
| Live streaming (WebSocket) | ✅ | ✅ |
| Audio quality analysis | ✅ | ✅ |
| Side-by-side comparison | ✅ | ✅ |

## Tabs

- **Pre-recorded**: Upload audio or provide a URL; choose AssemblyAI, Deepgram, or both
- **Live Streaming**: Real-time transcription via browser mic (WebRTC) or local microphone
- **Compare**: Transcribe the same audio with both providers; compare transcript, confidence, latency, and word count
- **Debug / API Inspector**: Health checks, raw JSON, curl command generator, transcript management

## Quick Start

### 1. Get API keys

- AssemblyAI: <https://www.assemblyai.com> → Dashboard → API key
- Deepgram: <https://console.deepgram.com> → Create API key

### 2. Install & run

```bash
git clone https://github.com/anubodhkarki/voice-ai-explorer
cd voice-ai-explorer
poetry install
cp .env.example .env
# Edit .env and add your keys
poetry run explorer
```

Or with pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Optional: live microphone

```bash
# macOS
brew install portaudio
poetry install --extras live
```

## Environment Variables

```
ASSEMBLYAI_API_KEY=your_assemblyai_key
DEEPGRAM_API_KEY=your_deepgram_key
```

Keys can also be entered in the app sidebar — useful for the live Hugging Face demo without environment setup.

## Architecture

```
src/voice_ai_explorer/
├── config.py            # Key loading (env vars + UI override)
├── api.py               # AssemblyAI REST client
├── providers/
│   └── deepgram_api.py  # Deepgram REST client
├── streaming.py         # AssemblyAI + Deepgram WebSocket streaming
├── audio_quality.py     # Waveform, spectrogram, clipping/silence detection
├── rendering.py         # Transcript rendering (diarization, sentiment, entities)
├── payloads.py          # Request payload builders (pure functions)
├── state.py             # Streamlit session state defaults
└── ui.py                # Main Streamlit app (tabs, sidebar, compare)
```

## Running Tests

```bash
poetry run pytest -q
```

## Deployment

Auto-deploys to [Hugging Face Spaces](https://huggingface.co/spaces/anubodhkarki/voice-ai-explorer) on every push to `main` via GitHub Actions.

Set these secrets in your HF Space for a hosted demo:
- `ASSEMBLYAI_API_KEY`
- `DEEPGRAM_API_KEY`
