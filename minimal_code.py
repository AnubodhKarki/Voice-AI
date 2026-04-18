"""Minimal AssemblyAI v3 streaming example using a custom PyAudio microphone iterator."""

import os

import pyaudio
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

SAMPLE_RATE = 16000
CHUNK_FRAMES = 1024


def on_turn(client: StreamingClient, event: TurnEvent) -> None:
    if event.turn_is_formatted:
        print(event.transcript)


def mic_stream(pa: pyaudio.PyAudio):
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )
    try:
        while True:
            yield stream.read(CHUNK_FRAMES, exception_on_overflow=False)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()


def main() -> None:
    pa = pyaudio.PyAudio()
    client = StreamingClient(
        StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"])
    )
    client.on(StreamingEvents.Turn, on_turn)
    client.connect(StreamingParameters(sample_rate=SAMPLE_RATE, format_turns=True))
    try:
        client.stream(mic_stream(pa))
    finally:
        client.disconnect(terminate=True)
        pa.terminate()


if __name__ == "__main__":
    main()
