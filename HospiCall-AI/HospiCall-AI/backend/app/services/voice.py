import os

import httpx


class Voice:
    """STT + TTS wrapper around OmniVoice-Studio's OpenAI-compatible API."""

    def __init__(self, base_url: str, tts_engine: str, stt_engine: str):
        self.base_url = base_url.rstrip("/")
        self.tts_engine = tts_engine
        self.stt_engine = stt_engine
        # Long read timeout (CPU synthesis/transcription is slow), but a
        # short connect timeout so a missing OmniVoice fails in seconds,
        # not minutes. Localhost only — bypass any system proxy.
        self._client = httpx.Client(timeout=httpx.Timeout(120, connect=3.0),
                                    trust_env=False)

    def close(self):
        self._client.close()

    def transcribe(self, audio_path: str, content_type: str | None = None) -> str:
        """Speech-to-text: audio file -> text. content_type must reflect the
        actual upload format (the browser demo sends audio/webm, not wav)."""
        url = f"{self.base_url}/v1/audio/transcriptions"
        mime = content_type or "audio/wav"
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, mime)}
            data = {"model": self.stt_engine, "response_format": "json"}
            resp = self._client.post(url, files=files, data=data)
            resp.raise_for_status()
            return (resp.json().get("text") or "").strip()

    def synthesize(self, text: str, voice: str, out_path: str) -> str:
        """Text-to-speech: text -> wav file."""
        url = f"{self.base_url}/v1/audio/speech"
        payload = {
            "model": self.tts_engine,
            "voice": voice,
            "input": text,
            "response_format": "wav",
        }
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        # An error page that arrives with HTTP 200 would otherwise be written
        # as a "wav" file and played as noise.
        if resp.headers.get("content-type", "").startswith("application/json"):
            raise RuntimeError(f"TTS service returned an error payload: {resp.text[:200]}")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
