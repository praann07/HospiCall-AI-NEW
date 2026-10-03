from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "HospiCall"
    # 127.0.0.1, not "localhost": avoids the IPv6 (::1) connect attempt that
    # doubles the timeout when the service is down.
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    omnivoice_url: str = "http://127.0.0.1:3900"
    tts_engine: str = "kittentts"
    tts_voice: str = "en-us-libritts-high"
    stt_engine: str = "whisperx"
    # Absolute so the DB location does not depend on the working directory.
    # Override with the DB_PATH env var (e.g. for tests).
    db_path: str = str(BACKEND_DIR / "data" / "hospicall.db")
    session_ttl_seconds: int = 900
    max_text_chars: int = 1000
    max_upload_bytes: int = 15 * 1024 * 1024
    # LLM calls are capped so a hung/slow Ollama can never stall a call turn.
    llm_timeout_seconds: float = 5.0
    llm_connect_timeout_seconds: float = 1.0
    # Ask Ollama to keep the model loaded between calls (no cold start per turn).
    llm_keep_alive: str = "30m"
    # Optional smaller/faster model used ONLY for intent classification
    # (e.g. "llama3.2:3b"). Empty = reuse llm_model.
    llm_classify_model: str = ""
    # /health dependency probes: short, parallel, and cached so the endpoint
    # stays fast even when Ollama/OmniVoice are down (dead ports are slow).
    health_probe_timeout_seconds: float = 0.5
    health_cache_seconds: float = 5.0


settings = Settings()
