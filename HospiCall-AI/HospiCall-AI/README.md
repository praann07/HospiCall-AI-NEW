<div align="center">

# 🏥 HospiCall

### A self-hosted AI voice receptionist for hospitals — running 100% on your own machine

Call in → speak to the AI → get a doctor's appointment booked. No cloud voice APIs,
no per-minute fees, no patient data leaving the laptop.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-62%20passing-2FD4B5)](backend/tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ What it does

```
📞 Caller connects (browser demo today, FreeSWITCH planned)
   │
   ▼
🎙️ OmniVoice (STT)      "I need a cardiology appointment tomorrow"
   │
   ▼
🧠 HospiCall brain      fast-path keyword+DB engine answers in ~80 ms
   │                     mid-tier matcher catches paraphrases
   │                     Llama 3.1 (optional) writes the long tail
   ▼
📅 Booking engine       real slots from SQLite, multi-doctor availability,
   │                     conflict-safe writes, "9 AM / the second one / ten o'clock"
   ▼
🔊 OmniVoice (TTS)      sentence-streamed reply starts playing while
                         the rest is still being synthesized
```

**It runs with nothing installed.** No Ollama and no OmniVoice on the machine? The demo
detects it in seconds, tells you what's missing, and keeps working in text mode with a
built-in offline brain that still books real appointments. Start the services later and
voice + LLM light up without touching anything.

| Turn | Brain path | Latency |
|---|---|---|
| "Book a cardiologist tomorrow" | Fast-path → real DB availability | **~80 ms** |
| "I need to schedule something" | Mid-tier matcher (no LLM) | **~80 ms** |
| "Emergency!" | Safety script — checked before everything | **~80 ms** |
| Anything unusual | Llama 3.1 8B, **hard-capped at 5 s**; on timeout/absence → composed offline reply | ≤ 5 s |
| LLM/OmniVoice down | Never hangs, never 500s — one capped attempt, then graceful mode | ~1 s |

## 📦 Tech stack

| Layer | Tool |
|---|---|
| Speech-to-Text | [OmniVoice Studio](https://omnivoice.dev) — Parakeet TDT v3 (fast CPU) or Whisper large-v3 |
| Text-to-Speech | OmniVoice Studio — KittenTTS, sentence-streamed |
| Brain | Keyword fast-path + mid-tier matcher + offline composer, [Ollama](https://ollama.com) Llama 3.1 8B when available |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + SQLite (WAL, unique slot constraint) |
| Demo UI | Single-file browser call screen — hold-to-talk mic, live call timer, transcript with booking chips |
| Telephony (planned) | FreeSWITCH for real phone lines |

## 🛠️ Quick start

### 1. Run the backend

```bash
cd backend
pip install -r requirements.txt
python scripts/seed_db.py                     # demo doctors + patient
python -m uvicorn app.main:app --port 8000
```

Open **http://127.0.0.1:8000** — you're on a call. Type or hold the mic.
No other setup needed for text mode.

### 2. (Optional) Add the LLM brain

```bash
# install Ollama, then:
ollama pull llama3.1:8b-instruct-q4_K_M
# faster intent classification (recommended):
ollama pull llama3.2:3b
# then run the server with:
set LLM_CLASSIFY_MODEL=llama3.2:3b        # (use export on macOS/Linux)
```

### 3. (Optional) Add real voice

Install [OmniVoice Studio](https://omnivoice.dev) (free, local) with the **KittenTTS**
and **Parakeet TDT v3** engines, press **Start Server**, and set the active ASR backend
to *Sherpa-ONNX dictation*. The demo page detects it automatically and enables the mic
and spoken replies.

### 4. Make a test call over the API

```bash
curl -X POST http://127.0.0.1:8000/calls/start \
     -H "Content-Type: application/json" \
     -d '{"call_id":"demo-1","phone":"9441321662"}'

curl -X POST http://127.0.0.1:8000/calls/message \
     -H "Content-Type: application/json" \
     -d '{"call_id":"demo-1","text":"I want a cardiology appointment tomorrow"}'
# → {"intent":"book","response":"Sure. Our cardiology department has openings
#     tomorrow at 9 AM, 11 AM, 2 PM. Which one works best for you?"}

curl -X POST http://127.0.0.1:8000/calls/message \
     -H "Content-Type: application/json" \
     -d '{"call_id":"demo-1","text":"11 AM works for me"}'
# → {"intent":"booked","response":"Booked: Dr. Mehta, cardiology tomorrow at 11 AM. Anything else?"}
```

## 🧠 How the brain stays fast — and never stalls

Three layers, cheapest first:

1. **Fast-path (~80 ms)** — keyword rules grounded in live database availability:
   booking offers, greetings, doctor list, hospital info, emergency triage.
   Emergencies are checked before anything else, every turn.
2. **Mid-tier matcher (~0 ms)** — broader synonym sets label paraphrases
   ("I need to *schedule* something") so most turns never reach an LLM.
3. **LLM fallback (≤ 5 s, hard cap)** — Ollama classifies intent and writes the
   reply for genuinely unusual turns, with `keep_alive` so the model stays warm.
   If Ollama is absent or slow, an **offline composer** answers from templates
   and the same DB — including a full "which department? → cardiology → pick a
   time → booked" flow. Scripted intents (cancel, reschedule, medical advice,
   emergency) never pay for an LLM round-trip at all.

Booking is transactional: availability is the union across every doctor in a
department, a slot is re-checked right before insert, and a database unique
constraint makes double-booking impossible even under concurrent requests.

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status + per-dependency checks (parallel, cached — always fast) |
| `POST` | `/calls/start` | Start a call, returns greeting (includes recording consent) |
| `POST` | `/calls/message` | Send patient text, get `intent` + `response` |
| `POST` | `/calls/audio?call_id=…` | Upload a mic recording (multipart): STT + brain in one turn |
| `POST` | `/calls/transcribe` | STT an audio file (multipart), returns text only |
| `POST` | `/synthesize` | Text → spoken WAV (OmniVoice) |
| `GET` | `/calls/{call_id}/transcript` | Full ordered conversation |
| `GET` | `/calls/{call_id}/pending` | Pending booking state (debug) |
| `POST` | `/calls/{call_id}/end` | End the call |

### Configuration (env vars)

| Variable | Default | What it does |
|---|---|---|
| `DB_PATH` | `backend/data/hospicall.db` | SQLite location (auto-created) |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LLM_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Model for generated replies |
| `LLM_CLASSIFY_MODEL` | *(same as LLM_MODEL)* | Smaller/faster model for intent labels |
| `LLM_TIMEOUT_SECONDS` | `5` | Hard cap per LLM call |
| `OMNIVOICE_URL` | `http://127.0.0.1:3900` | OmniVoice Studio server |
| `STT_ENGINE` / `TTS_ENGINE` | `whisperx` / `kittentts` | Engines requested from OmniVoice |

Full list with defaults in [`backend/app/config.py`](backend/app/config.py).

## ✅ Tests

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q        # 62 tests — no Ollama or OmniVoice needed
```

CI runs the same suite on every push ([workflow](.github/workflows/ci.yml)).

## 🗂️ Project structure

```
HospiCall-AI/
├── backend/
│   ├── app/
│   │   ├── main.py               # routes, booking engine, offline composer
│   │   ├── config.py             # env-driven settings
│   │   ├── db.py                 # SQLite models (WAL, slot uniqueness)
│   │   ├── services/
│   │   │   ├── brain.py          # fast-path, mid-tier, Ollama wrapper
│   │   │   ├── voice.py          # OmniVoice STT/TTS client
│   │   │   └── session.py        # per-call state
│   │   └── static/index.html     # browser call-screen demo
│   ├── scripts/seed_db.py        # demo doctors/patient
│   └── tests/                    # 62 pytest cases
├── .github/workflows/ci.yml      # CI: install + test
├── docs/                         # design notes
└── README.md
```

## 🗺️ Roadmap

- [x] STT → intent → response → TTS pipeline end-to-end
- [x] DB-backed, conflict-safe appointment booking (multi-doctor)
- [x] Browser demo: mic → STT → brain → streamed TTS, live call screen
- [x] Sub-100 ms fast-path + mid-tier matcher
- [x] Graceful offline mode — fully working demo with zero AI services
- [x] Test suite + CI
- [ ] Cancel / reschedule against the database
- [ ] Hindi / Telugu keyword sets (OmniVoice already supports the audio side)
- [ ] FreeSWITCH phone-line integration
- [ ] PostgreSQL migration for production

## ⚠️ Privacy & ethics

> The demo uses **synthetic data only** — no real patient information. For any real
> deployment, comply with HIPAA/GDPR and your local healthcare regulations first.

## 📄 License

[MIT](LICENSE) © 2026
