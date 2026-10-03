import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from starlette.background import BackgroundTask

from .config import settings
from .db import Appointment, Conversation, Doctor, Intent, TranscriptSegment, User, init_db
from .services.brain import (CANCEL_KEYWORDS, CANCEL_SCRIPT, CAPABILITIES_SCRIPT,
                             EMERGENCY_KEYWORDS, EMERGENCY_SCRIPT,
                             HOSPITAL_INFO_SCRIPT, OUT_OF_SCOPE_SCRIPT,
                             RESCHEDULE_KEYWORDS, RESCHEDULE_SCRIPT,
                             TIMING_DISPLAY, Brain, extract_slots,
                             matches_any, parse_slot)
from .services.session import SessionManager
from .services.voice import Voice

app = FastAPI(title="HospiCall")
Session = init_db(settings.db_path)
brain = Brain(settings.ollama_url, settings.llm_model,
              timeout_s=settings.llm_timeout_seconds,
              connect_s=settings.llm_connect_timeout_seconds,
              keep_alive=settings.llm_keep_alive,
              classify_model=settings.llm_classify_model or None)
voice = Voice(settings.omnivoice_url, settings.tts_engine, settings.stt_engine)
sessions = SessionManager(settings.session_ttl_seconds)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_AUDIO_DIR = os.path.join(BACKEND_DIR, "data", "tmp")
os.makedirs(TMP_AUDIO_DIR, exist_ok=True)

# call_id also becomes part of temp filenames, so it is restricted to
# filename-safe characters (no separators, no traversal).
CALL_ID_PATTERN = r"^[\w.-]{1,64}$"
AUDIO_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "application/octet-stream": ".bin",
}

GREETING = ("Welcome to City Hospital. This call may be recorded for quality purposes. "
            "How can I help you today?")


class CallEvent(BaseModel):
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    phone: str = Field(min_length=1, max_length=32)


class ChatMessage(BaseModel):
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    text: str = Field(min_length=1, max_length=2000)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _slot_label(dt: datetime) -> str:
    return dt.strftime("%I %p").lstrip("0")


def _target_day(timing: str) -> datetime.date:
    """Part-of-day requests ('morning') refer to tomorrow, matching what the
    confirmation message says ('tomorrow morning')."""
    today = datetime.now().date()
    if timing in ("tomorrow", "morning", "afternoon", "evening"):
        return today + timedelta(days=1)
    return today


def _when_brief(timing: str) -> str:
    return {"today": "today", "tomorrow": "tomorrow", "morning": "tomorrow morning",
            "afternoon": "tomorrow afternoon", "evening": "tomorrow evening"}.get(timing, "tomorrow")


def _day_window(timing: str) -> list[str]:
    if timing in ("tomorrow", "today"):
        return ["9 AM", "10 AM", "11 AM", "2 PM", "3 PM", "4 PM"]
    return {
        "morning": ["9 AM", "10 AM", "11 AM"],
        "afternoon": ["2 PM", "3 PM"],
        "evening": ["4 PM", "5 PM"],
    }.get(timing, ["9 AM", "10 AM", "11 AM", "2 PM", "3 PM", "4 PM"])


def _slot_time(label: str) -> datetime:
    hh, mer = label.split()
    hh = int(hh)
    if mer == "PM" and hh != 12:
        hh += 12
    if mer == "AM" and hh == 12:
        hh = 0
    return datetime(2000, 1, 1, hh, 0)


def _slot_to_datetime(day: datetime.date, label: str) -> datetime:
    return datetime.combine(day, _slot_time(label).time())


def _doctors_for(db, specialty: str | None):
    if not specialty:
        return []
    return db.query(Doctor).filter(Doctor.specialization.ilike(specialty)).order_by(Doctor.id).all()


def find_slots(specialty: str | None, timing: str | None) -> list[str]:
    """Available slots for a specialty on the preferred day — the union
    across ALL doctors in the department (a slot is offered if any doctor is
    free at that time). Never offers a time already in the past. No LLM."""
    if not specialty:
        return []
    timing = timing or "tomorrow"
    day = _target_day(timing)
    day_start = datetime.combine(day, datetime.min.time())
    day_end = datetime.combine(day + timedelta(days=1), datetime.min.time())
    now = datetime.now()
    with Session() as db:
        doctors = _doctors_for(db, specialty)
        if not doctors:
            return []
        doc_ids = [d.id for d in doctors]
        booked = {(a.doctor_id, _slot_label(a.slot_start)) for a in db.query(Appointment).filter(
            Appointment.doctor_id.in_(doc_ids),
            Appointment.slot_start >= day_start,
            Appointment.slot_start < day_end,
        ).all()}
        free = [s for s in _day_window(timing)
                if _slot_to_datetime(day, s) > now
                and any((doc_id, s) not in booked for doc_id in doc_ids)]
        return free[:3]


def _list_doctors() -> list[str]:
    with Session() as db:
        rows = db.query(Doctor).all()
        return sorted({f"{d.name} ({d.specialization})" for d in rows})


def _ensure_conversation(call_id: str, phone: str):
    with Session() as db:
        if not db.query(Conversation).filter_by(call_id=call_id).first():
            db.add(Conversation(call_id=call_id, phone=phone))
            db.commit()


def process_text(call_id: str, text: str) -> dict:
    """Shared brain: booking confirmation, fast-path, and LLM fallback."""
    text = text.strip()
    if not text:
        raise HTTPException(400, "Message text is empty.")
    if len(text) > settings.max_text_chars:
        text = text[:settings.max_text_chars]

    session = sessions.get(call_id)
    if session is None:
        raise HTTPException(404, f"Unknown or expired call_id '{call_id}'. POST /calls/start first.")

    sessions.add_turn(call_id, "patient", text)
    low = text.lower()
    pending = sessions.get_pending_booking(call_id)

    if matches_any(low, EMERGENCY_KEYWORDS):
        # Safety checks run first, even mid-booking.
        if pending:
            sessions.clear_pending_booking(call_id)
        intent, response = "emergency", EMERGENCY_SCRIPT
    elif pending and matches_any(low, CANCEL_KEYWORDS):
        sessions.clear_pending_booking(call_id)
        intent, response = "cancel", "Okay, I've set that booking aside. Anything else I can help with?"
    elif pending:
        intent, response = _continue_booking(call_id, text, pending)
    else:
        fast = brain.fast_handle(text, find_slots, _list_doctors)
        if fast:
            intent, response = fast
            if intent == "book":
                specialty, timing = extract_slots(text)
                if specialty:
                    timing = timing or "tomorrow"
                    sessions.set_pending_booking(call_id, {
                        "specialty": specialty, "timing": timing,
                        "offered": find_slots(specialty, timing),
                    })
        else:
            intent, response = _fallback_brain(call_id, text)

    sessions.update_slots(call_id, intent=intent)
    specialty, timing = extract_slots(text)
    slots_record = {"specialty": specialty, "timing": timing}
    with Session() as db:
        db.add(Intent(call_id=call_id, intent_type=intent, slots_json=json.dumps(slots_record)))
        db.add(TranscriptSegment(call_id=call_id, speaker="patient", text=text))
        db.add(TranscriptSegment(call_id=call_id, speaker="ai", text=response))
        db.commit()
    sessions.add_turn(call_id, "ai", response)
    return {"intent": intent, "response": response}


def _continue_booking(call_id: str, text: str, pending: dict) -> tuple[str, str]:
    """The patient is answering a slot offer: pick a time, refine the request,
    or say something that needs the LLM."""
    low = text.lower()
    chosen = parse_slot(text, offered=pending.get("offered") or [])
    specialty, timing = extract_slots(text)

    if not pending.get("specialty"):
        # Department-picker flow (opened by the offline brain): we asked
        # which department — accept one now, or re-ask.
        if specialty:
            t = timing or pending.get("timing") or "tomorrow"
            updated = {"specialty": specialty, "timing": t,
                       "offered": find_slots(specialty, t)}
            if updated["offered"]:
                return _reoffer(call_id, updated, "")
            sessions.clear_pending_booking(call_id)
            return ("book", f"I couldn't find openings for {specialty} {TIMING_DISPLAY.get(t, t)}. "
                            "Could you try another day or department?")
        return ("book", _department_question())

    if chosen:
        fresh = find_slots(pending["specialty"], pending["timing"])
        if chosen in fresh:
            response, booked = _confirm_and_book(call_id, pending, chosen)
            if booked:
                sessions.clear_pending_booking(call_id)
                return ("booked", response)
            return ("book", response)  # conflict path already re-offered + updated pending
        return _reoffer(call_id, pending, f"Sorry, {chosen} isn't available.")

    if specialty or timing:
        # Patient refined the request ("actually neurology", "make it today").
        new_specialty = specialty or pending["specialty"]
        new_timing = timing or pending["timing"]
        updated = {"specialty": new_specialty, "timing": new_timing,
                   "offered": find_slots(new_specialty, new_timing)}
        if updated["offered"]:
            return _reoffer(call_id, updated, "")
        sessions.clear_pending_booking(call_id)
        return ("book", f"I couldn't find openings for {new_specialty} {new_timing}. "
                        "Could you try another day or department?")

    if len(low) < 30 and "?" not in low:
        # Short ack ("yes", "okay") — remind the offered times instead of
        # paying for an LLM round-trip.
        return _reoffer(call_id, pending, "")

    return _fallback_brain(call_id, text)


def _reoffer(call_id: str, pending: dict, prefix: str) -> tuple[str, str]:
    """(Re)offer pending slots. Always refreshes pending['offered'] so every
    time we just spoke is actually bookable on the next turn."""
    offered = pending.get("offered") or []
    if not offered:
        sessions.clear_pending_booking(call_id)
        return ("book", "Sorry, those times were just taken. Could you tell me the department and day you'd like?")
    sessions.set_pending_booking(call_id, pending)
    lead = f"{prefix} " if prefix else ""
    return ("book", f"{lead}Available times: {', '.join(offered)}. Which works best for you?")


def _confirm_and_book(call_id: str, pending: dict, chosen: str) -> tuple[str, bool]:
    """Write the appointment. Returns (response, booked). On failure the
    pending booking is refreshed with current availability."""
    specialty = pending["specialty"]
    timing = pending.get("timing", "tomorrow")
    slot_dt = _slot_to_datetime(_target_day(timing), chosen)
    phone = sessions.get(call_id)["phone"]

    def _conflict_msg():
        fresh = find_slots(specialty, timing)
        sessions.set_pending_booking(call_id, {**pending, "offered": fresh})
        return (f"Sorry, {chosen} just got taken. Available: {', '.join(fresh) or 'no other times that day'}. Which works?", False)

    with Session() as db:
        doctor = None
        for doc in _doctors_for(db, specialty):
            clash = db.query(Appointment).filter(
                Appointment.doctor_id == doc.id, Appointment.slot_start == slot_dt).first()
            if not clash:
                doctor = doc
                break
        if doctor is None:
            return _conflict_msg()

        if not db.query(User).filter_by(phone=phone).first():
            db.add(User(phone=phone))

        db.add(Appointment(user_phone=phone, doctor_id=doctor.id,
                           slot_start=slot_dt, slot_end=slot_dt + timedelta(minutes=30)))
        conv = db.query(Conversation).filter_by(call_id=call_id).first()
        if conv:
            conv.outcome = "booked"
        try:
            db.commit()
        except IntegrityError:
            # Lost a race for this exact slot: the unique constraint is the
            # real guarantee, the pre-check above is just fast-path UX.
            db.rollback()
            return _conflict_msg()
        return (f"Booked: {doctor.name}, {specialty} {_when_brief(timing)} at {chosen}. Anything else?", True)


def _departments_line() -> str:
    with Session() as db:
        return ", ".join(sorted({d.specialization for d in db.query(Doctor).all()}))


def _department_question() -> str:
    return f"Sure, I can book that. Which department do you need? We have {_departments_line()}."


def _offline_reply(call_id: str, text: str, intent: str) -> str:
    """Compose a real, data-backed reply with NO LLM. Used when Ollama is
    missing, slow, or timed out — the demo must converse sensibly either
    way. Booking flows plug into the same pending-booking machinery as the
    fast path, so 'which department?' → 'cardiology' → slot offer → confirm
    works end-to-end offline."""
    specialty, timing = extract_slots(text)
    if intent == "inquiry" and specialty and len(text.split()) <= 4:
        # Offline only: a bare department name ("cardiology") is booking
        # interest — with an LLM up, classify_intent handles the nuance.
        intent = "book"
    if intent == "book":
        if specialty:
            t = timing or "tomorrow"
            slots = find_slots(specialty, t)
            if slots:
                sessions.set_pending_booking(call_id, {"specialty": specialty, "timing": t, "offered": slots})
                when = TIMING_DISPLAY.get(t, t)
                return (f"Sure. Our {specialty} department has openings {when} at "
                        f"{', '.join(slots)}. Which one works best for you?")
            return f"I couldn't find openings for {specialty} {TIMING_DISPLAY.get(t, t)}. Could you try another day or department?"
        # No department mentioned yet: open the department-picker flow. A
        # pending booking with specialty=None makes the caller's next turn
        # ("cardiology") offer real slots via _continue_booking.
        sessions.set_pending_booking(call_id, {"specialty": None,
                                               "timing": timing or "tomorrow", "offered": []})
        return _department_question()
    if intent == "inquiry":
        return HOSPITAL_INFO_SCRIPT
    if intent == "human":
        return "Hello! I can book an appointment, check timings, or tell you about our departments. How can I help?"
    return CAPABILITIES_SCRIPT


def _fallback_brain(call_id: str, text: str) -> tuple[str, str]:
    session = sessions.get(call_id)
    grounding = {
        "hospital_info": "City Hospital, open Mon-Sat 8 AM-8 PM, 24/7 emergency, 100 Health Avenue",
        "departments": _list_doctors(),
    }
    specialty, timing = extract_slots(text)
    if specialty:
        grounding["available_slots"] = {f"{specialty} ({timing or 'tomorrow'})":
                                        find_slots(specialty, timing or "tomorrow")}

    # Mid-tier matcher first: when it is confident we skip the LLM classify.
    intent = brain.classify_intent_fast(text)
    llm_unreachable = False
    if intent is None:
        try:
            intent = brain.classify_intent(text)
        except httpx.HTTPError:
            intent, llm_unreachable = "inquiry", True
        except Exception:
            intent = "inquiry"

    # Scripted intents never pay for a response-generation round-trip.
    scripts = {"emergency": EMERGENCY_SCRIPT, "cancel": CANCEL_SCRIPT,
               "reschedule": RESCHEDULE_SCRIPT, "out_of_scope": OUT_OF_SCOPE_SCRIPT}
    if intent in scripts:
        return (intent, scripts[intent])
    if llm_unreachable:
        # Ollama is down: no point attempting generation against the same dead
        # host — compose an offline reply immediately.
        return (intent, _offline_reply(call_id, text, intent))

    try:
        response = brain.generate_response(session["transcript"], session["slots"], grounding)
    except Exception:
        # Ollama slow / timed out / errored: compose instead of stalling.
        response = _offline_reply(call_id, text, intent)
    return (intent, response)


@app.exception_handler(httpx.HTTPError)
async def _upstream_unreachable(_request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": f"A required local service is unreachable ({exc.__class__.__name__}). "
                           "Check that Ollama (port 11434) and OmniVoice (port 3900) are running."},
    )


def _check_db() -> bool:
    try:
        with Session() as db:
            db.execute(sql_text("SELECT 1"))
        return True
    except Exception:
        return False


def _check_url(url: str) -> bool:
    # Localhost-only target: never route through any configured system proxy,
    # and fail fast — a dead local port must not stall the health check.
    try:
        with httpx.Client(timeout=settings.health_probe_timeout_seconds,
                          trust_env=False) as client:
            client.get(url)
        return True
    except Exception:
        return False


_health_cache = {"ts": 0.0, "payload": None}


@app.get("/health")
def health():
    # Probes are parallel (ThreadPoolExecutor) and the result is cached for
    # health_cache_seconds: two dead services cost one short timeout, not two,
    # and repeat polls are instant.
    now = time.time()
    if _health_cache["payload"] is not None and now - _health_cache["ts"] < settings.health_cache_seconds:
        return _health_cache["payload"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        db_f = pool.submit(_check_db)
        ollama_f = pool.submit(_check_url, f"{settings.ollama_url}/api/tags")
        omnivoice_f = pool.submit(_check_url, settings.omnivoice_url)
        checks = {"database": db_f.result(), "ollama": ollama_f.result(),
                  "omnivoice": omnivoice_f.result()}
    payload = {"status": "ok" if all(checks.values()) else "degraded",
               "app": settings.app_name, "dependencies": checks}
    _health_cache["ts"], _health_cache["payload"] = now, payload
    return payload


@app.post("/calls/start")
def start_call(event: CallEvent):
    with Session() as db:
        conv = db.query(Conversation).filter_by(call_id=event.call_id).first()
        # 409 only for a genuinely LIVE call: DB row open AND the session
        # still exists in this process. An open row without a live session
        # means the server restarted mid-call (crash) — safe to restart.
        if conv and conv.outcome == "open" and conv.ended_at is None \
                and sessions.get(event.call_id):
            raise HTTPException(409, f"call_id '{event.call_id}' already exists. "
                                     f"End it first via /calls/{event.call_id}/end or pick a new one.")
        if conv:
            # A previous call with this id ended or died with the server:
            # restart it as a fresh conversation instead of 409 forever —
            # the UI cannot recover otherwise.
            conv.phone = event.phone
            conv.outcome = "open"
            conv.ended_at = None
            conv.started_at = datetime.now()
            db.query(TranscriptSegment).filter_by(call_id=event.call_id).delete()
            db.query(Intent).filter_by(call_id=event.call_id).delete()
        else:
            db.add(Conversation(call_id=event.call_id, phone=event.phone))
        db.commit()
    sessions.create(event.call_id, event.phone)
    return {"call_id": event.call_id, "greeting": GREETING}


@app.post("/calls/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """STT an uploaded audio file. Returns text only; use /calls/message to
    run the brain (transcripts are persisted there)."""
    path = await _save_upload(file, prefix="stt")
    try:
        text = await run_in_threadpool(voice.transcribe, path, file.content_type)
    finally:
        _remove_file(path)
    return {"text": text}


@app.post("/calls/message")
def message(msg: ChatMessage):
    return process_text(msg.call_id, msg.text)


@app.post("/calls/audio")
async def audio(call_id: str = Query(pattern=CALL_ID_PATTERN),
                phone: str = Query("demo", max_length=32),
                file: UploadFile = File(...)):
    """STT an uploaded mic recording, run the brain, return the full turn."""
    path = await _save_upload(file, prefix=call_id)
    try:
        sessions.create(call_id, phone)
        _ensure_conversation(call_id, phone)
        text = await run_in_threadpool(voice.transcribe, path, file.content_type)
        if not text:
            return {"call_id": call_id, "user_text": "", "intent": "empty",
                    "response": "I didn't catch any speech. Please hold the mic button and speak again."}
        result = await run_in_threadpool(process_text, call_id, text)
    finally:
        _remove_file(path)
    result["call_id"] = call_id
    result["user_text"] = text
    return result


async def _save_upload(file: UploadFile, prefix: str) -> str:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio upload.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"Audio upload too large (limit {settings.max_upload_bytes // (1024 * 1024)} MB).")
    suffix = AUDIO_SUFFIXES.get(file.content_type or "", ".webm")
    safe_prefix = prefix if prefix.replace("-", "a").replace(".", "a").replace("_", "a").isalnum() else "audio"
    path = os.path.join(TMP_AUDIO_DIR, f"{safe_prefix}_{datetime.now().strftime('%H%M%S%f')}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    path = os.path.join(TMP_AUDIO_DIR, f"tts_{datetime.now().strftime('%H%M%S%f')}.wav")
    voice.synthesize(req.text, settings.tts_voice, path)
    # Delete the temp file after the response has been streamed.
    return FileResponse(path, media_type="audio/wav", background=BackgroundTask(_remove_file, path))


@app.post("/calls/{call_id}/end")
def end_call(call_id: str):
    with Session() as db:
        conv = db.query(Conversation).filter_by(call_id=call_id).first()
        if conv:
            if conv.outcome != "booked":
                conv.outcome = "completed"
            conv.ended_at = datetime.now()
            db.commit()
    sessions.end(call_id)
    return {"status": "ended"}


@app.get("/calls/{call_id}/transcript")
def get_transcript(call_id: str):
    with Session() as db:
        segs = db.query(TranscriptSegment).filter_by(call_id=call_id).order_by(TranscriptSegment.id).all()
        return [{"speaker": s.speaker, "text": s.text} for s in segs]


@app.get("/calls/{call_id}/pending")
def get_pending(call_id: str):
    return {"pending": sessions.get_pending_booking(call_id)}


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
