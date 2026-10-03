"""End-to-end API tests via TestClient — Ollama/OmniVoice not required
(the LLM fallback is monkeypatched where a turn would need it)."""
import re
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.db import Appointment, Doctor
from app.main import Session, app, _slot_to_datetime, _target_day

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def seed_doctors():
    with Session() as db:
        if not db.query(Doctor).filter_by(specialization="Cardiology").first():
            db.add_all([
                Doctor(name="Dr. Mehta", specialization="Cardiology", location="Block A, 2F"),
                Doctor(name="Dr. Rao", specialization="Cardiology", location="Block A, 2F"),
            ])
            db.commit()


def _call_id():
    return "t-" + uuid.uuid4().hex[:10]


def _start_call(phone="9000000000"):
    call_id = _call_id()
    r = client.post("/calls/start", json={"call_id": call_id, "phone": phone})
    assert r.status_code == 200
    return call_id, r.json()


class TestCallLifecycle:
    def test_greeting_mentions_recording(self):
        _, body = _start_call()
        assert "recorded" in body["greeting"].lower()

    def test_duplicate_call_id_conflict(self):
        call_id, _ = _start_call()
        r = client.post("/calls/start", json={"call_id": call_id, "phone": "9000000000"})
        assert r.status_code == 409

    def test_message_unknown_call_404(self):
        r = client.post("/calls/message", json={"call_id": "nope-123", "text": "hello"})
        assert r.status_code == 404

    def test_end_call(self):
        call_id, _ = _start_call()
        assert client.post(f"/calls/{call_id}/end").status_code == 200

    def test_transcript_is_ordered_pairwise(self):
        call_id, _ = _start_call()
        client.post("/calls/message", json={"call_id": call_id, "text": "hello"})
        r = client.get(f"/calls/{call_id}/transcript")
        segs = r.json()
        assert [s["speaker"] for s in segs] == ["patient", "ai"]


class TestHealth:
    def test_health_reports_dependencies(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert set(body["dependencies"]) == {"database", "ollama", "omnivoice"}
        assert body["dependencies"]["database"] is True  # our own DB always works


class TestFastPathTurns:
    def test_generic_appointment_phrase_not_routed_to_ent(self):
        # Regression: "book an appointment" used to hit the ENT department.
        call_id, _ = _start_call()
        r = client.post("/calls/message", json={"call_id": call_id, "text": "I want to book an appointment"})
        body = r.json()
        assert body["intent"] == "book"
        assert "consultation" in body["response"]  # no specialty was detected

    def test_emergency_keyword_is_instant(self):
        call_id, _ = _start_call()
        r = client.post("/calls/message", json={"call_id": call_id, "text": "this is an emergency, ambulance!"})
        body = r.json()
        assert body["intent"] == "emergency"
        assert "108" in body["response"]

    def test_urgent_statement_is_not_treated_as_a_name(self):
        call_id, _ = _start_call()
        r = client.post("/calls/message", json={"call_id": call_id, "text": "this is urgent"})
        body = r.json()
        assert "Nice to meet you" not in body["response"]


class TestBookingFlow:
    def test_book_confirm_and_persist(self):
        call_id, _ = _start_call()
        r1 = client.post("/calls/message",
                         json={"call_id": call_id, "text": "I want a cardiology appointment tomorrow"})
        assert r1.json()["intent"] == "book"

        offered = [s for s in ("9 AM", "10 AM", "11 AM", "2 PM") if s in r1.json()["response"]]
        assert offered, f"no slots offered: {r1.json()['response']}"
        chosen = offered[0]

        r2 = client.post("/calls/message", json={"call_id": call_id, "text": f"{chosen} works"})
        body = r2.json()
        assert body["intent"] == "booked"
        assert "Booked" in body["response"]

        with Session() as db:
            appts = db.query(Appointment).filter(Appointment.user_phone == "9000000000").all()
            assert any(a.slot_start == _slot_to_datetime(_target_day("tomorrow"), chosen) for a in appts)

        pending = client.get(f"/calls/{call_id}/pending").json()["pending"]
        assert pending is None  # cleared after successful booking

    def test_conflict_reoffer_is_bookable(self):
        # Book out the whole cardiology department at 10 AM tomorrow, then
        # verify the caller can still complete a booking after a conflict.
        slot_dt = _slot_to_datetime(_target_day("tomorrow"), "10 AM")
        with Session() as db:
            for doc in db.query(Doctor).filter_by(specialization="Cardiology").all():
                db.add(Appointment(user_phone="9000000000", doctor_id=doc.id,
                                   slot_start=slot_dt, slot_end=slot_dt + timedelta(minutes=30)))
            db.commit()

        call_id, _ = _start_call()
        r1 = client.post("/calls/message",
                         json={"call_id": call_id, "text": "book a cardiology appointment tomorrow"})
        offered = r1.json()["response"]
        assert "10 AM" not in offered  # fully booked slot not offered

        # Caller insists on 10 AM anyway -> conflict message with fresh slots.
        r2 = client.post("/calls/message", json={"call_id": call_id, "text": "10 AM please"})
        body2 = r2.json()
        assert "10 AM" in body2["response"] and "Available" in body2["response"]

        # A slot from the fresh offer must now complete the booking.
        fresh = [s for s in ("9 AM", "11 AM", "2 PM", "3 PM") if s in body2["response"]]
        assert fresh
        r3 = client.post("/calls/message", json={"call_id": call_id, "text": f"{fresh[0]} please"})
        assert r3.json()["intent"] == "booked"

    def test_ordinal_confirmation(self):
        call_id, _ = _start_call()
        r1 = client.post("/calls/message",
                         json={"call_id": call_id, "text": "book a cardiology appointment tomorrow"})
        import re as _re
        offered = _re.findall(r"\d{1,2} [AP]M", r1.json()["response"])
        assert offered, f"no slots offered: {r1.json()['response']}"
        r2 = client.post("/calls/message", json={"call_id": call_id, "text": "the first one"})
        body = r2.json()
        assert body["intent"] == "booked"
        assert offered[0] in body["response"]  # ordinal mapped to offered[0]

    def test_booking_uses_second_doctor_when_first_busy(self):
        # Offers are department-wide: if the first neurologist is busy at
        # 10 AM but the second is free, 10 AM is offered and books.
        with Session() as db:
            if not db.query(Doctor).filter_by(specialization="Neurology").first():
                db.add_all([
                    Doctor(name="Dr. Neu One", specialization="Neurology", location="Block B"),
                    Doctor(name="Dr. Neu Two", specialization="Neurology", location="Block B"),
                ])
                db.commit()
        slot_dt = _slot_to_datetime(_target_day("tomorrow"), "10 AM")
        with Session() as db:
            first = db.query(Doctor).filter_by(specialization="Neurology").order_by(Doctor.id).first()
            if not db.query(Appointment).filter_by(doctor_id=first.id, slot_start=slot_dt).first():
                db.add(Appointment(user_phone="9000000000", doctor_id=first.id,
                                   slot_start=slot_dt, slot_end=slot_dt + timedelta(minutes=30)))
                db.commit()

        call_id, _ = _start_call()
        r1 = client.post("/calls/message",
                         json={"call_id": call_id, "text": "book a neurology appointment tomorrow"})
        assert "10 AM" in r1.json()["response"]
        r2 = client.post("/calls/message", json={"call_id": call_id, "text": "10 AM works"})
        body = r2.json()
        assert body["intent"] == "booked"
        assert "Dr. Neu Two" in body["response"]

    def test_cancel_during_pending_booking(self):
        call_id, _ = _start_call()
        client.post("/calls/message",
                    json={"call_id": call_id, "text": "book a cardiology appointment tomorrow"})
        r = client.post("/calls/message", json={"call_id": call_id, "text": "cancel that please"})
        assert r.json()["intent"] == "cancel"
        assert client.get(f"/calls/{call_id}/pending").json()["pending"] is None


class TestValidation:
    def test_rejects_traversal_call_id(self):
        r = client.post("/calls/start", json={"call_id": "../../etc/passwd", "phone": "9000000000"})
        assert r.status_code == 422

    def test_rejects_empty_text(self):
        call_id, _ = _start_call()
        r = client.post("/calls/message", json={"call_id": call_id, "text": ""})
        assert r.status_code == 422

    def test_transcribe_without_omnivoice_returns_503_not_500(self):
        r = client.post("/calls/transcribe", files={"file": ("t.wav", b"RIFFfake", "audio/wav")})
        assert r.status_code == 503
        assert "unreachable" in r.json()["detail"].lower()

    def test_audio_upload_empty_file(self):
        r = client.post(f"/calls/audio?call_id={_call_id()}", files={"file": ("t.wav", b"", "audio/wav")})
        assert r.status_code == 400


class TestHealthCache:
    def test_health_cached_and_shape_stable(self):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.status_code == r2.status_code == 200
        assert set(r1.json()["dependencies"]) == {"database", "ollama", "omnivoice"}
        # Second call within the cache TTL returns the identical payload.
        assert r2.json() == r1.json()


class TestFallbackDegradation:
    def test_unmatched_turn_degrades_fast_when_llm_down(self, monkeypatch):
        """Ollama unreachable: one capped attempt, then the degrade script —
        never a 500 and never two timeouts."""
        import httpx
        call_id, _ = _start_call()
        calls = {"n": 0}
        def dead_llm(*a, **k):
            calls["n"] += 1
            raise httpx.ConnectError("refused")
        monkeypatch.setattr("app.main.brain.classify_intent", dead_llm)
        monkeypatch.setattr("app.main.brain.generate_response", dead_llm)
        r = client.post("/calls/message", json={
            "call_id": call_id, "text": "the vending machine on the third floor ate my coins"})
        assert r.status_code == 200
        body = r.json()
        assert body["intent"] == "inquiry"
        # No LLM -> composed offline reply, not a generic apology.
        assert "outpatient department" in body["response"]
        # classify failed at connect level -> generation must not be retried.
        assert calls["n"] == 1


class TestCallRestart:
    def test_ended_call_id_can_restart(self):
        """A call that ended (or whose server restarted) must be startable
        again — otherwise the browser demo bricks itself after a refresh."""
        call_id, _ = _start_call()
    # leave a turn behind so we can verify it is cleared on restart
        r = client.post("/calls/message", json={"call_id": call_id, "text": "hello"})
        assert r.status_code == 200
        assert client.post(f"/calls/{call_id}/end").status_code == 200

        r2 = client.post("/calls/start", json={"call_id": call_id, "phone": "9000000000"})
        assert r2.status_code == 200
        assert "greeting" in r2.json()
        # Fresh conversation: old transcript segments were cleared.
        assert client.get(f"/calls/{call_id}/transcript").json() == []
        # And the new session accepts messages immediately.
        r3 = client.post("/calls/message", json={"call_id": call_id, "text": "hello"})
        assert r3.status_code == 200

    def test_open_call_restarts_after_server_restart(self):
        """Crash simulation: RAM session wiped, DB row still 'open' —
        /calls/start must recover it, not 409 forever."""
        call_id, _ = _start_call()
        from app.main import sessions as app_sessions
        app_sessions.end(call_id)  # exactly what a server restart does
        r = client.post("/calls/start", json={"call_id": call_id, "phone": "9000000000"})
        assert r.status_code == 200
        r2 = client.post("/calls/message", json={"call_id": call_id, "text": "hello"})
        assert r2.status_code == 200


class TestOfflineBrain:
    """No Ollama at all: the demo must still converse and book via the
    offline composer + department-picker flow."""

    @pytest.fixture(autouse=True)
    def _dead_llm(self, monkeypatch):
        import httpx
        def dead(*a, **k):
            raise httpx.ConnectError("refused")
        monkeypatch.setattr("app.main.brain.classify_intent", dead)
        monkeypatch.setattr("app.main.brain.generate_response", dead)

    def test_full_booking_conversation_without_llm(self):
        call_id, _ = _start_call()

        # Turn 1: booking intent, no department -> asks which department.
        r1 = client.post("/calls/message", json={
            "call_id": call_id, "text": "I need to schedule something with a doctor"})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["intent"] == "book"
        assert "Which department" in b1["response"]

        # Turn 2: bare department -> real slot offer from the DB.
        r2 = client.post("/calls/message", json={
            "call_id": call_id, "text": "cardiology"})
        b2 = r2.json()
        assert b2["intent"] == "book"
        m = re.search(r"(\d{1,2} [AP]M)", b2["response"])
        assert m, f"no slot offered: {b2['response']}"

        # Turn 3: confirm the first offered slot -> booked in the DB.
        r3 = client.post("/calls/message", json={
            "call_id": call_id, "text": m.group(1) + " works"})
        b3 = r3.json()
        assert b3["intent"] == "booked"
        assert "Booked" in b3["response"]
        with Session() as db:
            assert db.query(Appointment).filter_by(
                user_phone="9000000000").count() >= 1

    def test_department_picker_reasks_on_smalltalk(self):
        call_id, _ = _start_call()
        client.post("/calls/message", json={
            "call_id": call_id, "text": "I need to schedule something with a doctor"})
        r = client.post("/calls/message", json={"call_id": call_id, "text": "hello"})
        body = r.json()
        assert body["intent"] == "book"
        assert "Which department" in body["response"]
