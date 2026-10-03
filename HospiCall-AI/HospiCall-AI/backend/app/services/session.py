import threading
from datetime import datetime, timedelta


class SessionManager:
    """In-memory session state per call. Redis replaces this in production."""

    def __init__(self, ttl_seconds: int = 900):
        self._sessions = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds

    def create(self, call_id: str, phone: str) -> dict:
        """Create a session, or return the existing one for this call_id
        (re-calling /calls/audio must not wipe an in-flight conversation)."""
        with self._lock:
            self._cleanup()
            existing = self._sessions.get(call_id)
            if existing is not None:
                return existing
            session = {
                "call_id": call_id,
                "phone": phone,
                "transcript": [],
                "slots": {},
                "pending_booking": None,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now(),
            }
            self._sessions[call_id] = session
            return session

    def get(self, call_id: str) -> dict | None:
        with self._lock:
            self._cleanup()
            s = self._sessions.get(call_id)
            if s:
                s["last_activity"] = datetime.now()
            return s

    def add_turn(self, call_id: str, role: str, content: str):
        s = self.get(call_id)
        if s:
            s["transcript"].append({"role": role, "content": content})

    def update_slots(self, call_id: str, **slots):
        s = self.get(call_id)
        if s:
            s["slots"].update(slots)

    def set_pending_booking(self, call_id: str, pending: dict):
        s = self.get(call_id)
        if s:
            s["pending_booking"] = pending

    def get_pending_booking(self, call_id: str) -> dict | None:
        s = self.get(call_id)
        return s["pending_booking"] if s else None

    def clear_pending_booking(self, call_id: str):
        s = self.get(call_id)
        if s:
            s["pending_booking"] = None

    def end(self, call_id: str):
        with self._lock:
            self._sessions.pop(call_id, None)

    def _cleanup(self):
        """Caller must hold self._lock."""
        cutoff = datetime.now() - timedelta(seconds=self.ttl)
        stale = [k for k, v in self._sessions.items() if v["last_activity"] < cutoff]
        for k in stale:
            self._sessions.pop(k, None)
