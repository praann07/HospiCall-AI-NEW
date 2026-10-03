import json
import re
from typing import Callable, Optional

import httpx

SPECIALTY_KEYWORDS = {
    # Keywords ending in '*' are prefix matches (cardio -> cardiology,
    # cardiologist); everything else matches on word boundaries so 'ent'
    # never matches 'appointment' and 'kid' never matches 'kidney'.
    "cardiology": ["cardio*", "heart", "hearts", "chest pain", "bp", "blood pressure"],
    "neurology": ["neuro*", "brain", "headache", "headaches", "migraine", "seizure", "memory"],
    "orthopedics": ["ortho*", "bone", "bones", "joint", "joints", "knee", "knees", "spine", "back pain", "fracture"],
    "pediatrics": ["pediatr*", "child", "children", "baby", "babies", "kid", "kids"],
    "gynecology": ["gynec*", "pregnancy", "pregnant"],
    "dermatology": ["derm*", "skin", "rash", "rashes"],
    "ophthalmology": ["ophthalm*", "eye", "eyes", "vision"],
    "ent": ["ent", "ear", "ears", "nose", "throat"],
}

TIMING_KEYWORDS = {
    "morning": ["morning"],
    "afternoon": ["afternoon"],
    "evening": ["evening"],
    "today": ["today"],
    "tomorrow": ["tomorrow"],
}

# How a timing key reads aloud in an offer ('morning' alone is ambiguous).
TIMING_DISPLAY = {
    "morning": "tomorrow morning",
    "afternoon": "tomorrow afternoon",
    "evening": "tomorrow evening",
}

BOOKING_KEYWORDS = ["appointment", "book", "see a", "consult", "visit", "checkup", "meet", "take an", "fix up", "slot"]
CANCEL_KEYWORDS = ["cancel"]
RESCHEDULE_KEYWORDS = ["reschedule", "move the", "change the", "shift"]
EMERGENCY_KEYWORDS = ["emergency", "ambulance", "heart attack", "unconscious", "not breathing", "bleeding", "stroke", "choking"]
INQUIRY_KEYWORDS = ["timing", "timings", "open", "hours", "location", "address", "reach", "parking", "fee", "cost", "charge", "where", "phone number", "contact"]
OUT_OF_SCOPE_KEYWORDS = ["medical advice", "symptom", "diagnos", "treatment for", "prescription", "should i take", "is it serious"]

GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste", "namaskar"]
DOCTOR_LIST_KEYWORDS = ["what doctor", "which doctor", "list of doctor", "list the doctor", "doctors do you have",
                        "what specialist", "all the doctor", "who are the doctor", "hospitals are present",
                        "what hospitals", "list of hospitals", "departments do you have"]

# Scripts must never promise actions the system cannot perform (no alerting,
# no transfer, no cancel/reschedule backend exists in the demo).
EMERGENCY_SCRIPT = ("If this is a medical emergency, please hang up and dial 108 for an ambulance right now. "
                    "Our emergency department at 100 Health Avenue is open 24/7.")
CANCEL_SCRIPT = ("I can't process cancellations on this line yet — our front desk at 080-100-20000 "
                 "(Monday to Saturday, 8 AM to 8 PM) can cancel it for you. Would you like to book a new appointment instead?")
RESCHEDULE_SCRIPT = ("For rescheduling, please call our front desk at 080-100-20000 (Monday to Saturday, 8 AM to 8 PM). "
                     "I can book a brand-new appointment right now if you'd like — just tell me the department and day.")
OUT_OF_SCOPE_SCRIPT = "I can't give medical advice, but I can book you an appointment or connect you to a doctor."
HOSPITAL_INFO_SCRIPT = ("Our outpatient department is open Monday to Saturday, 8 AM to 8 PM, and 24/7 for "
                        "emergencies. We're located at 100 Health Avenue, opposite the city park.")
CAPABILITIES_SCRIPT = ("I can book an appointment, check hospital timings, or list our doctors. "
                       "What would you like?")


def matches_any(text: str, keywords: list[str]) -> bool:
    """Keyword match with word boundaries so 'ent' doesn't match
    'appointment' and 'kid' doesn't match 'kidney'. A keyword ending in
    '*' matches any word starting with it (cardio -> cardiology)."""
    for k in keywords:
        if k.endswith("*"):
            if re.search(rf"\b{re.escape(k[:-1])}", text):
                return True
        elif re.search(rf"\b{re.escape(k)}\b", text):
            return True
    return False


def extract_slots(text: str) -> tuple[Optional[str], Optional[str]]:
    """Keyword-based (specialty, timing) extraction. Either may be None."""
    low = text.lower()
    specialty = next((s for s, kws in SPECIALTY_KEYWORDS.items() if matches_any(low, kws)), None)
    timing = next((t for t, kws in TIMING_KEYWORDS.items() if matches_any(low, kws)), None)
    return specialty, timing


# Words that follow "i am / this is" but are NOT names ("i am in pain",
# "this is urgent", "i am ravi and i need..."). extract_name stops at the
# first one and rejects the match if nothing readable came before it.
NAME_STOPWORDS = {
    "in", "not", "feeling", "feel", "calling", "trying", "looking", "here",
    "sick", "fine", "tired", "unwell", "well", "good", "bad", "ok", "okay",
    "urgent", "late", "early", "back", "again", "sorry", "happy", "glad",
    "interested", "booking", "book", "asking", "waiting", "checking",
    "going", "doing", "done", "ready", "available", "busy", "free",
    "dizzy", "feverish", "nauseous", "bleeding", "having", "getting",
    "and", "i", "please", "to", "because", "but", "need", "want",
    "would", "like", "have", "has",
}
NAME_RE = re.compile(r"\b(?:my name is|i am|i'm|this is)\s+([a-z][a-z .'-]*?)(?:[,.]|$)", re.IGNORECASE)

SLOT_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?(?:\s*(a\.?m\.?|p\.?m\.?)\b|(?!\d))")
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse_slot(text: str, offered: Optional[list[str]] = None) -> Optional[str]:
    """Extract a slot label like '10 AM' from patient text.

    Understands 'at 10 am', '2 PM please', '9:30 am' (minutes round to the
    hour), ordinals referring to the offered list ('the second one'), and
    number words ('ten o'clock'). Returns None when nothing parseable.
    """
    low = text.lower()
    offered = offered or []

    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", low) and idx < len(offered):
            return offered[idx]

    m = SLOT_RE.search(low)
    if m:
        hour = int(m.group(1))
        minutes = int(m.group(2) or 0)
        mer = (m.group(3) or "").replace(".", "")
        if not 1 <= hour <= 12:
            return None
        if mer in ("am", "pm"):
            return f"{hour} {mer.upper()}"
        if minutes == 0:  # bare hour ('at 10'): resolve AM/PM against the offer
            for suffix in ("AM", "PM"):
                if f"{hour} {suffix}" in offered:
                    return f"{hour} {suffix}"
        return None

    mer = "am" if re.search(r"\ba\.?m\.?\b", low) else ("pm" if re.search(r"\bp\.?m\.?\b", low) else None)
    for word, hour in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", low):
            if mer:
                return f"{hour} {mer.upper()}"
            for suffix in ("AM", "PM"):
                if f"{hour} {suffix}" in offered:
                    return f"{hour} {suffix}"
    return None


def extract_name(text: str) -> Optional[str]:
    """Extract a speaker name ('my name is Ravi Kumar'), rejecting common
    non-name continuations ('this is urgent', 'i am in pain')."""
    m = NAME_RE.search(text)
    if not m:
        return None
    words = []
    for w in m.group(1).strip().split():
        if w.lower() in NAME_STOPWORDS:
            break
        words.append(w)
    if not words or len(words) > 3:
        return None
    return " ".join(words).title()


VALID_INTENTS = ("book", "cancel", "reschedule", "inquiry", "emergency", "human", "out_of_scope")

# Mid-tier intent synonyms: broader than the fast-path scripts, used to label
# a turn WITHOUT an LLM round-trip. Keys must be VALID_INTENTS entries.
INTENT_SYNONYMS = {
    "book": ["appointment", "book", "booking", "schedule", "scheduled", "scheduling",
             "slot", "slots", "consult", "consultation", "checkup", "check up",
             "see a doctor", "see the doctor", "meet the doctor", "visit"],
    "cancel": CANCEL_KEYWORDS + ["cancellation", "cancell"],
    "reschedule": RESCHEDULE_KEYWORDS + ["postpone", "later date", "another day"],
    "emergency": EMERGENCY_KEYWORDS,
    "inquiry": INQUIRY_KEYWORDS + ["when", "what time", "how do i get", "directions",
                                   "how far", "visiting hours", "do you have"],
}


def normalize_intent(raw: str) -> str:
    """Map a raw LLM reply ('"Book."', 'The intent is emergency!') to a known
    label so downstream equality checks actually work."""
    label = re.sub(r"[^a-z ]", " ", raw.lower())
    for intent in sorted(VALID_INTENTS, key=len, reverse=True):
        if intent.replace("_", " ") in label:
            return intent
    return "inquiry"


class Brain:
    """LLM wrapper around Ollama (Llama 3.1 8B) plus the keyword fast-path.

    Every LLM call is hard-capped by llm_timeout_seconds: a hung or absent
    Ollama degrades to scripts, it never stalls the caller's turn."""

    def __init__(self, base_url: str, model: str, *, timeout_s: float = 5.0,
                 connect_s: float = 2.0, keep_alive: str = "30m",
                 classify_model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Optional smaller/faster model for pure intent labeling.
        self.classify_model = classify_model or model
        self.timeout = httpx.Timeout(timeout_s, connect=connect_s)
        self.keep_alive = keep_alive
        # Localhost-only target: bypass any configured system proxy.
        self._client = httpx.Client(timeout=self.timeout, trust_env=False)

    def close(self):
        self._client.close()

    def _request(self, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 300, model: Optional[str] = None) -> str:
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def confirm_booking(self, text: str, pending: dict) -> Optional[str]:
        """If a booking is pending, parse a chosen slot like '10 AM' from the
        text. Returns the slot label or None if no slot chosen."""
        return parse_slot(text, offered=pending.get("offered") or [])

    def fast_handle(self, text: str, find_slots: Callable, list_doctors: Optional[Callable] = None) -> Optional[tuple]:
        """Instant keyword-based response. Returns (intent, response) or None if the LLM is needed."""
        low = text.lower()

        if matches_any(low, EMERGENCY_KEYWORDS):
            return ("emergency", EMERGENCY_SCRIPT)
        if matches_any(low, CANCEL_KEYWORDS):
            return ("cancel", CANCEL_SCRIPT)
        if matches_any(low, RESCHEDULE_KEYWORDS):
            return ("reschedule", RESCHEDULE_SCRIPT)
        if matches_any(low, OUT_OF_SCOPE_KEYWORDS):
            return ("out_of_scope", OUT_OF_SCOPE_SCRIPT)

        name = extract_name(text)
        if name:
            return ("human", f"Nice to meet you, {name}! I can book an appointment, check timings, or tell you about our departments. What do you need?")

        if list_doctors and matches_any(low, DOCTOR_LIST_KEYWORDS):
            doctors = list_doctors()
            if doctors:
                return ("inquiry", f"Our hospital has {len(doctors)} departments. We have {', '.join(doctors)}. "
                                   "Which one would you like to visit?")

        if matches_any(low, GREETING_KEYWORDS) and len(low) < 20:
            return ("human", "Hello! Welcome to City Hospital. I can book an appointment, check timings, or tell you about our departments. How can I help?")

        specialty, timing = extract_slots(text)
        timing = timing or "tomorrow"

        if matches_any(low, BOOKING_KEYWORDS):
            slots = find_slots(specialty, timing)
            when = TIMING_DISPLAY.get(timing, timing)
            if slots:
                return ("book", f"Sure. Our {specialty or 'consultation'} department has openings {when} at "
                                f"{', '.join(slots)}. Which one works best for you?")
            return ("book", f"I can book you a {specialty or 'consultation'} appointment. Could you tell me the department and the day and time you'd prefer?")

        if matches_any(low, INQUIRY_KEYWORDS):
            return ("inquiry", HOSPITAL_INFO_SCRIPT)

        return None

    def classify_intent_fast(self, text: str) -> Optional[str]:
        """Mid-tier intent matcher: broader synonyms than the fast-path
        scripts, ~microseconds, no LLM. Returns a known intent or None when
        genuinely unsure (the LLM then decides)."""
        low = text.lower()
        for intent, synonyms in INTENT_SYNONYMS.items():
            if matches_any(low, synonyms):
                return intent
        return None

    def classify_intent(self, text: str) -> str:
        system = (
            "You are an intent classifier for a hospital call assistant. "
            'Respond with ONLY one label: book, cancel, reschedule, inquiry, emergency, human, out_of_scope.'
        )
        # Classification is a 10-token job: use the smaller classify model
        # when configured, under the same hard timeout as everything else.
        raw = self._request(system, text, temperature=0.0, max_tokens=10,
                            model=self.classify_model)
        return normalize_intent(raw)

    def generate_response(self, transcript: list[dict], slots: dict, grounding: Optional[dict] = None) -> str:
        system = (
            "You are the AI receptionist for City Hospital. "
            "You can ONLY schedule appointments, check availability, and answer general hospital questions. "
            "NEVER give medical advice, diagnoses, or prescriptions. "
            "If asked for medical advice, say: 'I can't give medical advice, but I can book you an appointment or connect you to a doctor.' "
            "Use ONLY the doctors and time slots listed in Known info — never invent availability, doctors, or times. "
            "If the information you need is not in Known info, ask the caller a clarifying question instead. "
            "Keep responses short, warm, and under 2 sentences when possible."
        )
        known = {"slots": slots}
        if grounding:
            known.update(grounding)
        context = "\n".join(f"{t['role']}: {t['content']}" for t in transcript[-10:])
        user = f"Known info: {json.dumps(known, default=str)}\nConversation so far:\n{context}\n\nRespond as the receptionist."
        return self._request(system, user)
