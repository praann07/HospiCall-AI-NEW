"""Unit tests for the keyword fast-path — no Ollama required."""
from app.services.brain import (Brain, extract_name, extract_slots,
                                matches_any, normalize_intent, parse_slot)


class TestKeywordMatching:
    def test_appointment_is_not_ent(self):
        # Regression: substring matching sent "book an appointment" to ENT.
        assert extract_slots("I want to book an appointment") == (None, None)

    def test_kidney_is_not_pediatrics(self):
        assert extract_slots("I have a kidney stone problem")[0] is None

    def test_which_is_not_a_greeting(self):
        assert not matches_any("which unit is cardiology", ["hi"])

    def test_ent_matches_as_a_word(self):
        assert extract_slots("I need an ENT doctor")[0] == "ent"

    def test_cardiology_variants(self):
        assert extract_slots("book a cardiology appointment tomorrow") == ("cardiology", "tomorrow")
        assert extract_slots("I want to see a cardiologist")[0] == "cardiology"
        assert extract_slots("my heart hurts") [0] == "cardiology"


class TestNameExtraction:
    def test_simple_name(self):
        assert extract_name("my name is Ravi Kumar") == "Ravi Kumar"

    def test_i_am_with_name(self):
        assert extract_name("i am Ravi") == "Ravi"

    def test_this_is_urgent_is_not_a_name(self):
        assert extract_name("this is urgent") is None

    def test_i_am_in_pain_is_not_a_name(self):
        assert extract_name("i am in pain") is None

    def test_i_am_not_feeling_well_is_not_a_name(self):
        assert extract_name("i am not feeling well") is None


class TestParseSlot:
    OFFERED = ["9 AM", "10 AM", "11 AM"]

    def test_explicit_times(self):
        assert parse_slot("10 AM works") == "10 AM"
        assert parse_slot("book 2 PM please") == "2 PM"
        assert parse_slot("sure, 9am") == "9 AM"
        assert parse_slot("9:30 am") == "9 AM"  # minutes round to the hour

    def test_rejections(self):
        assert parse_slot("maybe") is None
        assert parse_slot("13 PM") is None
        assert parse_slot("please call 108") is None  # no false slot from numbers

    def test_ordinals_map_to_offered(self):
        assert parse_slot("the second one", offered=self.OFFERED) == "10 AM"
        assert parse_slot("first one please", offered=self.OFFERED) == "9 AM"

    def test_number_words(self):
        assert parse_slot("ten o'clock", offered=self.OFFERED) == "10 AM"
        assert parse_slot("eleven am", offered=self.OFFERED) == "11 AM"


class TestIntentNormalization:
    def test_punctuated_label(self):
        assert normalize_intent('"Book."') == "book"

    def test_sentence_label(self):
        assert normalize_intent("The intent is emergency!") == "emergency"

    def test_out_of_scope_with_spaces(self):
        assert normalize_intent("out of scope") == "out_of_scope"

    def test_unknown_defaults_to_inquiry(self):
        assert normalize_intent("banana") == "inquiry"


class TestFastHandle:
    def _brain(self):
        return Brain("http://localhost:1", "unused")  # no network calls made

    def test_emergency_first(self):
        intent, response = self._brain().fast_handle("my father is having a heart attack", lambda *a: [])
        assert intent == "emergency"

    def test_greeting(self):
        intent, _ = self._brain().fast_handle("hello", lambda *a: [])
        assert intent == "human"

    def test_doctor_list(self):
        intent, response = self._brain().fast_handle(
            "which doctors do you have", lambda *a: [], lambda: ["Dr. Mehta (Cardiology)"])
        assert intent == "inquiry"
        assert "Dr. Mehta" in response

    def test_booking_with_slots(self):
        intent, response = self._brain().fast_handle(
            "book a cardiology appointment tomorrow", lambda s, t: ["9 AM", "10 AM"])
        assert intent == "book"
        assert "9 AM" in response

    def test_cancel_is_honest(self):
        # Must not promise a cancellation flow that doesn't exist.
        intent, response = self._brain().fast_handle("I want to cancel my appointment", lambda *a: [])
        assert intent == "cancel"
        assert "front desk" in response

    def test_emergency_script_makes_no_false_promises(self):
        intent, response = self._brain().fast_handle("emergency", lambda *a: [])
        assert "alerted" not in response
        assert "108" in response

    def test_unmatched_returns_none(self):
        assert self._brain().fast_handle("insurance papers", lambda *a: []) is None


class TestClassifyIntentWithFakeLLM:
    def test_punctuated_llm_reply_is_normalized(self, monkeypatch):
        brain = Brain("http://localhost:1", "unused")
        monkeypatch.setattr(brain, "_request", lambda *a, **k: 'Sure! "Book."')
        assert brain.classify_intent("I need an appointment") == "book"


class TestClassifyIntentFast:
    """Mid-tier matcher: labels common paraphrases with no LLM round-trip."""

    def _brain(self):
        return Brain("http://localhost:1", "unused")

    def test_booking_paraphrase_beyond_fast_path(self):
        # "schedule" is not a fast-path BOOKING keyword, but the mid-tier
        # should still label it instead of paying for the LLM.
        assert self._brain().classify_intent_fast(
            "I need to schedule something with a doctor") == "book"

    def test_inquiry_paraphrase(self):
        assert self._brain().classify_intent_fast(
            "when do you open on sundays") == "inquiry"

    def test_word_boundaries_still_apply(self):
        # "book" must not match inside other words.
        assert self._brain().classify_intent_fast("facebook is down") is None

    def test_unsure_returns_none(self):
        assert self._brain().classify_intent_fast(
            "the vending machine ate my coins") is None


class TestLlmRequestCaps:
    def test_keep_alive_and_classify_model_in_payload(self, monkeypatch):
        brain = Brain("http://localhost:1", "main-model",
                      keep_alive="30m", classify_model="small-model")
        seen = {}
        import app.services.brain as b
        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "book"}}
        def fake_post(url, json=None):
            seen["url"], seen["payload"] = url, json
            return FakeResp()
        monkeypatch.setattr(brain._client, "post", fake_post)
        assert brain.classify_intent("I need an appointment") == "book"
        assert seen["payload"]["keep_alive"] == "30m"
        assert seen["payload"]["model"] == "small-model"
        # Response generation uses the main model.
        brain.generate_response([{"role": "patient", "content": "hi"}], {})
        assert seen["payload"]["model"] == "main-model"
