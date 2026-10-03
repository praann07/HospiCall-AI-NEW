"""Tests for slot availability and the day/time logic — no Ollama required."""
from datetime import date, datetime, timedelta

from app.db import Appointment, Doctor
from app.main import (Session, _slot_to_datetime, _target_day,
                      _when_brief, find_slots)


def _add_doctor(specialization, name="Dr. Test"):
    with Session() as db:
        doc = Doctor(name=name, specialization=specialization, location="Block A")
        db.add(doc)
        db.commit()
        return doc.id


class TestDayLogic:
    def test_part_of_day_means_tomorrow(self):
        # Regression: 'morning' slots were computed for today but confirmed
        # as 'tomorrow morning'. Both must agree now.
        for timing in ("morning", "afternoon", "evening"):
            assert _target_day(timing) == date.today() + timedelta(days=1)
            assert "tomorrow" in _when_brief(timing)

    def test_today_and_tomorrow(self):
        assert _target_day("today") == date.today()
        assert _target_day("tomorrow") == date.today() + timedelta(days=1)

    def test_slot_to_datetime(self):
        tomorrow = _target_day("tomorrow")
        assert _slot_to_datetime(tomorrow, "10 AM").hour == 10
        assert _slot_to_datetime(tomorrow, "2 PM").hour == 14
        assert _slot_to_datetime(tomorrow, "12 PM").hour == 12


class TestFindSlots:
    def test_excludes_booked_slots(self):
        _add_doctor("Dermatology-Test")
        with Session() as db:
            doc = db.query(Doctor).filter_by(specialization="Dermatology-Test").first()
            tomorrow_10 = _slot_to_datetime(_target_day("tomorrow"), "10 AM")
            db.add(Appointment(user_phone="9000000000", doctor_id=doc.id,
                               slot_start=tomorrow_10, slot_end=tomorrow_10 + timedelta(minutes=30)))
            db.commit()
        slots = find_slots("dermatology-test", "tomorrow")
        assert "10 AM" not in slots
        assert slots  # 9 AM / 11 AM / 2 PM still free

    def test_second_doctor_covers_full_first_doctor(self):
        # Department availability: a slot is offered if ANY doctor is free.
        _add_doctor("Ortho-Test", name="Dr. Full")
        _add_doctor("Ortho-Test", name="Dr. Free")
        with Session() as db:
            doc = db.query(Doctor).filter_by(specialization="Ortho-Test", name="Dr. Full").first()
            for label in ("9 AM", "10 AM", "11 AM"):
                dt = _slot_to_datetime(_target_day("tomorrow"), label)
                db.add(Appointment(user_phone="9000000000", doctor_id=doc.id,
                                   slot_start=dt, slot_end=dt + timedelta(minutes=30)))
            db.commit()
        slots = find_slots("ortho-test", "tomorrow")
        assert slots == ["9 AM", "10 AM", "11 AM"]  # all still free via Dr. Free

    def test_today_never_offers_past_times(self):
        _add_doctor("Pastcheck-Test")
        slots = find_slots("pastcheck-test", "today")
        for label in slots:
            assert _slot_to_datetime(date.today(), label) > datetime.now()

    def test_unknown_specialty_returns_empty(self):
        assert find_slots("nonexistent-dept", "tomorrow") == []
