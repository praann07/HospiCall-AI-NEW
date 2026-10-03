import os
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, UniqueConstraint, create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def _now():
    # Local naive timestamps, consistent with the slot logic in main.py.
    return datetime.now()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    call_id = Column(String, unique=True, index=True)
    phone = Column(String, index=True)
    outcome = Column(String, default="open")
    started_at = Column(DateTime, default=_now)
    ended_at = Column(DateTime, nullable=True)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    id = Column(Integer, primary_key=True)
    call_id = Column(String, index=True)
    speaker = Column(String)
    text = Column(Text)
    confidence = Column(Float, default=1.0)
    ts = Column(DateTime, default=_now)


class Intent(Base):
    __tablename__ = "intents"
    id = Column(Integer, primary_key=True)
    call_id = Column(String, index=True)
    intent_type = Column(String)
    slots_json = Column(Text)
    confidence = Column(Float, default=1.0)
    ts = Column(DateTime, default=_now)


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    specialization = Column(String, index=True)
    location = Column(String)
    rating = Column(Float, default=5.0)


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    user_phone = Column(String, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), index=True)
    slot_start = Column(DateTime)
    slot_end = Column(DateTime)
    status = Column(String, default="booked")
    __table_args__ = (
        # DB-level guarantee against double-booking a doctor's slot, even
        # under concurrent requests (the check-then-insert race).
        UniqueConstraint("doctor_id", "slot_start", name="uq_appointment_doctor_slot"),
    )


def init_db(db_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
