from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime,
    Float, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from web.database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    full_name       = Column(String(128), nullable=False)
    department      = Column(String(256), nullable=True)
    hashed_password = Column(String(256), nullable=False)
    is_active       = Column(Boolean, default=True)
    role            = Column(String(32), default="investigator")  
    lang            = Column(String(4), default="ru")             
    theme           = Column(String(8), default="dark")           
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_login      = Column(DateTime, nullable=True)

    cases = relationship("Case", back_populates="owner", cascade="all, delete-orphan")

    @property
    def initials(self) -> str:
        parts = [p for p in self.full_name.split() if p]
        if not parts:
            return self.username[:2].upper()
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[1][0]).upper()

    def __repr__(self):
        return f"<User {self.username}>"


class Case(Base):
    __tablename__ = "cases"

    id              = Column(Integer, primary_key=True, index=True)
    case_number     = Column(String(64), nullable=False, index=True)  
    search_date     = Column(String(32), nullable=True)               # DD.MM.YYYY
    address         = Column(String(512), nullable=True)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    owner_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner           = relationship("User", back_populates="cases")

    analysis        = relationship(
        "Analysis", back_populates="case", uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Case #{self.case_number}>"


class Analysis(Base):
    __tablename__ = "analyses"

    id              = Column(Integer, primary_key=True, index=True)
    case_id         = Column(Integer, ForeignKey("cases.id"), nullable=False)
    case            = relationship("Case", back_populates="analysis")

    video_filename  = Column(String(256), nullable=True)
    video_path      = Column(String(512), nullable=True)
    video_duration   = Column(Float, nullable=True)          

    status          = Column(String(32), default="pending")
    current_step    = Column(Integer, default=0)             
    step_label      = Column(String(64), nullable=True)
    error_message   = Column(Text, nullable=True)
    started_at      = Column(DateTime, nullable=True)
    finished_at     = Column(DateTime, nullable=True)

    participant_count       = Column(Integer, nullable=True)   
    participant_by_video    = Column(Integer, nullable=True)   
    participant_by_voice    = Column(Integer, nullable=True)   
  
    camera_obstructed       = Column(Boolean, nullable=True)
    camera_obstruction_sec  = Column(Float, nullable=True)

    language                = Column(String(8), nullable=True)
    full_transcript          = Column(Text, nullable=True)

    verdict                 = Column(String(64), nullable=True)
    violation_count         = Column(Integer, nullable=True)
    llm_summary             = Column(Text, nullable=True)
    model_used              = Column(String(64), nullable=True)

    result_json             = Column(JSON, nullable=True)

    created_at              = Column(DateTime, default=datetime.utcnow)

    @property
    def processing_time_seconds(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def status_label(self) -> dict:
        labels = {
            "pending":    {"ru": "Ожидание",  "kk": "Күту",     "en": "Pending"},
            "processing": {"ru": "Обработка", "kk": "Өңделуде", "en": "Processing"},
            "done":       {"ru": "Завершён",  "kk": "Аяқталды", "en": "Completed"},
            "error":      {"ru": "Ошибка",    "kk": "Қате",     "en": "Error"},
            "cancelled": {"ru": "Отменено","kk": "Бас тартылды","en": "Cancelled"},
        }
        return labels.get(self.status, labels["pending"])

    @property
    def verdict_label(self):
        if not self.verdict:
            return None
        labels = {
            "СООТВЕТСТВУЕТ": {
                "ru": "Соответствует", "kk": "Сәйкес келеді", "en": "Compliant",
            },
            "ЧАСТИЧНО_СООТВЕТСТВУЕТ": {
                "ru": "Частично соответствует",
                "kk": "Ішінара сәйкес келеді",
                "en": "Partially compliant",
            },
            "НЕ_СООТВЕТСТВУЕТ": {
                "ru": "Не соответствует", "kk": "Сәйкес келмейді", "en": "Non-compliant",
            },
        }
        return labels.get(self.verdict)

    def __repr__(self):
        return f"<Analysis case_id={self.case_id} status={self.status}>"