import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "inspectra.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from web.models import Base, User  # local import avoids circular import
    from web.auth import hash_password

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            test_users = [
                User(
                    username="admin",
                    full_name="Администратор",
                    department="dept_prevention",
                    hashed_password=hash_password("admin123"),
                    is_active=True,
                    role="admin",
                    lang="ru",
                ),
                User(
                    username="asel",
                    full_name="Асель Сейткали",
                    department="dept_prevention",
                    hashed_password=hash_password("asel123"),
                    is_active=True,
                    role="investigator",
                    lang="ru",
                ),
                User(
                    username="daulet",
                    full_name="Даулет Нурланов",
                    department="dept_financial",
                    hashed_password=hash_password("daulet123"),
                    is_active=True,
                    role="investigator",
                    lang="kk",
                ),
            ]
            db.add_all(test_users)
            db.commit()
            print("Test users created: admin/admin123, asel/asel123, daulet/daulet123")
    finally:
        db.close()
