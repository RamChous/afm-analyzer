import secrets
from datetime import datetime
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from web.database import get_db
from web.models import User

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# token -> user_id
SESSION_STORE: dict[str, int] = {}


def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_ctx.verify(plain, hashed)
    except ValueError:
        return False


def create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    SESSION_STORE[token] = user_id
    return token


def delete_session(token: str) -> None:
    SESSION_STORE.pop(token, None)


def get_user_id_from_session(token: str) -> Optional[int]:
    return SESSION_STORE.get(token)


def get_current_user(
    session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Raises 401 if not authenticated — caught globally and redirected to /login."""
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = get_user_id_from_session(session)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")

    return user


def get_current_user_optional(
    session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    try:
        return get_current_user(session=session, db=db)
    except HTTPException:
        return None


def authenticate_user(username: str, password: str, db: Session) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def login_user(user: User, db: Session) -> str:
    token = create_session(user.id)
    user.last_login = datetime.utcnow()
    db.commit()
    return token