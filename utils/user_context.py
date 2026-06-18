from contextvars import ContextVar

_active_user_id: ContextVar[str | None] = ContextVar("active_user_id", default=None)


def set_active_user_id(user_id: str | None) -> None:
    _active_user_id.set(user_id)


def get_active_user_id() -> str | None:
    return _active_user_id.get()
