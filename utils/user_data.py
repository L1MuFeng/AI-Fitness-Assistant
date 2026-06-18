import os

from utils.user_profile import (
    ensure_local_user,
    get_local_user_id,
    load_records,
)


_records_cache: dict[str, list[dict]] = {}


def invalidate_records_cache(user_id: str | None = None) -> None:
    user_id = user_id or get_local_user_id()
    _records_cache.pop(user_id, None)


def get_user_records(user_id: str | None = None) -> list[dict]:
    """读取用户训练与补剂记录（本地 JSON，按时间排序）。"""
    user_id = user_id or get_local_user_id()
    ensure_local_user()

    if user_id not in _records_cache:
        records = load_records(user_id)
        _records_cache[user_id] = sorted(records, key=lambda item: item.get("时间", ""))

    return _records_cache[user_id]
