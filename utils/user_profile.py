import csv
import json
import os
import shutil
from datetime import datetime

from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

USERS_ROOT = get_abs_path(agent_conf.get("user_data_dir", "data/users"))
LOCAL_USER_ID = agent_conf.get("local_user_id", "local")
DEMO_DATA_PATH = get_abs_path(
    agent_conf.get(
        "demo_data_path",
        agent_conf.get("external_data_path", "data/external/10用户10天训练与补剂数据.csv"),
    )
)

DEFAULT_PROFILE = {
    "nickname": "我",
    "gender": "男",
    "age": 25,
    "height_cm": 175.0,
    "weight_kg": 70.0,
    "body_fat_pct": 15.0,
    "training_goal": "",
    "activity_level": "medium",
    "notes": "",
}


def get_local_user_id() -> str:
    return LOCAL_USER_ID


def _user_dir(user_id: str) -> str:
    return os.path.join(USERS_ROOT, user_id)


def _profile_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "profile.json")


def _records_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "records.json")


def ensure_local_user() -> str:
    """首次运行时创建本地用户目录与默认档案。"""
    user_id = get_local_user_id()
    os.makedirs(_user_dir(user_id), exist_ok=True)
    if not os.path.exists(_profile_path(user_id)):
        profile = {
            **DEFAULT_PROFILE,
            "user_id": user_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(_profile_path(user_id), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    if not os.path.exists(_records_path(user_id)):
        with open(_records_path(user_id), "w", encoding="utf-8") as f:
            json.dump([], f)
    return user_id


def load_profile(user_id: str | None = None) -> dict:
    user_id = user_id or get_local_user_id()
    ensure_local_user()
    path = _profile_path(user_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**DEFAULT_PROFILE, **data, "user_id": user_id}
    return {**DEFAULT_PROFILE, "user_id": user_id}


def save_profile(user_id: str, updates: dict) -> dict:
    ensure_local_user()
    profile = load_profile(user_id)
    allowed = set(DEFAULT_PROFILE) | {"user_id"}
    profile.update({k: v for k, v in updates.items() if k in allowed})
    profile["user_id"] = user_id
    profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_profile_path(user_id), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


def format_body_data(profile: dict) -> str:
    gender = profile.get("gender", "男")
    age = profile.get("age", 25)
    height = profile.get("height_cm", 175)
    weight = profile.get("weight_kg", 70)
    body_fat = profile.get("body_fat_pct", 15)
    return f"{gender},{age}岁,{height}cm,{weight}kg,体脂{body_fat}%"


def get_profile_summary(user_id: str | None = None, latest_body_data: str | None = None) -> dict:
    user_id = user_id or get_local_user_id()
    profile = load_profile(user_id)
    summary = {
        "用户ID": user_id,
        "昵称": profile.get("nickname") or "未设置",
        "训练目标": profile.get("training_goal") or "未设置",
        "活动强度": profile.get("activity_level") or "medium",
        "个人备注": profile.get("notes") or "无",
        "档案更新时间": profile.get("updated_at", profile.get("created_at", "尚未保存")),
    }
    if latest_body_data:
        summary["最新体测数据"] = latest_body_data
    else:
        summary["档案体测数据"] = format_body_data(profile)
    return summary


def load_records(user_id: str | None = None) -> list[dict]:
    user_id = user_id or get_local_user_id()
    ensure_local_user()
    path = _records_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_records(user_id: str, records: list[dict]) -> None:
    os.makedirs(_user_dir(user_id), exist_ok=True)
    with open(_records_path(user_id), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def count_records(user_id: str | None = None) -> int:
    return len(load_records(user_id))


def import_demo_records(demo_user_id: str = "U001", target_user_id: str | None = None) -> int:
    """从演示 CSV 导入指定用户的数据到本地个人记录。"""
    target_user_id = target_user_id or get_local_user_id()
    if not os.path.exists(DEMO_DATA_PATH):
        raise FileNotFoundError(f"演示数据不存在：{DEMO_DATA_PATH}")

    imported: list[dict] = []
    with open(DEMO_DATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["用户ID"].strip() != demo_user_id:
                continue
            imported.append(
                {
                    "时间": row["时间"].strip(),
                    "身体数据": row["用户身体数据"].strip(),
                    "训练计划": row["用户锻炼计划"].strip(),
                    "摄入情况": row["补剂摄入情况"].strip(),
                }
            )

    if not imported:
        raise ValueError(f"演示数据中未找到 {demo_user_id}")

    imported.sort(key=lambda item: item["时间"])
    save_records(target_user_id, imported)

    latest = imported[-1]
    body = _parse_demo_body(latest["身体数据"])
    if body:
        save_profile(target_user_id, body)

    return len(imported)


def _parse_demo_body(body_str: str) -> dict:
    import re

    updates = {}
    if "女" in body_str.split(",")[0]:
        updates["gender"] = "女"
    else:
        updates["gender"] = "男"
    if match := re.search(r"(\d+)岁", body_str):
        updates["age"] = int(match.group(1))
    if match := re.search(r"(\d+(?:\.\d+)?)cm", body_str):
        updates["height_cm"] = float(match.group(1))
    if match := re.search(r"(\d+(?:\.\d+)?)kg", body_str):
        updates["weight_kg"] = float(match.group(1))
    if match := re.search(r"体脂([\d.]+)%", body_str):
        updates["body_fat_pct"] = float(match.group(1))
    return updates


def clear_records(user_id: str | None = None) -> None:
    save_records(user_id or get_local_user_id(), [])


def export_user_data_backup(export_dir: str | None = None) -> str:
    """备份本地用户 profile + records 到指定目录。"""
    user_id = get_local_user_id()
    export_dir = export_dir or get_abs_path("data/backups")
    os.makedirs(export_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(export_dir, f"{user_id}_{stamp}")
    shutil.copytree(_user_dir(user_id), dest)
    return dest
