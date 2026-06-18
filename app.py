import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="健身助手", page_icon="💪", layout="wide")

if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("未检测到 DASHSCOPE_API_KEY。请在项目根目录创建 .env 文件并填入阿里云 DashScope API Key。")
    st.code("DASHSCOPE_API_KEY=你的密钥", language="bash")
    st.stop()

if "kb_sync_result" not in st.session_state:
    from rag.vector_store import ensure_knowledge_base_loaded

    with st.spinner("正在同步知识库..."):
        st.session_state["kb_sync_result"] = ensure_knowledge_base_loaded()

from agent.reactAgent import ReactAgent
from utils.user_data import invalidate_records_cache
from utils.user_profile import (
    clear_records,
    count_records,
    ensure_local_user,
    get_local_user_id,
    import_demo_records,
    load_profile,
    save_profile,
)

LOCAL_USER_ID = ensure_local_user()

st.title("健身助手")
st.caption("本地个人版 · 数据保存在本机，不会上传云端")

kb_result = st.session_state.get("kb_sync_result", {})
kb_loaded = len(kb_result.get("loaded", []))
kb_skipped = len(kb_result.get("skipped", []))
if kb_loaded:
    st.info(f"知识库已同步：本次新索引 {kb_loaded} 个文件，已有 {kb_skipped} 个文件跳过。")
elif kb_skipped:
    st.caption(f"知识库就绪（{kb_skipped} 个文档已索引）。")

GOAL_OPTIONS = ["", "增肌", "减脂", "力量提升", "综合体能", "康复恢复"]
ACTIVITY_OPTIONS = ["low", "medium", "high"]
GENDER_OPTIONS = ["男", "女"]

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

profile = load_profile(LOCAL_USER_ID)
record_count = count_records(LOCAL_USER_ID)

with st.sidebar:
    st.header("我的档案")
    st.caption(f"用户 ID：`{LOCAL_USER_ID}` · 训练记录 {record_count} 条")

    nickname = st.text_input("昵称", value=profile.get("nickname", ""))
    gender = st.selectbox(
        "性别",
        GENDER_OPTIONS,
        index=GENDER_OPTIONS.index(profile.get("gender", "男")),
    )
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=10, max_value=90, value=int(profile.get("age", 25)))
        height_cm = st.number_input("身高(cm)", min_value=100.0, max_value=250.0, value=float(profile.get("height_cm", 175)))
    with col2:
        weight_kg = st.number_input("体重(kg)", min_value=30.0, max_value=200.0, value=float(profile.get("weight_kg", 70)))
        body_fat_pct = st.number_input("体脂(%)", min_value=3.0, max_value=60.0, value=float(profile.get("body_fat_pct", 15)))

    training_goal = st.selectbox(
        "训练目标",
        GOAL_OPTIONS,
        index=GOAL_OPTIONS.index(profile.get("training_goal", ""))
        if profile.get("training_goal", "") in GOAL_OPTIONS
        else 0,
    )
    activity_level = st.selectbox(
        "活动强度",
        ACTIVITY_OPTIONS,
        index=ACTIVITY_OPTIONS.index(profile.get("activity_level", "medium"))
        if profile.get("activity_level", "medium") in ACTIVITY_OPTIONS
        else 1,
    )
    notes = st.text_area("个人备注", value=profile.get("notes", ""), height=80)

    if st.button("保存档案", use_container_width=True):
        save_profile(
            LOCAL_USER_ID,
            {
                "nickname": nickname,
                "gender": gender,
                "age": age,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "body_fat_pct": body_fat_pct,
                "training_goal": training_goal,
                "activity_level": activity_level,
                "notes": notes,
            },
        )
        st.success("档案已保存到本机")

    st.divider()
    st.header("训练数据")
    st.caption("数据目录：`data/users/local/`")

    if st.button("导入 10 天示例数据", use_container_width=True):
        try:
            imported = import_demo_records("U001", LOCAL_USER_ID)
            invalidate_records_cache(LOCAL_USER_ID)
            st.success(f"已导入 {imported} 条示例记录，并同步体测数据到档案")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.button("清空训练记录", use_container_width=True):
        clear_records(LOCAL_USER_ID)
        invalidate_records_cache(LOCAL_USER_ID)
        st.warning("训练记录已清空（档案保留）")
        st.rerun()

    st.divider()
    st.header("知识库")
    kb_stats = st.session_state.get("kb_sync_result", {})
    st.caption(
        f"已索引 {len(kb_stats.get('loaded', [])) + len(kb_stats.get('skipped', []))} 个文档"
        if kb_stats
        else "尚未同步"
    )
    if st.button("重新同步知识库", use_container_width=True):
        from rag.vector_store import ensure_knowledge_base_loaded

        with st.spinner("同步中..."):
            st.session_state["kb_sync_result"] = ensure_knowledge_base_loaded()
        st.rerun()

    st.divider()
    st.header("会话")
    if st.button("清空对话", use_container_width=True):
        st.session_state["message"] = []
        st.rerun()

st.divider()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("输入健身、补给或训练报告相关问题…")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("思考中..."):
        res_stream = st.session_state["agent"].execute_stream(st.session_state["message"])

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        if response_messages:
            st.session_state["message"].append(
                {"role": "assistant", "content": response_messages[-1]}
            )
