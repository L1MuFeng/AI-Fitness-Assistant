import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，支持直接运行本文件
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools import (
    get_user_id,
    get_user_profile,
    rag_summarize,
    Get_User_Training_Log,
    Get_Supplement_Intake_History,
    Analyze_Physiological_Data,
    Search_Equipment_Database,
    Calculate_Nutrient_Needs,
    Check_Supplement_Interaction,
    fill_context_for_report,
)
from utils.user_context import set_active_user_id
from utils.user_profile import get_local_user_id
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

_REPORT_KEYWORDS = (
    "报告",
    "复盘",
    "诊断",
    "训练日志",
    "阶段性",
    "瓶颈",
    "补剂无效",
    "睡眠变差",
)


def _is_report_request(query: str) -> bool:
    return any(keyword in query for keyword in _REPORT_KEYWORDS)


def _should_use_report_mode(messages: list[dict[str, str]]) -> bool:
    """当前轮或历史任一轮用户消息含报告关键词时，启用报告提示词。"""
    for msg in messages:
        if msg.get("role") == "user" and _is_report_request(msg.get("content", "")):
            return True
    return False


def _build_agent_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """将 UI 会话历史转为 Agent 可接受的 messages 格式。"""
    agent_messages = []
    for msg in history:
        content = msg.get("content", "").strip()
        if not content:
            continue
        role = msg.get("role", "user")
        if role == "assistant":
            role = "ai"
        agent_messages.append({"role": role, "content": content})
    return agent_messages


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [
                fill_context_for_report,
                get_user_id,
                get_user_profile,
                rag_summarize,
                Get_User_Training_Log,
                Get_Supplement_Intake_History,
                Analyze_Physiological_Data,
                Search_Equipment_Database,
                Calculate_Nutrient_Needs,
                Check_Supplement_Interaction,
            ],
            middleware = [monitor_tool, log_before_model, report_prompt_switch],
        )
    
    def execute_stream(self, history: list[dict[str, str]], user_id: str | None = None):
        """流式执行 Agent。history 为完整对话历史（含本轮用户消息）。"""
        agent_messages = _build_agent_messages(history)
        if not agent_messages:
            return

        user_id = user_id or get_local_user_id()
        set_active_user_id(user_id)
        input_dict = {"messages": agent_messages}

        for chunk in self.agent.stream(
            input_dict,
            stream_mode="messages",
            context={"report": _should_use_report_mode(history), "user_id": user_id},
        ):
            messages = chunk.get("messages", [])
            if not messages:
                continue
            latest_message = messages[-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

if __name__ == "__main__":
    agent = ReactAgent()
    demo_history = [{"role": "user", "content": "给我生成我的训练报告"}]
    for chunk in agent.execute_stream(demo_history):
        print(chunk, end="", flush=True)