import json
import os
import re
import sys
from pathlib import Path
from typing import Union

# 将项目根目录加入模块搜索路径，支持直接运行本文件或从任意工作目录导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
from utils.user_context import get_active_user_id
from utils.user_data import get_user_records
from utils.user_profile import (
    format_body_data,
    get_local_user_id,
    get_profile_summary,
    load_profile,
)

rag = RagSummarizeService()

_ACTIVITY_MULTIPLIERS = {
    "low": 1.2,
    "sedentary": 1.2,
    "medium": 1.55,
    "moderate": 1.55,
    "high": 1.725,
    "intense": 1.725,
}

_SUPPLEMENT_INTERACTIONS: dict[frozenset[str], str] = {
    frozenset({"钙", "铁"}): "钙与铁存在肠道吸收竞争，建议间隔 2 小时以上服用。",
    frozenset({"钙", "锌"}): "钙与锌存在吸收竞争，建议间隔 2 小时以上服用。",
    frozenset({"铁", "锌"}): "铁与锌存在吸收竞争，建议间隔 2 小时以上服用。",
    frozenset({"咖啡因", "肌酸"}): "可同服；肌酸建议随碳水摄入，咖啡因不影响肌酸长期饱和。",
    frozenset({"咖啡因", "前充能泵"}): "前充能泵通常含高剂量咖啡因，注意每日总咖啡因不超过 400mg。",
}


def _read_data_text(filename: str) -> str:
    path = get_abs_path(os.path.join("data", filename))
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def _get_user_records(user_id: str) -> list[dict]:
    return get_user_records(user_id)


def _missing_user_message(user_id: str) -> str:
    profile = load_profile(user_id)
    return (
        f"用户 {user_id}（{profile.get('nickname', '我')}）尚无训练与补剂记录。"
        "请在应用侧边栏填写个人档案，或导入示例数据后再生成报告。"
        f"当前档案体测：{format_body_data(profile)}"
    )


def _parse_body_data(body_str: str) -> dict:
    gender = "女" if "女" in body_str.split(",")[0] else "男"
    age = re.search(r"(\d+)岁", body_str)
    height = re.search(r"(\d+(?:\.\d+)?)cm", body_str)
    weight = re.search(r"(\d+(?:\.\d+)?)kg", body_str)
    body_fat = re.search(r"体脂([\d.]+)%", body_str)
    return {
        "性别": gender,
        "年龄": int(age.group(1)) if age else None,
        "身高_cm": float(height.group(1)) if height else None,
        "体重_kg": float(weight.group(1)) if weight else None,
        "体脂率_%": float(body_fat.group(1)) if body_fat else None,
    }


def _is_rest_day(plan: str) -> bool:
    rest_keywords = ("休息日", "休息或瑜伽", "恢复：")
    return any(k in plan for k in rest_keywords) or plan.strip() == ""


def _is_high_intensity(plan: str) -> bool:
    if re.search(r"(8[5-9]|9\d|100)%1RM", plan):
        return True
    high_rpe_keywords = ("5×1", "5×2", "5×3", "3×1", "3×2", "3×3", "HIIT", "AMRAP", "间歇训练")
    return any(k in plan for k in high_rpe_keywords)


def _parse_intake_items(intake_str: str) -> list[dict]:
    if not intake_str.strip():
        return []
    items = []
    for part in intake_str.split("|"):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(.+?)\((.+?)\)-(.+)$", part)
        if match:
            items.append({
                "名称": match.group(1).strip(),
                "剂量": match.group(2).strip(),
                "时机": match.group(3).strip(),
            })
        else:
            items.append({"名称": part, "剂量": "", "时机": ""})
    return items


def _extract_protein_grams(dose: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*g", dose, re.I)
    return float(match.group(1)) if match else 0.0


def _extract_caffeine_mg(dose: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*mg", dose, re.I)
    return float(match.group(1)) if match else 0.0


def _normalize_supplement_name(name: str) -> str:
    aliases = {
        "蛋白粉": "乳清蛋白",
        "乳清": "乳清蛋白",
        "分离乳清蛋白": "乳清蛋白",
        "乳清分离蛋白": "乳清蛋白",
        "pre": "前充能泵",
        "pre-workout": "前充能泵",
        "训练前": "前充能泵",
        "beta-alanine": "β-丙氨酸",
        "丙氨酸": "β-丙氨酸",
    }
    normalized = name.strip()
    return aliases.get(normalized.lower(), normalized)


def _parse_supplement_list(supplement_list: Union[list[str], str]) -> list[str]:
    if isinstance(supplement_list, str):
        items = supplement_list.replace("，", ",").replace("、", ",").split(",")
    else:
        items = supplement_list
    return [_normalize_supplement_name(item) for item in items if item.strip()]


def _search_data_text(text: str, keywords: list[str], context_lines: int = 3) -> list[str]:
    lines = text.splitlines()
    hits: list[str] = []
    for i, line in enumerate(lines):
        if any(k.lower() in line.lower() for k in keywords):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            block = "\n".join(lines[start:end]).strip()
            if block and block not in hits:
                hits.append(block)
    return hits


@tool(
    description=(
        "获取当前本地用户的 ID。"
        "本助手为单机个人版，返回当前档案对应的 user_id（通常为 local）。"
    )
)
def get_user_id() -> str:
    return get_active_user_id() or get_local_user_id()


@tool(
    description=(
        "获取当前用户的个人档案，包含昵称、性别、年龄、身高体重、体脂、训练目标与备注。"
        "无训练记录时也会返回档案体测数据，生成个性化建议前应优先调用。"
    )
)
def get_user_profile(user_id: str) -> str:
    records = _get_user_records(user_id)
    profile = load_profile(user_id)

    if records:
        latest_body = records[-1]["身体数据"]
        summary = get_profile_summary(user_id, latest_body_data=latest_body)
        data_period = f"{records[0]['时间']} ~ {records[-1]['时间']}"
        record_count = len(records)
    else:
        summary = get_profile_summary(user_id)
        data_period = "尚无训练记录"
        record_count = 0

    result = {
        "个人档案": summary,
        "训练记录条数": record_count,
        "数据周期": data_period,
        "个性化提示": (
            f"用户训练目标为「{summary['训练目标']}」，活动强度为「{summary['活动强度']}」。"
            if profile.get("training_goal")
            else "用户尚未填写训练目标，可结合档案体测数据给出通用建议。"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="从向量存储中检索参考资料，用于回答健身补给与营养相关问题")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="检索用户最近一个周期的训练监控数据，包含1RM变化、训练量、频次及RPE")
def Get_User_Training_Log(user_id: str) -> str:
    records = _get_user_records(user_id)
    if not records:
        return _missing_user_message(user_id)

    training_days = []
    rest_days = []
    high_intensity_days = []
    daily_plans = []
    squat_bench_deadlift_volume: list[str] = []

    for rec in records:
        plan = rec["训练计划"]
        time_key = rec["时间"]
        if _is_rest_day(plan):
            rest_days.append(time_key)
        else:
            training_days.append(time_key)
            daily_plans.append({"时间": time_key, "训练内容": plan})
            if _is_high_intensity(plan):
                high_intensity_days.append(time_key)
            for lift in ("深蹲", "卧推", "硬拉"):
                if lift in plan:
                    sets_reps = re.findall(rf"{lift}[^，,]*?(\d+×\d+)", plan)
                    if sets_reps:
                        squat_bench_deadlift_volume.append(f"{time_key} {lift} {sets_reps[0]}")

    first_body = _parse_body_data(records[0]["身体数据"])
    last_body = _parse_body_data(records[-1]["身体数据"])

    period_days = len(records)
    training_freq = f"{len(training_days)}次/{period_days}天"

    one_rm_trend = {
        "体重变化": (
            f"{first_body['体重_kg']}kg → {last_body['体重_kg']}kg"
            if first_body["体重_kg"] and last_body["体重_kg"]
            else "数据不足"
        ),
        "体脂变化": (
            f"{first_body['体脂率_%']}% → {last_body['体脂率_%']}%"
            if first_body["体脂率_%"] and last_body["体脂率_%"]
            else "数据不足"
        ),
        "大重量动作容量记录": squat_bench_deadlift_volume or ["近周期内未记录深蹲/卧推/硬拉结构化组次"],
        "备注": "CSV 未包含实测 1RM，以上为训练计划中的组次与体成分趋势参考",
    }

    result = {
        "用户ID": user_id,
        "数据周期": f"{records[0]['时间']} ~ {records[-1]['时间']}",
        "训练频次": training_freq,
        "休息日": rest_days,
        "高负荷训练日": high_intensity_days,
        "高RPE训练日频次": f"{len(high_intensity_days)}次（含≥85%1RM或高强度模式）",
        "1RM与体成分趋势": one_rm_trend,
        "每日训练明细": daily_plans,
        "关节不适自述": "当前数据源未记录关节不适字段",
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="调取用户近期补给品与膳食营养素摄入日志，包含剂量、时机及耐受反馈")
def Get_Supplement_Intake_History(user_id: str) -> str:
    records = _get_user_records(user_id)
    if not records:
        return _missing_user_message(user_id)

    daily_intake_log = []
    supplement_counter: dict[str, int] = {}
    caffeine_days = 0
    protein_supplement_days = 0
    total_protein_from_supplements = 0.0
    caffeine_total_mg = 0.0

    for rec in records:
        intake_str = rec["摄入情况"]
        items = _parse_intake_items(intake_str)
        day_protein = 0.0
        day_caffeine = 0.0
        for item in items:
            name = item["名称"]
            supplement_counter[name] = supplement_counter.get(name, 0) + 1
            if "蛋白" in name:
                day_protein += _extract_protein_grams(item["剂量"])
            if "咖啡因" in name:
                day_caffeine += _extract_caffeine_mg(item["剂量"])
        if day_protein > 0:
            protein_supplement_days += 1
            total_protein_from_supplements += day_protein
        if day_caffeine > 0:
            caffeine_days += 1
            caffeine_total_mg += day_caffeine
        daily_intake_log.append({
            "时间": rec["时间"],
            "摄入明细": items if items else ["当日无补剂记录"],
        })

    avg_supplement_protein = (
        round(total_protein_from_supplements / protein_supplement_days, 1)
        if protein_supplement_days
        else 0
    )
    body = _parse_body_data(records[-1]["身体数据"])
    weight = body.get("体重_kg") or 70
    estimated_total_protein = round(weight * 1.8, 1)

    result = {
        "用户ID": user_id,
        "数据周期": f"{records[0]['时间']} ~ {records[-1]['时间']}",
        "每日摄入日志": daily_intake_log,
        "补剂出现频次": supplement_counter,
        "每日蛋白质估算": {
            "补剂来源日均蛋白_g": avg_supplement_protein,
            "结合饮食估算总蛋白_g": estimated_total_protein,
            "参考标准_g_per_kg": "1.6-2.2（增肌）/ 2.3-3.1（减脂保肌）",
            "氮平衡评估": (
                "正氮平衡倾向（蛋白摄入充足）"
                if estimated_total_protein >= weight * 1.6
                else "蛋白摄入可能不足，建议提高膳食或补剂蛋白"
            ),
        },
        "咖啡因摄入": {
            "摄入天数": caffeine_days,
            "累计摄入_mg": caffeine_total_mg,
            "日均_mg": round(caffeine_total_mg / max(caffeine_days, 1), 1),
            "安全提示": "建议每日咖啡因总量 ≤ 400mg，午后避免摄入",
        },
        "胃肠道耐受记录": "当前数据源未记录耐受反馈，需用户主观反馈补充",
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="提取用户最新体成分、静息心率、睡眠质量等生理指标，评估恢复与CNS疲劳状态")
def Analyze_Physiological_Data(user_id: str) -> str:
    records = _get_user_records(user_id)
    if not records:
        return _missing_user_message(user_id)

    first_body = _parse_body_data(records[0]["身体数据"])
    last_body = _parse_body_data(records[-1]["身体数据"])

    training_count = sum(1 for r in records if not _is_rest_day(r["训练计划"]))
    high_intensity_count = sum(
        1 for r in records if not _is_rest_day(r["训练计划"]) and _is_high_intensity(r["训练计划"])
    )
    rest_count = len(records) - training_count
    training_ratio = training_count / len(records)

    weight_delta = (last_body["体重_kg"] or 0) - (first_body["体重_kg"] or 0)
    fat_delta = (last_body["体脂率_%"] or 0) - (first_body["体脂率_%"] or 0)

    if high_intensity_count >= 4 and rest_count < 3:
        cns_score = min(90, 55 + high_intensity_count * 5)
        overtraining_risk = "偏高（高强度训练密集且休息不足）"
    elif high_intensity_count >= 2:
        cns_score = 50 + high_intensity_count * 3
        overtraining_risk = "中等"
    else:
        cns_score = 35
        overtraining_risk = "较低"

    if fat_delta < -0.5 and weight_delta <= 0:
        body_composition = "体脂下降且体重稳定，体成分优化趋势良好"
    elif fat_delta > 0.5:
        body_composition = "体脂率上升，需关注饮食热量与训练恢复"
    else:
        body_composition = "体成分变化平稳"

    indicators = [
        f"最新体成分：{records[-1]['身体数据']}",
        f"周期体成分变化：体重 {first_body['体重_kg']}→{last_body['体重_kg']}kg，"
        f"体脂 {first_body['体脂率_%']}%→{last_body['体脂率_%']}%",
        f"骨骼肌与脂肪趋势：{body_composition}",
        f"水分调节状态：数据源未含 InBody 水分指标，建议结合体测仪补充",
        f"静息心率/睡眠质量：数据源未记录 RHR 与睡眠，无法直接评估",
        f"CNS 疲劳指数（估算）：{cns_score}/100",
        f"过度训练风险：{overtraining_risk}（训练日占比 {training_ratio:.0%}，"
        f"高强度日 {high_intensity_count} 次，休息 {rest_count} 天）",
    ]

    result = {
        "用户ID": user_id,
        "数据周期": f"{records[0]['时间']} ~ {records[-1]['时间']}",
        "生理指标分析": indicators,
        "恢复建议方向": (
            "增加低强度恢复日或减载周"
            if cns_score >= 65
            else "当前训练恢复平衡尚可，可维持现有节奏"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="根据品牌和设备类型检索健身装备的核心规格、材质、适用人群及市场参考价格")
def Search_Equipment_Database(brand: str, type: str) -> str:
    guide_text = _read_data_text("健身补给选购指南.txt")
    keywords = [brand.strip(), type.strip()]
    if "腰带" in type or "belt" in type.lower():
        keywords.extend(["腹内压", "IAP", "护具"])
    if "护膝" in type or "sleeve" in type.lower():
        keywords.extend(["护膝", "氯丁橡胶", "关节"])
    if "拉力" in type or "strap" in type.lower():
        keywords.extend(["握力", "硬拉", "引体"])

    local_hits = _search_data_text(guide_text, keywords)
    rag_query = f"{brand} {type} 健身装备 规格 材质 适用人群 承重 价格 护具"
    rag_result = rag.rag_summarize(rag_query)

    result = {
        "检索条件": {"品牌": brand, "类型": type},
        "本地参考资料摘要": local_hits or ["本地知识库主要为补给选购指南，未找到装备专条"],
        "向量检索补充": rag_result,
        "选购提示": (
            "大重量深蹲/硬拉建议评估腰带厚度（10mm/13mm）与护膝刚度；"
            "护具选择需结合动作力臂与关节稳定性需求。"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="基于体重、身高、活动强度与训练目标，计算BMR、TDEE及三大营养素推荐摄入量")
def Calculate_Nutrient_Needs(
    weight: Union[float, str],
    height: Union[float, str],
    activity_level: str,
    goal: str,
) -> str:
    weight_kg = float(weight)
    height_cm = float(height)
    dosage_text = _read_data_text("健身补剂建议补充量.txt")

    age = 30
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    activity_key = activity_level.strip().lower().split("/")[0]
    multiplier = _ACTIVITY_MULTIPLIERS.get(activity_key, 1.55)
    tdee = bmr * multiplier

    goal_key = goal.strip().lower().split("/")[0]
    if goal_key in {"fat_loss", "减脂"}:
        target_calories = tdee * 0.85
        protein_g = round(weight_kg * 2.2, 1)
        fat_g = round(weight_kg * 0.9, 1)
        protein_ref = "2.3-3.1 g/kg/d（减脂保肌）"
    elif goal_key in {"muscle_gain", "增肌"}:
        target_calories = tdee * 1.1
        protein_g = round(weight_kg * 2.0, 1)
        fat_g = round(weight_kg * 1.0, 1)
        protein_ref = "1.6-2.2 g/kg/d（增肌）"
    else:
        target_calories = tdee * 1.05
        protein_g = round(weight_kg * 2.0, 1)
        fat_g = round(weight_kg * 1.0, 1)
        protein_ref = "1.6-2.2 g/kg/d（力量训练）"

    protein_cal = protein_g * 4
    fat_cal = fat_g * 9
    carb_cal = max(target_calories - protein_cal - fat_cal, 0)
    carb_g = round(carb_cal / 4, 1)

    dosage_hits = _search_data_text(
        dosage_text,
        ["蛋白质", "肌酸", "咖啡因", "鱼油", "维生素"],
        context_lines=2,
    )

    result = {
        "BMR_kcal": round(bmr, 1),
        "TDEE_kcal": round(tdee, 1),
        "目标热量_kcal": round(target_calories, 1),
        "蛋白质_g": protein_g,
        "碳水化合物_g": carb_g,
        "脂肪_g": fat_g,
        "热量占比": {
            "蛋白质": f"{round(protein_cal / target_calories * 100, 1)}%",
            "碳水化合物": f"{round(carb_cal / target_calories * 100, 1)}%",
            "脂肪": f"{round(fat_cal / target_calories * 100, 1)}%",
        },
        "蛋白质参考标准": protein_ref,
        "计算假设": "默认男性、30岁；如需更精确请补充年龄与性别",
        "参考资料摘录": dosage_hits[:3],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="评估多种补给品或药物组合的安全性、摄入时机及潜在吸收竞争或副反应风险")
def Check_Supplement_Interaction(supplement_list: Union[list[str], str]) -> str:
    supplements = _parse_supplement_list(supplement_list)
    if not supplements:
        return "请提供至少一种补给品名称。"

    guide_text = _read_data_text("健身补给选购指南.txt")
    dosage_text = _read_data_text("健身补剂建议补充量.txt")

    warnings: list[str] = []
    for i, item_a in enumerate(supplements):
        for item_b in supplements[i + 1 :]:
            pair = frozenset({item_a, item_b})
            if pair in _SUPPLEMENT_INTERACTIONS:
                warnings.append(f"[{item_a} + {item_b}] {_SUPPLEMENT_INTERACTIONS[pair]}")

    caffeine_sources = sum(1 for s in supplements if "咖啡因" in s or "前充能泵" in s)
    if caffeine_sources >= 2:
        warnings.append(
            "[咖啡因叠加] 多种含咖啡因来源并用，注意每日总量 ≤ 400mg，避免影响睡眠。"
        )

    reference_blocks = []
    for sup in supplements:
        hits = _search_data_text(guide_text, [sup], context_lines=4)
        hits += _search_data_text(dosage_text, [sup], context_lines=3)
        if hits:
            reference_blocks.append({"补剂": sup, "资料摘录": hits[:2]})

    rag_query = (
        f"评估以下补给品同时服用的安全性、最佳摄入时机及相互作用："
        f"{', '.join(supplements)}"
    )
    rag_detail = rag.rag_summarize(rag_query)

    result = {
        "在服清单": supplements,
        "已知相互作用提示": warnings or ["未发现已知高风险矿物质竞争组合"],
        "本地资料摘录": reference_blocks,
        "向量检索分析": rag_detail,
        "通用建议": (
            "矿物质（钙/铁/锌）避免同服；训前咖啡因与睡前酪蛋白注意时序；"
            "肌酸建议维持 3-5g/d，无需周期。"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(
    description=(
        "切换至「个人运动表现与营养膳食诊断报告」撰写模式。"
        "当用户需要生成/查询训练报告、阶段性复盘、瓶颈诊断、补剂效果评估等报告型输出时，"
        "必须作为第一步调用本工具（无入参），再调用 get_user_id 及数据类工具拉取用户信息。"
    )
)
def fill_context_for_report() -> str:
    return "已切换至报告撰写模式，请继续调用 get_user_id 及 Get_User_Training_Log 等工具获取数据。"


ALL_TOOLS = [
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
]

if __name__ == "__main__":
    from utils.user_profile import ensure_local_user

    uid = ensure_local_user()
    print(json.dumps(get_user_profile.invoke({"user_id": uid}), ensure_ascii=False, indent=2))
