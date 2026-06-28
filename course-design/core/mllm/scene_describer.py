# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

"""Scene describer — structured prompt templates and JSON output parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.mllm.scene_context_buffer import SceneContext

logger = logging.getLogger(__name__)

SCENE_DESCRIPTION_PROMPT = """你是一个校园安全监控场景分析助手。下面提供的是**同一监控画面按时间先后排列的连续多帧图像**（从早到晚），请结合这一组帧观察人物动作的变化过程，并参考规则引擎的检测结果进行分析。

规则引擎检测结果（重要先验，必须采信）：
{context}

关键说明：
- 这组图像是连续帧,请对比帧与帧之间人物姿态、肢体、位置的**变化**来判断动作。例如打架斗殴会表现为帧间快速挥拳、肢体大幅摆动、相互推搡;跌倒会表现为人体姿态从直立快速变为倒地。
- 规则引擎基于**多帧连续的人体骨架与运动轨迹**分析得出上述结果,可靠性高。单看某一帧可能只是两人靠近,容易误判为"交谈"或"张望",但结合多帧的动作变化和规则引擎结论应识别出真实行为。

因此请遵循以下规则：
1. 当规则引擎检测到异常事件（跌倒/打架/奔跑/聚集/入侵）且置信度较高（≥50%）时，activity_type 必须采用该检测结果，不要因为某一帧看起来平静就改判为"正常"或"闲聊"。
2. 你的任务是结合多帧画面,对该事件做出**合理的细节描述**（谁、在哪、和谁、什么姿态动作、动作如何变化），而不是推翻规则引擎的判断。
3. 仅当多帧画面有充分证据证明规则引擎明显误报时，才可在 anomaly_details 中说明理由。

请以JSON格式输出分析结果：
{{
  "scene_summary": "场景简要描述（一句话，需与检测到的事件一致）",
  "activity_type": "主要活动类型（正常/奔跑/聚集/跌倒/打架/入侵/其他）——优先采用规则引擎的高置信度检测结果",
  "confidence": 0.0-1.0的置信度,
  "anomaly_detected": true/false,
  "anomaly_details": "异常详情描述（如无异常则为空字符串）",
  "risk_level": "低/中/高",
  "suggested_action": "建议采取的行动",
  "narrative": "用中文写一段50-100字的自然语言场景描述。结合多帧的动作变化和检测到的事件类型，描述人物外观（服装颜色、体型姿态）、行为动作及其变化过程、人物互动、场景环境（室内外、光线、地面）。若检测到打架/跌倒等异常，需在描述中体现对应的危险行为。"
}}

仅输出JSON，不要输出其他内容。"""

ALARM_VERIFICATION_PROMPT = """你是校园安全告警验证助手。系统规则引擎产生了以下告警，请仔细观察监控画面并结合场景数据，判断该告警是否为误报。

告警类型：{alarm_type}
告警详情：{alarm_details}
场景数据（参考）：{context}

判断规则（verdict 字段只能填以下三个英文词之一）：
- validate：画面确实存在该告警描述的异常行为，告警成立
- dismiss：画面是正常行为被误判（如打闹、蹲下、奔跑锻炼），属于误报
- escalate：画面比告警描述的更危险，需要升级处理

严格要求：
1. reasoning 必须描述你在【本画面】中实际看到的人物动作与依据，不要套用任何固定句式。
2. reasoning 的结论必须与 verdict 一致：填 validate 就说明为何告警成立，填 dismiss 才说是正常行为。两者矛盾视为错误。
3. confidence 填 0 到 1 之间的具体小数（如 0.85），表示把握程度。

按以下 JSON 结构输出，把每个尖括号占位替换成基于本画面的真实内容：
{{
  "verdict": "<validate 或 dismiss 或 escalate 三选一>",
  "confidence": <0到1的小数>,
  "reasoning": "<你在本画面看到的具体动作依据，须与 verdict 一致>",
  "suggested_action": "<针对本画面的处置建议>",
  "narrative": "<50-100字中文场景描述，基于画面中的真实人物特征和行为>"
}}

只输出一个 JSON 对象，不要输出任何其他文字。"""


def parse_json_response(text: str) -> dict[str, Any] | None:
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse VLM JSON response: %s", text[:200])
    return None


def build_scene_prompt(context: SceneContext) -> str:
    return SCENE_DESCRIPTION_PROMPT.format(context=context.to_prompt_context())


def build_alarm_prompt(
    alarm_type: str, alarm_details: str, context: SceneContext
) -> str:
    return ALARM_VERIFICATION_PROMPT.format(
        alarm_type=alarm_type,
        alarm_details=alarm_details,
        context=context.to_prompt_context(),
    )


def normalize_scene_result(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_summary": str(raw.get("scene_summary", "")),
        "activity_type": str(raw.get("activity_type", "unknown")),
        "confidence": float(raw.get("confidence", 0.0)),
        "anomaly_detected": bool(raw.get("anomaly_detected", False)),
        "anomaly_details": str(raw.get("anomaly_details", "")),
        "risk_level": str(raw.get("risk_level", "低")),
        "suggested_action": str(raw.get("suggested_action", "")),
        "narrative": str(raw.get("narrative", "")),
    }


# Detector event type -> (activity_type label, risk level) used to reconcile a
# VLM scene result against high-confidence rule-engine detections.
_EVENT_TO_ACTIVITY = {
    "fight": ("打架", "高"),
    "fall": ("跌倒", "高"),
    "intrusion": ("入侵", "高"),
    "running": ("奔跑", "中"),
    "crowd": ("聚集", "中"),
}
# activity_type strings that indicate the VLM judged the scene as benign.
_BENIGN_ACTIVITIES = {"正常", "其他", "闲聊", "张望", "unknown", ""}


def reconcile_with_detector(
    result: dict[str, Any],
    event_counts: dict[str, int],
    event_confidences: dict[str, float],
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Override a benign VLM verdict when the motion-aware detector is confident.

    A 2B VLM reading a single still frame can misread an in-progress fight as
    casual standing/chatting. The rule engine analyses multi-frame skeleton
    motion and is the authority on *what activity* is happening, so when it
    reports a high-confidence anomaly we force the activity_type / anomaly
    fields to agree rather than let the VLM downgrade it to "正常/闲聊".
    """
    # Pick the highest-confidence detected anomaly above the threshold.
    best_type, best_conf = None, 0.0
    for etype, count in event_counts.items():
        if count <= 0 or etype not in _EVENT_TO_ACTIVITY:
            continue
        conf = event_confidences.get(etype, 0.0)
        if conf >= conf_threshold and conf > best_conf:
            best_type, best_conf = etype, conf
    if best_type is None:
        return result

    activity_label, risk = _EVENT_TO_ACTIVITY[best_type]
    if result.get("activity_type", "") in _BENIGN_ACTIVITIES:
        result["activity_type"] = activity_label
        result["anomaly_detected"] = True
        # The VLM may have written a benign anomaly_details ("无异常"); when the
        # detector overrides the verdict, replace it so it doesn't contradict.
        result["anomaly_details"] = (
            f"规则引擎以{best_conf:.0%}置信度检测到{activity_label}行为"
        )
        # Escalate risk only upward.
        if risk == "高" or result.get("risk_level", "低") == "低":
            result["risk_level"] = risk
        # The 2B VLM may still describe a calm scene (it cannot perceive the
        # fight from frames). Rewrite the summary/narrative so the prose does
        # not contradict the detected activity, while preserving whatever
        # visual detail the VLM did extract (people, clothing, environment).
        visual = (result.get("narrative") or result.get("scene_summary") or "").strip()
        visual = visual.rstrip("。.，, ")
        result["scene_summary"] = f"检测到{activity_label}行为（置信度{best_conf:.0%}）"
        if visual:
            result["narrative"] = (
                f"监控检测到{activity_label}行为（置信度{best_conf:.0%}）。"
                f"画面中：{visual}。请立即核查现场并采取相应措施。"
            )
        else:
            result["narrative"] = (
                f"监控检测到{activity_label}行为（置信度{best_conf:.0%}），"
                f"请立即核查现场并采取相应措施。"
            )
        result["reconciled_by_detector"] = True
    return result


_VALID_VERDICTS = ("validate", "dismiss", "escalate")


def normalize_alarm_result(raw: dict[str, Any]) -> dict[str, Any]:
    # Guard against a small model echoing the template placeholder
    # ("validate/dismiss/escalate") or an unknown word. An unrecognized verdict
    # falls back to "validate" — the safe choice for an alarm system, since it
    # keeps the alarm live rather than auto-dismissing on a parse failure.
    raw_verdict = str(raw.get("verdict", "validate")).strip().lower()
    verdict = raw_verdict if raw_verdict in _VALID_VERDICTS else "validate"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(raw.get("reasoning", "")),
        "suggested_action": str(raw.get("suggested_action", "")),
        "narrative": str(raw.get("narrative", "")),
    }
