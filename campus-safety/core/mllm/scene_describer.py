# Copyright (c) 2025 YOLO Course Design Contributors
# SPDX-License-Identifier: Apache-2.0

# ──────────────────────────────────────────────────────────
# 【MLLM子系统】scene_describer.py — 场景描述生成
# 依赖：inference_engine.py（推理引擎）
# 被调用：mllm_sidecar.py（旁路调用，不阻塞主循环）
# 核心职责：
#   ① 构建结构化 prompt（场景中的人数/行为/风险等级）
#   ② 传入关键帧图片 → VLM推理 → JSON格式的场景描述
#   ③ 解析返回的JSON（人数、主要行为、风险等级、建议）
# 输出示例：{"people_count":5, "behavior":"crowd", "risk":"medium"}
# ──────────────────────────────────────────────────────────

"""Scene describer — structured prompt templates and JSON output parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.mllm.scene_context_buffer import SceneContext

logger = logging.getLogger(__name__)

SCENE_DESCRIPTION_PROMPT = """你是校园安全监控AI助手。请结合监控画面和检测数据，识别以下常见校园安全场景：
- 有人摔倒（人员躺倒、姿态异常）
- 人群聚集（多人近距离停留、密度高）
- 车辆违停（机动车/电动车停在禁停区域）
- 奔跑追逐、打架斗殴、区域入侵

检测数据摘要：
{context}

请仔细观察画面中的：人物数量与分布、身体姿态、车辆、区域占用情况。

以JSON格式输出（仅JSON，无其他文字）：
{{
  "scene_summary": "一句话场景摘要",
  "activity_type": "正常/奔跑/人群聚集/有人摔倒/打架/车辆违停/区域入侵/其他",
  "confidence": 0.0到1.0,
  "anomaly_detected": true或false,
  "anomaly_details": "异常详情，无异常则为空字符串",
  "risk_level": "低/中/高",
  "suggested_action": "建议处置措施",
  "narrative": "50-100字中文场景描述，含人物特征、行为、环境，便于安保人员快速判断"
}}"""

ALARM_VERIFICATION_PROMPT = """你是校园安全告警验证助手。系统规则引擎产生了以下告警，请根据场景数据判断该告警是否为误报。

告警类型：{alarm_type}
告警详情：{alarm_details}
场景数据：{context}

请以JSON格式输出验证结果：
{{
  "verdict": "validate/dismiss/escalate",
  "confidence": 0.0-1.0,
  "reasoning": "用中文说明判断理由",
  "suggested_action": "用中文给出建议行动",
  "narrative": "用中文写一段50-100字的自然语言场景描述，包含人物特征和行为细节"
}}

仅输出JSON，不要输出其他内容。"""


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


def normalize_alarm_result(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": str(raw.get("verdict", "validate")),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": str(raw.get("reasoning", "")),
        "suggested_action": str(raw.get("suggested_action", "")),
        "narrative": str(raw.get("narrative", "")),
    }
