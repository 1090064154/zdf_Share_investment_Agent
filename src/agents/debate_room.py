from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
import ast
import logging

# 获取日志记录器
logger = logging.getLogger('debate_room')


@resilient_agent(critical=True)
@agent_endpoint("debate_room", "辩论室，分析多空双方观点，得出平衡的投资结论")
def debate_room_agent(state: AgentState):
    """Facilitates debate between bull and bear researchers to reach a balanced conclusion."""
    show_workflow_status("辩论室")
    show_reasoning = state["metadata"]["show_reasoning"]
    logger.info("="*50)
    logger.info("⚖️ [DEBATE_ROOM] 开始辩论室分析")
    logger.info("="*50)

    # 收集所有研究员信息 - 向前兼容设计（添加防御性检查）
    researcher_messages = {}
    for msg in state["messages"]:
        # 添加防御性检查，确保 msg 和 msg.name 不为 None
        if msg is None:
            continue
        if not hasattr(msg, 'name') or msg.name is None:
            continue
        if isinstance(msg.name, str) and msg.name.startswith("researcher_") and msg.name.endswith("_agent"):
            researcher_messages[msg.name] = msg
            logger.debug(f"收集到研究员信息: {msg.name}")

    # 确保至少有看多和看空两个研究员
    if "researcher_bull_agent" not in researcher_messages or "researcher_bear_agent" not in researcher_messages:
        logger.error(
            "缺少必要的研究员数据: researcher_bull_agent 或 researcher_bear_agent")
        raise ValueError(
            "Missing required researcher_bull_agent or researcher_bear_agent messages")

# 处理研究员数据
    researcher_data = {}
    for name, msg in researcher_messages.items():
        # 添加防御性检查，确保 msg.content 不为 None
        if not hasattr(msg, 'content') or msg.content is None:
            logger.warning(f"研究员 {name} 的消息内容为空")
            continue
        try:
            data = json.loads(msg.content)
            logger.debug(f"成功解析 {name} 的 JSON 内容")
        except (json.JSONDecodeError, TypeError):
            try:
                data = ast.literal_eval(msg.content)
                logger.debug(f"通过 ast.literal_eval 解析 {name} 的内容")
            except (ValueError, SyntaxError, TypeError):
                # 如果无法解析内容，跳过此消息
                logger.warning(f"无法解析 {name} 的消息内容，已跳过")
                continue
        researcher_data[name] = data

    # ============================================================
    # Step 2: 获取研究员论点
    # ============================================================
    logger.info("📥 Step 2: 获取研究员论点...")
    if "researcher_bull_agent" not in researcher_data or "researcher_bear_agent" not in researcher_data:
        logger.error("无法解析必要的研究员数据")
        raise ValueError(
            "Could not parse required researcher_bull_agent or researcher_bear_agent messages")

    bull_thesis = researcher_data["researcher_bull_agent"]
    bear_thesis = researcher_data["researcher_bear_agent"]
    bull_confidence = bull_thesis.get("confidence", 0)
    bear_confidence = bear_thesis.get("confidence", 0)

    logger.info("────────────────────────────────────────────────────────")
    logger.info("📊 研究员论点汇总:")
    logger.info(f"  🐂 看多研究员: confidence={bull_confidence}")
    logger.info(f"     论点数: {len(bull_thesis.get('thesis_points', []))}")
    logger.info(f"  🐻 看空研究员: confidence={bear_confidence}")
    logger.info(f"     风险点数: {len(bear_thesis.get('risk_points', []))}")
    logger.info("────────────────────────────────────────────────────────")

    # 分析辩论观点
    debate_summary = []
    debate_summary.append("看多观点:")
    for point in bull_thesis.get("thesis_points", []):
        debate_summary.append(f"+ {point}")

    debate_summary.append("\n看空观点:")
    for point in bear_thesis.get("thesis_points", []):
        debate_summary.append(f"- {point}")

# ============================================================
    # Step 3: 构建prompt并调用LLM进行辩论
    # ============================================================
    logger.info("🔧 Step 3: 构建prompt并调用LLM...")
    all_perspectives = {}
    for name, data in researcher_data.items():
        perspective = data.get("perspective", name.replace(
            "researcher_", "").replace("_agent", ""))
        all_perspectives[perspective] = {
            "confidence": data.get("confidence", 0),
            "thesis_points": data.get("thesis_points", data.get("risk_points", []))
        }

    # 构建发送给 LLM 的提示
    llm_prompt = """
你是一位专业的金融分析师，请分析以下投资研究员的观点，并给出你的第三方分析:

"""
    for perspective, data in all_perspectives.items():
        llm_prompt += f"\n{perspective.upper()} 观点 (置信度: {data['confidence']}):\n"
        for point in data['thesis_points']:
            llm_prompt += f"- {point}\n"

    llm_prompt += """
请提供以下格式的 JSON 回复:
{
    "analysis": "你的详细分析，评估各方观点的优劣，并指出你认为最有说服力的论点",
    "score": 0.5,  // 你的评分，从 -1.0(极度看空) 到 1.0(极度看多)，0 表示中性
    "reasoning": "你给出这个评分的简要理由"
}

务必确保你的回复是有效的 JSON 格式，且包含上述所有字段。请使用中文回复。
"""

    # 调用 LLM 获取第三方观点
    llm_response = None
    llm_analysis = None
    llm_score = 0  # 默认为中性
    try:
        logger.info("🤖 调用LLM进行第三方辩论分析...")
        messages = [
            {"role": "system", "content": "你是一位专业的金融分析师。请用中文提供你的分析。"},
            {"role": "user", "content": llm_prompt}
        ]

        # 使用log_llm_interaction装饰器记录LLM交互
        llm_response = log_llm_interaction(state)(
            lambda: get_chat_completion(
                messages,
                max_retries=1,
                initial_retry_delay=0.5,
            )
        )()

        logger.info("✅ LLM辩论分析完成")

        # 解析 LLM 返回的 JSON
        if llm_response:
            try:
                # 尝试提取 JSON 部分
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                    llm_score = float(llm_analysis.get("score", 0))
                    # 确保分数在有效范围内
                    llm_score = max(min(llm_score, 1.0), -1.0)
                    logger.info(f"成功解析 LLM 回复，评分: {llm_score}")
                    logger.debug(
                        f"LLM 分析内容: {llm_analysis.get('analysis', '未提供分析')[:100]}...")
            except Exception as e:
                # 如果解析失败，记录错误并使用默认值
                logger.error(f"解析 LLM 回复失败: {e}")
                llm_analysis = {"analysis": "Failed to parse LLM response",
                                "score": 0, "reasoning": "Parsing error"}
    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        llm_analysis = {"analysis": "LLM API call failed",
                        "score": 0, "reasoning": "API error"}

    # ============================================================
    # Step 4: 计算混合置信度
    # ============================================================
    logger.info("🔢 Step 4: 计算混合置信度...")
    confidence_diff = bull_confidence - bear_confidence

    # 发送研究员论点汇总到前端
    bull_points = bull_thesis.get("thesis_points", []) if bull_thesis else []
    bear_points = bear_thesis.get("risk_points", []) if bear_thesis else []
    show_agent_reasoning({
        "看多置信度": f"{bull_confidence:.2%}",
        "看空置信度": f"{bear_confidence:.2%}",
        "差距": f"{abs(confidence_diff):.2%}",
        "看多论点": [p[:80] + '...' if len(p) > 80 else p for p in bull_points[:3]] if bull_points else [],
        "看空论点": [p[:80] + '...' if len(p) > 80 else p for p in bear_points[:3]] if bear_points else []
    }, "辩论室")
    llm_weight = 0.3 * (llm_score if llm_analysis else 0.5)
    researcher_direction = 1 if confidence_diff > 0 else -1
    llm_direction = 1 if llm_score > 0 else -1 if llm_analysis else 0
    consistency_bonus = 0.1 if (researcher_direction == llm_direction and llm_direction != 0) else -0.1
    mixed_confidence_diff = (
        (1 - llm_weight) * confidence_diff +
        llm_weight * llm_score +
        consistency_bonus
    )

    logger.info("────────────────────────────────────────────────────────")
    logger.info("📊 置信度计算:")
    logger.info(f"  看多置信度: {bull_confidence:.4f}")
    logger.info(f"  看空置信度: {bear_confidence:.4f}")
    logger.info(f"  原始差异:  {confidence_diff:.4f}")
    logger.info(f"  LLM评分:   {llm_score:.4f}")
    logger.info(f"  LLM权重:   {llm_weight:.2f}")
    logger.info(f"  一致性:    {'✅ 一致' if consistency_bonus > 0 else '❌ 不一致'}")
    logger.info(f"  混合差异:  {mixed_confidence_diff:.4f}")
    logger.info("────────────────────────────────────────────────────────")

    # 确定最终信号
    if abs(mixed_confidence_diff) < 0.1:
        final_signal = "neutral"
        final_signal_cn = "中性"
        reasoning = "多空双方论点均衡"
        confidence = max(bull_confidence, bear_confidence)
    elif mixed_confidence_diff > 0:
        final_signal = "bullish"
        final_signal_cn = "看涨"
        reasoning = "看多观点更具说服力"
        confidence = bull_confidence
    else:
        final_signal = "bearish"
        final_signal_cn = "看跌"
        reasoning = "看空观点更具说服力"
        confidence = bear_confidence

    logger.info(f"🎯 最终信号: {final_signal_cn}, 置信度: {confidence}")

    # 构建返回消息，包含 LLM 分析
    message_content = {
        "signal": final_signal,
        "confidence": confidence,
        "bull_confidence": bull_confidence,
        "bear_confidence": bear_confidence,
        "confidence_diff": confidence_diff,
        "llm_score": llm_score if llm_analysis else None,
        "llm_analysis": llm_analysis["analysis"] if llm_analysis and "analysis" in llm_analysis else None,
        "llm_reasoning": llm_analysis["reasoning"] if llm_analysis and "reasoning" in llm_analysis else None,
        "mixed_confidence_diff": mixed_confidence_diff,
        "debate_summary": debate_summary,
        "reasoning": reasoning
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="debate_room_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    show_agent_reasoning({
        "最终信号": final_signal_cn,
        "置信度": f"{confidence:.2%}",
        "决策逻辑": reasoning,
        "看多论点": len(bull_points),
        "看空论点": len(bear_points),
        "LLM评估": f"{llm_score:.2%}" if llm_analysis else "无",
        "signal": final_signal,
        "bull_points": bull_points,
        "bear_points": bear_points
    }, "辩论室")

    show_workflow_complete(
        "辩论室",
        signal=final_signal,
        confidence=confidence,
        details=message_content,
        message=f"辩论完成，结论:{final_signal_cn}，置信度:{confidence:.2%}"
    )
    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "debate_analysis": message_content
        },
        "metadata": state["metadata"],
    }
