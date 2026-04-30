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


def _safe_parse_confidence(value):
    """安全解析confidence值为float，处理百分号字符串等情况"""
    if isinstance(value, str):
        value = value.strip().replace('%', '')
        try:
            val = float(value)
            if val > 1:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0
    try:
        val = float(value or 0)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.0


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
    bull_confidence = _safe_parse_confidence(bull_thesis.get("confidence", 0))
    bear_confidence = _safe_parse_confidence(bear_thesis.get("confidence", 0))

    logger.info("────────────────────────────────────────────────────────")
    logger.info("📊 研究员论点汇总:")
    logger.info(f"  🐂 看多研究员: confidence={bull_confidence:.4f}")
    logger.info(f"     论点数: {len(bull_thesis.get('thesis_points', []))}")
    logger.info(f"  🐻 看空研究员: confidence={bear_confidence:.4f}")
    logger.info(f"     风险点数: {len(bear_thesis.get('risk_points', []))}")
    logger.info("────────────────────────────────────────────────────────")

    # 获取论点列表
    bull_points = bull_thesis.get("thesis_points", []) if bull_thesis else []
    bear_points = bear_thesis.get("risk_points", []) if bear_thesis else []

    # ============================================================
    # Step 2.5: 双方不确定守门
    # ============================================================
    # 如果双方置信度都低于0.4，说明底层分析高度不确定，直接输出中性
    MAX_UNCERTAIN_CONFIDENCE = 0.4
    if bull_confidence < MAX_UNCERTAIN_CONFIDENCE and bear_confidence < MAX_UNCERTAIN_CONFIDENCE:
        logger.warning(
            f"⚠️ 双方研究员置信度均过低(牛市={bull_confidence:.2f}, 熊市={bear_confidence:.2f})，"
            f"跳过LLM辩论，直接输出低置信度中性信号"
        )
        message_content = {
            "signal": "neutral",
            "confidence": 0.2,
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "confidence_diff": bull_confidence - bear_confidence,
            "llm_score": None,
            "llm_analysis": None,
            "llm_reasoning": None,
            "mixed_confidence_diff": 0.0,
            "debate_summary": ["分析数据不足，双方研究员均无法形成有说服力的观点"],
            "reasoning": "底层分析数据不足，双方研究员置信度过低，无法形成有效辩论结论",
            "consistency": "uncertain",
            "跳过LLM": True
        }

        message = HumanMessage(
            content=json.dumps(message_content, ensure_ascii=False),
            name="debate_room_agent",
        )

        if show_reasoning:
            state["metadata"]["agent_reasoning"] = message_content

        show_agent_reasoning({
            "最终信号": "中性",
            "置信度": "20%",
            "决策逻辑": "双方研究员置信度过低，无法形成有效辩论",
            "看多置信度": f"{bull_confidence:.0%}",
            "看空置信度": f"{bear_confidence:.0%}",
            "signal": "neutral",
            "bull_points": bull_points,
            "bear_points": bear_points
        }, "辩论室")

        show_workflow_complete(
            "辩论室",
            signal="neutral",
            confidence=0.2,
            details=message_content,
            message="辩论完成：双方研究员置信度过低，输出中性信号"
        )
        return {
            "messages": state["messages"] + [message],
            "data": {
                **state["data"],
                "debate_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # 分析辩论观点
    debate_summary = []
    debate_summary.append("看多观点:")
    for point in bull_points:
        debate_summary.append(f"+ {point}")

    debate_summary.append("\n看空观点:")
    for point in bear_points:
        debate_summary.append(f"- {point}")

    # ============================================================
    # Step 3: 构建prompt并调用LLM进行辩论
    # ============================================================
    logger.info("🔧 Step 3: 构建prompt并调用LLM...")

    # 获取9维度基础信号上下文（让LLM看到原始信号分布）
    ticker = state["data"].get("ticker", "未知")
    horizon = state["data"].get("investment_horizon", "medium")
    horizon_cn = {"short": "短线(1周-1月)", "medium": "中线(1-3月)", "long": "长线(6月+)"}.get(horizon, horizon)

    # 收集各基础Agent的原始信号
    agent_signals_summary = []
    for msg in state["messages"]:
        if msg is None or not hasattr(msg, 'name') or not hasattr(msg, 'content'):
            continue
        if msg.name and msg.name not in ("researcher_bull_agent", "researcher_bear_agent", "debate_room_agent"):
            try:
                data = json.loads(msg.content)
                sig = data.get("signal", data.get("最终信号", "unknown"))
                conf = data.get("confidence", data.get("置信度", "unknown"))
                agent_signals_summary.append(f"  - {msg.name}: 信号={sig}, 置信度={conf}")
            except:
                pass

    all_perspectives = {}
    for name, data in researcher_data.items():
        perspective = data.get("perspective", name.replace(
            "researcher_", "").replace("_agent", ""))
        all_perspectives[perspective] = {
            "confidence": data.get("confidence", 0),
            "thesis_points": data.get("thesis_points", data.get("risk_points", []))
        }

    # 构建增强版 LLM prompt（含股票上下文和原始信号）
    llm_prompt = f"""你是一位专业的A股金融分析师。请基于以下信息，对股票 {ticker}（持仓周期：{horizon_cn}）的多空辩论进行独立第三方分析。

【9维度基础信号分布】
{chr(10).join(agent_signals_summary) if agent_signals_summary else "  无原始信号数据"}

【研究员观点】
"""
    for perspective, persp_data in all_perspectives.items():
        llm_prompt += f"\n{perspective.upper()} 观点 (置信度: {persp_data['confidence']}):\n"
        for point in persp_data['thesis_points']:
            llm_prompt += f"- {point}\n"

    llm_prompt += f"""
请评估各方论点的说服力，特别关注：
1. 哪些论点有数据支撑，哪些只是推测
2. 在{horizon_cn}的持仓周期下，哪些因素最关键
3. 当前A股市场环境下，多空双方谁的观点更符合实际

请提供以下格式的 JSON 回复:
{{
    "analysis": "你的详细分析，评估各方观点的优劣，并指出最有说服力的论点",
    "score": 0.5,
    "reasoning": "评分理由"
}}
score范围: -1.0(极度看空) 到 1.0(极度看多)，0 表示中性。
务必确保回复是有效JSON且包含所有字段。使用中文回复。
"""

    # 调用 LLM 获取第三方观点
    llm_response = None
    llm_analysis = None
    llm_score = 0  # 默认为中性
    try:
        logger.info("🤖 调用LLM进行第三方辩论分析...")
        messages = [
            {"role": "system", "content": "你是一位专业的A股金融分析师。请用中文提供独立、客观的分析。不要简单复述研究员的观点，要给出你自己的判断。"},
            {"role": "user", "content": llm_prompt}
        ]

        llm_response = log_llm_interaction(state)(
            lambda: get_chat_completion(
                messages,
                max_retries=1,
                initial_retry_delay=0.5,
            )
        )()

        logger.info("✅ LLM辩论分析完成")

        if llm_response:
            try:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                    llm_score = _safe_parse_confidence(llm_analysis.get("score", 0))
                    llm_score = max(min(llm_score, 1.0), -1.0)
                    logger.info(f"成功解析 LLM 回复，评分: {llm_score:.4f}")
                    logger.debug(
                        f"LLM 分析内容: {llm_analysis.get('analysis', '未提供分析')[:100]}...")
            except Exception as e:
                logger.error(f"解析 LLM 回复失败: {e}")
                llm_analysis = {"analysis": "Failed to parse LLM response",
                                "score": 0, "reasoning": "Parsing error"}
    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        llm_analysis = {"analysis": "LLM API call failed",
                        "score": 0, "reasoning": "API error"}

    # ============================================================
    # Step 4: 计算混合置信度（修正版）
    # ============================================================
    logger.info("🔢 Step 4: 计算混合置信度...")
    confidence_diff = bull_confidence - bear_confidence

    # 发送研究员论点汇总到前端
    show_agent_reasoning({
        "看多置信度": f"{bull_confidence:.2%}",
        "看空置信度": f"{bear_confidence:.2%}",
        "差距": f"{confidence_diff:+.2%}",
        "看多论点": bull_points[:3] if bull_points else [],
        "看空论点": bear_points[:3] if bear_points else []
    }, "辩论室")

    # 修正: llm_weight 为常数，LLM贡献30%
    LLM_WEIGHT = 0.3
    RESEARCHER_WEIGHT = 1.0 - LLM_WEIGHT  # 70%

    # 修正: 零点方向判定
    researcher_direction = 1 if confidence_diff > 0 else (-1 if confidence_diff < 0 else 0)
    llm_direction = 1 if llm_score > 0 else (-1 if llm_score < 0 else 0) if llm_analysis else 0

    # 修正: 一致性奖金分级
    if researcher_direction == llm_direction and llm_direction != 0:
        # 同向：按共识强度分级
        consensus_strength = abs(confidence_diff) + abs(llm_score)
        if consensus_strength > 0.6:
            consistency_bonus = 0.15   # 强共识
        elif consensus_strength > 0.3:
            consistency_bonus = 0.08   # 中等共识
        else:
            consistency_bonus = 0.03   # 弱共识
    elif researcher_direction != 0 and llm_direction != 0:
        # 反向：按分歧强度分级
        divergence_strength = abs(confidence_diff - llm_score)
        if divergence_strength > 0.6:
            consistency_bonus = -0.15  # 严重分歧
        elif divergence_strength > 0.3:
            consistency_bonus = -0.08  # 中等分歧
        else:
            consistency_bonus = -0.03  # 轻微分歧
    else:
        # 任一方为中性
        consistency_bonus = 0.0

    # 混合置信度差异 = 研究员差异(70%) + LLM评分(30%) + 一致性调整
    mixed_confidence_diff = (
        RESEARCHER_WEIGHT * confidence_diff +
        LLM_WEIGHT * llm_score +
        consistency_bonus
    )

    logger.info("────────────────────────────────────────────────────────")
    logger.info("📊 置信度计算:")
    logger.info(f"  看多置信度: {bull_confidence:.4f}")
    logger.info(f"  看空置信度: {bear_confidence:.4f}")
    logger.info(f"  研究员差异: {confidence_diff:+.4f} (权重{RESEARCHER_WEIGHT:.0%})")
    logger.info(f"  LLM评分:    {llm_score:+.4f} (权重{LLM_WEIGHT:.0%})")
    logger.info(f"  一致性奖金: {consistency_bonus:+.3f} ({'✅ 一致' if consistency_bonus > 0 else '❌ 分歧' if consistency_bonus < 0 else '➖ 中性'})")
    logger.info(f"  混合差异:   {mixed_confidence_diff:+.4f}")
    logger.info("────────────────────────────────────────────────────────")

    # ============================================================
    # Step 5: 确定最终信号（修正版）
    # ============================================================
    # 修正: 最终置信度融入LLM分析和一致性调整
    if abs(mixed_confidence_diff) < 0.08:  # 从0.1缩窄到0.08
        final_signal = "neutral"
        final_signal_cn = "中性"
        reasoning = "多空双方论点接近均衡"
        # 中性时置信度反映双方不确定性
        base_confidence = max(bull_confidence, bear_confidence)
        confidence = base_confidence * (1.0 + consistency_bonus)
        consistency_label = "balanced"
    elif mixed_confidence_diff > 0:
        final_signal = "bullish"
        final_signal_cn = "看涨"
        reasoning = "综合多空辩论及LLM第三方分析，看多观点更具说服力"
        # 修正: 置信度融入LLM影响
        base_confidence = bull_confidence
        confidence = min(base_confidence * (1.0 + consistency_bonus), 0.95)
        consistency_label = "bullish_dominant"
    else:
        final_signal = "bearish"
        final_signal_cn = "看跌"
        reasoning = "综合多空辩论及LLM第三方分析，看空观点更具说服力"
        base_confidence = bear_confidence
        confidence = min(base_confidence * (1.0 + consistency_bonus), 0.95)
        consistency_label = "bearish_dominant"

    # 确保置信度在合理范围
    confidence = max(0.1, min(confidence, 0.95))

    logger.info(f"🎯 最终信号: {final_signal_cn}, 置信度: {confidence:.4f}")

    # ============================================================
    # Step 6: 构建返回消息
    # ============================================================
    message_content = {
        "signal": final_signal,
        "confidence": round(confidence, 4),
        "bull_confidence": bull_confidence,
        "bear_confidence": bear_confidence,
        "confidence_diff": confidence_diff,
        "llm_score": llm_score if llm_analysis else None,
        "llm_weight": LLM_WEIGHT,
        "researcher_weight": RESEARCHER_WEIGHT,
        "llm_analysis": llm_analysis["analysis"] if llm_analysis and "analysis" in llm_analysis else None,
        "llm_reasoning": llm_analysis["reasoning"] if llm_analysis and "reasoning" in llm_analysis else None,
        "mixed_confidence_diff": round(mixed_confidence_diff, 4),
        "consistency_bonus": round(consistency_bonus, 4),
        "consistency": consistency_label,
        "debate_summary": debate_summary,
        "reasoning": reasoning,
        "跳过LLM": False
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="debate_room_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    # 推送辩论总结到前端（修正：传列表而非int）
    show_agent_reasoning({
        "最终信号": final_signal_cn,
        "置信度": f"{confidence:.2%}",
        "决策逻辑": reasoning,
        "看多置信度": f"{bull_confidence:.0%}",
        "看空置信度": f"{bear_confidence:.0%}",
        "LLM评分": f"{llm_score:+.2f}" if llm_analysis else "不可用",
        "一致性": "一致" if consistency_bonus > 0 else ("分歧" if consistency_bonus < 0 else "中性"),
        "signal": final_signal,
        "bull_points": bull_points[:5] if bull_points else [],
        "bear_points": bear_points[:5] if bear_points else [],
        "debate_summary": debate_summary
    }, "辩论室")

    show_workflow_complete(
        "辩论室",
        signal=final_signal,
        confidence=confidence,
        details=message_content,
        message=f"辩论完成，结论:{final_signal_cn}，置信度:{confidence:.2%}，一致性:{consistency_label}"
    )
    return {
        "messages": state["messages"] + [message],
        "data": {
            **state["data"],
            "debate_analysis": message_content
        },
        "metadata": state["metadata"],
    }
