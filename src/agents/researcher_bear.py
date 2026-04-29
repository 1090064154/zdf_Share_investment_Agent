from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
import ast
import logging

logger = logging.getLogger('researcher_bear')


def _safe_parse_signal(message, default_signal: str = "neutral") -> dict:
    """安全解析消息content为JSON，失败时返回默认值"""
    if not message or not message.content:
        return {"signal": default_signal, "confidence": 0.0, "reasoning": f"消息内容为空"}

    content = message.content.strip()
    if not content:
        return {"signal": default_signal, "confidence": 0.0, "reasoning": f"消息内容为空"}

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        pass

    return {"signal": default_signal, "confidence": 0.0, "reasoning": f"无法解析消息内容: {content[:100]}"}


def _format_9agent_summary(signals_dict: dict, name: str) -> str:
    """格式化9个agent的分析结果为文本"""
    signal = signals_dict.get("signal", "neutral")
    signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
    confidence = signals_dict.get("confidence", 0.0)
    reasoning = signals_dict.get("reasoning", "")
    return f"{name}: 信号={signal_cn}, 置信度={confidence}, 理由={reasoning}"


@resilient_agent
@agent_endpoint("researcher_bear", "空方研究员，从看空角度综合9个agent分析并提出风险警示")
def researcher_bear_agent(state: AgentState):
    """Analyzes signals from a bearish perspective using LLM to generate professional risk warnings."""
    show_workflow_status("看空研究员")
    logger.info("="*50)
    logger.info("🐻 [RESEARCHER_BEAR] 开始空方研究分析（LLM扩展）")
    logger.info("="*50)
    show_reasoning = state["metadata"]["show_reasoning"]

    # ============================================================
    # Step 1: 收集9个Agent的消息
    # ============================================================
    logger.info("📥 Step 1: 收集9个Agent的消息...")
    technical_message = next(
        (msg for msg in state["messages"] if msg.name == "technical_analyst_agent"), None)
    fundamentals_message = next(
        (msg for msg in state["messages"] if msg.name == "fundamentals_agent"), None)
    sentiment_message = next(
        (msg for msg in state["messages"] if msg.name == "sentiment_agent"), None)
    valuation_message = next(
        (msg for msg in state["messages"] if msg.name == "valuation_agent"), None)
    industry_cycle_message = next(
        (msg for msg in state["messages"] if msg.name == "industry_cycle_agent"), None)
    institutional_message = next(
        (msg for msg in state["messages"] if msg.name == "institutional_agent"), None)
    expectation_diff_message = next(
        (msg for msg in state["messages"] if msg.name == "expectation_diff_agent"), None)
    macro_news_message = next(
        (msg for msg in state["messages"] if msg.name == "macro_news_agent"), None)
    macro_analyst_message = next(
        (msg for msg in state["messages"] if msg.name == "macro_analyst_agent"), None)

    # ============================================================
    # Step 2: 解析各Agent的信号
    # ============================================================
    logger.info("🔍 Step 2: 解析各Agent的信号...")
    technical_signals = _safe_parse_signal(technical_message)
    fundamental_signals = _safe_parse_signal(fundamentals_message)
    sentiment_signals = _safe_parse_signal(sentiment_message)
    valuation_signals = _safe_parse_signal(valuation_message)
    industry_cycle_signals = _safe_parse_signal(industry_cycle_message)
    institutional_signals = _safe_parse_signal(institutional_message)
    expectation_diff_signals = _safe_parse_signal(expectation_diff_message)
    macro_news_signals = _safe_parse_signal(macro_news_message)
    macro_analyst_signals = _safe_parse_signal(macro_analyst_message)

    # 打印各Agent的信号汇总
    logger.info("────────────────────────────────────────────────────────")
    logger.info("📊 9个Agent信号汇总:")
    def signal_cn(s):
        return {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(s, s)
    logger.info(f"  1️⃣ 技术分析:    信号={signal_cn(technical_signals.get('signal'))}, 置信度={technical_signals.get('confidence')}")
    logger.info(f"  2️⃣ 基本面:     信号={signal_cn(fundamental_signals.get('signal'))}, 置信度={fundamental_signals.get('confidence')}")
    logger.info(f"  3️⃣ 情绪分析:   信号={signal_cn(sentiment_signals.get('signal'))}, 置信度={sentiment_signals.get('confidence')}")
    logger.info(f"  4️⃣ 估值分析:   信号={signal_cn(valuation_signals.get('signal'))}, 置信度={valuation_signals.get('confidence')}")
    logger.info(f"  5️⃣ 行业周期:   信号={signal_cn(industry_cycle_signals.get('signal'))}, 置信度={industry_cycle_signals.get('confidence')}")
    logger.info(f"  6️⃣ 机构持仓:   信号={signal_cn(institutional_signals.get('signal'))}, 置信度={institutional_signals.get('confidence')}")
    logger.info(f"  7️⃣ 预期差:     信号={signal_cn(expectation_diff_signals.get('signal'))}, 置信度={expectation_diff_signals.get('confidence')}")
    logger.info(f"  8️⃣ 宏观新闻:   信号={signal_cn(macro_news_signals.get('signal'))}, 置信度={macro_news_signals.get('confidence')}")
    logger.info(f"  9️⃣ 宏观分析:   信号={signal_cn(macro_analyst_signals.get('signal'))}, 置信度={macro_analyst_signals.get('confidence')}")
    logger.info("────────────────────────────────────────────────────────")

    # 发送9维度信号汇总到前端
    show_agent_reasoning({
        "9维度信号汇总": {
            "技术分析": signal_cn(technical_signals.get('signal')),
            "基本面": signal_cn(fundamental_signals.get('signal')),
            "情绪分析": signal_cn(sentiment_signals.get('signal')),
            "估值分析": signal_cn(valuation_signals.get('signal')),
            "行业周期": signal_cn(industry_cycle_signals.get('signal')),
            "机构持仓": signal_cn(institutional_signals.get('signal')),
            "预期差": signal_cn(expectation_diff_signals.get('signal')),
            "宏观新闻": signal_cn(macro_news_signals.get('signal')),
            "宏观分析": signal_cn(macro_analyst_signals.get('signal'))
        }
    }, "看空研究员")

    # 统计信号分布
    bearish_count = sum(1 for s in [technical_signals, fundamental_signals, sentiment_signals,
                                   valuation_signals, industry_cycle_signals, institutional_signals,
                                   expectation_diff_signals] if s.get("signal") == "bearish")
    neutral_count = sum(1 for s in [technical_signals, fundamental_signals, sentiment_signals,
                                  valuation_signals, industry_cycle_signals, institutional_signals,
                                  expectation_diff_signals] if s.get("signal") == "neutral")
    logger.info(f"📉 信号统计: 看跌={bearish_count}, 中性={neutral_count}, 看涨={7-bearish_count-neutral_count}")

    # ============================================================
    # Step 3: 构建prompt并调用LLM
    # ============================================================
    logger.info("🔧 Step 3: 构建prompt并调用LLM...")
    analyst_summaries = f"""
技术分析: {_format_9agent_summary(technical_signals, "技术分析")}
基本面: {_format_9agent_summary(fundamental_signals, "基本面")}
情绪分析: {_format_9agent_summary(sentiment_signals, "情绪分析")}
估值分析: {_format_9agent_summary(valuation_signals, "估值分析")}
行业周期: {_format_9agent_summary(industry_cycle_signals, "行业周期")}
机构持仓: {_format_9agent_summary(institutional_signals, "机构持仓")}
预期差: {_format_9agent_summary(expectation_diff_signals, "预期差")}
宏观新闻: {_format_9agent_summary(macro_news_signals, "宏观新闻")}
宏观分析: {_format_9agent_summary(macro_analyst_signals, "宏观分析")}
"""

    # 调用LLM生成看空论点
    llm_prompt = f"""你是一位专业的金融分析师，请从投资者的角度，基于以下9个维度的分析结果，从看空角度生成专业的风险警示。

【9个维度分析结果】
{analyst_summaries}

【任务要求】
1. 综合分析以上9个维度的信息，提取看空的风险点和不利因素
2. 如果某些维度是看多的，分析其可能的潜在风险
3. 生成3-5个有说服力的看空论点，每个论点需要说明风险依据
4. 给出整体看空置信度（0.0-1.0）
5. 用专业、理性的语言进行风险分析，避免过度悲观

请以JSON格式回复：
{{
    "perspective": "bearish",
    "confidence": 0.0-1.0,
    "risk_points": ["风险点1", "风险点2", ...],
    "reasoning": "综合分析的主要风险理由"
}}
"""

    llm_response = None
    llm_analysis = None

    try:
        logger.info("🤖 调用LLM生成看空论点...")
        messages = [
            {"role": "system", "content": "你是一位专业的金融风险分析师。请用中文提供专业的风险分析。"},
            {"role": "user", "content": llm_prompt}
        ]
        llm_response = log_llm_interaction(state)(
            lambda: get_chat_completion(
                messages,
                max_retries=2,
                initial_retry_delay=1.0,
            )
        )()

        # 解析LLM返回的JSON
        if llm_response:
            try:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    llm_analysis = json.loads(json_str)
                    logger.info("✅ LLM看空分析完成")
                    logger.info(f"   置信度: {llm_analysis.get('confidence')}")
                    logger.info(f"   风险点数: {len(llm_analysis.get('risk_points', []))}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ 解析LLM响应失败: {e}")
    except Exception as e:
        logger.error(f"❌ 调用LLM失败: {e}")

    # 如果LLM失败，使用fallback逻辑
    if not llm_analysis:
        logger.warning("⚠️ 使用fallback规则生成看空论点")
        llm_analysis = _generate_bearish_fallback(
            technical_signals, fundamental_signals, sentiment_signals,
            valuation_signals, industry_cycle_signals, institutional_signals,
            expectation_diff_signals
        )

    # ============================================================
    # Step 4: 构建返回消息
    # ============================================================
    logger.info("📤 Step 4: 构建返回消息...")
    message_content = {
        "perspective": "bearish",
        "signal": "bearish",
        "confidence": llm_analysis.get("confidence", 0.5),
        "risk_points": llm_analysis.get("risk_points", []),
        "reasoning": llm_analysis.get("reasoning", "基于LLM分析得出看空观点"),
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "llm_used": llm_response is not None
    }

    logger.info("✅ 看空研究员执行完成")
    logger.info(f"   最终置信度: {message_content['confidence']}")
    logger.info(f"   风险点数: {len(message_content['risk_points'])}")

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="researcher_bear_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    show_agent_reasoning({
        "signal": "bearish",
        "看跌信号数": bearish_count,
        "最终信号": "看跌",
        "置信度": message_content.get("confidence"),
        "风险论点数量": len(message_content.get("risk_points", [])),
        "risk_points": message_content.get("risk_points", [])
    }, "看空研究员")

    show_workflow_complete(
        "看空研究员",
        signal="bearish",
        confidence=message_content.get("confidence"),
        details=message_content,
        message=f"看空研究完成，置信度:{message_content.get('confidence')}，{len(message_content.get('risk_points', []))}条风险点"
    )
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def _generate_bearish_fallback(technical_signals, fundamental_signals, sentiment_signals,
                               valuation_signals, industry_cycle_signals, institutional_signals,
                               expectation_diff_signals) -> dict:
    """当LLM失败时的fallback逻辑"""
    def parse_confidence(sig):
        try:
            val = float(str(sig.get("confidence", "0")).replace("%", "")) / 100
            return min(max(val, 0.0), 1.0)
        except:
            return 0.3

    risk_points = []
    confidence_scores = []

    # Technical
    if technical_signals.get("signal") == "bearish":
        risk_points.append(f"技术指标显示下跌动能，置信度{technical_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(technical_signals))
    else:
        risk_points.append("技术指标存在回调风险")
        confidence_scores.append(0.3)

    # Fundamentals
    if fundamental_signals.get("signal") == "bearish":
        risk_points.append(f"基本面恶化风险，置信度{fundamental_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(fundamental_signals))
    else:
        risk_points.append("基本面可能不及预期")
        confidence_scores.append(0.3)

    # Sentiment
    if sentiment_signals.get("signal") == "bearish":
        risk_points.append(f"市场情绪过度乐观，置信度{sentiment_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(sentiment_signals))
    else:
        risk_points.append("情绪过热可能引发回调")
        confidence_scores.append(0.3)

    # Valuation
    if valuation_signals.get("signal") == "bearish":
        risk_points.append(f"估值偏高，置信度{valuation_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(valuation_signals))
    else:
        risk_points.append("估值存在高估风险")
        confidence_scores.append(0.3)

    # Industry Cycle
    if industry_cycle_signals.get("signal") == "bearish":
        risk_points.append(f"行业周期下行，置信度{industry_cycle_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(industry_cycle_signals))
    else:
        risk_points.append("行业周期可能进入下行阶段")
        confidence_scores.append(0.3)

    # Institutional
    if institutional_signals.get("signal") == "bearish":
        risk_points.append(f"机构抛售风险，置信度{institutional_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(institutional_signals))
    else:
        risk_points.append("机构持仓可能下降")
        confidence_scores.append(0.3)

    # Expectation Diff
    if expectation_diff_signals.get("signal") == "bearish":
        risk_points.append(f"预期差负向，置信度{expectation_diff_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(expectation_diff_signals))
    else:
        risk_points.append("预期差存在下行风险")
        confidence_scores.append(0.3)

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.3

    return {
        "confidence": avg_confidence,
        "risk_points": risk_points,
        "reasoning": "基于9维度分析的综合看空观点"
    }