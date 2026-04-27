from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
import ast
import logging

logger = logging.getLogger('researcher_bull')


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
    confidence = signals_dict.get("confidence", 0.0)
    reasoning = signals_dict.get("reasoning", "")
    return f"{name}: 信号={signal}, 置信度={confidence}, 理由={reasoning}"


@resilient_agent
@agent_endpoint("researcher_bull", "多方研究员，从看多角度综合9个agent分析并提出投资论点")
def researcher_bull_agent(state: AgentState):
    """Analyzes signals from a bullish perspective using LLM to generate professional investment thesis."""
    show_workflow_status("看多研究员")
    logger.info("="*50)
    logger.info("🐂 [RESEARCHER_BULL] 开始多方研究分析（LLM扩展）")
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
    logger.info(f"  1️⃣ 技术分析:    signal={technical_signals.get('signal')}, confidence={technical_signals.get('confidence')}")
    logger.info(f"  2️⃣ 基本面:     signal={fundamental_signals.get('signal')}, confidence={fundamental_signals.get('confidence')}")
    logger.info(f"  3️⃣ 情绪分析:   signal={sentiment_signals.get('signal')}, confidence={sentiment_signals.get('confidence')}")
    logger.info(f"  4️⃣ 估值分析:   signal={valuation_signals.get('signal')}, confidence={valuation_signals.get('confidence')}")
    logger.info(f"  5️⃣ 行业周期:   signal={industry_cycle_signals.get('signal')}, confidence={industry_cycle_signals.get('confidence')}")
    logger.info(f"  6️⃣ 机构持仓:   signal={institutional_signals.get('signal')}, confidence={institutional_signals.get('confidence')}")
    logger.info(f"  7️⃣ 预期差:     signal={expectation_diff_signals.get('signal')}, confidence={expectation_diff_signals.get('confidence')}")
    logger.info(f"  8️⃣ 宏观新闻:   signal={macro_news_signals.get('signal')}, confidence={macro_news_signals.get('confidence')}")
    logger.info(f"  9️⃣ 宏观分析:   signal={macro_analyst_signals.get('signal')}, confidence={macro_analyst_signals.get('confidence')}")
    logger.info("────────────────────────────────────────────────────────")

    # 统计信号分布
    bullish_count = sum(1 for s in [technical_signals, fundamental_signals, sentiment_signals,
                                   valuation_signals, industry_cycle_signals, institutional_signals,
                                   expectation_diff_signals] if s.get("signal") == "bullish")
    neutral_count = sum(1 for s in [technical_signals, fundamental_signals, sentiment_signals,
                                  valuation_signals, industry_cycle_signals, institutional_signals,
                                  expectation_diff_signals] if s.get("signal") == "neutral")
    logger.info(f"📈 信号统计: bullish={bullish_count}, neutral={neutral_count}, bearish={7-bullish_count-neutral_count}")

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

    # 调用LLM生成看多论点
    llm_prompt = f"""你是一位专业的金融分析师，请从投资者的角度，基于以下9个维度的分析结果，从看多角度生成专业的投资论点。

【9个维度分析结果】
{analyst_summaries}

【任务要求】
1. 综合分析以上9个维度的信息，提取看多的有利论据
2. 如果某些维度不利于看多，也要分析其可能的改善空间或被低估的原因
3. 生成3-5个有说服力的看多论点，每个论点需要说明依据
4. 给出整体看多置信度（0.0-1.0）
5. 用专业、理性的语言，避免过度乐观

请以JSON格式回复：
{{
    "perspective": "bullish",
    "confidence": 0.0-1.0,
    "thesis_points": ["论点1", "论点2", ...],
    "reasoning": "综合分析的主要理由"
}}
"""

    llm_response = None
    llm_analysis = None

    try:
        logger.info("🤖 调用LLM生成看多论点...")
        messages = [
            {"role": "system", "content": "你是一位专业的金融分析师。请用中文提供专业的投资分析。"},
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
                    logger.info("✅ LLM看多分析完成")
                    logger.info(f"   置信度: {llm_analysis.get('confidence')}")
                    logger.info(f"   论点数: {len(llm_analysis.get('thesis_points', []))}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ 解析LLM响应失败: {e}")
    except Exception as e:
        logger.error(f"❌ 调用LLM失败: {e}")

    # 如果LLM失败，使用fallback逻辑
    if not llm_analysis:
        logger.warning("⚠️ 使用fallback规则生成看多论点")
        llm_analysis = _generate_bullish_fallback(
            technical_signals, fundamental_signals, sentiment_signals,
            valuation_signals, industry_cycle_signals, institutional_signals,
            expectation_diff_signals
        )

    # ============================================================
    # Step 4: 构建返回消息
    # ============================================================
    logger.info("📤 Step 4: 构建返回消息...")
    message_content = {
        "perspective": "bullish",
        "confidence": llm_analysis.get("confidence", 0.5),
        "thesis_points": llm_analysis.get("thesis_points", []),
        "reasoning": llm_analysis.get("reasoning", "基于LLM分析得出看多观点"),
        "bullish_count": bullish_count,
        "neutral_count": neutral_count,
        "llm_used": llm_response is not None
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="researcher_bull_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "看多研究员")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("看多研究员", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }


def _generate_bullish_fallback(technical_signals, fundamental_signals, sentiment_signals,
                         valuation_signals, industry_cycle_signals, institutional_signals,
                         expectation_diff_signals) -> dict:
    """当LLM失败时的fallback逻辑"""
    def parse_confidence(sig):
        try:
            val = float(str(sig.get("confidence", "0")).replace("%", "")) / 100
            return min(max(val, 0.0), 1.0)
        except:
            return 0.3

    bullish_points = []
    confidence_scores = []

    # Technical
    if technical_signals.get("signal") == "bullish":
        bullish_points.append(f"技术指标显示看多动能，置信度{technical_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(technical_signals))
    else:
        bullish_points.append("技术指标处于相对低位，提供向上空间")
        confidence_scores.append(0.3)

    # Fundamentals
    if fundamental_signals.get("signal") == "bullish":
        bullish_points.append(f"基本面表现强劲，置信度{fundamental_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(fundamental_signals))
    else:
        bullish_points.append("基本面存在改善潜力")
        confidence_scores.append(0.3)

    # Sentiment
    if sentiment_signals.get("signal") == "bullish":
        bullish_points.append(f"市场情绪积极，置信度{sentiment_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(sentiment_signals))
    else:
        bullish_points.append("悲观情绪可能过度，估值有修复空间")
        confidence_scores.append(0.3)

    # Valuation
    if valuation_signals.get("signal") == "bullish":
        bullish_points.append(f"估值偏低，置信度{valuation_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(valuation_signals))
    else:
        bullish_points.append("当前估值未充分反映增长潜力")
        confidence_scores.append(0.3)

    # Industry Cycle
    if industry_cycle_signals.get("signal") == "bullish":
        bullish_points.append(f"行业周期有利，置信度{industry_cycle_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(industry_cycle_signals))
    else:
        bullish_points.append("行业周期可能迎来拐点")
        confidence_scores.append(0.3)

    # Institutional
    if institutional_signals.get("signal") == "bullish":
        bullish_points.append(f"机构资金流入，置信度{institutional_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(institutional_signals))
    else:
        bullish_points.append("机构持仓有提升空间")
        confidence_scores.append(0.3)

    # Expectation Diff
    if expectation_diff_signals.get("signal") == "bullish":
        bullish_points.append(f"预期差正向，置信度{expectation_diff_signals.get('confidence')}")
        confidence_scores.append(parse_confidence(expectation_diff_signals))
    else:
        bullish_points.append("预期差存在改善空间")
        confidence_scores.append(0.3)

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.3

    return {
        "confidence": avg_confidence,
        "thesis_points": bullish_points,
        "reasoning": "基于9维度分析的综合看多观点"
    }