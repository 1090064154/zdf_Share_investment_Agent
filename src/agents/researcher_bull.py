from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
import ast


@resilient_agent
@agent_endpoint("researcher_bull", "多方研究员，从看多角度分析市场数据并提出投资论点")
def researcher_bull_agent(state: AgentState):
    """Analyzes signals from a bullish perspective and generates optimistic investment thesis."""
    show_workflow_status("看多研究员")
    import logging
    logger = logging.getLogger('researcher_bull')
    logger.info("="*50)
    logger.info("🐂 [RESEARCHER_BULL] 开始多方研究分析")
    logger.info("="*50)
    show_reasoning = state["metadata"]["show_reasoning"]

    # Fetch messages from analysts
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")

    try:
        fundamental_signals = json.loads(fundamentals_message.content)
        technical_signals = json.loads(technical_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
        valuation_signals = json.loads(valuation_message.content)
    except Exception as e:
        fundamental_signals = ast.literal_eval(fundamentals_message.content)
        technical_signals = ast.literal_eval(technical_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)
        valuation_signals = ast.literal_eval(valuation_message.content)

    analyst_signals = [
        technical_signals.get("signal"),
        fundamental_signals.get("signal"),
        sentiment_signals.get("signal"),
        valuation_signals.get("signal"),
    ]
    if all(signal == "neutral" for signal in analyst_signals):
        message_content = {
            "perspective": "bullish",
            "confidence": 0.0,
            "thesis_points": ["缺乏非中性的证据来建立看多观点。"],
            "reasoning": "所有上游分析师信号都是中性或不可用的。"
        }
        message = HumanMessage(
            content=json.dumps(message_content),
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

    def parse_confidence(confidence_str):
        raw_val = float(str(confidence_str).replace("%", "")) / 100
        return min(max(raw_val, 0.0), 1.0)

    # Analyze from bullish perspective
    bullish_points = []
    confidence_scores = []

    # Technical Analysis
    if technical_signals["signal"] == "bullish":
        bullish_points.append(
            f"技术指标显示看多动能，置信度为{technical_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(technical_signals["confidence"]))
    else:
        bullish_points.append(
            "技术指标可能较为保守，提供了买入机会")
        confidence_scores.append(0.3)

    # Fundamental Analysis
    if fundamental_signals["signal"] == "bullish":
        bullish_points.append(
            f"基本面强劲，置信度为{fundamental_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(fundamental_signals["confidence"]))
    else:
        bullish_points.append(
            "公司基本面有改善潜力")
        confidence_scores.append(0.3)

    # Sentiment Analysis
    if sentiment_signals["signal"] == "bullish":
        bullish_points.append(
            f"市场情绪积极，置信度为{sentiment_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(sentiment_signals["confidence"]))
    else:
        bullish_points.append(
            "市场情绪可能过于悲观，创造了价值机会")
        confidence_scores.append(0.3)

    # Valuation Analysis
    if valuation_signals["signal"] == "bullish":
        bullish_points.append(
            f"股票被低估，置信度为{valuation_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(valuation_signals["confidence"]))
    else:
        bullish_points.append(
            "当前估值可能未完全反映增长潜力")
        confidence_scores.append(0.3)

    # Calculate overall bullish confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores)

    message_content = {
        "perspective": "bullish",
        "confidence": avg_confidence,
        "thesis_points": bullish_points,
        "reasoning": "基于技术、基本面、情绪和估值因素的综合分析得出看多观点"
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="researcher_bull_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "看多研究员")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("看多研究员", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }
