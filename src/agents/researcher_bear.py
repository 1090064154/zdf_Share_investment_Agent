from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
import ast


@resilient_agent
@agent_endpoint("researcher_bear", "空方研究员，从看空角度分析市场数据并提出风险警示")
def researcher_bear_agent(state: AgentState):
    """Analyzes signals from a bearish perspective and generates cautionary investment thesis."""
    show_workflow_status("看空研究员")
    import logging
    logger = logging.getLogger('researcher_bear')
    logger.info("="*50)
    logger.info("🐻 [RESEARCHER_BEAR] 开始空方研究分析")
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
            "perspective": "bearish",
            "confidence": 0.0,
            "thesis_points": ["缺乏非中性的证据来建立看空观点。"],
            "reasoning": "所有上游分析师信号都是中性或不可用的。"
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="researcher_bear_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "看空研究员")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("看空研究员", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": state["data"],
            "metadata": state["metadata"],
        }

    def parse_confidence(confidence_str):
        raw_val = float(str(confidence_str).replace("%", "")) / 100
        return min(max(raw_val, 0.0), 1.0)

    # Analyze from bearish perspective
    bearish_points = []
    confidence_scores = []

    # Technical Analysis
    if technical_signals["signal"] == "bearish":
        bearish_points.append(
            f"技术指标显示看空动能，置信度为{technical_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(technical_signals["confidence"]))
    else:
        bearish_points.append(
            "技术反弹可能是暂时的，暗示潜在反转")
        confidence_scores.append(0.3)

    # Fundamental Analysis
    if fundamental_signals["signal"] == "bearish":
        bearish_points.append(
            f"基本面令人担忧，置信度为{fundamental_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(fundamental_signals["confidence"]))
    else:
        bearish_points.append(
            "当前基本面的强劲可能不可持续")
        confidence_scores.append(0.3)

    # Sentiment Analysis
    if sentiment_signals["signal"] == "bearish":
        bearish_points.append(
            f"负面市场情绪，置信度为{sentiment_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(sentiment_signals["confidence"]))
    else:
        bearish_points.append(
            "市场情绪可能过于乐观，表明存在潜在风险")
        confidence_scores.append(0.3)

    # Valuation Analysis
    if valuation_signals["signal"] == "bearish":
        bearish_points.append(
            f"股票被高估，置信度为{valuation_signals['confidence']}")
        confidence_scores.append(
            parse_confidence(valuation_signals["confidence"]))
    else:
        bearish_points.append(
            "当前估值可能未完全反映下行风险")
        confidence_scores.append(0.3)

    # Calculate overall bearish confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores)

    message_content = {
        "perspective": "bearish",
        "confidence": avg_confidence,
        "thesis_points": bearish_points,
        "reasoning": "基于技术、基本面、情绪和估值因素的综合分析得出看空观点"
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="researcher_bear_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "看空研究员")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("看空研究员", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
        "metadata": state["metadata"],
    }
