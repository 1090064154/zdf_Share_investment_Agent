from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from src.utils.optimization_config import get_config
from src.utils.logging_config import setup_logger
from src.utils.decision_engine import DecisionEngine, create_decision_engine

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion
from src.utils.api_utils import agent_endpoint, log_llm_interaction

# 初始化 logger
logger = setup_logger('portfolio_management_agent')

##### Portfolio Management Agent #####

# Helper function to get the latest message by agent name


def get_latest_message_by_name(messages: list, name: str):
    for msg in reversed(messages):
        if msg.name == name:
            return msg
    logger.warning(
        f"Message from agent '{name}' not found in portfolio_management_agent.")
    # Return a dummy message object or raise an error, depending on desired handling
    # For now, returning a dummy message to avoid crashing, but content will be None.
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"Message from {name} not found"}), name=name)


def _parse_message_json(message_content: str):
    try:
        return json.loads(message_content)
    except (TypeError, json.JSONDecodeError):
        return None


def _normalize_confidence(value) -> float:
    try:
        if isinstance(value, str):
            value = value.strip().replace("%", "")
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))


def _extract_signal_entry(agent_name: str, payload: dict, signal_key: str = "signal", confidence_key: str = "confidence") -> dict:
    signal = "neutral"
    confidence = 0.0

    if isinstance(payload, dict):
        signal = payload.get(signal_key, signal)
        confidence = _normalize_confidence(payload.get(confidence_key, confidence))

    return {
        "agent_name": agent_name,
        "signal": signal,
        "confidence": confidence,
    }


def _build_fallback_portfolio_decision(
    technical_payload: dict,
    fundamentals_payload: dict,
    sentiment_payload: dict,
    valuation_payload: dict,
    risk_payload: dict,
    macro_payload: dict,
    debate_payload: dict,
    has_macro_news_summary: bool,
) -> str:
    risk_entry = {
        "agent_name": "risk_management",
        "signal": (risk_payload or {}).get("交易行动", "hold"),
        "confidence": 1.0 if risk_payload else 0.0,
    }

    debate_entry = _extract_signal_entry("debate_room", debate_payload)

    agent_signals = [
        _extract_signal_entry("technical_analysis", technical_payload),
        _extract_signal_entry("fundamental_analysis", fundamentals_payload),
        _extract_signal_entry("sentiment_analysis", sentiment_payload),
        _extract_signal_entry("valuation_analysis", valuation_payload),
        risk_entry,
        {
            "agent_name": "macro_analysis",
            "signal": macro_payload.get("impact_on_stock", "neutral") if isinstance(macro_payload, dict) else "neutral",
            "confidence": 0.5 if macro_payload and isinstance(macro_payload, dict) and macro_payload.get("key_factors") else 0.3,
        },
        {
            "agent_name": "macro_news_analysis",
            "signal": "neutral" if has_macro_news_summary else "unavailable",
            "confidence": 0.1 if has_macro_news_summary else 0.0,
        },
        debate_entry,
    ]

    risk_signal = risk_entry["signal"]
    risk_score = 0.0
    if isinstance(risk_payload, dict):
        try:
            risk_score = float(risk_payload.get("风险评分", 0))
        except (TypeError, ValueError):
            risk_score = 0.0

    reasoning = "LLM不可用，已改用确定性保守决策，并保留上游Agent的真实信号。"
    if risk_signal in {"sell", "reduce"}:
        reasoning += f" 当前风险管理建议为{risk_signal}，因此不执行买入。"
    elif risk_score >= 7:
        reasoning += f" 当前风险评分为{risk_score:.0f}/10，优先保持观望。"
    else:
        reasoning += " 在缺少最终LLM综合裁决时，默认持有并等待更可靠输入。"

    return json.dumps({
        "action": "hold",
        "quantity": 0,
        "confidence": 0.35,
        "agent_signals": agent_signals,
        "reasoning": reasoning,
    })


def _has_usable_macro_news_summary(summary: str) -> bool:
    if not summary:
        return False
    invalid_markers = ["暂不可用", "未提供", "未获取到", "跳过该模块"]
    return not any(marker in summary for marker in invalid_markers)


@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """Responsible for portfolio management"""
    agent_name = "portfolio_management_agent"
    logger.info("="*60)
    logger.info("🎯 [PORTFOLIO_MANAGER] 开始执行投资组合管理")
    logger.info("="*60)

    # Clean and unique messages by agent name, taking the latest if duplicates exist
    unique_incoming_messages = {}
    for msg in state["messages"]:
        unique_incoming_messages[msg.name] = msg

    cleaned_messages_for_processing = list(unique_incoming_messages.values())

    logger.info(f"  收集到 {len(cleaned_messages_for_processing)} 个 Agent 的分析结果:")
    for msg in cleaned_messages_for_processing:
        logger.info(f"    - {msg.name}")

    show_workflow_status(f"{agent_name}: --- 正在执行投资组合管理 ---")
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    logger.info(f"  当前投资组合: 现金={portfolio.get('cash', 0):.2f}元, 持仓={portfolio.get('stock', 0)}股")

    # Get messages from other agents using the cleaned list
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analyst_agent")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_agent")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_agent")
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_management_agent")
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_analyst_agent")
    debate_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "debate_room_agent")
    industry_cycle_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "industry_cycle_agent")
    institutional_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "institutional_agent")
    expectation_diff_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "expectation_diff_agent")

    # Extract content, handling potential None if message not found by get_latest_message_by_name
    technical_content = technical_message.content if technical_message else json.dumps(
        {"signal": "error", "details": "Technical message missing"})
    fundamentals_content = fundamentals_message.content if fundamentals_message else json.dumps(
        {"signal": "error", "details": "Fundamentals message missing"})
    sentiment_content = sentiment_message.content if sentiment_message else json.dumps(
        {"signal": "error", "details": "Sentiment message missing"})
    valuation_content = valuation_message.content if valuation_message else json.dumps(
        {"signal": "error", "details": "Valuation message missing"})
    risk_content = risk_message.content if risk_message else json.dumps(
        {"signal": "error", "details": "Risk message missing"})
    tool_based_macro_content = tool_based_macro_message.content if tool_based_macro_message else json.dumps(
        {"signal": "error", "details": "Tool-based Macro message missing"})
    industry_cycle_content = industry_cycle_message.content if industry_cycle_message else json.dumps(
        {"signal": "error", "details": "Industry cycle message missing"})
    institutional_content = institutional_message.content if institutional_message else json.dumps(
        {"signal": "error", "details": "Institutional message missing"})
    expectation_diff_content = expectation_diff_message.content if expectation_diff_message else json.dumps(
        {"signal": "error", "details": "Expectation diff message missing"})

    # Market-wide news summary from macro_news_agent (already correctly fetched from state["data"])
    market_wide_news_summary_content = state["data"].get(
        "macro_news_analysis_result", "大盘宏观新闻分析不可用或未提供。")
    macro_news_agent_message_obj = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_news_agent")

    # Debate room analysis - newly added
    debate_content = debate_message.content if debate_message else json.dumps(
        {"signal": "neutral", "confidence": 0.5, "details": "Debate room analysis not available"})

    technical_payload = _parse_message_json(technical_content) or {}
    fundamentals_payload = _parse_message_json(fundamentals_content) or {}
    sentiment_payload = _parse_message_json(sentiment_content) or {}
    valuation_payload = _parse_message_json(valuation_content) or {}
    risk_payload = _parse_message_json(risk_content) or {}
    macro_payload = _parse_message_json(tool_based_macro_content) or {}
    debate_payload = _parse_message_json(debate_content) or {}

    # 构建信号列表供强制规则使用
    risk_entry = {
        "agent_name": "risk_management",
        "signal": (risk_payload or {}).get("交易行动", "hold"),
        "confidence": 1.0 if risk_payload else 0.0,
    }
    agent_signals = [
        _extract_signal_entry("technical_analysis", technical_payload),
        _extract_signal_entry("fundamental_analysis", fundamentals_payload),
        _extract_signal_entry("sentiment_analysis", sentiment_payload),
        _extract_signal_entry("valuation_analysis", valuation_payload),
        risk_entry,
        {
            "agent_name": "macro_analysis",
            "signal": macro_payload.get("impact_on_stock", "neutral") if isinstance(macro_payload, dict) else "neutral",
            "confidence": 0.5 if macro_payload and isinstance(macro_payload, dict) and macro_payload.get("key_factors") else 0.3,
        },
        {
            "agent_name": "macro_news_analysis",
            "signal": "neutral",
            "confidence": 0.1,
        },
        _extract_signal_entry("debate_room", debate_payload),
        _extract_signal_entry("industry_cycle", _parse_message_json(industry_cycle_content) or {}),
        _extract_signal_entry("institutional", _parse_message_json(institutional_content) or {}),
        _extract_signal_entry("expectation_diff", _parse_message_json(expectation_diff_content) or {}),
    ]

    system_message_content = """你是一位专业的投资组合经理，负责做出最终的交易决策。
            你的任务是在严格遵守风险管理约束的前提下，根据团队的分析做出交易决策。

            ========== 风险管理约束（最高优先级）==========
            1. 禁止超过风险管理师规定的最大持仓量
            2. 必须遵循风险管理师推荐的交易操作（买入/卖出/持有）
            3. 如果风险评分 >= 7，强制执行 hold
            4. 如果风险管理建议 sell 或 reduce，必须执行卖出或减持
            ========== 以上是硬性约束，不可违背 ==========

            ========== 各分析维度权重 ==========
            1. 估值分析 (25%权重) - DCF、所有者收益法
            2. 基本面分析 (20%权重) - 盈利能力、增长、财务健康
            3. 技术分析 (15%权重) - 趋势、动量、RSI等
            4. 宏观分析 (10%权重) - 宏观经济对个股影响
            5. 情绪分析 (5%权重) - 新闻情绪
            6. 辩论室分析 (15%权重) - 多空辩论的平衡结论
            7. 风险管理 (10%权重) - 风险评分和仓位控制
            ======================================

            决策流程：
            1. 首先检查风险管理约束（硬性）
            2. 然后评估辩论室的多空平衡结论
            3. 接着评估估值信号
            4. 评估基本面信号
            5. 综合考虑宏观环境和新闻
            6. 使用技术分析把握时机
            7. 参考情绪分析进行最终调整

            请按以下JSON格式输出：
            - "action": "buy" | "sell" | "hold"（买入/卖出/持有）
            - "quantity": <正整数，交易数量>
            - "confidence": <0到1之间的浮点数，表示你对最终决策的置信度>
            - "agent_signals": <包含各Agent信号的列表，每个信号是一个对象>
              你的 'agent_signals' 列表必须包含以下Agent的条目：
                - "technical_analysis"（技术分析）
                - "fundamental_analysis"（基本面分析）
                - "sentiment_analysis"（情绪分析）
                - "valuation_analysis"（估值分析）
                - "risk_management"（风险管理）
                - "macro_analysis"（宏观分析）
                - "debate_room"（辩论室多空平衡）
            - "reasoning": <简洁解释你的决策过程>

            ========== 交易规则（强制校验）==========
            1. 买入前校验：数量 * 当前股价 <= 可用现金
            2. 卖出前校验：卖出数量 <= 当前持仓量
            3. 禁止超过风险管理允许的最大持仓量
            4. 如果风控强制 hold，必须遵守
            ========================================"""
    system_message = {
        "role": "system",
        "content": system_message_content
    }

    # Get current stock price from market data
    current_price = 0.0
    prices = state["data"].get("prices", [])
    if prices and len(prices) > 0:
        latest_price = prices[-1]
        if isinstance(latest_price, dict) and "close" in latest_price:
            current_price = float(latest_price.get("close", 0))
        elif isinstance(latest_price, dict) and "收盘" in latest_price:
            current_price = float(latest_price.get("收盘", 0))

    user_message_content = f"""根据团队的分析结果，做出您的交易决策。

            技术分析信号: {technical_content}
            基本面分析信号: {fundamentals_content}
            情绪分析信号: {sentiment_content}
            估值分析信号: {valuation_content}
            风险管理信号: {risk_content}
            宏观分析信号（来自宏观分析师）: {tool_based_macro_content}
            每日市场新闻摘要（来自宏观新闻分析师）: {market_wide_news_summary_content}
            辩论室分析（多空平衡）: {debate_content}

            当前投资组合:
            现金: {portfolio['cash']:.2f} 元
            当前持仓: {portfolio['stock']} 股
            当前股价: {current_price:.2f} 元/股
            风险管理允许的最大持仓: {risk_payload.get('最大持仓规模', '未指定')}

            请仅输出JSON格式。确保'agent_signals'包含所有必需的agent信息。"""
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    show_agent_reasoning(
        agent_name, f"准备LLM调用，包含: 技术分析、基本面、情绪、估值、风险管理、宏观、新闻")

    # [NEW] 尝试使用DecisionEngine决策
    config = get_config()
    use_decision_engine = config.enable_decision_engine if config._config else False

    # 提取risk_score和trading_action（在LLM决策之前）
    risk_score = 0.0
    trading_action = "hold"
    if risk_payload:
        try:
            risk_score = float(risk_payload.get("风险评分", 0))
        except (TypeError, ValueError):
            risk_score = 0.0
        trading_action = risk_payload.get("交易行动", "hold")

    if use_decision_engine:
        logger.info("🎯 启用DecisionEngine规则化决策")
        try:
            # 构建信号字典
            signals = {
                'technical': {
                    'signal': technical_payload.get('signal', 'neutral'),
                    'confidence': _normalize_confidence(technical_payload.get('confidence', 0.3))
                },
                'fundamentals': {
                    'signal': fundamentals_payload.get('signal', 'neutral'),
                    'confidence': _normalize_confidence(fundamentals_payload.get('confidence', 0.4))
                },
                'sentiment': {
                    'signal': sentiment_payload.get('signal', 'neutral'),
                    'confidence': _normalize_confidence(sentiment_payload.get('confidence', 0.25))
                },
                'valuation': {
                    'signal': valuation_payload.get('signal', 'neutral'),
                    'confidence': _normalize_confidence(valuation_payload.get('confidence', 0.35))
                },
                'macro': {
                    'signal': macro_payload.get('impact_on_stock', 'neutral') if isinstance(macro_payload, dict) else 'neutral',
                    'confidence': 0.5 if macro_payload and isinstance(macro_payload, dict) and macro_payload.get('key_factors') else 0.3
                },
                'risk': {
                    'signal': risk_payload.get('交易行动', 'hold'),
                    'confidence': 1.0 if risk_payload else 0.0
                }
            }

            # 添加新增模块的信号
            industry_cycle_message = get_latest_message_by_name(cleaned_messages_for_processing, "industry_cycle_agent")
            if industry_cycle_message:
                ic_payload = _parse_message_json(industry_cycle_message.content)
                if ic_payload:
                    signals['industry_cycle'] = {
                        'signal': ic_payload.get('signal', 'neutral'),
                        'confidence': _normalize_confidence(ic_payload.get('confidence', 0.4))
                    }

            institutional_message = get_latest_message_by_name(cleaned_messages_for_processing, "institutional_agent")
            if institutional_message:
                inst_payload = _parse_message_json(institutional_message.content)
                if inst_payload:
                    signals['institutional'] = {
                        'signal': inst_payload.get('signal', 'neutral'),
                        'confidence': _normalize_confidence(inst_payload.get('confidence', 0.4))
                    }

            expectation_diff_message = get_latest_message_by_name(cleaned_messages_for_processing, "expectation_diff_agent")
            if expectation_diff_message:
                exp_payload = _parse_message_json(expectation_diff_message.content)
                if exp_payload:
                    signals['expectation_diff'] = {
                        'signal': exp_payload.get('signal', 'neutral'),
                        'confidence': _normalize_confidence(exp_payload.get('confidence', 0.3))
                    }

            # 获取macro_factor
            macro_factor = 1.0
            if isinstance(macro_payload, dict):
                position_factor = macro_payload.get('position_factor')
                if position_factor:
                    try:
                        macro_factor = float(position_factor)
                    except (ValueError, TypeError):
                        macro_factor = 1.0

            # 创建DecisionEngine并决策
            engine = create_decision_engine(config.get_agent_weights())
            engine_decision = engine.make_decision(
                signals=signals,
                risk_score=risk_score,
                risk_action=trading_action,
                macro_factor=macro_factor,
                portfolio=portfolio
            )

            logger.info(f"🎯 DecisionEngine决策: {engine_decision}")

            # 使用DecisionEngine的决策结果
            final_action = engine_decision.get('action', 'hold')
            final_quantity = engine_decision.get('quantity', 0)
            engine_reason = engine_decision.get('reason', '')

            llm_response_content = json.dumps({
                'action': final_action,
                'quantity': final_quantity,
                'confidence': engine_decision.get('confidence', 0.5),
                'agent_signals': agent_signals,
                'reasoning': f"[DecisionEngine] {engine_reason}"
            }, ensure_ascii=False)

            show_agent_reasoning(agent_name, f"DecisionEngine决策: {final_action} {final_quantity}股")
            decision_json = json.loads(llm_response_content)

        except Exception as e:
            logger.warning(f"DecisionEngine决策失败，回退到LLM: {e}")
            use_decision_engine = False

    if not use_decision_engine or llm_response_content is None:
        llm_interaction_messages = [system_message, user_message]
        llm_response_content = get_chat_completion(
            llm_interaction_messages,
            max_retries=1,
            initial_retry_delay=0.5,
        )

    current_metadata = state["metadata"]
    current_metadata["current_agent_name"] = agent_name

    def get_llm_result_for_logging_wrapper():
        return llm_response_content
    log_llm_interaction(state)(get_llm_result_for_logging_wrapper)()

    if llm_response_content is None:
        show_agent_reasoning(
            agent_name, "LLM call failed. Using default conservative decision.")
        llm_response_content = _build_fallback_portfolio_decision(
            technical_payload=technical_payload,
            fundamentals_payload=fundamentals_payload,
            sentiment_payload=sentiment_payload,
            valuation_payload=valuation_payload,
            risk_payload=risk_payload,
            macro_payload=macro_payload,
            debate_payload=debate_payload,
            has_macro_news_summary=_has_usable_macro_news_summary(market_wide_news_summary_content),
        )

    final_decision_message = HumanMessage(
        content=llm_response_content,
        name=agent_name,
    )

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"Final LLM decision JSON: {llm_response_content}")

    agent_decision_details_value = {}
    decision_json = {}
    try:
        decision_json = json.loads(llm_response_content)
        agent_decision_details_value = {
            "action": decision_json.get("action"),
            "quantity": decision_json.get("quantity"),
            "confidence": decision_json.get("confidence"),
            "reasoning_snippet": decision_json.get("reasoning", "")[:150] + "..."
        }
    except json.JSONDecodeError:
        agent_decision_details_value = {
            "error": "Failed to parse LLM decision JSON from portfolio manager",
            "raw_response_snippet": llm_response_content[:200] + "..."
        }

    # ===== 强制风险校验 =====
    # 保存从 LLM 响应中解析的 agent_signals，如果没有则使用前面构建的信号
    parsed_agent_signals = decision_json.get("agent_signals", None)
    final_action = decision_json.get("action", "hold")
    final_quantity = decision_json.get("quantity", 0)
    risk_score = 0.0
    risk_signal = "hold"
    max_position = 0.0
    trading_action = "hold"

    if risk_payload:
        try:
            risk_score = float(risk_payload.get("风险评分", 0))
        except (TypeError, ValueError):
            risk_score = 0.0
        risk_signal = risk_payload.get("交易行动", "hold")
        try:
            max_position = float(risk_payload.get("最大持仓规模", 0))
        except (TypeError, ValueError):
            max_position = 0.0
        trading_action = risk_payload.get("交易行动", "hold")

    # 规则1: 风险评分 >= 7，强制 hold
    if risk_score >= 7:
        logger.warning(f"风险评分 {risk_score} >= 7，强制执行 hold")
        final_action = "hold"
        final_quantity = portfolio.get("stock", 0)
        signals_to_use = parsed_agent_signals if parsed_agent_signals else agent_signals
        llm_response_content = json.dumps({
            "action": "hold",
            "quantity": final_quantity,
            "confidence": decision_json.get("confidence", 0.5),
            "agent_signals": agent_signals,
            "reasoning": f"风险评分{risk_score:.0f}/10 >= 7，强制执行持有。风险管理约束优先。"
        }, ensure_ascii=False)
        final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)
    # 规则2: 风险管理建议 - 一票否决或强制执行
    elif trading_action in ["sell", "reduce", "减仓", "清仓"]:
        config = get_config()
        current_position = portfolio.get("stock", 0)
        
        if config.enable_veto_power and trading_action == "sell":
            # 新逻辑: 一票否决 - 阻止买入但不强制卖出
            if current_position > 0:
                # 有持仓时,执行卖出
                logger.warning(f"风险管理建议 {trading_action}，有持仓执行卖出")
                final_action = "sell"
                final_quantity = min(final_quantity, current_position)
                llm_response_content = json.dumps({
                    "action": final_action,
                    "quantity": final_quantity,
                    "confidence": decision_json.get("confidence", 0.5),
                    "agent_signals": agent_signals,
                    "reasoning": f"风险管理建议{trading_action}，有持仓执行卖出"
                }, ensure_ascii=False)
            else:
                # 无持仓时,阻止买入但不强制卖出
                logger.warning(f"风险管理建议{trading_action}，但无持仓可卖，阻止新买入")
                final_action = "hold"
                final_quantity = 0
                llm_response_content = json.dumps({
                    "action": final_action,
                    "quantity": final_quantity,
                    "confidence": decision_json.get("confidence", 0.5),
                    "agent_signals": agent_signals,
                    "reasoning": f"风控一票否决:风险管理建议{trading_action}，无持仓可卖，阻止新买入"
                }, ensure_ascii=False)
            final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)
        else:
            # 旧逻辑: 强制执行
            logger.warning(f"风险管理建议 {trading_action}，强制执行")
            final_action = "sell"
            if available_stock <= 0:
                final_quantity = 0
                llm_response_content = json.dumps({
                    "action": final_action,
                    "quantity": final_quantity,
                    "confidence": decision_json.get("confidence", 0.5),
                    "agent_signals": agent_signals,
                    "reasoning": f"风险管理建议{trading_action}，但无持仓可卖"
                }, ensure_ascii=False)
            else:
                final_quantity = min(final_quantity, available_stock)
                llm_response_content = json.dumps({
                    "action": final_action,
                    "quantity": final_quantity,
                    "confidence": decision_json.get("confidence", 0.5),
                    "agent_signals": agent_signals,
                    "reasoning": f"风险管理建议{trading_action}，强制执行。"
                }, ensure_ascii=False)
            final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)

    # 规则3: 资金校验 - 买入时检查现金是否充足
    if final_action == "buy" and current_price > 0:
        required_cash = final_quantity * current_price
        available_cash = portfolio.get("cash", 0)
        if required_cash > available_cash:
            max_shares = int(available_cash / current_price)
            logger.warning(f"现金不足，需要{required_cash:.2f}元，现有{available_cash:.2f}元，调整为{max_shares}股")
            final_quantity = max_shares
            if final_quantity <= 0:
                final_action = "hold"
            llm_response_content = json.dumps({
                "action": final_action,
                "quantity": final_quantity,
                "confidence": decision_json.get("confidence", 0.5),
                "agent_signals": agent_signals,
                "reasoning": f"现金不足，原计划买入{final_quantity}股，现调整为{max_shares}股"
            }, ensure_ascii=False)
            final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)

    # 规则4: 持仓校验 - 卖出时检查持仓是否充足
    if final_action == "sell":
        available_stock = portfolio.get("stock", 0)
        if final_quantity > available_stock:
            logger.warning(f"持仓不足，需要卖出{final_quantity}股，现有{available_stock}股，调整为{available_stock}股")
            final_quantity = available_stock
            if final_quantity <= 0:
                final_action = "hold"
                final_quantity = 0
            llm_response_content = json.dumps({
                "action": final_action,
                "quantity": final_quantity,
                "confidence": decision_json.get("confidence", 0.5),
                "agent_signals": agent_signals,
                "reasoning": f"持仓不足，原计划卖出{final_quantity}股，现调整为{available_stock}股"
            }, ensure_ascii=False)
            final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)

    # 规则5: 最大持仓校验
    if max_position > 0 and final_action == "buy":
        current_position = portfolio.get("stock", 0)
        potential_position = current_position + final_quantity
        if potential_position > max_position:
            allowed_quantity = int(max_position - current_position)
            if allowed_quantity < 0:
                allowed_quantity = 0
            logger.warning(f"超过最大持仓限制，需要{final_quantity}股，最大允许{max_position}股，调整为{allowed_quantity}股")
            final_quantity = allowed_quantity
            if final_quantity <= 0:
                final_action = "hold"
            llm_response_content = json.dumps({
                "action": final_action,
                "quantity": final_quantity,
                "confidence": decision_json.get("confidence", 0.5),
                "agent_signals": agent_signals,
                "reasoning": f"超过最大持仓限制{max_position}股，调整为{final_quantity}股"
            }, ensure_ascii=False)
            final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)

    logger.info("="*60)
    logger.info("🎯 [PORTFOLIO_MANAGER] 最终决策")
    logger.info("="*60)
    logger.info(f"  📊 风险评分: {risk_score}/10")
    logger.info(f"  📊 风控建议: {trading_action}")
    logger.info(f"  📊 最大持仓: {max_position} 股")
    logger.info(f"  📊 当前价格: {current_price:.2f} 元")
    logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"  ✅ 最终行动: {final_action.upper()}")
    logger.info(f"  ✅ 交易数量: {final_quantity} 股")
    logger.info(f"  ✅ 置信度: {decision_json.get('confidence', 0)*100:.0f}%")
    logger.info("="*60)

    show_workflow_status(f"{agent_name}: --- 投资组合管理完成 ---")

    # The portfolio_management_agent is a terminal or near-terminal node in terms of new message generation for the main state.
    # It should return its own decision, and an updated state["messages"] that includes its decision.
    # As it's a汇聚点, it should ideally start with a cleaned list of messages from its inputs.
    # The cleaned_messages_for_processing already did this. We append its new message to this cleaned list.

    # If we strictly want to follow the pattern of `state["messages"] + [new_message]` for all non-leaf nodes,
    # then the `cleaned_messages_for_processing` should become the new `state["messages"]` for this node's context.
    # However, for simplicity and robustness, let's assume its output `messages` should just be its own message added to the cleaned input it processed.

    final_messages_output = cleaned_messages_for_processing + [final_decision_message]
    # Alternative if we want to be super strict about adding to the raw incoming state["messages"]:
    # final_messages_output = state["messages"] + [final_decision_message]
    # But this ^ is prone to the duplication we are trying to solve if not careful.
    # The most robust is that portfolio_manager provides its clear output, and the graph handles accumulation if needed for further steps (none in this case as it's END).

    # logger.info(
    # f"--- DEBUG: {agent_name} RETURN messages: {[msg.name for msg in final_messages_output]} ---")

    return {
        "messages": final_messages_output,
        "data": state["data"],
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str, market_wide_news_summary: str = "未提供") -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""

    fundamental_signal = next(
        (s for s in agent_signals if s["agent_name"] == "fundamental_analysis"), None)
    valuation_signal = next(
        (s for s in agent_signals if s["agent_name"] == "valuation_analysis"), None)
    technical_signal = next(
        (s for s in agent_signals if s["agent_name"] == "technical_analysis"), None)
    sentiment_signal = next(
        (s for s in agent_signals if s["agent_name"] == "sentiment_analysis"), None)
    risk_signal = next(
        (s for s in agent_signals if s["agent_name"] == "risk_management"), None)
    # Existing macro signal from macro_analyst_agent (tool-based)
    general_macro_signal = next(
        (s for s in agent_signals if s["agent_name"] == "macro_analyst_agent"), None)
    # New market-wide news summary signal from macro_news_agent
    market_wide_news_signal = next(
        (s for s in agent_signals if s["agent_name"] == "macro_news_agent"), None)

    def signal_to_chinese(signal_data):
        if not signal_data:
            return "无数据"
        if signal_data.get("signal") == "bullish":
            return "看多"
        if signal_data.get("signal") == "bearish":
            return "看空"
        return "中性"

    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal['confidence']*100:.0f if fundamental_signal else 0}%
   要点:
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal['confidence']*100:.0f if valuation_signal else 0}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal['confidence']*100:.0f if technical_signal else 0}%
   要点:
   - 趋势跟踪: ADX={technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', 0.0):.2f if technical_signal else 0.0:.2f}
   - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', 0.0):.2f if technical_signal else 0.0:.2f}
   - 动量指标:
     * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', 0.0):.2% if technical_signal else 0.0:.2%}
     * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', 0.0):.2% if technical_signal else 0.0:.2%}
     * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', 0.0):.2% if technical_signal else 0.0:.2%}
   - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', 0.0):.2% if technical_signal else 0.0:.2%}

4. 宏观分析 (综合权重15%):
   a) 常规宏观分析 (来自 Macro Analyst Agent):
      信号: {signal_to_chinese(general_macro_signal)}
      置信度: {general_macro_signal['confidence']*100:.0f if general_macro_signal else 0}%
      宏观环境: {general_macro_signal.get(
          'macro_environment', '无数据') if general_macro_signal else '无数据'}
      对股票影响: {general_macro_signal.get(
          'impact_on_stock', '无数据') if general_macro_signal else '无数据'}
      关键因素: {', '.join(general_macro_signal.get(
          'key_factors', ['无数据']) if general_macro_signal else ['无数据'])}

   b) 大盘宏观新闻分析 (来自 Macro News Agent):
      信号: {signal_to_chinese(market_wide_news_signal)}
      置信度: {market_wide_news_signal['confidence']*100:.0f if market_wide_news_signal else 0}%
      摘要或结论: {market_wide_news_signal.get(
          'reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

5. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal['confidence']*100:.0f if sentiment_signal else 0}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')
                             if sentiment_signal else '无详细分析'}

二、风险评估
风险评分: {risk_signal.get('risk_score', '无数据') if risk_signal else '无数据'}/10
主要指标:
- 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', 0.0)*100:.1f if risk_signal else 0.0}%
- 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', 0.0)*100:.1f if risk_signal else 0.0}%
- VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', 0.0)*100:.1f if risk_signal else 0.0}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据') if risk_signal else '无数据'}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence*100:.0f}%

四、决策依据
{reasoning}

===================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }
