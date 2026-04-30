from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from src.utils.optimization_config import get_config
from src.utils.logging_config import setup_logger
from src.utils.decision_engine import DecisionEngine, create_decision_engine
from src.utils.error_handler import resilient_agent
from src.utils.decision_validator import create_decision_validator, DecisionPriority

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
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
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"Message from {name} not found"}, ensure_ascii=False), name=name)


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
    industry_cycle_payload: dict = None,
    institutional_payload: dict = None,
    expectation_diff_payload: dict = None,
    has_macro_news_summary: bool = False,
) -> str:
    risk_score = 0.0
    if isinstance(risk_payload, dict):
        try:
            risk_score = float(risk_payload.get("风险评分", 0))
        except (TypeError, ValueError):
            risk_score = 0.0

    risk_entry = {
        "agent_name": "risk_management",
        "signal": (risk_payload or {}).get("交易行动", "hold"),
        "confidence": max(0.1, 1.0 - risk_score / 10.0) if risk_payload else 0.0,
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
        _extract_signal_entry("industry_cycle", industry_cycle_payload or {}),
        _extract_signal_entry("institutional", institutional_payload or {}),
        _extract_signal_entry("expectation_diff", expectation_diff_payload or {}),
    ]

    risk_signal = risk_entry["signal"]

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
    }, ensure_ascii=False)


def _has_usable_macro_news_summary(summary: str) -> bool:
    if not summary:
        return False
    invalid_markers = ["暂不可用", "未提供", "未获取到", "跳过该模块"]
    return not any(marker in summary for marker in invalid_markers)


@resilient_agent(critical=True)
@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """
    投资组合管理Agent - 最终交易决策者
    
    【整体功能】
    综合所有分析师的意见，结合风险管理约束，做出最终的交易决策（买入/卖出/持有）
    
    【工作流程】
    1. 收集所有上游Agent的分析结果
    2. 提取各Agent的信号和置信度
    3. 构建决策上下文
    4. 使用DecisionEngine规则引擎或LLM进行决策
    5. 应用风险管理强制规则
    6. 输出最终交易决策
    
    【输入依赖】
    - technical_analyst_agent: 技术分析结果
    - fundamentals_agent: 基本面分析结果
    - sentiment_agent: 情绪分析结果
    - valuation_agent: 估值分析结果
    - risk_management_agent: 风险评估结果
    - macro_analyst_agent: 宏观分析结果
    - debate_room_agent: 辩论室结论
    - industry_cycle_agent: 行业周期分析
    - institutional_agent: 机构持仓分析
    - expectation_diff_agent: 预期差分析
    - macro_news_agent: 宏观新闻摘要
    
    【输出】
    - action: 交易行动 (buy/sell/hold)
    - quantity: 交易数量
    - confidence: 决策置信度
    - reasoning: 决策理由
    - agent_signals: 各分析师信号汇总
    """
    agent_name = "portfolio_management_agent"
    logger.info("="*60)
    logger.info("🎯 [PORTFOLIO_MANAGER] 开始执行投资组合管理")
    logger.info("="*60)

    # ==================== 步骤1: 消息去重与整理 ====================
    # 目的: 避免重复的Agent消息影响决策
    # 策略: 对每个Agent只保留最新的消息
    unique_incoming_messages = {}
    for msg in state["messages"]:
        unique_incoming_messages[msg.name] = msg

    cleaned_messages_for_processing = list(unique_incoming_messages.values())

    logger.info(f"  收集到 {len(cleaned_messages_for_processing)} 个 Agent 的分析结果:")
    for msg in cleaned_messages_for_processing:
        logger.info(f"    - {msg.name}")

    show_workflow_status(agent_name)
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    logger.info(f"  当前投资组合: 现金={portfolio.get('cash', 0):.2f}元, 持仓={portfolio.get('stock', 0)}股")

    # ==================== 步骤2: 提取各Agent的分析结果 ====================
    # 从消息列表中提取每个Agent的最新分析结果
    # 如果某个Agent的消息不存在，返回错误占位符
    # 提取基础分析Agent的结果
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analyst_agent")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_agent")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_agent")
    
    # 提取风险管理和辩论结果
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_management_agent")
    debate_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "debate_room_agent")
    
    # 提取宏观分析结果
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_analyst_agent")
    
    # 提取新增的三个Agent结果
    industry_cycle_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "industry_cycle_agent")
    institutional_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "institutional_agent")
    expectation_diff_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "expectation_diff_agent")

    # ==================== 步骤3: 解析JSON内容 ====================
    # 将各Agent的消息内容从JSON字符串解析为字典对象
    # 如果解析失败，使用错误占位符
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

    # ==================== 步骤4: 构建信号列表 ====================
    # 将所有Agent的信号标准化为统一格式，便于后续决策引擎处理
    # 注意: agent_name字段必须使用中文名称，以便终端输出显示
    risk_entry = {
        "agent_name": "风险管理",
        "signal": (risk_payload or {}).get("交易行动", "hold"),
        "confidence": max(0.1, 1.0 - float((risk_payload or {}).get("风险评分", 5)) / 10.0) if risk_payload else 0.0,
    }
    agent_signals = [
        _extract_signal_entry("技术分析", technical_payload),
        _extract_signal_entry("基本面分析", fundamentals_payload),
        _extract_signal_entry("情绪分析", sentiment_payload),
        _extract_signal_entry("估值分析", valuation_payload),
        risk_entry,
        {
            "agent_name": "宏观分析",
            "signal": macro_payload.get("impact_on_stock", "neutral") if isinstance(macro_payload, dict) else "neutral",
            "confidence": 0.5 if macro_payload and isinstance(macro_payload, dict) and macro_payload.get("key_factors") else 0.3,
        },
        {
            "agent_name": "宏观新闻分析",
            "signal": "neutral",
            "confidence": 0.1,
        },
        _extract_signal_entry("辩论室", debate_payload),
        _extract_signal_entry("行业周期", _parse_message_json(industry_cycle_content) or {}),
        _extract_signal_entry("机构持仓", _parse_message_json(institutional_content) or {}),
        _extract_signal_entry("预期差", _parse_message_json(expectation_diff_content) or {}),
    ]

    # 发送各Agent信号汇总到前端
    logger.info("  📊 各Agent信号汇总:")
    for s in agent_signals:
        logger.info(f"    {s['agent_name']}: {s['signal']} ({s['confidence']:.0%})")
    show_agent_reasoning({"各分析模块信号": agent_signals}, "投资组合管理")

    # 发送所有Agent信号汇总到前端
    signals_summary = {s["agent_name"]: f"{s['signal']}({s['confidence']:.0%})" for s in agent_signals}
    show_agent_reasoning({"各模块信号汇总": signals_summary}, agent_name)

    # ==================== 步骤5: 构建LLM决策上下文 ====================
    investment_horizon = state["data"].get('investment_horizon', 'medium')
    horizon_cn = {"short": "短线(1周-1月)", "medium": "中线(1-3月)", "long": "长线(6月+)"}.get(investment_horizon, "中线")

    # 根据投资期限调整维度权重
    if investment_horizon == 'short':
        dim_weights = "技术分析35%、情绪分析20%、辩论室15%、机构持仓10%、风险管理10%、基本面5%、估值5%"
        focus = "短期价格动量和市场情绪是最关键的驱动力，基本面和估值的权重降低"
    elif investment_horizon == 'long':
        dim_weights = "基本面分析30%、估值分析25%、行业周期15%、辩论室10%、宏观分析10%、风险管理5%、技术分析5%"
        focus = "长期价值创造能力是核心，短期波动和技术指标的重要性大幅降低"
    else:
        dim_weights = "辩论室20%、基本面分析20%、估值分析15%、风险管理15%、技术分析10%、宏观分析10%、情绪分析5%、行业周期5%"
        focus = "多空平衡结论和基本面估值的综合判断最为关键"

    system_message_content = f"""你是一位专业的A股投资组合经理。当前投资期限为{horizon_cn}。

            {focus}

            ========== 风险管理约束（最高优先级，不可违背）==========
            1. 禁止超过风险管理师规定的最大持仓量
            2. 风险评分 >= 7 时，强制执行 hold
            3. 风险管理建议 sell/reduce 时，必须执行卖出或减持
            4. 买入校验：数量 × 当前股价 ≤ 可用现金 × 80%（留20%安全垫）
            5. 卖出校验：卖出数量 ≤ 当前持仓量
            ========================================

            ========== 当前投资期限维度权重 ==========
            {dim_weights}
            ======================================

            ========== 仓位规模决策方法 ==========
            1. 高置信度(>0.7)买入：可用资金的 40%-60%
            2. 中等置信度(0.5-0.7)买入：可用资金的 15%-35%
            3. 低置信度(<0.5)：不交易或仅试探性(≤10%)
            4. 卖出时：卖出当前持仓的 30%-100%，置信度越高卖出越多
            5. 持有：quantity=0
            ======================================

            决策流程：
            1. 检查风控硬约束 → 2. 评估辩论室多空结论 → 3. 综合估值和基本面
            → 4. 考虑行业周期和机构动向 → 5. 用技术分析把握时机 → 6. 参考情绪和预期差微调

            输出JSON格式：
            {{
                "action": "buy"|"sell"|"hold",
                "quantity": <正整数>,
                "confidence": <0到1之间>,
                "agent_signals": [{{"agent_name":"技术分析","signal":"bullish|bearish|neutral","confidence":0.0-1.0}}, ...],
                "reasoning": "<决策过程简述>"
            }}
            agent_signals必须包含: 技术分析, 基本面分析, 情绪分析, 估值分析, 风险管理, 宏观分析, 宏观新闻分析, 辩论室, 行业周期, 机构持仓, 预期差
            """
    system_message = {
        "role": "system",
        "content": system_message_content
    }

    # ==================== 步骤6: 获取当前股价 ====================
    # 从市场数据中提取最新收盘价，用于计算交易数量和验证资金充足性
    current_price = 0.0
    prices = state["data"].get("prices", [])
    if prices and len(prices) > 0:
        latest_price = prices[-1]
        if isinstance(latest_price, dict) and "close" in latest_price:
            current_price = float(latest_price.get("close", 0))
        elif isinstance(latest_price, dict) and "收盘" in latest_price:
            current_price = float(latest_price.get("收盘", 0))

    # 构建精简的摘要式user message（避免原始JSON超长浪费token）
    def _summarize_agent(name_cn, payload, extra_keys=None):
        if not payload or not isinstance(payload, dict):
            return f"{name_cn}: 数据不可用"
        signal = payload.get('signal', payload.get('交易行动', 'unknown'))
        conf = payload.get('confidence', payload.get('置信度', 'unknown'))
        reason = payload.get('reasoning', payload.get('reason', payload.get('summary', '')))
        if isinstance(reason, str) and len(reason) > 150:
            reason = reason[:150] + '...'
        parts = [f"信号={signal}"]
        if conf not in ('unknown', None):
            parts.append(f"置信度={conf}")
        if reason:
            parts.append(f"摘要={reason}")
        if extra_keys:
            for k in extra_keys:
                v = payload.get(k, '')
                if v:
                    parts.append(f"{k}={v}")
        return f"{name_cn}: {'; '.join(parts)}"

    industry_cycle_payload = _parse_message_json(industry_cycle_content) or {}
    institutional_payload = _parse_message_json(institutional_content) or {}
    expectation_diff_payload = _parse_message_json(expectation_diff_content) or {}

    # [NEW] 提取risk_score和trading_action（必须在使用前定义）
    risk_score = 0.0
    trading_action = "hold"
    dynamic_threshold = 5.0

    if risk_payload:
        try:
            risk_score = float(risk_payload.get("风险评分", 0))
        except (TypeError, ValueError):
            risk_score = 0.0
        trading_action = risk_payload.get("交易行动", "hold")
        dynamic_threshold = float(risk_payload.get("动态阈值", 5.0))

    user_message_content = f"""请根据团队分析结果做出{horizon_cn}交易决策。

【技术分析】{_summarize_agent('技术分析', technical_payload, ['strategy_signals'])}
【基本面分析】{_summarize_agent('基本面', fundamentals_payload)}
【情绪分析】{_summarize_agent('情绪', sentiment_payload)}
【估值分析】{_summarize_agent('估值', valuation_payload)}
【风险管理】{_summarize_agent('风险管理', risk_payload, ['风险评分', '动态阈值', '最大持仓规模'])}
【宏观分析】{_summarize_agent('宏观', macro_payload, ['macro_environment', 'impact_on_stock'])}
【宏观新闻】{market_wide_news_summary_content[:200] if market_wide_news_summary_content else '不可用'}
【辩论室】{_summarize_agent('辩论', debate_payload, ['bull_confidence', 'bear_confidence', 'consistency'])}
【行业周期】{_summarize_agent('行业周期', industry_cycle_payload, ['cycle_type_cn', 'phase', 'weight_factor'])}
【机构持仓】{_summarize_agent('机构持仓', institutional_payload)}
【预期差】{_summarize_agent('预期差', expectation_diff_payload)}

当前投资组合: 现金={portfolio['cash']:.2f}元, 持仓={portfolio['stock']}股, 股价≈{current_price:.2f}元
风险管理: 最大持仓={risk_payload.get('最大持仓规模', '未指定')}, 风险评分={risk_score}/10, 风控建议={trading_action}

请仅输出JSON格式。"""
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    show_agent_reasoning(
        agent_name, f"准备LLM调用，包含: 技术分析、基本面、情绪、估值、风险管理、宏观、新闻")

    # ==================== 步骤7: 尝试使用DecisionEngine决策 ====================
    # DecisionEngine是规则化决策引擎，可以替代LLM进行更快速、更稳定的决策
    config = get_config()
    use_decision_engine = config.enable_decision_engine if config._config else False

    # 注意: current_price优先使用市场数据真实值(已在步骤6获取)，不在此处覆盖
    
    # 如果市场数据没有价格，才尝试从风险分析中推算
    if current_price <= 0 and risk_payload:
        max_position = float(risk_payload.get("最大持仓规模", 0))
        total_value = portfolio.get('cash', 0) + portfolio.get('stock', 0) * 100
        if max_position > 0 and total_value > 0:
            current_price = max_position / (0.25 * total_value / 100) if total_value > 0 else 0.0

    # 更新portfolio中的current_price
    portfolio_with_price = dict(portfolio)
    if current_price > 0:
        portfolio_with_price['current_price'] = current_price

    if use_decision_engine:
        logger.info("🎯 启用DecisionEngine规则化决策")
        try:
            # ==================== 步骤7.1: 构建信号字典 ====================
            # 将所有Agent的信号整理为DecisionEngine需要的格式
            # 每个信号包含: signal (bullish/bearish/neutral) 和 confidence (0-1)
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
                    'confidence': max(0.1, 1.0 - risk_score / 10.0) if risk_payload else 0.0
                },
                'debate': {
                    'signal': debate_payload.get('signal', 'neutral'),
                    'confidence': _normalize_confidence(debate_payload.get('confidence', 0.5))
                }
            }

            # 添加新增模块的信号（行业周期、机构持仓、预期差）
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

            # ==================== 步骤7.2: 执行DecisionEngine决策 ====================
            # DecisionEngine根据预定义规则进行决策，比LLM更快更稳定
            # 注意：macro因子已通过9维度信号中的macro信号加权计算，不再单独检查
            investment_horizon = state["data"].get('investment_horizon', 'medium')
            engine = create_decision_engine(config.get_agent_weights(), investment_horizon)
            engine_decision = engine.make_decision(
                signals=signals,
                risk_score=risk_score,
                risk_action=trading_action,
                portfolio=portfolio_with_price,
                dynamic_threshold=dynamic_threshold
            )

            logger.info(f"🎯 DecisionEngine决策: {engine_decision}")

            # 使用DecisionEngine的决策结果构造JSON响应
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

    # ==================== 步骤8: LLM决策（如果DecisionEngine不可用） ====================
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

    # ==================== 步骤9: 处理LLM失败情况 ====================
    # 如果LLM调用失败，使用保守的fallback策略
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
            industry_cycle_payload=industry_cycle_payload,
            institutional_payload=institutional_payload,
            expectation_diff_payload=expectation_diff_payload,
            has_macro_news_summary=_has_usable_macro_news_summary(market_wide_news_summary_content),
        )

    # ==================== 步骤10: 构造最终决策消息 ====================
    final_decision_message = HumanMessage(
        content=llm_response_content,
        name=agent_name,
    )

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"Final LLM decision JSON: {llm_response_content}")

    # 解析决策JSON
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
        # 发送最终决策到前端
        show_agent_reasoning({
            "最终决策": decision_json.get("action"),
            "数量": f"{decision_json.get('quantity')}股",
            "置信度": f"{decision_json.get('confidence', 0):.0%}",
            "推理": decision_json.get("reasoning", "")[:100]
        }, agent_name)
    except json.JSONDecodeError:
        agent_decision_details_value = {
            "error": "Failed to parse LLM decision JSON from portfolio manager",
            "raw_response_snippet": llm_response_content[:200] + "..."
        }

    # ==================== 步骤11: 统一决策验证 ====================
    # 使用DecisionValidator框架进行统一的决策验证，消除规则冲突

    # 保存从 LLM 响应中解析的 agent_signals
    parsed_agent_signals = decision_json.get("agent_signals", None)
    final_action = decision_json.get("action", "hold")
    final_quantity = decision_json.get("quantity", 0)

    # 提取风险管理信息（risk_score已在步骤7提取，此处补充max_position）
    max_position = 0.0
    if risk_payload:
        try:
            max_position = float(risk_payload.get("最大持仓规模", 0))
        except (TypeError, ValueError):
            max_position = 0.0
    
    # 创建决策验证器并执行验证
    validator = create_decision_validator()
    config = get_config()
    
    validation_context = {
        "risk_score": risk_score,
        "risk_action": trading_action,
        "max_position": max_position,
        "portfolio": portfolio,
        "current_price": current_price,
        "enable_veto_power": config.enable_veto_power if config._config else False,
    }
    
    initial_decision = {
        "action": final_action,
        "quantity": final_quantity,
        "confidence": decision_json.get("confidence", 0.5),
        "reasoning": decision_json.get("reasoning", ""),
    }
    
    # 执行验证
    validation_result = validator.validate(initial_decision, validation_context)
    
    # 应用验证结果
    final_action = validation_result.action
    final_quantity = validation_result.quantity
    
    # 构造最终决策消息
    llm_response_content = json.dumps({
        "action": final_action,
        "quantity": final_quantity,
        "confidence": validation_result.confidence,
        "agent_signals": parsed_agent_signals if parsed_agent_signals else agent_signals,
        "reasoning": validation_result.reason,
        "validation_overridden": validation_result.overridden,
    }, ensure_ascii=False)
    
    final_decision_message = HumanMessage(content=llm_response_content, name=agent_name)

    # ==================== 步骤12: 输出最终决策日志 ====================
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

    show_workflow_complete(
        agent_name,
        signal=final_action,
        confidence=validation_result.confidence,
        details={
            "action": final_action,
            "quantity": final_quantity,
            "confidence": validation_result.confidence,
            "reasoning": decision_json.get("reasoning", ""),
            "agent_signals": parsed_agent_signals if parsed_agent_signals else agent_signals,
            "signal_summary": signals_summary,
        },
        message=f"最终决策完成，动作:{final_action}，数量:{final_quantity}股，置信度:{validation_result.confidence*100:.0f}%"
    )

    # ==================== 步骤13: 构造返回结果 ====================
    # 将最终决策消息添加到清理后的消息列表中
    # 这样可以避免重复消息，同时保留完整的决策链路
    final_messages_output = cleaned_messages_for_processing + [final_decision_message]

    return {
        "messages": final_messages_output,
        "data": state["data"],
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }
