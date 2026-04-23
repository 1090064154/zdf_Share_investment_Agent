import math

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import prices_to_df
from src.utils.api_utils import agent_endpoint, log_llm_interaction

import json
import ast
from src.utils.logging_config import setup_logger
from src.utils.error_handler import resilient_agent

##### Risk Management Agent #####

logger = setup_logger('risk_management_agent')


@resilient_agent(critical=True)
@agent_endpoint("risk_management", "风险管理专家，评估投资风险并给出风险调整后的交易建议")
def risk_management_agent(state: AgentState):
    """Responsible for risk management"""
    show_workflow_status("风险管理师")
    logger.info("="*50)
    logger.info("⚠️ [RISK_MANAGER] 开始风险管理分析")
    logger.info("="*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    prices_df = prices_to_df(data["prices"])
    logger.info(
        "Risk manager received price frame: rows=%s columns=%s",
        len(prices_df),
        list(prices_df.columns),
    )

    debate_message = next(
        (msg for msg in state["messages"] if msg.name == "debate_room_agent"),
        None,
    )

    if prices_df.empty or len(prices_df) < 2 or debate_message is None:
        reason = (
            f"Insufficient inputs for risk analysis: prices_rows={len(prices_df)}, "
            f"debate_message_present={debate_message is not None}"
        )
        logger.warning(reason)
        message_content = {
            "最大持仓规模": 0.0,
            "风险评分": 10,
            "交易行动": "hold",
            "风险指标": {
                "波动率": 0.0,
                "95%风险价值": 0.0,
                "最大回撤": 0.0,
                "市场风险评分": 10,
                "压力测试结果": {}
            },
            "辩论分析": {
                "多方置信度": 0.0,
                "空方置信度": 0.0,
                "辩论置信度": 0.0,
                "辩论信号": "neutral"
            },
            "推理": reason
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="risk_management_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "Risk Management Agent")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("风险管理师", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": {
                **data,
                "risk_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    try:
        debate_results = json.loads(debate_message.content)
    except Exception as e:
        debate_results = ast.literal_eval(debate_message.content)

    # 1. Calculate Risk Metrics
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    # Annualized volatility approximation
    volatility = daily_vol * (252 ** 0.5)

    # 计算波动率的历史分布
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = 0 if volatility_std == 0 or math.isnan(volatility_std) else (
        (volatility - volatility_mean) / volatility_std
    )

    # Simple historical VaR at 95% confidence
    var_95 = returns.quantile(0.05)
    # 使用60天窗口计算最大回撤
    max_drawdown = (
        prices_df['close'] / prices_df['close'].rolling(window=60).max() - 1).min()

    # 2. Market Risk Assessment
    market_risk_score = 0

    # Volatility scoring based on percentile
    if volatility_percentile > 1.5:     # 高于1.5个标准差
        market_risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        market_risk_score += 1

    # VaR scoring
    # Note: var_95 is typically negative. The more negative, the worse.
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    # Max Drawdown scoring
    if max_drawdown < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits
    # Consider total portfolio value, not just cash
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # Start with 25% max position of total portfolio
    base_position_size = total_portfolio_value * 0.25

    if market_risk_score >= 4:
        # Reduce position for high risk
        max_position_size = base_position_size * 0.5
    elif market_risk_score >= 2:
        # Slightly reduce for moderate risk
        max_position_size = base_position_size * 0.75
    else:
        # Keep base size for low risk
        max_position_size = base_position_size

    # 4. Stress Testing
    stress_test_scenarios = {
        "market_crash": -0.20,
        "moderate_decline": -0.10,
        "slight_decline": -0.05
    }

    stress_test_results = {}
    current_position_value = current_stock_value

    if current_position_value == 0 and max_position_size > 0:
        current_position_value = max_position_size

    for scenario, decline in stress_test_scenarios.items():
        if current_position_value == 0:
            stress_test_results[scenario] = {
                "潜在损失": None,
                "组合影响": None
            }
        else:
            potential_loss = current_position_value * decline
            portfolio_impact = potential_loss / (portfolio['cash'] + current_position_value) if (
                portfolio['cash'] + current_position_value) != 0 else math.nan
            stress_test_results[scenario] = {
                "潜在损失": potential_loss,
                "组合影响": portfolio_impact
            }

    # 5. Risk-Adjusted Signal Analysis
    # Consider debate room confidence levels
    bull_confidence = debate_results["bull_confidence"]
    bear_confidence = debate_results["bear_confidence"]
    debate_confidence = debate_results["confidence"]

    # Add to risk score if confidence is low or debate was close
    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:  # Close debate
        market_risk_score += 1
    if debate_confidence < 0.3:  # Low overall confidence
        market_risk_score += 1

    # Cap risk score at 10
    risk_score = min(round(market_risk_score), 10)

    # 6. Generate Trading Action
    # Consider debate room signal along with risk assessment
    debate_signal = debate_results["signal"]

    # 获取当前持仓
    current_position = data.get("portfolio", {}).get("stock", 0)

    if risk_score >= 9:
        trading_action = "hold"
    elif risk_score >= 7:
        # 如果当前持仓为0，reduce没有意义，改为hold
        if current_position <= 0:
            trading_action = "hold"
        else:
            trading_action = "reduce"
    else:
        if debate_signal == "bullish" and debate_confidence > 0.5:
            trading_action = "buy"
        elif debate_signal == "bearish" and debate_confidence > 0.5:
            trading_action = "sell"
        else:
            trading_action = "hold"

    message_content = {
        "最大持仓规模": float(max_position_size),
        "风险评分": risk_score,
        "交易行动": trading_action,
        "风险指标": {
            "波动率": float(volatility),
            "95%风险价值": float(var_95),
            "最大回撤": float(max_drawdown),
            "市场风险评分": market_risk_score,
            "压力测试结果": {
                "市场崩盘": stress_test_results.get("market_crash", {}),
                "中度下跌": stress_test_results.get("moderate_decline", {}),
                "轻度下跌": stress_test_results.get("slight_decline", {})
            }
        },
        "辩论分析": {
            "多方置信度": bull_confidence,
            "空方置信度": bear_confidence,
            "辩论置信度": debate_confidence,
            "辩论信号": debate_signal
        },
        "推理": f"风险评分 {risk_score}/10: 市场风险={market_risk_score}, "
                f"波动率={volatility:.2%}, 风险价值={var_95:.2%}, "
                f"最大回撤={max_drawdown:.2%}, 辩论信号={debate_signal}"
    }

    # Create the risk management message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("风险管理师", "completed")
    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "risk_analysis": message_content
        },
        "metadata": state["metadata"],
    }
