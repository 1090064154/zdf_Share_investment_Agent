import math
import akshare as ak

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


def _get_market_index_data(end_date: str = None, window: int = 60) -> dict:
    """
    获取沪深300指数数据，计算大盘风险指标
    
    Args:
        end_date: 结束日期
        window: 回测期间（天数），根据投资周期动态调整
    """
    try:
        # 获取沪深300指数数据
        index_df = ak.stock_zh_index_daily(symbol="sh000300")
        if index_df is None or len(index_df) < 60:
            return None

        # 转换日期格式
        if end_date:
            try:
                from datetime import datetime
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                # 过滤到end_date之后的数据（取最后60天）
                index_df = index_df[index_df['date'] <= end_dt].tail(60)
            except:
                index_df = index_df.tail(60)
        else:
            index_df = index_df.tail(60)

        if len(index_df) < 30:
            return None

        returns = index_df['close'].pct_change().dropna()

        # 计算大盘风险指标
        daily_vol = returns.std()
        volatility = daily_vol * (252 ** 0.5)  # 年化波动率

        # CVaR (95%) - 条件在险价值，更适合A股尾部风险
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # 最大回撤（根据window动态调整）
        window = min(window, len(index_df))
        max_drawdown = (
            index_df['close'] / index_df['close'].rolling(window=window).max() - 1).min()

        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'returns': returns
        }
    except Exception as e:
        logger.warning(f"获取大盘数据失败: {e}")
        return None


def _calculate_component_score(value: float, thresholds: list, scores: list) -> float:
    """
    根据阈值和分数计算单个风险指标得分

    Args:
        value: 风险指标值
        thresholds: [低阈值, 高阈值] 如 [-0.02, -0.05]
        scores: [低分, 高分] 如 [0, 2]

    Returns:
        风险得分 0-2
    """
    if value <= thresholds[0]:
        return scores[1]
    elif value <= thresholds[1]:
        return (scores[1] + scores[0]) / 2
    else:
        return scores[0]


def _calculate_dynamic_threshold(historical_values: list, multiplier: float = 1.5) -> float:
    """
    计算动态阈值：均值 + multiplier × 标准差

    这种方法让市场自己决定风险高低，比固定阈值更合理
    """
    if not historical_values or len(historical_values) < 10:
        return 5.0  # 默认阈值

    mean = sum(historical_values) / len(historical_values)
    variance = sum((x - mean) ** 2 for x in historical_values) / len(historical_values)
    std = math.sqrt(variance)

    threshold = mean + multiplier * std
    return max(min(threshold, 8.0), 3.0)  # 限制在3-8之间


def _calculate_risk_score(
    market_metrics: dict,
    stock_metrics: dict,
    market_returns: list,
    debate_signal: str,
    bull_confidence: float,
    bear_confidence: float,
    debate_confidence: float,
    historical_risk_scores: list = None
) -> dict:
    """
    综合计算风险评分

    Args:
        market_metrics: 大盘风险指标
        stock_metrics: 个股风险指标
        market_returns: 大盘日收益率（用于计算相对风险）
        debate_signal: 辩论信号
        bull_confidence: 多方置信度
        bear_confidence: 空方置信度
        debate_confidence: 辩论置信度
        historical_risk_scores: 历史风险评分（用于动态阈值计算）

    Returns:
        dict with risk_score, trading_action, threshold
    """
    # ==================== 1. 大盘风险分 (0-10) ====================
    market_risk = 0

    # 波动率评分（年化）
    vol = market_metrics.get('volatility', 0)
    if vol > 0.30:
        market_risk += 3
    elif vol > 0.20:
        market_risk += 2
    elif vol > 0.15:
        market_risk += 1

    # VaR评分
    var = market_metrics.get('var_95', 0)
    if var < -0.03:
        market_risk += 3
    elif var < -0.02:
        market_risk += 2
    elif var < -0.01:
        market_risk += 1

    # CVaR评分（尾部风险，更适合A股）
    cvar = market_metrics.get('cvar_95', 0)
    if cvar < -0.05:
        market_risk += 2
    elif cvar < -0.03:
        market_risk += 1

    # 回撤评分
    dd = market_metrics.get('max_drawdown', 0)
    if dd < -0.15:
        market_risk += 4
    elif dd < -0.10:
        market_risk += 2
    elif dd < -0.05:
        market_risk += 1

    # ==================== 2. 个股风险分 (0-10) ====================
    stock_risk = 0

    stock_vol = stock_metrics.get('volatility', 0)
    if stock_vol > 0.50:
        stock_risk += 3
    elif stock_vol > 0.35:
        stock_risk += 2
    elif stock_vol > 0.25:
        stock_risk += 1

    stock_var = stock_metrics.get('var_95', 0)
    if stock_var < -0.04:
        stock_risk += 3
    elif stock_var < -0.025:
        stock_risk += 2
    elif stock_var < -0.015:
        stock_risk += 1

    stock_dd = stock_metrics.get('max_drawdown', 0)
    if stock_dd < -0.25:
        stock_risk += 4
    elif stock_dd < -0.15:
        stock_risk += 2
    elif stock_dd < -0.08:
        stock_risk += 1

    # ==================== 3. 相对风险分 (0-10) ====================
    relative_risk = 5  # 默认中性

    # 计算个股与大盘的beta（波动率比值）
    if market_metrics.get('volatility', 0) > 0 and stock_vol > 0:
        beta = stock_vol / market_metrics['volatility']
        if beta > 2.0:
            relative_risk += 3
        elif beta > 1.5:
            relative_risk += 2
        elif beta > 1.2:
            relative_risk += 1
        elif beta < 0.5:
            relative_risk -= 2
        elif beta < 0.8:
            relative_risk -= 1

    # ==================== 4. 综合风险评分 ====================
    # 加权平均：大盘40% + 个股40% + 相对20%
    raw_risk_score = (
        market_risk * 0.4 +
        stock_risk * 0.4 +
        relative_risk * 0.2
    )

    # 辩论信号调整（±1分）
    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:  # 多空极度一致
        # 可能过度乐观或悲观，加1分风险
        raw_risk_score += 1
    elif debate_confidence < 0.25:  # 辩论置信度极低
        raw_risk_score += 1

    # 封顶
    risk_score = min(round(raw_risk_score), 10)

    # ==================== 5. 动态阈值 ====================
    dynamic_threshold = _calculate_dynamic_threshold(historical_risk_scores or [], 1.5)

    # ==================== 6. 交易行动 ====================
    if risk_score >= 9:
        trading_action = "hold"
    elif risk_score >= dynamic_threshold:
        # 动态阈值：处于高风险区
        trading_action = "hold"
    else:
        if debate_signal == "bullish" and debate_confidence > 0.4:
            trading_action = "buy"
        elif debate_signal == "bearish" and debate_confidence > 0.4:
            trading_action = "sell"
        else:
            trading_action = "hold"

    return {
        'risk_score': risk_score,
        'trading_action': trading_action,
        'dynamic_threshold': dynamic_threshold,
        'components': {
            'market_risk': market_risk,
            'stock_risk': stock_risk,
            'relative_risk': relative_risk
        },
        'metrics': {
            'market_volatility': market_metrics.get('volatility', 0),
            'market_var': market_metrics.get('var_95', 0),
            'market_drawdown': market_metrics.get('max_drawdown', 0),
            'stock_volatility': stock_metrics.get('volatility', 0),
            'stock_var': stock_metrics.get('var_95', 0),
            'stock_drawdown': stock_metrics.get('max_drawdown', 0),
        }
    }


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
    end_date = data.get("end_date")
    investment_horizon = data.get('investment_horizon', 'medium')
    
    # 根据持仓周期动态调整回撤窗口
    horizon_windows = {'short': 20, 'medium': 60, 'long': 120}
    drawdown_window = horizon_windows.get(investment_horizon, 60)
    
    logger.info(f"  持仓周期: {investment_horizon}, 回撤窗口: {drawdown_window}天")

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
            "动态阈值": 5.0,
            "风险指标": {
                "波动率": 0.0,
                "95%风险价值": 0.0,
                "最大回撤": 0.0,
                "市场风险评分": 10,
                "大盘风险评分": 10,
                "相对风险评分": 5,
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
            content=json.dumps(message_content, ensure_ascii=False),
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

    # ==================== 1. 获取大盘风险指标 ====================
    logger.info("获取沪深300大盘数据...")
    market_data = _get_market_index_data(end_date, drawdown_window)

    if market_data is None:
        logger.warning("无法获取大盘数据，使用个股数据作为大盘代理")

    # ==================== 2. 计算个股风险指标 ====================
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    max_drawdown = (
        prices_df['close'] / prices_df['close'].rolling(window=drawdown_window).max() - 1).min()

    stock_metrics = {
        'volatility': volatility,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown
    }

    # 如果没有大盘数据，使用个股波动率×1.5作为大盘代理
    if market_data is None:
        market_metrics = {
            'volatility': volatility * 0.8,  # 大盘通常比个股波动小
            'var_95': var_95 * 0.8,
            'max_drawdown': max_drawdown * 0.8
        }
    else:
        market_metrics = {
            'volatility': market_data.get('volatility', volatility * 0.8),
            'var_95': market_data.get('var_95', var_95 * 0.8),
            'max_drawdown': market_data.get('max_drawdown', max_drawdown * 0.8)
        }

    # ==================== 3. 获取辩论信号 ====================
    bull_confidence = debate_results.get("bull_confidence", 0.5)
    bear_confidence = debate_results.get("bear_confidence", 0.5)
    debate_confidence = debate_results.get("confidence", 0.5)
    debate_signal = debate_results.get("signal", "neutral")

    # ==================== 4. 计算综合风险评分 ====================
    logger.info("计算综合风险评分...")

    # 从state中获取历史评分（用于动态阈值计算）
    historical_scores = state.get("metadata", {}).get("historical_risk_scores", [])

    risk_result = _calculate_risk_score(
        market_metrics=market_metrics,
        stock_metrics=stock_metrics,
        market_returns=market_data.get('returns', []) if market_data else [],
        debate_signal=debate_signal,
        bull_confidence=bull_confidence,
        bear_confidence=bear_confidence,
        debate_confidence=debate_confidence,
        historical_risk_scores=historical_scores
    )

    risk_score = risk_result['risk_score']
    trading_action = risk_result['trading_action']
    dynamic_threshold = risk_result['dynamic_threshold']

    logger.info(f"风险评分: {risk_score}/10, 动态阈值: {dynamic_threshold:.1f}")
    logger.info(f"大盘风险: {risk_result['components']['market_risk']}, "
                f"个股风险: {risk_result['components']['stock_risk']}, "
                f"相对风险: {risk_result['components']['relative_risk']}")

    # ==================== 5. 计算最大持仓规模 ====================
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value
    current_price = prices_df['close'].iloc[-1]

    # 基础最大仓位25%
    base_position_size = total_portfolio_value * 0.25

    # 根据风险评分调整
    if risk_score >= 8:
        max_position_size = base_position_size * 0.3
    elif risk_score >= dynamic_threshold:
        max_position_size = base_position_size * 0.5
    elif risk_score >= 4:
        max_position_size = base_position_size * 0.75
    else:
        max_position_size = base_position_size

    # ==================== 6. 压力测试 ====================
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

    # ==================== 7. 构建输出消息 ====================
    message_content = {
        "最大持仓规模": float(max_position_size),
        "风险评分": risk_score,
        "交易行动": trading_action,
        "动态阈值": dynamic_threshold,
        "持仓周期": investment_horizon,
        "回撤窗口": drawdown_window,
        "风险指标": {
            "波动率": float(volatility),
            "95%风险价值(VaR)": float(var_95),
            "95%条件风险价值(CVaR)": float(cvar_95),
            "最大回撤": float(max_drawdown),
            "大盘波动率": float(market_metrics['volatility']),
            "大盘VaR": float(market_metrics['var_95']),
            "大盘CVaR": float(market_metrics.get('cvar_95', 0)),
            "大盘回撤": float(market_metrics['max_drawdown']),
            "大盘风险评分": risk_result['components']['market_risk'],
            "个股风险评分": risk_result['components']['stock_risk'],
            "相对风险评分": risk_result['components']['relative_risk'],
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
        "推理": f"综合风险评分 {risk_score}/10: "
                f"大盘风险={risk_result['components']['market_risk']}/10, "
                f"个股风险={risk_result['components']['stock_risk']}/10, "
                f"相对风险={risk_result['components']['relative_risk']}/10, "
                f"动态阈值={dynamic_threshold:.1f}, "
                f"波动率={volatility:.2%}, 大盘波动率={market_metrics['volatility']:.2%}, "
                f"VaR={var_95:.2%}, 回撤={max_drawdown:.2%}"
    }

    # 保存当前评分到metadata（用于未来动态阈值计算）
    if "historical_risk_scores" not in state["metadata"]:
        state["metadata"]["historical_risk_scores"] = []
    state["metadata"]["historical_risk_scores"].append(risk_score)
    # 只保留最近20个评分
    state["metadata"]["historical_risk_scores"] = state["metadata"]["historical_risk_scores"][-20:]

    # Create the risk management message
    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("风险管理师", "completed")

    # 打印最终分析结果
    logger.info("────────────────────────────────────────────────────────")
    logger.info("✅ 风险管理分析完成:")
    logger.info(f"  ⚠️  风险评分: {risk_score}/10")
    logger.info(f"  📊 大盘风险: {risk_result['components']['market_risk']}/10")
    logger.info(f"  📊 个股风险: {risk_result['components']['stock_risk']}/10")
    logger.info(f"  📉 波动率: {volatility:.2%}")
    logger.info(f"  💰 VaR(95%): {var_95:.2%}")
    logger.info(f"  💰 CVaR(95%): {cvar_95:.2%}")
    logger.info(f"  📉 最大回撤: {max_drawdown:.2%}")
    logger.info(f"  🎯 交易行动: {trading_action}")
    logger.info(f"  📈 辩论室信号: {debate_signal} ({debate_confidence:.2%})")
    logger.info("────────────────────────────────────────────────────────")

    # 保存当前价格到portfolio（供后续决策使用）
    updated_portfolio = dict(portfolio)
    updated_portfolio['current_price'] = current_price

    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "risk_analysis": message_content,
            "portfolio": updated_portfolio
        },
        "metadata": state["metadata"],
    }
