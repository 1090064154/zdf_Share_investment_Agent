from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
import math

# 初始化 logger
logger = setup_logger('valuation_agent')


def _safe_first(items):
    return items[0] if isinstance(items, list) and items else {}


def _safe_second(items):
    if isinstance(items, list) and len(items) > 1:
        return items[1]
    return _safe_first(items)


def _is_meaningful_number(value) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric != 0


def _has_meaningful_valuation_inputs(current_financial_line_item: dict) -> bool:
    required_keys = [
        "net_income",
        "depreciation_and_amortization",
        "capital_expenditure",
        "working_capital",
        "free_cash_flow",
    ]
    return any(_is_meaningful_number(current_financial_line_item.get(key)) for key in required_keys)


@agent_endpoint("valuation", "估值分析师，使用DCF和所有者收益法评估公司内在价值")
def valuation_agent(state: AgentState):
    """Responsible for valuation analysis"""
    show_workflow_status("估值Agent")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = _safe_first(data.get("financial_metrics", []))
    current_financial_line_item = _safe_first(data.get("financial_line_items", []))
    previous_financial_line_item = _safe_second(data.get("financial_line_items", []))
    market_cap = data.get("market_cap") or 0
    logger.info(
        "Valuation inputs prepared: market_cap=%s metrics_keys=%s current_line_item_keys=%s previous_line_item_keys=%s",
        market_cap,
        list(metrics.keys()) if isinstance(metrics, dict) else type(metrics).__name__,
        list(current_financial_line_item.keys()) if isinstance(current_financial_line_item, dict) else type(current_financial_line_item).__name__,
        list(previous_financial_line_item.keys()) if isinstance(previous_financial_line_item, dict) else type(previous_financial_line_item).__name__,
    )

    reasoning = {}

    if market_cap <= 0 or not current_financial_line_item:
        reason = "估值输入不足：市值缺失或财务报表不可用。"
        logger.warning(reason)
        message_content = {
            "signal": "neutral",
            "confidence": "0%",
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "估值分析Agent")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("估值Agent", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "valuation_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    if not _has_meaningful_valuation_inputs(current_financial_line_item):
        reason = "估值输入不足：关键现金流和利润字段缺失或均为0，回退为中性。"
        logger.warning(reason)
        message_content = {
            "signal": "neutral",
            "confidence": "0%",
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "估值分析Agent")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("估值Agent", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "valuation_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # Calculate working capital change
    working_capital_change = (current_financial_line_item.get(
        'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

    # Owner Earnings Valuation (Buffett Method)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=metrics.get("earnings_growth", 0),
        required_return=0.15,
        margin_of_safety=0.25
    )

    # DCF Valuation
    # 如果free_cash_flow为0或负数，使用净利润作为替代
    fcff = current_financial_line_item.get('free_cash_flow', 0)
    if not fcff or fcff <= 0:
        fcff = current_financial_line_item.get('net_income', 0)
    dcf_value = calculate_intrinsic_value(
        free_cash_flow=fcff,
        growth_rate=metrics.get("earnings_growth", 0),
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    # 收集有效的估值结果
    valid_valuations = []
    reasoning = {}

    # DCF分析
    if dcf_value > 0:
        dcf_gap = (dcf_value - market_cap) / market_cap
        capped_dcf_gap = max(-1.0, min(1.0, dcf_gap))
        valid_valuations.append({
            "method": "dcf",
            "gap": dcf_gap,
            "weight": 0.4,  # DCF权重40%
            "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.25 else "neutral"
        })
        reasoning["dcf_analysis"] = {
            "signal": valid_valuations[-1]["signal"],
            "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {capped_dcf_gap:.1%}"
        }
    else:
        reasoning["dcf_analysis"] = {
            "signal": "neutral",
            "details": f"DCF计算无效（自由现金流: ${fcff:,.2f}），已跳过"
        }

    # 所有者收益分析
    if owner_earnings_value > 0:
        owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
        capped_owner_gap = max(-1.0, min(1.0, owner_earnings_gap))
        valid_valuations.append({
            "method": "owner_earnings",
            "gap": owner_earnings_gap,
            "weight": 0.35,  # 所有者收益权重35%
            "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.25 else "neutral"
        })
        reasoning["owner_earnings_analysis"] = {
            "signal": valid_valuations[-1]["signal"],
            "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {capped_owner_gap:.1%}"
        }
    else:
        reasoning["owner_earnings_analysis"] = {
            "signal": "neutral",
            "details": "所有者收益计算无效，已跳过"
        }

    # 市盈率分析（作为补充）
    pe_ratio = metrics.get("pe_ratio", 0)
    if pe_ratio > 0:
        # 行业平均市盈率参考
        industry_avg_pe = 20  # 默认使用20倍作为参考
        pe_gap = (industry_avg_pe - pe_ratio) / pe_ratio
        pe_signal = "bullish" if pe_gap > 0.3 else "bearish" if pe_gap < -0.3 else "neutral"
        valid_valuations.append({
            "method": "pe_ratio",
            "gap": pe_gap,
            "weight": 0.25,  # 市盈率权重25%
            "signal": pe_signal
        })
        reasoning["pe_analysis"] = {
            "signal": pe_signal,
            "details": f"市盈率: {pe_ratio:.2f}, 行业平均: {industry_avg_pe}, 相对估值: {'低估' if pe_gap > 0 else '高估' if pe_gap < 0 else '合理'}"
        }

    # 如果没有有效的估值结果
    if not valid_valuations:
        reason = "所有估值方法都未得到有效结果，无法进行估值分析。"
        logger.warning(reason)
        message_content = {
            "signal": "neutral",
            "confidence": "0%",
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "估值分析Agent")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("估值Agent", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "valuation_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # 计算加权平均估值差距
    total_weight = sum(v["weight"] for v in valid_valuations)
    weighted_gap = sum(v["gap"] * v["weight"] for v in valid_valuations) / total_weight

    # 统计各信号的数量
    bullish_count = sum(1 for v in valid_valuations if v["signal"] == "bullish")
    bearish_count = sum(1 for v in valid_valuations if v["signal"] == "bearish")
    neutral_count = sum(1 for v in valid_valuations if v["signal"] == "neutral")

    # 确定最终信号
    # 如果多数方法看空，则看空；如果多数看多，则看多；否则中性
    if bearish_count > bullish_count and bearish_count >= len(valid_valuations) / 2:
        signal = "bearish"
        # 置信度基于加权差距的绝对值
        confidence = min(abs(weighted_gap) * 100, 90)
    elif bullish_count > bearish_count and bullish_count >= len(valid_valuations) / 2:
        signal = "bullish"
        confidence = min(abs(weighted_gap) * 100, 90)
    else:
        signal = "neutral"
        # 中性信号的置信度基于估值方法的一致性
        confidence = 50 - abs(bullish_count - bearish_count) * 10

    # 确保置信度在合理范围内
    confidence = max(10, min(95, confidence))
    confidence_str = f"{confidence:.0f}%"

    message_content = {
        "signal": signal,
        "confidence": confidence_str,
        "reasoning": reasoning
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("估值Agent", "completed")
    # logger.info(
    # f"--- DEBUG: valuation_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "valuation_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5


) -> float:
    """
    使用改进的所有者收益法计算公司价值。

    Args:
        net_income: 净利润
        depreciation: 折旧和摊销
        capex: 资本支出
        working_capital_change: 营运资金变化
        growth_rate: 预期增长率
        required_return: 要求回报率
        margin_of_safety: 安全边际
        num_years: 预测年数

    Returns:
        float: 计算得到的公司价值
    """
    try:
        # 数据有效性检查
        if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
            return 0

        # 计算初始所有者收益
        owner_earnings = (
            net_income +
            depreciation -
            capex -
            working_capital_change
        )

        if owner_earnings <= 0:
            return 0

        # 调整增长率，确保合理性
        growth_rate = min(max(growth_rate, 0), 0.25)  # 限制在0-25%之间

        # 计算预测期收益现值
        future_values = []
        for year in range(1, num_years + 1):
            # 使用递减增长率模型
            year_growth = growth_rate * (1 - year / (2 * num_years))  # 增长率逐年递减
            future_value = owner_earnings * (1 + year_growth) ** year
            discounted_value = future_value / (1 + required_return) ** year
            future_values.append(discounted_value)

        # 计算永续价值
        terminal_growth = min(growth_rate * 0.4, 0.03)  # 永续增长率取增长率的40%或3%的较小值
        terminal_value = (
            future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
        terminal_value_discounted = terminal_value / \
            (1 + required_return) ** num_years

        # 计算总价值并应用安全边际
        intrinsic_value = sum(future_values) + terminal_value_discounted
        value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

        return max(value_with_safety_margin, 0)  # 确保不返回负值

    except Exception as e:
        print(f"所有者收益计算错误: {e}")
        return 0


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    使用改进的DCF方法计算内在价值，考虑增长率和风险因素。

    Args:
        free_cash_flow: 自由现金流
        growth_rate: 预期增长率
        discount_rate: 基础折现率
        terminal_growth_rate: 永续增长率
        num_years: 预测年数

    Returns:
        float: 计算得到的内在价值
    """
    try:
        if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
            return 0

        # 调整增长率，确保合理性
        growth_rate = min(max(growth_rate, 0), 0.25)  # 限制在0-25%之间

        # 调整永续增长率，不能超过经济平均增长
        terminal_growth_rate = min(growth_rate * 0.4, 0.03)  # 取增长率的40%或3%的较小值

        # 计算预测期现金流现值
        present_values = []
        for year in range(1, num_years + 1):
            future_cf = free_cash_flow * (1 + growth_rate) ** year
            present_value = future_cf / (1 + discount_rate) ** year
            present_values.append(present_value)

        # 计算永续价值
        terminal_year_cf = free_cash_flow * (1 + growth_rate) ** num_years
        terminal_value = terminal_year_cf * \
            (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        terminal_present_value = terminal_value / \
            (1 + discount_rate) ** num_years

        # 总价值
        total_value = sum(present_values) + terminal_present_value

        return max(total_value, 0)  # 确保不返回负值

    except Exception as e:
        print(f"DCF计算错误: {e}")
        return 0


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital
