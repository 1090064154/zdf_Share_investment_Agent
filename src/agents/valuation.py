from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
from src.agents.fundamentals import INDUSTRY_CYCLE_CLASSIFICATION
import json
import math

# 初始化 logger
logger = setup_logger('valuation_agent')

# [NEW] 动态折现率（基于无风险利率）
def _get_dynamic_discount_rate() -> float:
    """
    根据市场环境动态调整折现率
    基准：10年期国债收益率 + 股权风险溢价
    """
    try:
        import akshare as ak
        try:
            bond_df = ak.china_bond_yield曲线()
            if bond_df is not None and len(bond_df) > 0:
                ten_year_yield = float(bond_df.iloc[0].get('10年', 0) or 2.5)
                risk_premium = 0.05
                return min(0.15, max(0.08, ten_year_yield / 100 + risk_premium))
        except Exception as e:
            logger.debug(f"国债收益率获取失败: {e}")
    except ImportError:
        pass
    return 0.10


def _calculate_peg_ratio(pe_ratio: float, growth_rate: float) -> dict:
    """
    [NEW] 计算PEG估值
    PEG = PE / 增长率
    PEG < 0.8: 低估
    0.8-1.2: 合理
    PEG > 1.2: 高估
    """
    if not pe_ratio or pe_ratio <= 0 or not growth_rate or growth_rate <= 0:
        return {'peg': None, 'signal': 'neutral', 'confidence': 0}
    
    peg = pe_ratio / (growth_rate * 100)
    
    if peg < 0.8:
        signal = 'bullish'
        confidence = min(0.7, 0.5 + (0.8 - peg) * 0.5)
        reason = f"PEG={peg:.2f}，估值偏低"
    elif peg > 1.2:
        signal = 'bearish'
        confidence = min(0.7, 0.5 + (peg - 1.2) * 0.3)
        reason = f"PEG={peg:.2f}，估值偏高"
    else:
        signal = 'neutral'
        confidence = 0.5
        reason = f"PEG={peg:.2f}，估值合理"
    
    return {'peg': round(peg, 2), 'signal': signal, 'confidence': confidence, 'reason': reason}


def _calculate_dividend_yield(market_cap: float, dividend_per_share: float, price: float) -> dict:
    """
    [NEW] 计算股息率估值
    对比无风险利率（10年期国债）
    """
    if not price or price <= 0 or not dividend_per_share or dividend_per_share <= 0:
        return {'yield': None, 'signal': 'neutral', 'confidence': 0}
    
    dividend_yield = (dividend_per_share / price) * 100
    
    try:
        import akshare as ak
        risk_free_rate = 2.5
        try:
            bond_df = ak.china_bond_yield曲线()
            if bond_df is not None:
                risk_free_rate = float(bond_df.iloc[0].get('10年', 0) or 2.5)
        except:
            pass
    except ImportError:
        risk_free_rate = 2.5
    
    if dividend_yield > risk_free_rate * 1.5:
        signal = 'bullish'
        confidence = min(0.7, 0.5 + (dividend_yield - risk_free_rate * 1.5) / risk_free_rate)
        reason = f"股息率{dividend_yield:.2f}%超过无风险利率{risk_free_rate:.2f}%，具吸引力"
    elif dividend_yield < risk_free_rate * 0.5:
        signal = 'bearish'
        confidence = min(0.6, 0.4 + (risk_free_rate * 0.5 - dividend_yield) / risk_free_rate)
        reason = f"股息率{dividend_yield:.2f}%低于无风险利率，吸引力不足"
    else:
        signal = 'neutral'
        confidence = 0.4
        reason = f"股息率{dividend_yield:.2f}%处于合理区间"
    
    return {'yield': round(dividend_yield, 2), 'signal': signal, 'confidence': confidence, 'reason': reason, 'risk_free_rate': risk_free_rate}


# [OPTIMIZED] A股行业分类 - 从fundamentals.py统一导入，避免重复定义


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


def _identify_stock_type(ticker: str, industry: str = None) -> str:
    """
    [OPTIMIZED] 识别股票类型
    返回: 'cyclical', 'growth', 'blue_chip', 'other'
    """
    if not industry:
        return 'other'

    for cycle_type, industries in INDUSTRY_CYCLE_CLASSIFICATION.items():
        if any(ind in industry for ind in industries):
            if cycle_type == '强周期':
                return 'cyclical'
            elif cycle_type == '成长':
                return 'growth'
            elif cycle_type in ['防御', '弱周期']:
                return 'blue_chip'

    return 'other'


def _calculate_pe_pb_percentile(pe_ratio: float, pb_ratio: float) -> dict:
    """
    [OPTIMIZED] PE/PB分位点简化判断
    注意：实际应使用10年历史数据计算分位点，此处使用简化规则
    """
    result = {}

    # PE分位点简化判断
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < 15:
            pe_level = 'low'
        elif pe_ratio < 25:
            pe_level = 'medium'
        elif pe_ratio < 40:
            pe_level = 'high'
        else:
            pe_level = 'very_high'
        result['pe'] = {'level': pe_level, 'value': pe_ratio}

    # PB分位点简化判断
    if pb_ratio and pb_ratio > 0:
        if pb_ratio < 2:
            pb_level = 'low'
        elif pb_ratio < 4:
            pb_level = 'medium'
        elif pb_ratio < 6:
            pb_level = 'high'
        else:
            pb_level = 'very_high'
        result['pb'] = {'level': pb_level, 'value': pb_ratio}

    return result


def _calculate_liquidation_value(financial_line_items: dict) -> float:
    """
    [OPTIMIZED] 计算清算价值（适用于周期股）
    """
    if not financial_line_items:
        return 0

    try:
        cash = financial_line_items.get('cash', 0) or 0
        accounts_receivable = financial_line_items.get('accounts_receivable', 0) or 0
        inventory = financial_line_items.get('inventory', 0) or 0
        fixed_assets = financial_line_items.get('fixed_assets', 0) or 0
        total_liabilities = financial_line_items.get('total_liabilities', 0) or 0

        # 清算价值计算
        # 现金：100%清算
        liquidation_value = cash
        # 应收款：80%清算
        liquidation_value += accounts_receivable * 0.8
        # 存货：50%清算（周期行业存货可能贬值）
        liquidation_value += inventory * 0.5
        # 固定资产：30%清算
        liquidation_value += fixed_assets * 0.3
        # 减去负债
        liquidation_value -= total_liabilities

        return max(liquidation_value, 0)

    except Exception as e:
        logger.debug(f"清算价值计算失败: {e}")
        return 0


def _generate_relative_valuation_signal(percentile_data: dict, industry: str = None) -> dict:
    """
    [OPTIMIZED] 相对估值信号判定
    """
    signals = []
    reasoning = {}

    pe_data = percentile_data.get('pe', {})
    pb_data = percentile_data.get('pb', {})

    # PE分位点判断
    if pe_data:
        pe_level = pe_data.get('level', 'medium')
        pe_value = pe_data.get('value', 0)

        if pe_level == 'low':
            signal = 'bullish'
            confidence = 0.7
            reason = f"PE({pe_value:.1f})处于历史低位"
        elif pe_level == 'medium':
            signal = 'neutral'
            confidence = 0.5
            reason = f"PE({pe_value:.1f})处于合理区间"
        elif pe_level == 'high':
            signal = 'bearish'
            confidence = 0.6
            reason = f"PE({pe_value:.1f})处于历史高位"
        else:  # very_high
            signal = 'bearish'
            confidence = 0.7
            reason = f"PE({pe_value:.1f})极高，估值风险大"

        signals.append((signal, confidence, reason))
        reasoning['pe_analysis'] = {'signal': signal, 'details': reason}

    # PB分位点判断
    if pb_data:
        pb_level = pb_data.get('level', 'medium')
        pb_value = pb_data.get('value', 0)

        if pb_level == 'low':
            pb_signal = 'bullish'
            pb_conf = 0.7
            pb_reason = f"PB({pb_value:.1f})处于历史低位"
        elif pb_level == 'medium':
            pb_signal = 'neutral'
            pb_conf = 0.5
            pb_reason = f"PB({pb_value:.1f})处于合理区间"
        elif pb_level == 'high':
            pb_signal = 'bearish'
            pb_conf = 0.6
            pb_reason = f"PB({pb_value:.1f})处于历史高位"
        else:
            pb_signal = 'bearish'
            pb_conf = 0.7
            pb_reason = f"PB({pb_value:.1f})极高"

        signals.append((pb_signal, pb_conf, pb_reason))
        reasoning['pb_analysis'] = {'signal': pb_signal, 'details': pb_reason}

    # 综合判定
    if not signals:
        return {'signal': 'neutral', 'confidence': 0.3, 'reasoning': '估值数据不足'}

    bullish_count = sum(1 for s, c, _ in signals if s == 'bullish')
    bearish_count = sum(1 for s, c, _ in signals if s == 'bearish')

    if bullish_count > bearish_count:
        avg_conf = sum(c for s, c, _ in signals if s == 'bullish') / bullish_count
        return {'signal': 'bullish', 'confidence': avg_conf, 'reasoning': '估值偏低'}
    elif bearish_count > bullish_count:
        avg_conf = sum(c for s, c, _ in signals if s == 'bearish') / bearish_count
        return {'signal': 'bearish', 'confidence': avg_conf, 'reasoning': '估值偏高'}
    else:
        return {'signal': 'neutral', 'confidence': 0.4, 'reasoning': '估值合理'}


@resilient_agent
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

    show_agent_reasoning({"市值": f"{market_cap/1e8:.1f}亿"}, "估值Agent")

    reasoning = {}

    if market_cap <= 0 or not current_financial_line_item:
        reason = "估值输入不足：市值缺失或财务报表不可用。"
        message_content = {
            "signal": "neutral",
            "confidence": 0.0,
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content, ensure_ascii=False),
            name="valuation_agent",
        )
        if show_reasoning:
            state["metadata"]["agent_reasoning"] = message_content
        show_agent_reasoning({"最终信号": "中性", "置信度": "0%", "原因": "估值输入不足"}, "估值Agent")
        show_workflow_complete(
            "估值Agent",
            signal="neutral",
            confidence=0.0,
            details=message_content,
            message="估值分析完成：数据输入不足"
        )
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
        message_content = {
            "signal": "neutral",
            "confidence": 0.0,
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content, ensure_ascii=False),
            name="valuation_agent",
        )
        if show_reasoning:
            state["metadata"]["agent_reasoning"] = message_content
        show_agent_reasoning({"最终信号": "中性", "置信度": "0%", "原因": "关键财务数据缺失"}, "估值Agent")
        show_workflow_complete(
            "估值Agent",
            signal="neutral",
            confidence=0.0,
            details=message_content,
            message="估值分析完成：关键财务数据缺失"
        )
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

    # [NEW] 获取动态折现率（必须在使用前定义）
    dynamic_discount_rate = _get_dynamic_discount_rate()
    logger.info(f"[估值] 动态折现率: {dynamic_discount_rate:.2%}")

    # Owner Earnings Valuation (Buffett Method)
    required_return = dynamic_discount_rate + 0.05
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=metrics.get("earnings_growth", 0),
        required_return=required_return,
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
        discount_rate=dynamic_discount_rate,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    # [OPTIMIZED] 收集有效的估值结果 - 调整权重，使用相对估值为主
    valid_valuations = []
    reasoning = {}

    # 获取股票类型
    industry = data.get('industry', '')
    stock_type = _identify_stock_type(data.get('ticker', ''), industry)
    
    # [OPTIMIZED] 根据股票类型选择估值方法优先级
    # 周期股：相对估值(PB分位点) > 清算价值 > DCF
    # 成长股：DCF > PEG > 相对估值
    # 其他：相对估值 > DCF

    # 1. PE/PB分位点分析（相对估值）
    pe_ratio = metrics.get("pe_ratio", 0)
    pb_ratio = metrics.get("price_to_book", 0)
    percentile_data = _calculate_pe_pb_percentile(pe_ratio, pb_ratio)
    relative_signal = _generate_relative_valuation_signal(percentile_data, industry)

    if relative_signal['signal'] != 'neutral' or percentile_data:
        weight = 0.40 if stock_type == 'cyclical' else 0.25
        valid_valuations.append({
            "method": "relative_valuation",
            "gap": relative_signal.get('confidence', 0.5) * (1 if relative_signal['signal'] == 'bullish' else -1),
            "weight": weight,
            "signal": relative_signal['signal']
        })
        reasoning["relative_analysis"] = {
            "signal": relative_signal['signal'],
            "details": f"分位点分析: {relative_signal.get('reasoning', '')}"
        }
        logger.info(f"  📊 相对估值: {relative_signal['signal']} - {relative_signal.get('reasoning', '')}")
        show_agent_reasoning({"reasoning": {"relative_analysis": {"signal": relative_signal['signal'], "details": relative_signal.get('reasoning', '')}}}, "估值分析师")

    # [NEW] 2. PEG估值（成长股专用）
    growth_rate = metrics.get("earnings_growth", 0) or 0
    if stock_type == 'growth' and pe_ratio > 0 and growth_rate > 0:
        peg_result = _calculate_peg_ratio(pe_ratio, growth_rate)
        if peg_result.get('peg'):
            valid_valuations.append({
                "method": "peg_valuation",
                "gap": peg_result.get('confidence', 0.5) * (1 if peg_result['signal'] == 'bullish' else -1),
                "weight": 0.30,
                "signal": peg_result['signal']
            })
            reasoning["peg_analysis"] = {
                "signal": peg_result['signal'],
                "details": f"PEG={peg_result['peg']:.2f}，{peg_result.get('reason', '')}"
            }
            logger.info(f"  📈 PEG估值: {peg_result['signal']} - {peg_result.get('reason', '')}")
            show_agent_reasoning({"reasoning": {"peg_analysis": {"signal": peg_result['signal'], "details": f"PEG={peg_result['peg']:.2f} ({peg_result.get('reason', '')})"}}}, "估值分析师")

    # [NEW] 3. 股息率估值
    dividend_per_share = metrics.get("dividend_per_share", 0) or 0
    peg_result = {'peg': None, 'signal': 'neutral'}
    dividend_result = {'yield': None, 'signal': 'neutral'}
    
    if current_price > 0 and dividend_per_share > 0:
        dividend_result = _calculate_dividend_yield(market_cap, dividend_per_share, current_price)
        if dividend_result.get('yield'):
            valid_valuations.append({
                "method": "dividend_valuation",
                "gap": dividend_result.get('confidence', 0.4) * (1 if dividend_result['signal'] == 'bullish' else -1),
                "weight": 0.20,
                "signal": dividend_result['signal']
            })
            reasoning["dividend_analysis"] = {
                "signal": dividend_result['signal'],
                "details": f"股息率{dividend_result['yield']:.2f}%，{dividend_result.get('reason', '')}"
            }
            logger.info(f"  💰 股息率: {dividend_result['signal']} - {dividend_result.get('reason', '')}")
            show_agent_reasoning({"reasoning": {"dividend_analysis": {"signal": dividend_result['signal'], "details": f"股息率{dividend_result['yield']:.2f}%"}}}, "估值分析师")

    # [OPTIMIZED] 2. 周期股清算价值
    if stock_type == 'cyclical' and current_financial_line_item:
        liquidation_value = _calculate_liquidation_value(current_financial_line_item)
        if liquidation_value > 0 and market_cap > 0:
            liq_gap = (liquidation_value - market_cap) / market_cap
            liq_signal = "bullish" if liq_gap > 0.3 else "bearish" if liq_gap < -0.3 else "neutral"
            if liq_signal != "neutral":
                valid_valuations.append({
                    "method": "liquidation_value",
                    "gap": liq_gap,
                    "weight": 0.30,
                    "signal": liq_signal
                })
                reasoning["liquidation_analysis"] = {
                    "signal": liq_signal,
                    "details": f"清算价值: {liquidation_value/1e8:.1f}亿, 市值: {market_cap/1e8:.1f}亿"
                }
                logger.info(f"  💰 清算价值: {liq_signal} - 清算价值{liquidation_value/1e8:.1f}亿 vs 市值{market_cap/1e8:.1f}亿")
                show_agent_reasoning({"reasoning": {"liquidation_analysis": {"signal": liq_signal, "details": f"清算价值{liquidation_value/1e8:.1f}亿 vs 市值{market_cap/1e8:.1f}亿"}}}, "估值分析师")

    # 3. DCF分析（权重降低）
    if dcf_value > 0:
        dcf_gap = (dcf_value - market_cap) / market_cap
        capped_dcf_gap = max(-1.0, min(1.0, dcf_gap))
        # DCF权重降低：周期股15%，其他25%
        dcf_weight = 0.15 if stock_type == 'cyclical' else 0.25
        valid_valuations.append({
            "method": "dcf",
            "gap": dcf_gap,
            "weight": dcf_weight,
            "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.25 else "neutral"
        })
        reasoning["dcf_analysis"] = {
            "signal": valid_valuations[-1]["signal"],
            "details": f"DCF内在价值: {dcf_value/1e8:.1f}亿, 市值: {market_cap/1e8:.1f}亿, 差距: {capped_dcf_gap:.1%}"
        }
        logger.info(f"  📈 DCF估值: {valid_valuations[-1]['signal']} - DCF内在价值{dcf_value/1e8:.1f}亿 vs 市值{market_cap/1e8:.1f}亿")
        show_agent_reasoning({"reasoning": {"dcf_analysis": {"signal": valid_valuations[-1]["signal"], "details": f"DCF内在价值{dcf_value/1e8:.1f}亿 vs 市值{market_cap/1e8:.1f}亿"}}}, "估值分析师")
    else:
        reasoning["dcf_analysis"] = {
            "signal": "neutral",
            "details": "现金流数据不足，跳过DCF"
        }

    # 4. 所有者收益分析（权重降低）
    if owner_earnings_value > 0:
        owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
        capped_owner_gap = max(-1.0, min(1.0, owner_earnings_gap))
        valid_valuations.append({
            "method": "owner_earnings",
            "gap": owner_earnings_gap,
            "weight": 0.15,  # 降低权重
            "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.25 else "neutral"
        })
        reasoning["owner_earnings_analysis"] = {
            "signal": valid_valuations[-1]["signal"],
            "details": f"所有者收益: {owner_earnings_value/1e8:.1f}亿"
        }
        logger.info(f"  💎 所有者收益: {valid_valuations[-1]['signal']} - 所有者收益{owner_earnings_value/1e8:.1f}亿")
        show_agent_reasoning({"reasoning": {"owner_earnings_analysis": {"signal": valid_valuations[-1]["signal"], "details": f"所有者收益{owner_earnings_value/1e8:.1f}亿"}}}, "估值分析师")
    else:
        reasoning["owner_earnings_analysis"] = {
            "signal": "neutral",
            "details": "盈利数据不足，跳过所有者收益"
        }

    # [OPTIMIZED] 注意：PE/PB分位点分析已在上面relative_valuation中完成

    # 检查是否是亏损公司（保持原有逻辑用于营收分析）
    net_income = current_financial_line_item.get("net_income", 0)
    is_profitable = net_income and net_income > 0

    # 对于亏损公司，添加基本面估值分析
    if not is_profitable and (pe_ratio <= 0 or pb_ratio > 0):
        # 基于营收和资产给出一个参考分析
        revenue = current_financial_line_item.get("operating_revenue", 0)
        if revenue and revenue > 0:
            # 营收增速
            prev_revenue = previous_financial_line_item.get("operating_revenue", 0) if previous_financial_line_item else 0
            if prev_revenue and prev_revenue > 0:
                revenue_growth = (revenue - prev_revenue) / prev_revenue
                revenue_signal = "bullish" if revenue_growth > 0.1 else "bearish" if revenue_growth < -0.1 else "neutral"
                valid_valuations.append({
                    "method": "revenue_analysis",
                    "gap": revenue_growth,
                    "weight": 0.15,
                    "signal": revenue_signal
                })
                reasoning["revenue_analysis"] = {
                    "signal": revenue_signal,
                    "details": f"营收: {revenue/1e8:.2f}亿, 同比增速: {revenue_growth*100:.1f}%"
                }

    # 如果没有有效的估值结果
    if not valid_valuations:
        reason = "所有估值方法都未得到有效结果，无法进行估值分析。"
        logger.warning(reason)
        message_content = {
            "signal": "neutral",
            "confidence": 0.0,
            "reasoning": {
                "fallback": {
                    "signal": "neutral",
                    "details": reason
                }
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content, ensure_ascii=False),
            name="valuation_agent",
        )
        if show_reasoning:
            state["metadata"]["agent_reasoning"] = message_content
        show_agent_reasoning({
            "最终信号": "中性",
            "置信度": "0%",
            "原因": message_content.get("reasoning", {}).get("fallback", {}).get("details", "数据不足")
        }, "估值Agent")
        show_workflow_complete(
            "估值Agent",
            signal="neutral",
            confidence=0.0,
            details=message_content,
            message="估值分析完成：数据不足，无法估值"
        )
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

    # 获取当前股价
    prices = data.get("prices", [])
    current_price = 0
    if isinstance(prices, list) and len(prices) > 0:
        current_price = prices[-1].get("close", 0) if isinstance(prices[-1], dict) else 0

    # 推断合理价格（基于加权差距）
    if current_price > 0 and abs(weighted_gap) < 100:
        fair_value = current_price * (1 + weighted_gap)
    else:
        fair_value = 0

    # 计算折扣率（正=低估，负=高估）
    if current_price > 0 and fair_value > 0:
        discount = (fair_value - current_price) / current_price
    else:
        discount = 0

    # 构建清晰的方法列表
    methods = []
    for v in valid_valuations:
        method_name_map = {
            "dcf": "DCF估值",
            "owner_earnings": "所有者收益",
            "relative_valuation": "相对估值(PE/PB)",
            "liquidation_value": "清算价值",
            "revenue_analysis": "营收分析",
            "peg_valuation": "PEG估值",
            "dividend_valuation": "股息率估值",
        }
        name = method_name_map.get(v["method"], v["method"])
        signal_cn = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(v["signal"], v["signal"])
        methods.append({
            "name": name,
            "signal": v["signal"],
            "signal_cn": signal_cn,
            "weight": v["weight"],
        })

    # 统计各信号的数量
    bullish_count = sum(1 for v in valid_valuations if v["signal"] == "bullish")
    bearish_count = sum(1 for v in valid_valuations if v["signal"] == "bearish")
    neutral_count = sum(1 for v in valid_valuations if v["signal"] == "neutral")

    # 确定最终信号
    if bearish_count > bullish_count and bearish_count >= len(valid_valuations) / 2:
        signal = "bearish"
        confidence = min(abs(weighted_gap) * 100, 90) / 100
    elif bullish_count > bearish_count and bullish_count >= len(valid_valuations) / 2:
        signal = "bullish"
        confidence = min(abs(weighted_gap) * 100, 90) / 100
    else:
        signal = "neutral"
        confidence = max(10, 50 - abs(bullish_count - bearish_count) * 10) / 100

    confidence = max(0.1, min(0.95, confidence))

    # 生成人类可读的摘要
    signal_cn = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(signal, signal)
    price_info = ""
    if current_price > 0 and fair_value > 0:
        discount_pct = discount * 100
        if discount > 0.05:
            price_info = f"当前价{current_price:.2f}元低于合理价{fair_value:.2f}元，折扣率{discount_pct:.1f}%"
        elif discount < -0.05:
            price_info = f"当前价{current_price:.2f}元高于合理价{fair_value:.2f}元，溢价率{abs(discount_pct):.1f}%"
        else:
            price_info = f"当前价{current_price:.2f}元接近合理价{fair_value:.2f}元"
    elif current_price > 0:
        price_info = f"当前价{current_price:.2f}元（无法确定合理价）"
    else:
        price_info = "无价格数据"

    method_names = [m["name"] for m in methods]
    summary = f"估值{signal_cn}（置信度{confidence*100:.0f}%），{price_info}。采用方法：{'、'.join(method_names)}"

    message_content = {
        "signal": signal,
        "confidence": round(confidence, 4),
        "current_price": round(current_price, 2) if current_price else None,
        "fair_value": round(fair_value, 2) if fair_value else None,
        "discount": round(discount, 4) if current_price else None,
        "methods": methods,
        "summary": summary,
        "reasoning": reasoning,
        "pe_ratio": pe_ratio if pe_ratio else None,
        "pb_ratio": pb_ratio if pb_ratio else None,
        "stock_type": stock_type,
        "discount_rate": dynamic_discount_rate,
        "dcf_value": round(dcf_value / 1e8, 2) if dcf_value else None,
        "owner_earnings_value": round(owner_earnings_value / 1e8, 2) if owner_earnings_value else None,
        "peg_ratio": peg_result.get('peg') if stock_type == 'growth' else None,
        "dividend_yield": dividend_result.get('yield') if dividend_per_share > 0 else None,
        "growth_rate": round(growth_rate * 100, 2) if growth_rate else None,
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="valuation_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(message_content.get('signal', 'neutral'), message_content.get('signal', 'neutral'))

    method_descriptions = {
        'DCF': '现金流折现法',
        '所有者收益': '巴菲特估值法（净利润+折旧摊销-资本支出）',
        '相对估值': 'PE/PB分位点对比',
        '清算价值': '资产变现能力',
        'PE估值': '市盈率法',
        'PB估值': '市净率法'
    }

    # 构建友好的推理输出
    show_reasoning = {}
    show_reasoning["最终信号"] = signal_cn
    show_reasoning["置信度"] = f"{message_content.get('confidence', 0)*100:.0f}%"
    if message_content.get("current_price"):
        show_reasoning["当前股价"] = f"{message_content['current_price']:.2f}元"
    if message_content.get("fair_value"):
        show_reasoning["合理价"] = f"{message_content['fair_value']:.2f}元"
    if message_content.get("discount") is not None:
        pct = message_content["discount"] * 100
        show_reasoning["估值对比"] = f"{'低估' if pct > 0 else '高估' if pct < 0 else '公允'}({pct:+.1f}%)"

    for m in message_content.get("methods", []):
        method_desc = method_descriptions.get(m["name"], m["name"])
        show_reasoning[m["name"]] = f"{m['signal_cn']}（权重{m['weight']*100:.0f}%）"

    if 'DCF' in message_content.get('reasoning', {}):
        dcf_data = message_content['reasoning'].get('DCF', {})
        if dcf_data.get('details'):
            show_reasoning["DCF说明"] = dcf_data['details'][:50] + "..."

    if '所有者收益' in message_content.get('reasoning', {}):
        oe_data = message_content['reasoning'].get('所有者收益', {})
        if oe_data.get('details'):
            show_reasoning["所有者收益"] = oe_data['details'][:50] + "..."

    show_agent_reasoning(show_reasoning, "估值Agent")

    show_workflow_complete(
        "估值Agent",
        signal=message_content.get('signal', 'neutral'),
        confidence=message_content.get('confidence', 0) if isinstance(message_content.get('confidence'), (int, float)) else 0,
        details=message_content,
        message=f"估值分析完成：{message_content.get('summary', '估值分析')}"
    )

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
