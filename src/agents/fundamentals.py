from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent

import json

# 初始化 logger
logger = setup_logger('fundamentals_agent')

# [OPTIMIZED] A股行业周期分类
INDUSTRY_CYCLE_CLASSIFICATION = {
    '强周期': ['农林牧渔', '钢铁', '煤炭', '有色金属', '化工', '建材', '房地产', '汽车', '交通运输', '工程机械'],
    '弱周期': ['食品饮料', '医药生物', '家用电器', '纺织服装', '日用化工'],
    '成长': ['电子', '计算机', '通信', '传媒', '新能源'],
    '防御': ['公用事业', '银行', '保险', '券商']
}


def _is_missing_metric(value) -> bool:
    return value is None or value == 0


def _identify_cyclical_stock(ticker: str, industry: str = None) -> tuple:
    """
    [OPTIMIZED] 识别是否为周期股
    返回: (is_cyclical: bool, reason: str)
    """
    if not industry:
        # 简单根据股票代码判断（仅作为后备方案）
        # 农林牧渔养殖类通常是周期股
        if ticker.startswith('00') or ticker.startswith('30'):
            # 中小创股票可能包含养殖类
            return False, "无法确定行业，默认非周期股"

    # 行业判断
    for cycle_type, industries in INDUSTRY_CYCLE_CLASSIFICATION.items():
        if industry and any(ind in industry for ind in industries):
            return True, f"属于周期行业: {industry}"

    return False, "非周期行业"


def _analyze_pb_roe_percentile(pb_ratio: float, roe: float) -> dict:
    """
    [OPTIMIZED] PB-ROE分位点简化判断
    注意：实际应使用10年历史数据计算分位点，此处使用简化规则
    """
    # 简化规则：
    # 低PB (<3) + 高ROE (>10%) = 价值区域
    # 高PB (>8) = 风险区域
    # 周期股：PB更重要，ROE可正可负

    if pb_ratio is None or pb_ratio <= 0:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reasoning': 'PB数据无效',
            'pb_percentile': None,
            'roe_percentile': None
        }

    # 简化分位点估算
    if pb_ratio < 3:
        pb_level = 'low'
    elif pb_ratio < 6:
        pb_level = 'medium'
    elif pb_ratio < 10:
        pb_level = 'high'
    else:
        pb_level = 'very_high'

    if roe is None or roe <= 0:
        roe_level = 'low'
    elif roe < 10:
        roe_level = 'medium'
    elif roe < 20:
        roe_level = 'high'
    else:
        roe_level = 'very_high'

    # 投资象限分析
    if pb_level == 'low' and roe_level in ['high', 'very_high']:
        signal = 'bullish'
        confidence = 0.7
        reasoning = f"低PB({pb_ratio:.1f})+高ROE({roe:.1f}%)，最佳价值区域"
    elif pb_level == 'very_high':
        signal = 'bearish'
        confidence = 0.7
        reasoning = f"PB({pb_ratio:.1f})过高，存在估值风险"
    elif pb_level == 'high' and roe_level == 'low':
        signal = 'bearish'
        confidence = 0.6
        reasoning = f"高PB({pb_ratio:.1f})+低ROE({roe:.1f}%)，盈利支撑不足"
    elif pb_level == 'low' and roe_level == 'low':
        signal = 'neutral'
        confidence = 0.4
        reasoning = f"低PB({pb_ratio:.1f})+低ROE({roe:.1f}%)，可能处于行业周期底部"
    else:
        signal = 'neutral'
        confidence = 0.5
        reasoning = f"PB({pb_ratio:.1f}), ROE({roe:.1f}%)处于合理区间"

    return {
        'signal': signal,
        'confidence': confidence,
        'reasoning': reasoning,
        'pb_percentile': {'level': pb_level, 'value': pb_ratio},
        'roe_percentile': {'level': roe_level, 'value': roe}
    }


def _analyze_revenue_quality(financial_line_items: dict = None) -> dict:
    """
    [OPTIMIZED] 营收质量分析
    分析应收款/营收比例
    """
    if not financial_line_items:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reasoning': '无财务报表数据'
        }

    # 尝试从financial_line_items获取数据
    accounts_receivable = financial_line_items.get('accounts_receivable', 0)
    revenue = financial_line_items.get('operating_revenue', 1)

    if not revenue or revenue == 0 or not accounts_receivable:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reasoning': '营收或应收款数据缺失'
        }

    ar_ratio = accounts_receivable / revenue if revenue else 0

    if ar_ratio < 0.1:
        signal = 'bullish'
        confidence = 0.7
        reasoning = f"应收款占比仅{ar_ratio:.1%}，营收质量高"
    elif ar_ratio < 0.2:
        signal = 'neutral'
        confidence = 0.5
        reasoning = f"应收款占比{ar_ratio:.1%}，处于正常范围"
    else:
        signal = 'bearish'
        confidence = 0.6
        reasoning = f"应收款占比{ar_ratio:.1%}过高，存在坏账风险"

    return {
        'signal': signal,
        'confidence': confidence,
        'reasoning': reasoning,
        'ar_ratio': ar_ratio
    }

##### Fundamental Agent #####

@resilient_agent
@agent_endpoint("fundamentals", "基本面分析师，分析公司财务指标、盈利能力和增长潜力")
def fundamentals_agent(state: AgentState):
    """Responsible for fundamental analysis"""
    show_workflow_status("基本面分析师")
    logger.info("="*50)
    logger.info("📝 [FUNDAMENTALS] 开始基本面分析")
    logger.info("="*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    financial_metrics = data.get("financial_metrics", {})
    if isinstance(financial_metrics, dict):
        metrics = financial_metrics
    elif isinstance(financial_metrics, list) and len(financial_metrics) > 0:
        metrics = financial_metrics[0]
    else:
        metrics = {}

    populated_metrics = [
        key for key, value in metrics.items()
        if not _is_missing_metric(value)
    ] if isinstance(metrics, dict) else []

    if not populated_metrics:
        reason = "Financial metrics unavailable, defaulting fundamentals analysis to neutral."
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
            name="fundamentals_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "基本面分析师")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("基本面分析师", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "fundamental_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # Initialize signals list for different fundamental aspects
    signals = []
    reasoning = {}

    def _valid_num(v) -> bool:
        """Check if value is a valid non-zero, non-NaN number"""
        import math
        if v is None:
            return False
        try:
            fv = float(v)
            return not (math.isnan(fv) or math.isinf(fv) or fv == 0)
        except (TypeError, ValueError):
            return False

    # 1. Profitability Analysis
    return_on_equity = metrics.get("return_on_equity", 0) or 0
    net_margin = metrics.get("net_margin", 0) or 0
    operating_margin = metrics.get("operating_margin", 0) or 0

    thresholds = [
        (return_on_equity, 0.15),
        (net_margin, 0.20),
        (operating_margin, 0.15)
    ]
    profitability_score = sum(
        _valid_num(metric) and metric > threshold
        for metric, threshold in thresholds
    )

    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')

    roe_str = f"净资产收益率(ROE): {return_on_equity:.2%}" if _valid_num(return_on_equity) else "净资产收益率(ROE): N/A"
    netm_str = f"净利率(Net Margin): {net_margin:.2%}" if _valid_num(net_margin) else "净利率(Net Margin): N/A"
    opm_str = f"营业利润率(Op Margin): {operating_margin:.2%}" if _valid_num(operating_margin) else "营业利润率(Op Margin): N/A"

    reasoning["profitability_signal"] = {
        "signal": signals[0],
        "details": f"{roe_str}, {netm_str}, {opm_str}"
    }

    # 2. Growth Analysis
    revenue_growth = metrics.get("revenue_growth", 0) or 0
    earnings_growth = metrics.get("earnings_growth", 0) or 0
    book_value_growth = metrics.get("book_value_growth", 0) or 0

    thresholds = [
        (revenue_growth, 0.10),
        (earnings_growth, 0.10),
        (book_value_growth, 0.10)
    ]
    growth_score = sum(
        _valid_num(metric) and metric > threshold
        for metric, threshold in thresholds
    )

    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')

    revg_str = f"营收增长率(Revenue Growth): {revenue_growth:.2%}" if _valid_num(revenue_growth) else "营收增长率(Revenue Growth): N/A"
    erng_str = f"盈利增长率(Earnings Growth): {earnings_growth:.2%}" if _valid_num(earnings_growth) else "盈利增长率(Earnings Growth): N/A"

    reasoning["growth_signal"] = {
        "signal": signals[1],
        "details": f"{revg_str}, {erng_str}"
    }

    # 3. Financial Health
    current_ratio = metrics.get("current_ratio", 0) or 0
    debt_to_equity = metrics.get("debt_to_equity", 0) or 0
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share", 0) or 0
    earnings_per_share = metrics.get("earnings_per_share", 0) or 0

    health_score = 0
    if _valid_num(current_ratio) and current_ratio > 1.5:
        health_score += 1
    if _valid_num(debt_to_equity) and debt_to_equity < 0.5:
        health_score += 1
    if _valid_num(free_cash_flow_per_share) and _valid_num(earnings_per_share) and free_cash_flow_per_share > earnings_per_share * 0.8:
        health_score += 1

    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')

    cr_str = f"流动比率(Current Ratio): {current_ratio:.2f}" if _valid_num(current_ratio) else "流动比率(Current Ratio): N/A"
    de_str = f"负债权益比(D/E): {debt_to_equity:.2f}" if _valid_num(debt_to_equity) else "负债权益比(D/E): N/A"

    reasoning["financial_health_signal"] = {
        "signal": signals[2],
        "details": f"{cr_str}, {de_str}"
    }

    # 4. Price to X ratios
    pe_ratio = metrics.get("pe_ratio", 0) or 0
    price_to_book = metrics.get("price_to_book", 0) or 0
    price_to_sales = metrics.get("price_to_sales", 0) or 0

    thresholds = [
        (pe_ratio, 25),
        (price_to_book, 3),
        (price_to_sales, 5)
    ]
    price_ratio_score = sum(
        _valid_num(metric) and metric < threshold
        for metric, threshold in thresholds
    )

    signals.append('bullish' if price_ratio_score >= 2 else 'bearish' if price_ratio_score == 0 else 'neutral')

    pe_str = f"市盈率(P/E): {pe_ratio:.2f}" if _valid_num(pe_ratio) else "市盈率(P/E): N/A"
    pb_str = f"市净率(P/B): {price_to_book:.2f}" if _valid_num(price_to_book) else "市净率(P/B): N/A"
    ps_str = f"市销率(P/S): {price_to_sales:.2f}" if _valid_num(price_to_sales) else "市销率(P/S): N/A"

    reasoning["price_ratios_signal"] = {
        "signal": signals[3],
        "details": f"{pe_str}, {pb_str}, {ps_str}"
    }

    # [OPTIMIZED] 5. PB-ROE分位点分析
    pb_roe_analysis = _analyze_pb_roe_percentile(price_to_book, return_on_equity)
    reasoning["pb_roe_analysis"] = pb_roe_analysis
    signals.append(pb_roe_analysis['signal'])

    # [OPTIMIZED] 6. 周期股识别
    ticker = data.get("ticker", "")
    industry = data.get("industry", "")
    is_cyclical, cyclical_reason = _identify_cyclical_stock(ticker, industry)
    reasoning["cyclical_analysis"] = {
        "is_cyclical": is_cyclical,
        "industry": industry,
        "reasoning": cyclical_reason
    }

    # [OPTIMIZED] 7. 营收质量分析
    financial_line_items = data.get("financial_line_items", [])
    if financial_line_items and isinstance(financial_line_items, list):
        fli = financial_line_items[0] if financial_line_items else {}
    else:
        fli = financial_line_items
    revenue_quality = _analyze_revenue_quality(fli)
    reasoning["revenue_quality_analysis"] = revenue_quality
    signals.append(revenue_quality['signal'])

    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')

    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'

    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "基本面分析师")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("基本面分析师", "completed")
    # logger.info(f"--- DEBUG: fundamentals_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "fundamental_analysis": message_content
        },
        "metadata": state["metadata"],
    }
