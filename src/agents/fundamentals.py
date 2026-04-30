from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent

import json

# 初始化 logger
logger = setup_logger('fundamentals_agent')

# [OPTIMIZED] A股行业周期分类 - 与industry_cycle.py保持一致
INDUSTRY_CYCLE_CLASSIFICATION = {
    '强周期': [
        '农林牧渔', '养殖', '猪', '禽', '饲料', '化肥', '化肥农药',
        '钢铁', '煤炭', '有色金属', '化工', '建材', '房地产',
        '汽车', '交通运输', '工程机械', '航运', '港口', '船舶',
        '石油', '石化', '炼化', '橡胶', '塑料', '化纤'
    ],
    '弱周期': [
        '食品饮料', '医药生物', '家用电器', '纺织服装', '日用化工',
        '商贸零售', '旅游', '酒店', '餐饮', '酿酒', '化妆品'
    ],
    '成长': [
        '电子', '半导体', '计算机', '软件', '通信', '5G', '物联网',
        '新能源', '光伏', '风电', '锂电池', '电动车', '芯片', '集成电路',
        '医疗器械', '新材料', '机器人', '人工智能', '云计算'
    ],
    '防御': [
        '公用事业', '电力', '燃气', '水务', '环保', '银行', '保险', '券商', '多元金融'
    ]
}


def _is_missing_metric(value) -> bool:
    import math
    if value is None:
        return True
    try:
        fv = float(value)
        return math.isnan(fv) or math.isinf(fv) or fv == 0
    except (TypeError, ValueError):
        return True


def _valid_num(v) -> bool:
    """检查是否为有效的非零、非NaN、非无穷大数值"""
    import math
    if v is None:
        return False
    try:
        fv = float(v)
        return not (math.isnan(fv) or math.isinf(fv) or fv == 0)
    except (TypeError, ValueError):
        return False


def _valid_positive_num(v) -> bool:
    """检查是否为有效的正数（用于P/E等必须>0的指标）"""
    if not _valid_num(v):
        return False
    return float(v) > 0


def _has_valid_data(metrics_dict, *keys) -> int:
    """统计有效指标数量"""
    return sum(1 for k in keys if _valid_num(metrics_dict.get(k)))


def _get_industry_adjustments(industry: str = '') -> dict:
    """
    根据行业返回差异化阈值调整因子
    返回: {threshold_key: multiplier}
    multiplier < 1 表示该行业阈值可以更宽松
    """
    # 金融行业：低ROE、高杠杆是常态
    financial_keywords = ['银行', '保险', '券商', '多元金融']
    if any(kw in (industry or '') for kw in financial_keywords):
        return {'roe': 0.7, 'de': 2.5, 'net_margin': 0.6, 'pe': 0.6}

    # 地产/基建：高杠杆行业
    heavy_asset = ['房地产', '建筑', '建材', '钢铁', '交通运输', '航运', '港口']
    if any(kw in (industry or '') for kw in heavy_asset):
        return {'roe': 0.8, 'de': 2.0, 'net_margin': 0.7}

    # 科技/成长：高PE合理
    tech_keywords = ['电子', '半导体', '计算机', '软件', '通信', '新能源', '芯片', '医药', '医疗器械']
    if any(kw in (industry or '') for kw in tech_keywords):
        return {'pe': 1.8, 'pb': 1.5}

    return {}


def _identify_cyclical_stock(ticker: str, industry: str = None) -> tuple:
    """
    [OPTIMIZED] 识别是否为周期股
    返回: (is_cyclical: bool, reason: str)

    判断优先级：
    1. 行业关键词匹配（优先）
    2. 股票代码辅助判断（仅作为后备，不单独决定）
    """
    # 优先行业判断
    for cycle_type, industries in INDUSTRY_CYCLE_CLASSIFICATION.items():
        if industry and any(ind in industry for ind in industries):
            return True, f"属于{cycle_type}行业: {industry}"

    # 后备方案：根据股票代码特征辅助判断
    # 农林牧渔养殖类（002xxx, 000xxx常见）
    if ticker.startswith('00') or ticker.startswith('30'):
        # 尝试更精细判断
        return False, "未匹配行业关键词，默认为非周期股（请检查行业数据）"

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
    revenue = financial_line_items.get('operating_revenue', 0)

    if not _valid_num(revenue) or not _valid_num(accounts_receivable):
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

    show_agent_reasoning({"有效财务指标": f"{len(populated_metrics)}项"}, "基本面分析师")

    if not populated_metrics:
        reason = "Financial metrics unavailable, defaulting fundamentals analysis to neutral."
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
            name="fundamentals_agent",
        )
        if show_reasoning:
            state["metadata"]["agent_reasoning"] = message_content
        show_agent_reasoning({"最终信号": "中性", "置信度": "0%"}, "基本面分析师")
        show_workflow_complete(
            "基本面分析师",
            signal="neutral",
            confidence=0.0,
            details=message_content,
            message="基本面分析完成：财务数据不足，无法判断"
        )

        return {
            "messages": [message],
            "data": {
                **data,
                "fundamental_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # --- 获取行业差异化阈值 ---
    ticker = data.get("ticker", "")
    industry = data.get("industry", "")
    adj = _get_industry_adjustments(industry)

    # Initialize weighted signals list for aggregation
    signals = []     # [(signal, weight), ...]
    reasoning = {}

    # ========== 1. 盈利能力分析 ==========
    # 阈值已根据A股实际调整：ROE>10%即优秀(原15%)，净利率>8%合理(原20%)
    return_on_equity = metrics.get("return_on_equity", 0) or 0
    net_margin = metrics.get("net_margin", 0) or 0
    operating_margin = metrics.get("operating_margin", 0) or 0

    roe_threshold = 0.10 * adj.get('roe', 1.0)
    nm_threshold = 0.08 * adj.get('net_margin', 1.0)
    opm_threshold = 0.08 * adj.get('net_margin', 1.0)

    has_roe = _valid_num(return_on_equity)
    has_nm = _valid_num(net_margin)
    has_opm = _valid_num(operating_margin)
    profit_data_count = sum([has_roe, has_nm, has_opm])

    if profit_data_count >= 2:
        profit_score = sum([
            has_roe and return_on_equity > roe_threshold,
            has_nm and net_margin > nm_threshold,
            has_opm and operating_margin > opm_threshold
        ])
        if profit_score >= 2:
            profit_signal = 'bullish'
        elif profit_score == 0:
            profit_signal = 'bearish'
        else:
            profit_signal = 'neutral'
    else:
        profit_signal = 'neutral'  # 数据不足 → 中性，不误判

    roe_str = f"ROE:{return_on_equity:.1%}" if has_roe else "ROE:N/A"
    nm_str = f"净利率:{net_margin:.1%}" if has_nm else "净利率:N/A"
    opm_str = f"营业利润率:{operating_margin:.1%}" if has_opm else "营业利润率:N/A"

    reasoning["profitability_signal"] = {
        "signal": profit_signal,
        "details": f"{roe_str}, {nm_str}, {opm_str}",
        "data_available": profit_data_count >= 2
    }
    signals.append((profit_signal, 0.25))  # 权重25%
    logger.info(f"  💰 盈利能力: {profit_signal} - {roe_str}, {nm_str}")
    show_agent_reasoning({"reasoning": {"profitability_signal": {"signal": profit_signal, "details": reasoning["profitability_signal"]["details"]}}}, "基本面分析师")

    # ========== 2. 成长性分析 ==========
    revenue_growth = metrics.get("revenue_growth", 0) or 0
    earnings_growth = metrics.get("earnings_growth", 0) or 0
    book_value_growth = metrics.get("book_value_growth", 0) or 0

    has_rg = _valid_num(revenue_growth)
    has_eg = _valid_num(earnings_growth)
    has_bvg = _valid_num(book_value_growth)
    growth_data_count = sum([has_rg, has_eg, has_bvg])

    if growth_data_count >= 2:
        growth_score = sum([
            has_rg and revenue_growth > 0.10,
            has_eg and earnings_growth > 0.10,
            has_bvg and book_value_growth > 0.05
        ])
        if growth_score >= 2:
            growth_signal = 'bullish'
        elif growth_score == 0 and all(not _valid_num(metrics.get(k, 0)) or float(metrics.get(k, 0)) < 0
                                        for k in ['revenue_growth', 'earnings_growth'] if _valid_num(metrics.get(k, 0))):
            growth_signal = 'bearish'
        elif growth_score == 0:
            growth_signal = 'neutral'  # 不达标但不是严重负增长
        else:
            growth_signal = 'neutral'
    else:
        growth_signal = 'neutral'

    revg_str = f"营收增长:{revenue_growth:.1%}" if has_rg else "营收增长:N/A"
    erng_str = f"盈利增长:{earnings_growth:.1%}" if has_eg else "盈利增长:N/A"

    reasoning["growth_signal"] = {
        "signal": growth_signal,
        "details": f"{revg_str}, {erng_str}",
        "data_available": growth_data_count >= 2
    }
    signals.append((growth_signal, 0.20))  # 权重20%
    logger.info(f"  📈 成长性: {growth_signal} - {revg_str}, {erng_str}")
    show_agent_reasoning({"reasoning": {"growth_signal": {"signal": growth_signal, "details": reasoning["growth_signal"]["details"]}}}, "基本面分析师")

    # ========== 3. 财务健康分析 ==========
    current_ratio = metrics.get("current_ratio", 0) or 0
    debt_to_equity = metrics.get("debt_to_equity", 0) or 0
    interest_coverage = metrics.get("interest_coverage", 0) or 0
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share", 0) or 0
    earnings_per_share = metrics.get("earnings_per_share", 0) or 0

    has_cr = _valid_num(current_ratio)
    has_de = _valid_num(debt_to_equity)
    has_ic = _valid_num(interest_coverage)
    health_data_count = sum([has_cr, has_de])

    cr_threshold = 1.2   # 从1.5降低
    de_threshold = 1.5 * adj.get('de', 1.0)  # 从0.5大幅提高到1.5(非金融业)

    if health_data_count >= 1:
        health_score = 0
        if has_cr and current_ratio > cr_threshold:
            health_score += 1
        if has_de and debt_to_equity < de_threshold:
            health_score += 1
        # 利息保障倍数 ≥ 3 表示偿债能力充足
        if has_ic and interest_coverage >= 3:
            health_score += 1
        elif has_ic and interest_coverage < 1:
            health_score -= 1  # 利息都还不起，扣分
        # FCF/EPS质量检查
        if _valid_num(free_cash_flow_per_share) and _valid_num(earnings_per_share):
            if free_cash_flow_per_share > earnings_per_share * 0.8:
                health_score += 0.5
            elif free_cash_flow_per_share < 0:
                health_score -= 0.5

        if health_score >= 2:
            health_signal = 'bullish'
        elif health_score <= 0:
            health_signal = 'bearish'
        else:
            health_signal = 'neutral'
    else:
        health_signal = 'neutral'

    cr_str = f"流动比率:{current_ratio:.2f}" if has_cr else "流动比率:N/A"
    de_str = f"D/E:{debt_to_equity:.2f}" if has_de else "D/E:N/A"
    ic_str = f"利息保障:{interest_coverage:.1f}x" if has_ic else "利息保障:N/A"

    reasoning["financial_health_signal"] = {
        "signal": health_signal,
        "details": f"{cr_str}, {de_str}, {ic_str}",
        "data_available": health_data_count >= 1
    }
    signals.append((health_signal, 0.25))  # 权重25%
    logger.info(f"  🏦 财务健康: {health_signal} - {cr_str}, {de_str}, {ic_str}")
    show_agent_reasoning({"reasoning": {"financial_health_signal": {"signal": health_signal, "details": reasoning["financial_health_signal"]["details"]}}}, "基本面分析师")

    # ========== 4. 估值比率分析 ==========
    pe_ratio = metrics.get("pe_ratio", 0) or 0
    price_to_book = metrics.get("price_to_book", 0) or 0
    price_to_sales = metrics.get("price_to_sales", 0) or 0

    # P/E必须>0才有意义（亏损公司PE为负，不参与估值判断）
    has_pe = _valid_positive_num(pe_ratio)
    has_pb = _valid_num(price_to_book)
    has_ps = _valid_num(price_to_sales)
    price_data_count = sum([has_pe or _valid_num(pe_ratio) and float(pe_ratio) < 0, has_pb, has_ps])

    pe_threshold = 25 * adj.get('pe', 1.0)
    pb_threshold = 3 * adj.get('pb', 1.0)

    if price_data_count >= 2:
        price_score = 0
        price_detail_parts = []

        if has_pe and pe_ratio < pe_threshold:
            price_score += 1
            price_detail_parts.append(f"P/E={pe_ratio:.1f} (低于{pe_threshold:.0f})")
        elif has_pe:
            price_detail_parts.append(f"P/E={pe_ratio:.1f} (高于{pe_threshold:.0f})")
        elif _valid_num(pe_ratio) and float(pe_ratio) < 0:
            price_detail_parts.append(f"P/E=负值(亏损)")

        if has_pb and price_to_book < pb_threshold:
            price_score += 1
            price_detail_parts.append(f"P/B={price_to_book:.1f} (低于{pb_threshold:.1f})")
        elif has_pb:
            price_detail_parts.append(f"P/B={price_to_book:.1f} (高于{pb_threshold:.1f})")

        if has_ps and price_to_sales < 5:
            price_score += 1
            price_detail_parts.append(f"P/S={price_to_sales:.1f} (低于5)")

        if price_score >= 2:
            price_signal = 'bullish'
        elif price_score == 0:
            price_signal = 'bearish'
        else:
            price_signal = 'neutral'
    else:
        price_signal = 'neutral'
        price_detail_parts = ["估值数据不足"]

    reasoning["price_ratios_signal"] = {
        "signal": price_signal,
        "details": "; ".join(price_detail_parts) if price_detail_parts else "P/E:N/A, P/B:N/A, P/S:N/A",
        "data_available": price_data_count >= 2
    }
    signals.append((price_signal, 0.15))  # 权重15%
    logger.info(f"  📊 估值比率: {price_signal} - {reasoning['price_ratios_signal']['details'][:80]}")
    show_agent_reasoning({"reasoning": {"price_ratios_signal": {"signal": price_signal, "details": reasoning["price_ratios_signal"]["details"]}}}, "基本面分析师")

    # ========== 5. PB-ROE分位点分析 ==========
    pb_roe_analysis = _analyze_pb_roe_percentile(price_to_book, return_on_equity)
    reasoning["pb_roe_analysis"] = pb_roe_analysis
    signals.append((pb_roe_analysis['signal'], 0.10))  # 权重10%
    logger.info(f"  📐 PB-ROE: {pb_roe_analysis['signal']} - {pb_roe_analysis.get('reasoning', '')}")
    show_agent_reasoning({"reasoning": {"pb_roe_analysis": {"signal": pb_roe_analysis['signal'], "details": pb_roe_analysis.get('reasoning', '')}}}, "基本面分析师")

    # ========== 6. 周期股识别 ==========
    is_cyclical, cyclical_reason = _identify_cyclical_stock(ticker, industry)
    # 周期识别不单独贡献信号，而是通过行业因子调整阈值(已在_get_industry_adjustments中体现)
    # 此处仅记录，不加入signals列表参与投票
    reasoning["cyclical_analysis"] = {
        "is_cyclical": is_cyclical,
        "industry": industry,
        "reasoning": cyclical_reason,
        "adjustments_applied": adj
    }
    logger.info(f"  🔄 周期/行业识别: {'周期股' if is_cyclical else '非周期股'} - {cyclical_reason}")
    show_agent_reasoning({"reasoning": {"cyclical_analysis": {"signal": "neutral", "details": f"{industry or '未知行业'}: {cyclical_reason}"}}}, "基本面分析师")

    # ========== 7. 营收质量分析 ==========
    financial_line_items = data.get("financial_line_items", [])
    if financial_line_items and isinstance(financial_line_items, list):
        fli = financial_line_items[0] if financial_line_items else {}
    else:
        fli = financial_line_items
    revenue_quality = _analyze_revenue_quality(fli)
    reasoning["revenue_quality_analysis"] = revenue_quality
    signals.append((revenue_quality['signal'], 0.05))  # 权重5%
    logger.info(f"  📋 营收质量: {revenue_quality['signal']} - {revenue_quality.get('reasoning', '')}")
    show_agent_reasoning({"reasoning": {"revenue_quality_analysis": {"signal": revenue_quality['signal'], "details": revenue_quality.get('reasoning', '')}}}, "基本面分析师")

    # ========== 8. 盈利质量/现金转换率分析 (新增) ==========
    operating_cash_flow = metrics.get("operating_cash_flow", 0) or 0
    net_income = metrics.get("net_income", 0) or 0

    if _valid_num(operating_cash_flow) and _valid_num(net_income):
        ocf_ni_ratio = operating_cash_flow / net_income
        if ocf_ni_ratio >= 1.0:
            eq_signal = 'bullish'
            eq_detail = f"经营现金流/净利润={ocf_ni_ratio:.1f}x，利润含金量高"
        elif ocf_ni_ratio >= 0.8:
            eq_signal = 'neutral'
            eq_detail = f"经营现金流/净利润={ocf_ni_ratio:.1f}x，利润质量正常"
        elif ocf_ni_ratio > 0:
            eq_signal = 'bearish'
            eq_detail = f"经营现金流/净利润={ocf_ni_ratio:.1f}x，利润含金量不足"
        else:
            eq_signal = 'bearish'
            eq_detail = f"经营现金流为负({operating_cash_flow/1e8:.1f}亿)，现金转换能力差"
    else:
        eq_signal = 'neutral'
        eq_detail = "经营现金流或净利润数据缺失"
        ocf_ni_ratio = None

    reasoning["earnings_quality"] = {
        "signal": eq_signal,
        "details": eq_detail,
        "ocf_ni_ratio": round(ocf_ni_ratio, 2) if ocf_ni_ratio is not None else None
    }
    signals.append((eq_signal, 0.10))  # 权重10%
    logger.info(f"  💵 盈利质量: {eq_signal} - {eq_detail}")
    show_agent_reasoning({"reasoning": {"earnings_quality": {"signal": eq_signal, "details": eq_detail}}}, "基本面分析师")

    # ========== 加权信号聚合 ==========
    # 核心三维度(盈利/成长/财务健康)各25%，估值15%，盈利质量10%，PB-ROE 10%，营收质量5%
    bullish_weight = sum(w for s, w in signals if s == 'bullish')
    bearish_weight = sum(w for s, w in signals if s == 'bearish')
    neutral_weight = sum(w for s, w in signals if s == 'neutral')
    total_weight = bullish_weight + bearish_weight + neutral_weight

    if total_weight > 0:
        if bullish_weight > bearish_weight + 0.15:  # 需要显著优势(>15%差距)
            overall_signal = 'bullish'
        elif bearish_weight > bullish_weight + 0.15:
            overall_signal = 'bearish'
        else:
            overall_signal = 'neutral'
        # 置信度 = 主导信号权重占比
        dominant_weight = max(bullish_weight, bearish_weight)
        confidence = min(dominant_weight / total_weight + 0.1, 0.95)
    else:
        overall_signal = 'neutral'
        confidence = 0.0

    # 构建完整message_content（包含所有子信号）
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 4),
        "confidence_raw": round(confidence, 4),
        "reasoning": reasoning,
        "profitability_signal": reasoning.get("profitability_signal", {}),
        "growth_signal": reasoning.get("growth_signal", {}),
        "financial_health_signal": reasoning.get("financial_health_signal", {}),
        "price_ratios_signal": reasoning.get("price_ratios_signal", {}),
        "pb_roe_analysis": reasoning.get("pb_roe_analysis", {}),
        "cyclical_analysis": reasoning.get("cyclical_analysis", {}),
        "revenue_quality_analysis": reasoning.get("revenue_quality_analysis", {}),
        "earnings_quality": reasoning.get("earnings_quality", {}),
        "signal_weights": {
            "bullish_weight": round(bullish_weight, 2),
            "bearish_weight": round(bearish_weight, 2),
            "neutral_weight": round(neutral_weight, 2)
        },
        "industry": industry,
        "is_cyclical": is_cyclical
    }

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(message_content.get('signal', 'neutral'), message_content.get('signal', 'neutral'))
    def to_cn(s):
        return {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(s, s) if s else 'N/A'

    profitability = message_content.get('profitability_signal', {}).get('signal', 'neutral')
    growth = message_content.get('growth_signal', {}).get('signal', 'neutral')
    financial = message_content.get('financial_health_signal', {}).get('signal', 'neutral')

    profit_details = message_content.get('profitability_signal', {}).get('details', '')
    growth_details = message_content.get('growth_signal', {}).get('details', '')
    financial_details = message_content.get('financial_health_signal', {}).get('details', '')
    earn_quality = message_content.get('earnings_quality', {}).get('signal', 'neutral')
    pb_roe_sig = message_content.get('pb_roe_analysis', {}).get('signal', 'neutral')
    rev_quality_sig = message_content.get('revenue_quality_analysis', {}).get('signal', 'neutral')

    is_data_insufficient = not profit_details and not growth_details and not financial_details

    logic_parts = []
    if is_data_insufficient:
        logic_parts.append("财务数据不足，无法准确判断")
    else:
        if profitability == 'bullish':
            logic_parts.append(f"盈利判断：ROE/净利率表现良好")
        elif profitability == 'bearish':
            logic_parts.append(f"盈利判断：ROE/净利率偏低")
        else:
            logic_parts.append(f"盈利判断：指标一般")

        if growth == 'bullish':
            logic_parts.append(f"成长判断：营收/盈利增长强劲")
        elif growth == 'bearish':
            logic_parts.append(f"成长判断：增长放缓或负增长")
        else:
            logic_parts.append(f"成长判断：增长平稳")

        if financial == 'bullish':
            logic_parts.append(f"财务判断：负债率低，流动性好，利息覆盖充足")
        elif financial == 'bearish':
            logic_parts.append(f"财务判断：负债率高，偿债压力大")
        else:
            logic_parts.append(f"财务判断：财务状况正常")

        # 新增维度
        if earn_quality == 'bullish':
            logic_parts.append(f"盈利质量：现金流充裕，利润含金量高")
        elif earn_quality == 'bearish':
            logic_parts.append(f"盈利质量：现金流不足，利润含金量偏低")

        if pb_roe_sig == 'bullish':
            logic_parts.append(f"PB-ROE：处于价值区域")
        elif pb_roe_sig == 'bearish':
            logic_parts.append(f"PB-ROE：估值风险偏高")

        if rev_quality_sig == 'bearish':
            logic_parts.append(f"营收质量：应收款占比过高")

    decision_logic = "；".join(logic_parts)

    show_agent_reasoning({
        "最终信号": signal_cn,
        "置信度": message_content.get('confidence'),
        "盈利": to_cn(profitability),
        "成长": to_cn(growth),
        "财务": to_cn(financial),
        "盈利质量": to_cn(earn_quality),
        "PB-ROE": to_cn(pb_roe_sig),
        "营收质量": to_cn(rev_quality_sig),
        "数据状态": "数据不足" if is_data_insufficient else "数据完整",
        "判断逻辑": decision_logic
    }, "基本面分析师")

    show_workflow_complete(
        "基本面分析师",
        signal=overall_signal,
        confidence=confidence,
        details=message_content,
        message=f"基本面分析完成，信号:{signal_cn}，置信度:{message_content.get('confidence')}"
    )

    return {
        "messages": [message],
        "data": {
            **data,
            "fundamental_analysis": message_content
        },
        "metadata": state["metadata"],
    }
