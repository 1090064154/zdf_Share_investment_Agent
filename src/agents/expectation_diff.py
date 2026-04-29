"""
预期差分析Agent
分析业绩预告、分析师预期与实际业绩的差异
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json

logger = setup_logger('expectation_diff_agent')


def _get_earnings_forecast(ticker: str) -> dict:
    """
    获取业绩预告/盈利预测数据
    """
    try:
        import akshare as ak
        try:
            # 获取盈利预测数据
            forecast_df = ak.stock_profit_forecast_em(symbol='')
            if forecast_df is not None and len(forecast_df) > 0:
                match = forecast_df[forecast_df['代码'] == ticker]
                if len(match) > 0:
                    latest = match.iloc[0]
                    # 获取2025年预测每股收益
                    eps_2025 = latest.get('2025预测每股收益')
                    eps_2026 = latest.get('2026预测每股收益')
                    if eps_2025 and str(eps_2025) not in ['nan', 'None']:
                        # 计算增长
                        if eps_2026 and str(eps_2026) not in ['nan', 'None']:
                            try:
                                growth = ((float(eps_2026) - float(eps_2025)) / float(eps_2025)) * 100
                                if growth > 20:
                                    signal = 'bullish'
                                    confidence = 0.7
                                    reason = f"盈利预测增长{growth:.1f}%"
                                elif growth > 0:
                                    signal = 'bullish'
                                    confidence = 0.5
                                    reason = f"盈利预测增长{growth:.1f}%"
                                elif growth > -20:
                                    signal = 'neutral'
                                    confidence = 0.4
                                    reason = f"盈利预测下降{growth:.1f}%"
                                else:
                                    signal = 'bearish'
                                    confidence = 0.7
                                    reason = f"盈利预测大幅下降{growth:.1f}%"
                                
                                return {
                                    'signal': signal,
                                    'confidence': float(confidence),
                                    'growth': float(growth),
                                    'eps_2025': float(eps_2025),
                                    'eps_2026': float(eps_2026),
                                    'reason': reason,
                                    'source': 'earnings_forecast'
                                }
                            except:
                                pass
        except Exception as e:
            logger.debug(f"盈利预测数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'growth': 0,
        'reason': '无法获取业绩预告数据',
        'source': 'earnings_forecast'
    }


def _get_research_rating(ticker: str) -> dict:
    """
    获取券商研报评级
    """
    try:
        import akshare as ak
        try:
            # 获取券商评级 - 使用 stock_research_report_em
            rating_df = ak.stock_research_report_em(symbol=ticker)
            if rating_df is not None and len(rating_df) > 0:
                # 统计评级分布
                if '东财评级' in rating_df.columns:
                    ratings = rating_df['东财评级'].value_counts()
                elif '评级' in rating_df.columns:
                    ratings = rating_df['评级'].value_counts()
                else:
                    return {
                        'signal': 'neutral',
                        'confidence': 0,
                        'buy_ratio': 0,
                        'total_reports': 0,
                        'reason': '研报数据无评级信息',
                        'source': 'research_rating'
                    }

                buy_count = int(ratings.get('买入', 0)) + int(ratings.get('强烈推荐', 0)) + int(ratings.get('增持', 0)) + int(ratings.get('推荐', 0))
                hold_count = int(ratings.get('持有', 0)) + int(ratings.get('中性', 0)) + int(ratings.get('观望', 0))
                sell_count = int(ratings.get('卖出', 0)) + int(ratings.get('减持', 0)) + int(ratings.get('回避', 0))

                total = buy_count + hold_count + sell_count
                if total > 0:
                    buy_ratio = buy_count / total

                    if buy_ratio > 0.7:
                        signal = 'bullish'
                        confidence = min(0.5 + buy_ratio * 0.3, 0.8)
                        reason = f"机构评级买入占比{buy_ratio:.0%}"
                    elif buy_ratio > 0.4:
                        signal = 'neutral'
                        confidence = 0.5
                        reason = f"机构评级分歧，买入占比{buy_ratio:.0%}"
                    else:
                        signal = 'bearish'
                        confidence = min(0.5 + (1 - buy_ratio) * 0.3, 0.8)
                        reason = f"机构看空居多，买入占比{buy_ratio:.0%}"

                    return {
                        'signal': signal,
                        'confidence': float(confidence),
                        'buy_ratio': float(buy_ratio),
                        'total_reports': int(total),
                        'reason': reason,
                        'source': 'research_rating'
                    }
        except Exception as e:
            logger.debug(f"研报评级数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'buy_ratio': 0,
        'total_reports': 0,
        'reason': '无法获取研报评级数据',
        'source': 'research_rating'
    }


def _analyze_expectation_diff(forecast_result: dict, rating_result: dict) -> dict:
    """
    综合分析预期差
    """
    signals = []
    confidences = []

    # 业绩预告
    if forecast_result.get('confidence', 0) > 0:
        signals.append(forecast_result['signal'])
        confidences.append(forecast_result['confidence'])

    # 分析师评级
    if rating_result.get('confidence', 0) > 0:
        signals.append(rating_result['signal'])
        confidences.append(rating_result['confidence'])

    if not signals:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reason': '无预期差数据'
        }

    # 多数投票
    bullish_count = signals.count('bullish')
    bearish_count = signals.count('bearish')

    if bullish_count > bearish_count:
        signal = 'bullish'
        confidence = sum(c for s, c in zip(signals, confidences) if s == 'bullish') / bullish_count
    elif bearish_count > bullish_count:
        signal = 'bearish'
        confidence = sum(c for s, c in zip(signals, confidences) if s == 'bearish') / bearish_count
    else:
        signal = 'neutral'
        confidence = 0.4

    reason = f"预期差分析: {forecast_result.get('reason', 'N/A')}, {rating_result.get('reason', 'N/A')}"

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'forecast_analysis': forecast_result,
        'rating_analysis': rating_result
    }


@resilient_agent
@agent_endpoint("expectation_diff", "预期差分析师，分析业绩预告、分析师预期与实际的差异")
def expectation_diff_agent(state: AgentState):
    """分析预期差"""
    show_workflow_status("预期差分析师")
    logger.info("="*50)
    logger.info("📊 [EXPECTATION_DIFF] 开始预期差分析")
    logger.info("="*50)

    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    ticker = data.get("ticker", "")

    logger.info(f"  股票代码: {ticker}")

    # 1. 获取业绩预告
    logger.info("  获取业绩预告数据...")
    forecast_result = _get_earnings_forecast(ticker)
    logger.info(f"  业绩预告: {forecast_result.get('reason', 'N/A')}")

    # 2. 获取券商研报评级
    logger.info("  获取券商研报评级...")
    rating_result = _get_research_rating(ticker)
    logger.info(f"  研报评级: {rating_result.get('reason', 'N/A')}")

    # 3. 综合分析
    combined = _analyze_expectation_diff(forecast_result, rating_result)

    # 发送预期差分析结果到前端
    show_agent_reasoning({
        "业绩预告": forecast_result.get('signal', '-'),
        "业绩预告详情": forecast_result.get('reason', '-'),
        "研报评级": rating_result.get('signal', '-'),
        "研报评级详情": rating_result.get('reason', '-'),
        "综合信号": combined.get('signal', '-'),
        "综合置信度": f"{combined.get('confidence', 0.3) * 100:.0f}%"
    }, "预期差分析师")

    message_content = {
        "signal": combined['signal'],
        "confidence": f"{combined.get('confidence', 0.3) * 100:.0f}%",
        "reason": combined.get('reason', ''),
        "earnings_forecast": forecast_result,
        "research_rating": rating_result,
        "combined_analysis": combined
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False, indent=2),
        name="expectation_diff_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    def to_cn(s):
        return {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(s, s) if s else 'N/A'

    forecast = message_content.get('earnings_forecast', {})
    rating = message_content.get('research_rating', {})
    forecast_signal = forecast.get('signal', 'neutral')
    rating_signal = rating.get('signal', 'neutral')

    logic_parts = []
    if forecast_signal == 'bullish':
        logic_parts.append("业绩预增")
    elif forecast_signal == 'bearish':
        logic_parts.append("业绩预减")
    else:
        logic_parts.append("业绩持平")

    if rating_signal == 'bullish':
        logic_parts.append("机构看多")
    elif rating_signal == 'bearish':
        logic_parts.append("机构看空")
    else:
        logic_parts.append("机构中性")

    decision_logic = "，".join(logic_parts)

    show_agent_reasoning({
        "最终信号": to_cn(combined.get('signal')),
        "置信度": f"{combined.get('confidence', 0.3)*100:.0f}%",
        "业绩预告": to_cn(forecast_signal),
        "研报评级": to_cn(rating_signal),
        "判断逻辑": decision_logic
    }, "预期差分析师")

    show_workflow_status("预期差分析师", "completed")

    return {
        "messages": [message],
        "data": {
            **data,
            "expectation_diff_analysis": message_content
        },
        "metadata": state["metadata"],
    }
