"""
预期差分析Agent
分析业绩预告、分析师预期与实际业绩的差异
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json

logger = setup_logger('expectation_diff_agent')


def _get_earnings_forecast(ticker: str) -> dict:
    """
    获取业绩预告数据
    """
    try:
        import akshare as ak
        try:
            # 获取业绩预告
            forecast_df = ak.stock_earnings_forecast(stock=ticker)
            if forecast_df is not None and len(forecast_df) > 0:
                latest = forecast_df.iloc[0]
                # 尝试获取预告增长率
                if '预告净利润增长率' in latest:
                    growth = str(latest['预告净利润增长率']).replace('%', '').replace('+', '')
                    try:
                        growth = float(growth)
                        if growth > 20:
                            signal = 'bullish'
                            confidence = 0.7
                            reason = f"业绩预增{growth:.1f}%"
                        elif growth > 0:
                            signal = 'bullish'
                            confidence = 0.5
                            reason = f"业绩预增{growth:.1f}%"
                        elif growth > -20:
                            signal = 'neutral'
                            confidence = 0.4
                            reason = f"业绩预降{growth:.1f}%"
                        else:
                            signal = 'bearish'
                            confidence = 0.7
                            reason = f"业绩大幅预降{growth:.1f}%"

                        return {
                            'signal': signal,
                            'confidence': confidence,
                            'growth': growth,
                            'reason': reason,
                            'source': 'earnings_forecast'
                        }
                    except:
                        pass
        except Exception as e:
            logger.debug(f"业绩预告数据获取失败: {e}")
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
            # 获取券商评级
            rating_df = ak.stock_research_report(stock=ticker)
            if rating_df is not None and len(rating_df) > 0:
                # 统计评级分布
                ratings = rating_df['评级'].value_counts() if '评级' in rating_df.columns else {}

                buy_count = ratings.get('买入', 0) + ratings.get('强烈推荐', 0) + ratings.get('增持', 0)
                hold_count = ratings.get('持有', 0) + ratings.get('中性', 0)
                sell_count = ratings.get('卖出', 0) + ratings.get('减持', 0)

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
                        'confidence': confidence,
                        'buy_ratio': buy_ratio,
                        'total_reports': total,
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
        show_agent_reasoning(message_content, "预期差分析")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("预期差分析师", "completed")
    logger.info(f"[EXPECTATION_DIFF] 分析完成: {combined.get('signal')}")

    return {
        "messages": [message],
        "data": {
            **data,
            "expectation_diff_analysis": message_content
        },
        "metadata": state["metadata"],
    }
