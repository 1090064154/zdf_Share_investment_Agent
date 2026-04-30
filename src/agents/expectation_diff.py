"""
预期差分析Agent
分析业绩预告、分析师预期与实际业绩的差异
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json

logger = setup_logger('expectation_diff_agent')


def _get_actual_earnings_announcement(ticker: str) -> dict:
    """
    [NEW] 获取实际业绩预告（业绩预增/预减/扭亏）
    """
    try:
        import akshare as ak
        try:
            # 获取业绩预告
            ann_df = ak.stock_em_yjyg(symbol=ticker)
            if ann_df is not None and len(ann_df) > 0:
                latest = ann_df.iloc[0]
                
                # 业绩预告类型
                ann_type = latest.get('业绩变动类型', '不确定')
                net_profit = latest.get('预计净利润', '0')
                
                if '扭亏' in str(ann_type):
                    signal = 'bullish'
                    confidence = 0.8
                    reason = "业绩扭亏为盈"
                elif '预增' in str(ann_type):
                    signal = 'bullish'
                    confidence = 0.7
                    reason = f"业绩预增：{net_profit}"
                elif '预减' in str(ann_type):
                    signal = 'bearish'
                    confidence = 0.7
                    reason = f"业绩预减：{net_profit}"
                elif '首亏' in str(ann_type):
                    signal = 'bearish'
                    confidence = 0.8
                    reason = "首次亏损"
                else:
                    signal = 'neutral'
                    confidence = 0.4
                    reason = f"业绩预告：{ann_type}"
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'announcement_type': str(ann_type),
                    'net_profit': str(net_profit),
                    'reason': reason,
                    'source': 'earnings_announcement'
                }
        except Exception as e:
            logger.debug(f"业绩预告数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'signal': 'neutral', 'confidence': 0, 'reason': '无业绩预告数据', 'source': 'earnings_announcement'}


def _get_target_price(ticker: str) -> dict:
    """
    [NEW] 获取券商目标价分析
    """
    try:
        import akshare as ak
        try:
            # 获取券商评级和目标价
            report_df = ak.stock_research_report_em(symbol=ticker)
            if report_df is not None and len(report_df) > 0:
                target_prices = []
                current_prices = []
                
                for _, row in report_df.iterrows():
                    target = row.get('目标价') or row.get('目标价格')
                    current = row.get('最新价') or row.get('当前价格')
                    if target and str(target) not in ['nan', 'None', '']:
                        try:
                            target_prices.append(float(target))
                        except:
                            pass
                    if current and str(current) not in ['nan', 'None', '']:
                        try:
                            current_prices.append(float(current))
                        except:
                            pass
                
                if target_prices and current_prices:
                    avg_target = sum(target_prices) / len(target_prices)
                    avg_current = sum(current_prices) / len(current_prices)
                    upside = (avg_target - avg_current) / avg_current
                    
                    if upside > 0.3:
                        signal = 'bullish'
                        confidence = min(0.6 + upside, 0.85)
                        reason = f"目标价{avg_target:.2f}元，相对当前{avg_current:.2f}元上涨{upside*100:.0f}%"
                    elif upside > 0.1:
                        signal = 'neutral'
                        confidence = 0.5
                        reason = f"目标价{avg_target:.2f}元，略高于当前{upside*100:.0f}%"
                    elif upside > -0.1:
                        signal = 'neutral'
                        confidence = 0.4
                        reason = f"目标价{avg_target:.2f}元，与当前接近"
                    else:
                        signal = 'bearish'
                        confidence = min(0.6 + abs(upside), 0.85)
                        reason = f"目标价{avg_target:.2f}元，低于当前{abs(upside)*100:.0f}%"
                    
                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'avg_target_price': round(avg_target, 2),
                        'avg_current_price': round(avg_current, 2),
                        'upside_pct': round(upside * 100, 1),
                        'target_count': len(target_prices),
                        'reason': reason,
                        'source': 'target_price'
                    }
        except Exception as e:
            logger.debug(f"目标价数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'signal': 'neutral', 'confidence': 0, 'reason': '无目标价数据', 'source': 'target_price'}


def _get_forecast_adjustment(ticker: str) -> dict:
    """
    [NEW] 获取盈利预测调整趋势（上调/下调）
    """
    try:
        import akshare as ak
        try:
            forecast_df = ak.stock_profit_forecast_em(symbol='')
            if forecast_df is not None and len(forecast_df) > 0:
                match = forecast_df[forecast_df['代码'] == ticker]
                if len(match) >= 2:
                    latest = match.iloc[0]
                    previous = match.iloc[1]
                    
                    eps_latest = latest.get('2025预测每股收益')
                    eps_previous = previous.get('2025预测每股收益')
                    
                    if eps_latest and eps_previous and str(eps_latest) not in ['nan', 'None'] and str(eps_previous) not in ['nan', 'None']:
                        change_pct = ((float(eps_latest) - float(eps_previous)) / float(eps_previous)) * 100
                        
                        if change_pct > 10:
                            signal = 'bullish'
                            confidence = min(0.6 + change_pct / 50, 0.85)
                            reason = f"盈利预测上调{change_pct:.1f}%"
                        elif change_pct < -10:
                            signal = 'bearish'
                            confidence = min(0.6 + abs(change_pct) / 50, 0.85)
                            reason = f"盈利预测下调{abs(change_pct):.1f}%"
                        else:
                            signal = 'neutral'
                            confidence = 0.5
                            reason = f"盈利预测调整{change_pct:+.1f}%"
                        
                        return {
                            'signal': signal,
                            'confidence': confidence,
                            'change_pct': round(change_pct, 1),
                            'eps_latest': float(eps_latest),
                            'eps_previous': float(eps_previous),
                            'reason': reason,
                            'source': 'forecast_adjustment'
                        }
        except Exception as e:
            logger.debug(f"预测调整数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'signal': 'neutral', 'confidence': 0, 'reason': '无预测调整数据', 'source': 'forecast_adjustment'}


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


def _analyze_expectation_diff(
    earnings_announcement: dict, 
    forecast_result: dict, 
    rating_result: dict,
    target_price_result: dict,
    adjustment_result: dict
) -> dict:
    """
    [OPTIMIZED] 综合分析预期差 - 使用加权组合
    权重：业绩预告30%，盈利预测20%，研报评级20%，目标价15%，预测调整15%
    """
    weights = {
        'announcement': 0.30,
        'forecast': 0.20,
        'rating': 0.20,
        'target': 0.15,
        'adjustment': 0.15
    }
    
    results = {
        'announcement': earnings_announcement,
        'forecast': forecast_result,
        'rating': rating_result,
        'target': target_price_result,
        'adjustment': adjustment_result
    }
    
    signal_values = {'bullish': 1, 'bearish': -1, 'neutral': 0}
    weighted_sum = 0
    total_weight = 0
    
    for source, result in results.items():
        if result and result.get('confidence', 0) > 0:
            signal_val = signal_values.get(result.get('signal', 'neutral'), 0)
            weight = weights.get(source, 0.1)
            confidence = result.get('confidence', 0)
            
            weighted_sum += signal_val * weight * confidence
            total_weight += weight * confidence
    
    if total_weight == 0:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reason': '无预期差数据',
            'announcement_analysis': earnings_announcement,
            'forecast_analysis': forecast_result,
            'rating_analysis': rating_result,
            'target_analysis': target_price_result,
            'adjustment_analysis': adjustment_result
        }
    
    final_score = weighted_sum / total_weight
    
    if final_score > 0.2:
        signal = 'bullish'
        confidence = min(abs(final_score) + 0.3, 0.9)
    elif final_score < -0.2:
        signal = 'bearish'
        confidence = min(abs(final_score) + 0.3, 0.9)
    else:
        signal = 'neutral'
        confidence = max(0.4, 0.6 - abs(final_score))
    
    reason_parts = []
    if earnings_announcement.get('confidence', 0) > 0:
        reason_parts.append(f"业绩预告{earnings_announcement.get('signal', 'neutral')}")
    if rating_result.get('confidence', 0) > 0:
        reason_parts.append(f"评级{rating_result.get('signal', 'neutral')}")
    if target_price_result.get('confidence', 0) > 0:
        reason_parts.append(f"目标价{target_price_result.get('signal', 'neutral')}")
    if adjustment_result.get('confidence', 0) > 0:
        reason_parts.append(f"预测调{adjustment_result.get('signal', 'neutral')}")
    
    reason = f"预期差：{' '.join(reason_parts)}"

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'announcement_analysis': earnings_announcement,
        'forecast_analysis': forecast_result,
        'rating_analysis': rating_result,
        'target_analysis': target_price_result,
        'adjustment_analysis': adjustment_result
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

    # [NEW] 1. 获取实际业绩预告
    logger.info("  获取实际业绩预告...")
    earnings_announcement = _get_actual_earnings_announcement(ticker)
    logger.info(f"  业绩预告: {earnings_announcement.get('reason', 'N/A')}")

    # 2. 获取盈利预测
    logger.info("  获取盈利预测数据...")
    forecast_result = _get_earnings_forecast(ticker)
    logger.info(f"  盈利预测: {forecast_result.get('reason', 'N/A')}")

    # 3. 获取券商研报评级
    logger.info("  获取券商研报评级...")
    rating_result = _get_research_rating(ticker)
    logger.info(f"  研报评级: {rating_result.get('reason', 'N/A')}")

    # [NEW] 4. 获取目标价
    logger.info("  获取目标价...")
    target_price_result = _get_target_price(ticker)
    logger.info(f"  目标价: {target_price_result.get('reason', 'N/A')}")

    # [NEW] 5. 获取预测调整
    logger.info("  获取预测调整趋势...")
    adjustment_result = _get_forecast_adjustment(ticker)
    logger.info(f"  预测调整: {adjustment_result.get('reason', 'N/A')}")

    # 6. 综合分析
    combined = _analyze_expectation_diff(earnings_announcement, forecast_result, rating_result, target_price_result, adjustment_result)

    # 发送预期差分析结果到前端
    def sig_cn(s):
        return {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(s, s) if s else '-'
    
    show_agent_reasoning({
        "业绩预告": f"{sig_cn(earnings_announcement.get('signal'))} | {earnings_announcement.get('reason', '-')}",
        "盈利预测": f"{sig_cn(forecast_result.get('signal'))} | {forecast_result.get('reason', '-')}",
        "研报评级": f"{sig_cn(rating_result.get('signal'))} | {rating_result.get('reason', '-')}",
        "目标价": f"{sig_cn(target_price_result.get('signal'))} | {target_price_result.get('reason', '-')}",
        "预测调整": f"{sig_cn(adjustment_result.get('signal'))} | {adjustment_result.get('reason', '-')}",
        "综合信号": sig_cn(combined.get('signal')),
        "置信度": f"{combined.get('confidence', 0.3) * 100:.0f}%"
    }, "预期差分析师")

    message_content = {
        "signal": combined['signal'],
        "confidence": combined.get('confidence', 0.3),
        "reason": combined.get('reason', ''),
        "earnings_announcement": earnings_announcement,
        "earnings_forecast": forecast_result,
        "research_rating": rating_result,
        "target_price": target_price_result,
        "forecast_adjustment": adjustment_result,
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
    
    ann_signal = earnings_announcement.get('signal', 'neutral')
    if ann_signal == 'bullish':
        logic_parts.append("业绩预增")
    elif ann_signal == 'bearish':
        logic_parts.append("业绩预减")
    else:
        logic_parts.append("业绩持平")

    if rating_signal == 'bullish':
        logic_parts.append("机构看多")
    elif rating_signal == 'bearish':
        logic_parts.append("机构看空")
    else:
        logic_parts.append("机构中性")
    
    tp_signal = target_price_result.get('signal', 'neutral')
    if tp_signal == 'bullish':
        logic_parts.append("目标价看涨")
    elif tp_signal == 'bearish':
        logic_parts.append("目标价看跌")
    
    adj_signal = adjustment_result.get('signal', 'neutral')
    if adj_signal == 'bullish':
        logic_parts.append("预测上调")
    elif adj_signal == 'bearish':
        logic_parts.append("预测下调")

    decision_logic = "，".join(logic_parts)

    show_agent_reasoning({
        "最终信号": to_cn(combined.get('signal')),
        "置信度": f"{combined.get('confidence', 0.3)*100:.0f}%",
        "业绩预告": to_cn(earnings_announcement.get('signal')),
        "盈利预测": to_cn(forecast_result.get('signal')),
        "研报评级": to_cn(rating_result.get('signal')),
        "判断逻辑": decision_logic
    }, "预期差分析师")

    show_workflow_complete(
        "预期差分析师",
        signal=combined['signal'],
        confidence=combined.get('confidence', 0.3),
        details=message_content,
        message=f"预期差分析完成：信号{to_cn(combined.get('signal'))}，置信度{combined.get('confidence', 0.3)*100:.0f}%"
    )

    return {
        "messages": [message],
        "data": {
            **data,
            "expectation_diff_analysis": message_content
        },
        "metadata": state["metadata"],
    }
