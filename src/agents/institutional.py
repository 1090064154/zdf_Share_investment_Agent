"""
机构持仓分析Agent
分析北向资金、基金持仓、社保持股等机构持仓变化
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json

logger = setup_logger('institutional_agent')


def _get_shareholder_count(ticker: str) -> dict:
    """
    [NEW] 获取股东户数变化（筹码集中度分析）
    股东户数减少 → 筹码集中 → 利好
    股东户数增加 → 筹码分散 → 利空
    """
    try:
        import akshare as ak
        try:
            holder_df = ak.stock_zh_a_gdhs(symbol=ticker)
            if holder_df is not None and len(holder_df) >= 2:
                latest = holder_df.iloc[0]
                previous = holder_df.iloc[1]
                
                latest_count = float(latest.get('股东户数', 0) or 0)
                previous_count = float(previous.get('股东户数', 0) or 0)
                
                if latest_count > 0 and previous_count > 0:
                    change_pct = (latest_count - previous_count) / previous_count
                    
                    if change_pct < -0.05:
                        signal = 'bullish'
                        confidence = min(0.6 + abs(change_pct) * 2, 0.85)
                        reason = f"股东户数减少{abs(change_pct)*100:.1f}%，筹码集中"
                    elif change_pct > 0.05:
                        signal = 'bearish'
                        confidence = min(0.6 + change_pct * 2, 0.85)
                        reason = f"股东户数增加{change_pct*100:.1f}%，筹码分散"
                    else:
                        signal = 'neutral'
                        confidence = 0.5
                        reason = f"股东户数变化{change_pct*100:.1f}%，筹码相对稳定"
                    
                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'holder_count': latest_count,
                        'change_pct': change_pct * 100,
                        'reason': reason,
                        'source': 'shareholder'
                    }
        except Exception as e:
            logger.debug(f"股东户数数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'signal': 'neutral', 'confidence': 0, 'reason': '无法获取股东户数数据', 'source': 'shareholder'}


def _get_holding_trend(ticker: str, days: int = 5) -> dict:
    """
    [NEW] 获取机构持仓趋势分析
    分析多日净流入趋势
    """
    try:
        import akshare as ak
        try:
            market = "sz" if ticker.startswith("00") or ticker.startswith("30") else "sh"
            money_df = ak.stock_individual_money_flow(stock=ticker, market=market)
            
            if money_df is not None and len(money_df) >= days:
                trend_inflow = 0
                trend_days = min(days, len(money_df))
                
                for i in range(trend_days):
                    super_large = float(money_df.iloc[i].get('超大单净流入净额', 0) or 0)
                    large = float(money_df.iloc[i].get('大单净流入净额', 0) or 0)
                    trend_inflow += (super_large + large) / 10000
                
                avg_inflow = trend_inflow / trend_days
                
                if avg_inflow > 1000:
                    signal = 'bullish'
                    confidence = min(0.6 + avg_inflow / 10000, 0.85)
                    reason = f"{trend_days}日主力净流入平均{avg_inflow:.0f}万，持续买入"
                elif avg_inflow < -1000:
                    signal = 'bearish'
                    confidence = min(0.6 + abs(avg_inflow) / 10000, 0.85)
                    reason = f"{trend_days}日主力净流出平均{abs(avg_inflow):.0f}万，持续卖出"
                else:
                    signal = 'neutral'
                    confidence = 0.5
                    reason = f"{trend_days}日主力净流入平均{avg_inflow:.0f}万，方向不明"
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'trend_days': trend_days,
                    'avg_inflow': avg_inflow,
                    'total_inflow': trend_inflow,
                    'reason': reason,
                    'source': 'trend'
                }
        except Exception as e:
            logger.debug(f"持仓趋势数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'signal': 'neutral', 'confidence': 0, 'reason': '无法获取趋势数据', 'source': 'trend'}


def _get_north_money_data(ticker: str) -> dict:
    """
    获取北向资金数据
    """
    try:
        import akshare as ak
        try:
            north_df = ak.stock_hsgt_individual_em(symbol=ticker)
            if north_df is not None and len(north_df) > 0:
                latest = north_df.iloc[0]
                
                # 今日增持股数（可能为负数表示减持）
                change_shares = latest.get('今日增持股数', 0) or 0
                # 持股数量占A股百分比
                holding_pct = latest.get('持股数量占A股百分比', 0) or 0
                
                # 计算持股比例变化（与前一天对比）
                if len(north_df) >= 2:
                    prev = north_df.iloc[1]
                    prev_pct = prev.get('持股数量占A股百分比', 0) or 0
                    pct_change = (holding_pct or 0) - (prev_pct or 0)
                else:
                    pct_change = 0
                
                # 使用持股比例变化判断
                if pct_change > 0.01:  # 增持超过0.01%
                    signal = 'bullish'
                    confidence = min(0.4 + pct_change * 10, 0.8)
                    reason = f"北向资金增持{holding_pct:.2f}%，变化+{pct_change:.3f}%"
                elif pct_change < -0.01:  # 减持超过0.01%
                    signal = 'bearish'
                    confidence = min(0.4 + abs(pct_change) * 10, 0.8)
                    reason = f"北向资金持股{holding_pct:.2f}%，变化{pct_change:.3f}%"
                else:
                    signal = 'neutral'
                    confidence = 0.4
                    reason = f"北向资金持股{holding_pct:.2f}%，变化不明显"
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'hold_change': pct_change,
                    'holding_pct': holding_pct,
                    'reason': reason,
                    'source': 'north_money'
                }
        except Exception as e:
            logger.debug(f"北向资金数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'hold_change': 0,
        'reason': '无法获取北向资金数据',
        'source': 'north_money'
    }


def _get_fund_holding_data(ticker: str) -> dict:
    """
    获取基金持仓数据
    """
    try:
        import akshare as ak
        try:
            # 判断市场
            market = "sz" if ticker.startswith("00") or ticker.startswith("30") else "sh"
            # 获取个股资金流向数据
            fund_df = ak.stock_individual_fund_flow(stock=ticker, market=market)
            if fund_df is not None and len(fund_df) > 0:
                # 取最新一天的主力净流入
                latest = fund_df.iloc[0]
                main_inflow = latest.get('主力净流入-净额', 0) or 0
                main_ratio = latest.get('主力净流入-净占比', 0) or 0
                
                # 判断信号
                if main_inflow > 0 and main_ratio > 5:
                    signal = 'bullish'
                    confidence = min(0.4 + main_ratio * 0.05, 0.8)
                    reason = f"主力净流入{main_inflow/10000:.1f}万({main_ratio:.1f}%)"
                elif main_inflow < 0 and main_ratio < -5:
                    signal = 'bearish'
                    confidence = min(0.4 + abs(main_ratio) * 0.05, 0.8)
                    reason = f"主力净流出{abs(main_inflow)/10000:.1f}万({abs(main_ratio):.1f}%)"
                else:
                    signal = 'neutral'
                    confidence = 0.4
                    reason = "主力资金流动不明显"

                return {
                    'signal': signal,
                    'confidence': float(confidence),
                    'change': float(main_ratio),
                    'reason': reason,
                    'source': 'fund'
                }
        except Exception as e:
            logger.debug(f"基金持仓数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'change': 0,
        'reason': '无法获取基金持仓数据',
        'source': 'fund'
    }


def _get_money_flow_data(ticker: str) -> dict:
    """
    获取资金流数据（大单净流入、散户资金流向）

    A股特色指标：
    - 超大单净流入：机构行为
    - 大单净流入：主力行为
    - 中单净流入：大户行为
    - 小单净流入：散户行为
    """
    try:
        import akshare as ak
        try:
            # 判断市场
            market = "sz" if ticker.startswith("00") or ticker.startswith("30") else "sh"
            # 获取资金流向数据
            money_df = ak.stock_individual_money_flow(stock=ticker, market=market)
            if money_df is not None and len(money_df) > 0:
                latest = money_df.iloc[0]

                # 超大单净流入（大单机构）
                super_large_inflow = float(latest.get('超大单净流入净额', 0) or 0)
                super_large_ratio = float(latest.get('超大单净流入净占比', 0) or 0)

                # 大单净流入（主力）
                large_inflow = float(latest.get('大单净流入净额', 0) or 0)
                large_ratio = float(latest.get('大单净流入净占比', 0) or 0)

                # 小单净流入（散户）
                small_inflow = float(latest.get('小单净流入净额', 0) or 0)
                small_ratio = float(latest.get('小单净流入净占比', 0) or 0)

                # 综合主力资金净流入
                main_net_inflow = super_large_inflow + large_inflow
                main_net_ratio = super_large_ratio + large_ratio

                # 判断信号：主力净流入为正，散户净流入为负 → 利好
                # 主力净流出，散户净流入 → 利空
                if main_net_inflow > 0 and small_inflow < 0:
                    signal = 'bullish'
                    confidence = min(0.4 + abs(main_net_ratio) * 0.03, 0.85)
                    reason = f"主力净流入{main_net_inflow/10000:.1f}万(占比{main_net_ratio:.1f}%)，散户净流出"
                elif main_net_inflow < 0 and small_inflow > 0:
                    signal = 'bearish'
                    confidence = min(0.4 + abs(main_net_ratio) * 0.03, 0.85)
                    reason = f"主力净流出{abs(main_net_inflow)/10000:.1f}万(占比{abs(main_net_ratio):.1f}%)，散户净流入"
                elif abs(main_net_ratio) < 2:
                    signal = 'neutral'
                    confidence = 0.5
                    reason = "资金流向不明显，多空平衡"
                else:
                    signal = 'neutral'
                    confidence = 0.4
                    reason = f"资金流分歧，主力{'净流入' if main_net_inflow > 0 else '净流出'}，散户{'净流入' if small_inflow > 0 else '净流出'}"

                return {
                    'signal': signal,
                    'confidence': float(confidence),
                    'main_net_inflow': float(main_net_inflow),
                    'main_net_ratio': float(main_net_ratio),
                    'super_large_inflow': float(super_large_inflow),
                    'large_inflow': float(large_inflow),
                    'small_inflow': float(small_inflow),
                    'reason': reason,
                    'source': 'money_flow'
                }
        except Exception as e:
            logger.debug(f"资金流数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'main_net_inflow': 0,
        'reason': '无法获取资金流数据',
        'source': 'money_flow'
    }


def _get_margin_financing_data(ticker: str) -> dict:
    """
    获取融资融券数据

    融资余额增加 → 市场乐观
    融券余额增加 → 市场看空
    """
    try:
        import akshare as ak
        try:
            margin_df = ak.stock_margin_detail_szse(symbol=ticker)
            if margin_df is not None and len(margin_df) >= 2:
                # 获取最近两天数据
                latest = margin_df.iloc[0]
                previous = margin_df.iloc[1]

                # 融资余额
                rzye_latest = float(latest.get('融资余额', 0) or 0)
                rzye_previous = float(previous.get('融资余额', 0) or 0)

                # 融券余额
                rqye_latest = float(latest.get('融券余额', 0) or 0)
                rqye_previous = float(previous.get('融券余额', 0) or 0)

                # 计算变化
                rz_change = (rzye_latest - rzye_previous) / rzye_previous if rzye_previous > 0 else 0
                rq_change = (rqye_latest - rqye_previous) / rqye_previous if rqye_previous > 0 else 0

                # 融资余额增加代表杠杆资金做多
                if rz_change > 0.05:  # 融资余额增长超5%
                    signal = 'bullish'
                    confidence = min(0.5 + rz_change * 2, 0.8)
                    reason = f"融资余额增长{rz_change:.1%}，杠杆资金做多"
                elif rz_change < -0.05:  # 融资余额下降超5%
                    signal = 'bearish'
                    confidence = min(0.5 + abs(rz_change) * 2, 0.8)
                    reason = f"融资余额下降{abs(rz_change):.1%}，杠杆资金减仓"
                elif rq_change > 0.1:  # 融券余额大幅增长
                    signal = 'bearish'
                    confidence = min(0.4 + rq_change, 0.7)
                    reason = f"融券余额增长{rq_change:.1%}，看空情绪增强"
                else:
                    signal = 'neutral'
                    confidence = 0.5
                    reason = "融资融券变化不大"

                return {
                    'signal': signal,
                    'confidence': float(confidence),
                    'rz_balance': rzye_latest,
                    'rz_change': float(rz_change),
                    'rq_balance': rqye_latest,
                    'rq_change': float(rq_change),
                    'reason': reason,
                    'source': 'margin'
                }
        except Exception as e:
            logger.debug(f"融资融券数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'signal': 'neutral',
        'confidence': 0,
        'reason': '无法获取融资融券数据',
        'source': 'margin'
    }


def _analyze_institutional_signals(north_result: dict, fund_result: dict, money_flow_result: dict = None, margin_result: dict = None, holder_result: dict = None, trend_result: dict = None) -> dict:
    """
    [OPTIMIZED] 综合分析机构持仓信号（含资金流分析、趋势分析、股东户数）
    使用加权组合：北向资金权重最高
    """
    # [NEW] 定义各数据源权重
    weights = {
        'north': 0.25,
        'fund': 0.15,
        'money_flow': 0.20,
        'margin': 0.15,
        'holder': 0.15,
        'trend': 0.10
    }
    
    signal_values = {'bullish': 1, 'bearish': -1, 'neutral': 0}
    results = {
        'north': north_result,
        'fund': fund_result,
        'money_flow': money_flow_result,
        'margin': margin_result,
        'holder': holder_result,
        'trend': trend_result
    }
    
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
            'reason': '无机构持仓数据'
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
    if north_result.get('confidence', 0) > 0:
        reason_parts.append(f"北向{north_result.get('signal', 'neutral')}")
    if fund_result.get('confidence', 0) > 0:
        reason_parts.append(f"主力{fund_result.get('signal', 'neutral')}")
    if holder_result and holder_result.get('confidence', 0) > 0:
        reason_parts.append(f"筹码{holder_result.get('signal', 'neutral')}")
    if trend_result and trend_result.get('confidence', 0) > 0:
        reason_parts.append(f"趋势{trend_result.get('signal', 'neutral')}")
    
    reason = f"机构持仓：{' '.join(reason_parts)}"

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'north_analysis': north_result,
        'fund_analysis': fund_result,
        'money_flow_analysis': money_flow_result,
        'margin_analysis': margin_result,
        'holder_analysis': holder_result,
        'trend_analysis': trend_result
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

    reason = f"机构持仓信号: 北向{north_result.get('reason', 'N/A')}, 主力{fund_result.get('reason', 'N/A')}, 资金流{money_flow_result.get('reason', 'N/A') if money_flow_result else 'N/A'}"

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'north_analysis': north_result,
        'fund_analysis': fund_result,
        'money_flow_analysis': money_flow_result,
        'margin_analysis': margin_result
    }


@resilient_agent
@agent_endpoint("institutional", "机构持仓分析师，分析北向资金、基金持仓等机构持仓变化")
def institutional_agent(state: AgentState):
    """分析机构持仓"""
    show_workflow_status("机构持仓分析师")
    logger.info("="*50)
    logger.info("🏦 [INSTITUTIONAL] 开始机构持仓分析")
    logger.info("="*50)

    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    ticker = data.get("ticker", "")

    logger.info(f"  股票代码: {ticker}")

    # 1. 获取北向资金数据
    logger.info("  获取北向资金数据...")
    north_result = _get_north_money_data(ticker)
    logger.info(f"  北向资金: {north_result.get('reason', 'N/A')}")

    # 2. 获取基金/主力资金数据
    logger.info("  获取基金/主力资金数据...")
    fund_result = _get_fund_holding_data(ticker)
    logger.info(f"  基金/主力: {fund_result.get('reason', 'N/A')}")

    # 3. 获取资金流数据（大单净流入、散户流向）
    logger.info("  获取资金流数据...")
    money_flow_result = _get_money_flow_data(ticker)
    logger.info(f"  资金流: {money_flow_result.get('reason', 'N/A')}")

    # 4. 获取融资融券数据
    logger.info("  获取融资融券数据...")
    margin_result = _get_margin_financing_data(ticker)
    logger.info(f"  融资融券: {margin_result.get('reason', 'N/A')}")

    # [NEW] 5. 获取股东户数变化
    logger.info("  获取股东户数变化...")
    holder_result = _get_shareholder_count(ticker)
    logger.info(f"  股东户数: {holder_result.get('reason', 'N/A')}")

    # [NEW] 6. 获取持仓趋势
    logger.info("  获取持仓趋势...")
    trend_result = _get_holding_trend(ticker)
    logger.info(f"  持仓趋势: {trend_result.get('reason', 'N/A')}")

    # 7. 综合分析（含资金流、融资融券、股东户数、趋势）
    combined = _analyze_institutional_signals(north_result, fund_result, money_flow_result, margin_result, holder_result, trend_result)

    conf_float = combined.get('confidence', 0.3)
    signal = combined['signal']
    signal_cn = {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(signal, signal)

    def _sig_cn(r):
        s = r.get('signal', 'neutral') if r else 'neutral'
        return {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(s, s)

    decision_preview = f"北向{_sig_cn(north_result)}，主力{_sig_cn(fund_result)}，资金流{_sig_cn(money_flow_result)}，融资融券{_sig_cn(margin_result)}"

    # 发送机构持仓分析结果到前端
    show_agent_reasoning({
        "北向资金": f"{_sig_cn(north_result)} | {north_result.get('reason', '-')}",
        "主力资金": f"{_sig_cn(fund_result)} | {fund_result.get('reason', '-')}",
        "资金流向": f"{_sig_cn(money_flow_result)} | {money_flow_result.get('reason', '-')}",
        "融资融券": f"{_sig_cn(margin_result)} | {margin_result.get('reason', '-')}",
        "股东户数": f"{_sig_cn(holder_result)} | {holder_result.get('reason', '-')}",
        "持仓趋势": f"{_sig_cn(trend_result)} | {trend_result.get('reason', '-')}",
        "综合信号": signal_cn,
        "置信度": f"{conf_float*100:.0f}%",
    }, "机构持仓分析师")

    # 构建详细的数据项
    def _to_detail(result, name):
        if not result:
            return None
        s = result.get('signal', 'neutral')
        sc = {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(s, s)
        return {"name": name, "signal": s, "signal_cn": sc, "reason": result.get('reason', ''), "confidence": result.get('confidence', 0)}

    details = []
    for r, n in [
        (north_result, "北向资金"), 
        (fund_result, "主力资金"), 
        (money_flow_result, "资金流向"), 
        (margin_result, "融资融券"),
        (holder_result, "股东户数"),
        (trend_result, "持仓趋势")
    ]:
        d = _to_detail(r, n)
        if d and d["confidence"] > 0:
            details.append(d)

    # 构建决策逻辑
    parts = []
    for d_item in details:
        parts.append(f"{d_item['name']}：{d_item['signal_cn']}")
    decision_logic = "；".join(parts) if parts else "无有效机构数据"

    message_content = {
        "signal": signal,
        "confidence": round(conf_float, 4),
        "signal_cn": signal_cn,
        "reason": combined.get('reason', ''),
        "decision_logic": decision_logic,
        "details": details,
        "summary": f"机构持仓{signal_cn}（置信度{conf_float*100:.0f}%），{decision_logic}",
        "north_money": north_result,
        "fund_holding": fund_result,
        "money_flow": money_flow_result,
        "margin": margin_result,
        "shareholder": holder_result,
        "trend": trend_result,
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False, indent=2),
        name="institutional_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    show_agent_reasoning({
        "最终信号": signal_cn,
        "置信度": f"{conf_float*100:.0f}%",
        "北向资金": f"{_sig_cn(north_result)} | {north_result.get('reason', '-')}",
        "主力资金": f"{_sig_cn(fund_result)} | {fund_result.get('reason', '-')}",
        "资金流向": f"{_sig_cn(money_flow_result)} | {money_flow_result.get('reason', '-')}",
        "融资融券": f"{_sig_cn(margin_result)} | {margin_result.get('reason', '-')}",
        "决策逻辑": decision_logic
    }, "机构持仓分析师")

    show_workflow_complete(
        "机构持仓分析师",
        signal=signal,
        confidence=conf_float,
        details=message_content,
        message=f"机构持仓分析完成：信号{signal_cn}，置信度{conf_float*100:.0f}%"
    )

    return {
        "messages": [message],
        "data": {
            **data,
            "institutional_analysis": message_content
        },
        "metadata": state["metadata"],
    }
