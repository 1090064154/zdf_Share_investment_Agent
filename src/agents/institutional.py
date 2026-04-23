"""
机构持仓分析Agent
分析北向资金、基金持仓、社保持股等机构持仓变化
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json

logger = setup_logger('institutional_agent')


def _get_north_money_data(ticker: str) -> dict:
    """
    获取北向资金数据
    """
    try:
        import akshare as ak
        try:
            # 获取北向资金持股数据
            north_df = ak.stock_hsgt_individual_em(symbol=ticker)
            if north_df is not None and len(north_df) > 0:
                # 获取最新数据
                latest = north_df.iloc[0]

                # 尝试获取持股变化
                hold_change = 0
                if '持股变化' in latest:
                    hold_change = float(str(latest['持股变化']).replace('%', '').replace(',', '').replace('+', ''))
                elif '持股比例变化' in latest:
                    hold_change = float(str(latest['持股比例变化']).replace('%', '').replace(',', '').replace('+', ''))

                # 判断方向
                if hold_change > 0.5:
                    signal = 'bullish'
                    confidence = min(0.3 + abs(hold_change) * 0.1, 0.8)
                    reason = f"北向资金增持{hold_change:.2f}%"
                elif hold_change < -0.5:
                    signal = 'bearish'
                    confidence = min(0.3 + abs(hold_change) * 0.1, 0.8)
                    reason = f"北向资金减持{abs(hold_change):.2f}%"
                else:
                    signal = 'neutral'
                    confidence = 0.4
                    reason = "北向资金持股变化不大"

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'hold_change': hold_change,
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


def _analyze_institutional_signals(north_result: dict, fund_result: dict) -> dict:
    """
    综合分析机构持仓信号
    """
    signals = []
    confidences = []

    # 北向资金
    if north_result.get('confidence', 0) > 0:
        signals.append(north_result['signal'])
        confidences.append(north_result['confidence'])

    # 基金持仓
    if fund_result.get('confidence', 0) > 0:
        signals.append(fund_result['signal'])
        confidences.append(fund_result['confidence'])

    if not signals:
        return {
            'signal': 'neutral',
            'confidence': 0.3,
            'reason': '无机构持仓数据'
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

    reason = f"机构持仓信号: 北向{north_result.get('reason', 'N/A')}, 基金{fund_result.get('reason', 'N/A')}"

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'north_analysis': north_result,
        'fund_analysis': fund_result
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

    # 2. 获取基金持仓数据
    logger.info("  获取基金持仓数据...")
    fund_result = _get_fund_holding_data(ticker)
    logger.info(f"  基金持仓: {fund_result.get('reason', 'N/A')}")

    # 3. 综合分析
    combined = _analyze_institutional_signals(north_result, fund_result)

    message_content = {
        "signal": combined['signal'],
        "confidence": f"{combined.get('confidence', 0.3) * 100:.0f}%",
        "reason": combined.get('reason', ''),
        "north_money": north_result,
        "fund_holding": fund_result,
        "combined_analysis": combined
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False, indent=2),
        name="institutional_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "机构持仓分析")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("机构持仓分析师", "completed")
    logger.info(f"[INSTITUTIONAL] 分析完成: {combined.get('signal')}")

    return {
        "messages": [message],
        "data": {
            **data,
            "institutional_analysis": message_content
        },
        "metadata": state["metadata"],
    }
