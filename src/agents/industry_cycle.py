"""
行业周期分析Agent
分析股票所属行业的周期位置，为周期股提供特殊分析逻辑
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
from src.agents.fundamentals import INDUSTRY_CYCLE_CLASSIFICATION
import json

logger = setup_logger('industry_cycle_agent')

# [OPTIMIZED] 统一从fundamentals导入行业分类
INDUSTRY_CYCLE = INDUSTRY_CYCLE_CLASSIFICATION


def _get_inventory_cycle() -> dict:
    """
    [NEW] 获取库存周期分析
    通过PMI荣枯线和企业库存变化判断库存周期阶段
    """
    try:
        import akshare as ak
        try:
            pmi_df = ak.cn_pmi()
            if pmi_df is not None and len(pmi_df) > 0:
                latest_pmi = float(pmi_df.iloc[0].get('pmi', 50) or 50)
                ppi_df = ak.cn_ppi()
                ppi_change = 0
                if ppi_df is not None and len(ppi_df) > 0:
                    ppi_change = float(ppi_df.iloc[0].get('ppi', 0) or 0)
                
                if latest_pmi > 55:
                    phase = '主动补库'
                    signal = 'bullish'
                    confidence = 0.7
                elif latest_pmi > 50:
                    phase = '被动补库'
                    signal = 'neutral'
                    confidence = 0.5
                elif latest_pmi > 45:
                    phase = '主动去库'
                    signal = 'bearish'
                    confidence = 0.6
                else:
                    phase = '被动去库'
                    signal = 'neutral'
                    confidence = 0.5
                    
                return {
                    'phase': phase,
                    'signal': signal,
                    'confidence': confidence,
                    'pmi': latest_pmi,
                    'ppi_change': ppi_change,
                    'reason': f'PMI={latest_pmi:.1f}，PPI同比{ppi_change:.1f}%'
                }
        except Exception as e:
            logger.debug(f"库存周期数据获取失败: {e}")
    except ImportError:
        pass
    
    return {'phase': '未知', 'signal': 'neutral', 'confidence': 0.3, 'pmi': None, 'ppi_change': None}


def _get_commodity_price(commodity_type: str) -> dict:
    """
    [NEW] 获取大宗商品价格
    支持：钢材(螺纹钢)、煤炭(动力煤)、化工(MDI等)
    """
    result = {'price': None, 'change': None, 'signal': 'neutral', 'confidence': 0}
    
    try:
        import akshare as ak
        try:
            if commodity_type == 'steel':
                # 螺纹钢价格
                steel_df = ak.reits_abstract_em(indicator='螺纹钢价:HRB400:20mm:上海')
                if steel_df is not None and len(steel_df) > 0:
                    price = float(steel_df.iloc[-1].get('close', 0) or 0)
                    change = float(steel_df.iloc[-1].get('pct_chg', 0) or 0)
                    result = _analyze_price_change(price, change, '螺纹钢')
            elif commodity_type == 'coal':
                # 动力煤价格
                coal_df = ak.reits_abstract_em(indicator='动力煤期货收盘价')
                if coal_df is not None and len(coal_df) > 0:
                    price = float(coal_df.iloc[-1].get('close', 0) or 0)
                    change = float(coal_df.iloc[-1].get('pct_chg', 0) or 0)
                    result = _analyze_price_change(price, change, '动力煤')
            elif commodity_type == 'chemical':
                # 化工品价格(MDI)
                chem_df = ak.reits_abstract_em(indicator='MDI聚合物价格')
                if chem_df is not None and len(chem_df) > 0:
                    price = float(chem_df.iloc[-1].get('close', 0) or 0)
                    change = float(chem_df.iloc[-1].get('pct_chg', 0) or 0)
                    result = _analyze_price_change(price, change, 'MDI')
        except Exception as e:
            logger.debug(f"商品价格获取失败: {e}")
    except ImportError:
        pass
    
    return result


def _analyze_price_change(price: float, change: float, name: str) -> dict:
    """分析价格变化"""
    if change > 5:
        signal = 'bullish'
        confidence = 0.7
    elif change > 0:
        signal = 'neutral'
        confidence = 0.5
    elif change < -5:
        signal = 'bearish'
        confidence = 0.7
    elif change < 0:
        signal = 'neutral'
        confidence = 0.5
    else:
        signal = 'neutral'
        confidence = 0.3
    
    return {
        'price': round(price, 2),
        'change': round(change, 2),
        'signal': signal,
        'confidence': confidence,
        'reason': f'{name}价格{"上涨" if change > 0 else "下跌" if change < 0 else "平稳"}{abs(change):.1f}%'
    }


def _identify_industry_cycle(industry: str) -> dict:
    """
    识别行业周期类型
    """
    if not industry:
        return {'cycle_type': 'unknown', 'signal': 'neutral', 'confidence': 0.3}

    for cycle_type, industries in INDUSTRY_CYCLE.items():
        if any(ind in industry for ind in industries):
            return {'cycle_type': cycle_type, 'signal': 'neutral', 'confidence': 0.5}

    return {'cycle_type': 'other', 'signal': 'neutral', 'confidence': 0.3}


def _analyze_pig_cycle(ticker: str = None) -> dict:
    """
    [简化版] 猪周期分析
    实际需要对接猪价、存栏等数据
    """
    # 这里使用简化逻辑，实际应该获取真实数据
    try:
        import akshare as ak
        try:
            # 尝试获取猪价数据
            pig_price = ak.zhu_liu_price()
            if pig_price is not None and len(pig_price) > 0:
                latest_price = pig_price.iloc[-1]
                price_change = pig_price['外三元(元/公斤)'].pct_change().iloc[-1] if '外三元(元/公斤)' in pig_price.columns else 0

                if price_change > 0.1:
                    signal = 'bullish'
                    confidence = 0.7
                    phase = '价格上涨周期'
                elif price_change < -0.1:
                    signal = 'bearish'
                    confidence = 0.7
                    phase = '价格下跌周期'
                else:
                    signal = 'neutral'
                    confidence = 0.5
                    phase = '价格平稳期'

                return {
                    'cycle_type': '强周期',
                    'phase': phase,
                    'signal': signal,
                    'confidence': confidence,
                    'key_indicator': '猪价',
                    'price': latest_price,
                    'reason': f'猪周期阶段: {phase}'
                }
        except Exception as e:
            logger.debug(f"猪周期数据获取失败: {e}")
    except ImportError:
        pass

    return {
        'cycle_type': '强周期',
        'phase': '未知',
        'signal': 'neutral',
        'confidence': 0.3,
        'reason': '无法获取猪周期数据'
    }


def _analyze_cyclical_industry(ticker: str, industry: str) -> dict:
    """
    分析周期行业当前位置
    """
    # 养殖行业 - 猪周期
    if any(keyword in industry for keyword in ['农林牧渔', '养殖', '猪', '牧原', '温氏']):
        return _analyze_pig_cycle(ticker)

    # 其他周期行业 - 简化处理
    return {
        'cycle_type': '强周期',
        'phase': '需进一步分析',
        'signal': 'neutral',
        'confidence': 0.3,
        'reason': f'行业:{industry}，周期位置需进一步分析'
    }


def _generate_cycle_signal(cycle_info: dict) -> dict:
    """
    根据周期位置生成投资信号
    """
    cycle_type = cycle_info.get('cycle_type', 'unknown')
    phase = cycle_info.get('phase', '')
    signal = cycle_info.get('signal', 'neutral')
    confidence = cycle_info.get('confidence', 0.3)

    # 根据周期类型和位置生成建议
    if cycle_type == '强周期':
        # 周期股：在周期底部/上行期增加配置，下行期减少配置
        if signal == 'bullish':
            reason = '周期上行期，增加配置'
            weight_factor = 1.2
        elif signal == 'bearish':
            reason = '周期下行期，减少配置'
            weight_factor = 0.5
        else:
            reason = '周期平稳期，正常配置'
            weight_factor = 1.0
    elif cycle_type == '成长':
        # 成长股：享受高估值，关注增长确定性
        reason = '成长股，关注业绩增长确定性'
        weight_factor = 1.0
    elif cycle_type == '防御':
        # 防御板块：稳定收益
        reason = '防御板块，追求稳定收益'
        weight_factor = 0.8
    else:
        reason = '非周期股，正常分析'
        weight_factor = 1.0

    return {
        'signal': signal,
        'confidence': confidence,
        'cycle_type': cycle_type,
        'phase': phase,
        'reason': reason,
        'weight_factor': weight_factor
    }


@resilient_agent
@agent_endpoint("industry_cycle", "行业周期分析师，分析行业周期位置，判断当前所处阶段")
def industry_cycle_agent(state: AgentState):
    """分析行业周期"""
    show_workflow_status("行业周期分析师")
    logger.info("="*50)
    logger.info("🔄 [INDUSTRY_CYCLE] 开始行业周期分析")
    logger.info("="*50)

    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    ticker = data.get("ticker", "")
    industry = data.get("industry", "")

    logger.info(f"  股票代码: {ticker}")
    logger.info(f"  行业: {industry}")

    # 1. 识别行业周期类型
    cycle_type_result = _identify_industry_cycle(industry)

    # 2. 如果是周期行业，分析当前位置
    if cycle_type_result['cycle_type'] == '强周期':
        cycle_position = _analyze_cyclical_industry(ticker, industry)
        
        # [NEW] 获取库存周期
        inventory_cycle = _get_inventory_cycle()
        cycle_position['inventory_cycle'] = inventory_cycle
        logger.info(f"  📊 库存周期: {inventory_cycle.get('phase', '未知')}")
        
        # [NEW] 获取对应大宗商品价格
        commodity_data = {}
        if any(k in industry for k in ['钢铁', '建材', '工程机械']):
            commodity_data = _get_commodity_price('steel')
        elif any(k in industry for k in ['煤炭', '能源']):
            commodity_data = _get_commodity_price('coal')
        elif any(k in industry for k in ['化工']):
            commodity_data = _get_commodity_price('chemical')
        
        if commodity_data.get('price'):
            cycle_position['commodity'] = commodity_data
            logger.info(f"  📈 大宗商品: {commodity_data.get('reason', '')}")
    else:
        cycle_position = cycle_type_result
        # 非强周期行业也获取库存周期作为参考
        inventory_cycle = _get_inventory_cycle()
        cycle_position['inventory_cycle'] = inventory_cycle

    # 3. 生成周期信号
    cycle_signal = _generate_cycle_signal(cycle_position)

    # 获取信号结果
    conf_float = cycle_signal.get('confidence', 0.3)
    cycle_type_cn_map = {'强周期': '强周期行业', '弱周期': '弱周期行业', '成长': '成长型行业', '防御': '防御型行业', 'other': '其他', 'unknown': '未知'}
    cycle_type_cn = cycle_type_cn_map.get(cycle_signal.get('cycle_type', 'unknown'), cycle_signal.get('cycle_type', 'unknown'))
    phase = cycle_signal.get('phase', '未知')
    reason = cycle_signal.get('reason', '')
    signal = cycle_signal.get('signal', 'neutral')
    signal_cn = {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(signal, signal)
    wf = cycle_signal.get('weight_factor', 1.0)

    # 构建决策逻辑
    if signal == 'bullish':
        decision_logic = f"{industry}属于{cycle_type_cn}，当前处于{phase}，建议增配（权重系数{wf}）"
    elif signal == 'bearish':
        decision_logic = f"{industry}属于{cycle_type_cn}，当前处于{phase}，建议减配（权重系数{wf}）"
    else:
        decision_logic = f"{industry}属于{cycle_type_cn}，当前处于{phase}，建议标配（权重系数{wf}）"

    # 发送行业周期分析结果到前端
    show_agent_reasoning({
        "行业": industry,
        "周期类型": cycle_type_cn,
        "周期阶段": phase,
        "信号": signal_cn,
        "置信度": f"{conf_float*100:.0f}%",
        "决策逻辑": decision_logic,
        "权重系数": f"{wf:.1f}",
        "分析依据": reason
    }, "行业周期分析师")

    message_content = {
        "signal": signal,
        "confidence": round(conf_float, 4),
        "industry": industry,
        "cycle_type": cycle_signal.get('cycle_type', 'unknown'),
        "cycle_type_cn": cycle_type_cn,
        "phase": phase,
        "signal_cn": signal_cn,
        "reason": reason,
        "decision_logic": decision_logic,
        "weight_factor": wf,
        "inventory_cycle": cycle_position.get('inventory_cycle', {}),
        "commodity": cycle_position.get('commodity', {}),
        "summary": f"{industry}：{cycle_type_cn}，{phase}，信号{signal_cn}（置信度{conf_float*100:.0f}%），{reason}"
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="industry_cycle_agent",
    )

    if show_reasoning:
        state["metadata"]["agent_reasoning"] = message_content

    def to_cn(s):
        return {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(s, s) if s else 'N/A'

    cycle_type_cn = {'cyclical': '周期性行业', 'growth': '成长型行业', 'defensive': '防御型行业', 'unknown': '未知'}.get(cycle_signal.get('cycle_type', 'unknown'), cycle_signal.get('cycle_type', 'unknown'))
    phase_cn = {'early': '复苏期', 'mid': '繁荣期', 'late': '衰退期', 'bottom': '萧条期', 'unknown': '未知'}.get(cycle_signal.get('phase', 'unknown'), cycle_signal.get('phase', 'unknown'))

    show_agent_reasoning({
        "最终信号": signal_cn,
        "置信度": f"{conf_float*100:.0f}%",
        "行业": industry,
        "周期类型": cycle_type_cn,
        "周期阶段": phase,
        "决策逻辑": decision_logic,
        "投资策略": reason
    }, "行业周期分析师")

    show_workflow_complete(
        "行业周期分析师",
        signal=signal,
        confidence=conf_float,
        details=message_content,
        message=f"行业周期分析完成：{cycle_type_cn}，{phase}，信号{signal_cn}，置信度{conf_float*100:.0f}%"
    )

    return {
        "messages": [message],
        "data": {
            **data,
            "industry_cycle_analysis": message_content
        },
        "metadata": state["metadata"],
    }
