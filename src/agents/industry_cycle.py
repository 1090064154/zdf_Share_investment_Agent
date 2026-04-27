"""
行业周期分析Agent
分析股票所属行业的周期位置，为周期股提供特殊分析逻辑
"""
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json

logger = setup_logger('industry_cycle_agent')

# 行业周期分类
INDUSTRY_CYCLE = {
    '强周期': [
        '农林牧渔', '养殖', '猪', '禽', '饲料', '化肥',
        '钢铁', '煤炭', '有色金属', '化工', '建材', '房地产',
        '汽车', '交通运输', '工程机械', '航运', '港口'
    ],
    '弱周期': [
        '食品饮料', '医药生物', '家用电器', '纺织服装', '日用化工',
        '商贸零售', '旅游', '酒店', '餐饮'
    ],
    '成长': [
        '电子', '半导体', '计算机', '软件', '通信', '5G',
        '新能源', '光伏', '风电', '锂电池', '电动车', '芯片'
    ],
    '防御': [
        '公用事业', '电力', '燃气', '水务', '银行', '保险', '券商'
    ]
}


def _identify_industry_cycle(industry: str) -> dict:
    """
    识别行业周期类型
    """
    if not industry:
        return {'cycle_type': 'unknown', 'signal': 'neutral', 'confidence': 0.3}

    for cycle_type, keywords in INDUSTRY_CYCLE.items():
        if any(keyword in industry for keyword in keywords):
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
    else:
        cycle_position = cycle_type_result

    # 3. 生成周期信号
    cycle_signal = _generate_cycle_signal(cycle_position)

    message_content = {
        "cycle_type": cycle_signal.get('cycle_type', 'unknown'),
        "phase": cycle_signal.get('phase', '未知'),
        "signal": cycle_signal.get('signal', 'neutral'),
        "confidence": f"{cycle_signal.get('confidence', 0.3) * 100:.0f}%",
        "reason": cycle_signal.get('reason', ''),
        "weight_factor": cycle_signal.get('weight_factor', 1.0),
        "industry": industry,
        "raw_analysis": cycle_position
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="industry_cycle_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "行业周期分析")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("行业周期分析师", "completed")
    logger.info("────────────────────────────────────────────────────────")
    logger.info("✅ 行业周期分析完成:")
    logger.info(f"  📊 最终信号: {cycle_signal.get('signal')}")
    logger.info(f"  📈 置信度: {cycle_signal.get('confidence')}")
    logger.info(f"  📈 周期阶段: {cycle_signal.get('stage', 'N/A')}")
    logger.info("────────────────────────────────────────────────────────")

    return {
        "messages": [message],
        "data": {
            **data,
            "industry_cycle_analysis": message_content
        },
        "metadata": state["metadata"],
    }
