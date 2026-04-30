from typing import Annotated, Any, Dict, Sequence, TypedDict
from contextvars import ContextVar
import operator
from langchain_core.messages import BaseMessage
import json
from src.utils.logging_config import setup_logger

# 设置日志记录
logger = setup_logger('agent_state')

# 线程安全的上下文变量，用于传递 run_id
_current_run_id: ContextVar[str] = ContextVar('current_run_id', default='')

_AGENT_NAME_ALIASES = {
    "technical_analyst": "technical_analyst_agent",
    "fundamentals": "fundamentals_agent",
    "sentiment": "sentiment_agent",
    "valuation": "valuation_agent",
    "industry_cycle": "industry_cycle_agent",
    "institutional": "institutional_agent",
    "expectation_diff": "expectation_diff_agent",
    "macro_analyst": "macro_analyst_agent",
    "researcher_bull": "researcher_bull_agent",
    "researcher_bear": "researcher_bear_agent",
    "debate_room": "debate_room_agent",
    "risk_management": "risk_management_agent",
    "portfolio_management": "portfolio_management_agent",
    "market_data": "market_data_agent",
    "技术分析师": "technical_analyst_agent",
    "基本面分析师": "fundamentals_agent",
    "情绪分析师": "sentiment_agent",
    "估值Agent": "valuation_agent",
    "行业周期分析师": "industry_cycle_agent",
    "机构持仓分析师": "institutional_agent",
    "预期差分析师": "expectation_diff_agent",
    "宏观新闻Agent": "macro_news_agent",
    "宏观分析师": "macro_analyst_agent",
    "看多研究员": "researcher_bull_agent",
    "看空研究员": "researcher_bear_agent",
    "辩论室": "debate_room_agent",
    "风险管理师": "risk_management_agent",
    "市场数据Agent": "market_data_agent",
    "投资组合管理": "portfolio_management_agent",
}

def set_current_run_id(run_id: str):
    """设置当前 run_id（在工作流开始时调用）"""
    _current_run_id.set(run_id)

def get_current_run_id() -> str:
    """获取当前 run_id"""
    return _current_run_id.get()


def normalize_agent_name(agent_name: str) -> str:
    """统一 SSE 中的 agent 名称，避免前后端和不同 agent 的别名不一致。"""
    if not agent_name:
        return agent_name

    cleaned = agent_name.split(":")[0].strip()
    return _AGENT_NAME_ALIASES.get(cleaned, cleaned)

def _send_sse_event(event_type: str, agent_name: str, status: str = None, message: str = None, level: str = None, data: dict = None):
    """发送 SSE 事件（如果可用）"""
    run_id = get_current_run_id()
    agent_name = normalize_agent_name(agent_name)
    print(f">>> [_send_sse_event] run_id={run_id}, event_type={event_type}, agent={agent_name}", flush=True)
    if not run_id:
        print(f">>> [_send_sse_event] 没有 run_id，跳过", flush=True)
        return

    try:
        from src.api.log_hook import send_agent_event
        kwargs = {}
        if status:
            kwargs['status'] = status
        if message:
            kwargs['message'] = message
        if level:
            kwargs['level'] = level
        if data:
            kwargs.update(data)

        if event_type == 'agent_start':
            send_agent_event(run_id, 'agent_start', agent_name, **kwargs)
        elif event_type == 'agent_complete':
            send_agent_event(run_id, 'agent_complete', agent_name, **kwargs)
        elif event_type == 'agent_log':
            send_agent_event(run_id, 'agent_log', agent_name, **kwargs)
    except Exception as e:
        # SSE 发送失败不应该中断主流程
        print(f">>> [_send_sse_event] 异常: {e}", flush=True)
        logger.debug(f"SSE 事件发送失败: {e}")


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
    # [OPTIMIZED] 新增：Agent结果缓存，避免重复遍历messages
    agent_results: Annotated[Dict[str, Any], merge_dicts]


def show_workflow_status(agent_name: str, status: str = "processing"):
    """Display agent workflow status in a clean format and send SSE event.

    Args:
        agent_name: Name of the agent
        status: Status of the agent's work ("processing" or "completed")
    """
    if status == "processing":
        logger.info(f"🔄 {agent_name} 正在分析...")
        _send_sse_event('agent_start', agent_name, message=f"{agent_name} 正在分析...")
    else:
        logger.info(f"✅ {agent_name} 分析完成")
        _send_sse_event('agent_complete', agent_name, message=f"{agent_name} 分析完成")


def show_workflow_complete(agent_name: str, signal: str = None, confidence: float = None, details: dict = None, message: str = None):
    """发送完整的agent完成事件，包含信号、置信度和详情

    Args:
        agent_name: Name of the agent
        signal: Agent's signal (bullish/bearish/neutral)
        confidence: Confidence level (0-1)
        details: Full result details dict
        message: Optional completion message
    """
    logger.info(f"✅ {agent_name} 分析完成")
    try:
        data = {}
        if signal:
            data['signal'] = signal
        if confidence is not None:
            data['confidence'] = confidence
        if details:
            data['details'] = details
        if message:
            data['message'] = message
        _send_sse_event('agent_complete', agent_name, message=message, data=data)
    except Exception as e:
        logger.warning(f"show_workflow_complete SSE发送失败: {e}")


def show_agent_reasoning(output, agent_name):
    """Display agent's analysis results and send key reasoning to frontend via SSE."""
    SIGNAL_MAP = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}
    ENV_MAP = {'favorable': '有利', 'unfavorable': '不利', 'neutral': '中性'}
    IMPACT_MAP = {'positive': '正面', 'negative': '负面', 'neutral': '中性'}

    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)

    def convert_signals(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == 'signal' and isinstance(value, str) and value in SIGNAL_MAP:
                    result[key] = SIGNAL_MAP[value]
                elif key == 'macro_environment' and isinstance(value, str) and value in ENV_MAP:
                    result[key] = ENV_MAP[value]
                elif key == 'impact_on_stock' and isinstance(value, str) and value in IMPACT_MAP:
                    result[key] = IMPACT_MAP[value]
                elif key in ['最终信号', '信号'] and isinstance(value, str) and value in SIGNAL_MAP:
                    result[key] = SIGNAL_MAP[value]
                elif isinstance(value, (dict, list)):
                    result[key] = convert_signals(value)
                else:
                    result[key] = value
            return result
        elif isinstance(obj, list):
            return [convert_signals(item) for item in obj]
        else:
            return obj

    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        serializable_output = convert_signals(serializable_output)
        logger.info(json.dumps(serializable_output, indent=2, ensure_ascii=False))
        _send_key_reasoning(agent_name, serializable_output)
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2, ensure_ascii=False))
            # 发送关键推理日志到前端
            _send_key_reasoning(agent_name, parsed_output)
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)
            _send_sse_event('agent_log', agent_name, level='info', message=str(output)[:500])


def _send_key_reasoning(agent_name: str, reasoning_data: dict):
    """发送关键推理数据到前端，包括各子模块的详细分析结果"""
    if not isinstance(reasoning_data, dict):
        return

    # 根据不同Agent类型，提取并发送关键信息
    agent_name_lower = agent_name.lower() if isinstance(agent_name, str) else ""

    try:
        # 技术分析：发送各子策略信号
        if 'technical' in agent_name_lower or '技术' in agent_name or 'strategy_signals' in reasoning_data:
            strategy_signals = reasoning_data.get('strategy_signals', {})
            strategy_map = {
                'trend_following': '趋势跟踪',
                'mean_reversion': '均值回归',
                'momentum': '动量',
                'volatility': '波动率',
                'stat_arb': '统计套利',
                'stat_arb_analysis': '统计套利'
            }
            for strategy, signal_data in strategy_signals.items():
                if isinstance(signal_data, dict):
                    signal = signal_data.get('signal','-')
                    conf = signal_data.get('confidence','-')
                    signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                    strategy_cn = strategy_map.get(strategy, strategy)
                    msg = f"【{strategy_cn}】信号:{signal_cn} 置信度:{conf}"
                    _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 基本面分析：发送各维度信号
        elif 'fundamental' in agent_name_lower or '基本面' in agent_name:
            reasoning = reasoning_data.get('reasoning', {})
            key_map = {
                'profitability': '盈利能力',
                'profitability_signal': '盈利能力',
                'growth': '成长性',
                'growth_signal': '成长性',
                'financial_health': '财务健康',
                'financial_health_signal': '财务健康',
                'price_ratios_signal': '估值比率',
                'pb_roe_analysis': 'PB-ROE分析',
                'cyclical_analysis': '周期分析',
                'revenue_quality_analysis': '营收质量分析',
                'earnings_quality': '盈利质量'
            }
            if isinstance(reasoning, dict):
                for key, value in reasoning.items():
                    if isinstance(value, dict) and 'signal' in value:
                        signal = value.get('signal','-')
                        signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                        details = value.get('details','-')
                        key_cn = key_map.get(key, key)
                        msg = f"【{key_cn}】信号:{signal_cn} 详情:{details}"
                        _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 估值分析：发送各估值方法结果
        elif 'valuation' in agent_name_lower or '估值' in agent_name:
            reasoning = reasoning_data.get('reasoning', {})
            valuation_map = {
                'dcf_analysis': 'DCF分析',
                'owner_earnings_analysis': '所有者收益分析',
                'relative_analysis': '相对估值分析',
                'liquidation_analysis': '清算价值分析',
                'pe_analysis': '市盈率分析',
                'pb_analysis': '市净率分析'
            }
            if isinstance(reasoning, dict):
                for key, value in reasoning.items():
                    if isinstance(value, dict):
                        signal = value.get('signal', '-')
                        signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                        details = value.get('details', '-')
                        key_cn = valuation_map.get(key, key)
                        msg = f"【{key_cn}】{signal_cn}: {details}"
                        _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 情绪分析：发送各数据源结果
        elif 'sentiment' in agent_name_lower or '情绪' in agent_name:
            components = reasoning_data.get('components', {})
            source_map = {'news': '新闻舆情', 'guba': '股吧热度', 'quant': '量化指标', 'north_money': '北向资金'}
            if isinstance(components, dict):
                for source, data in components.items():
                    if isinstance(data, dict):
                        source_cn = source_map.get(source, source)
                        score = data.get('score', data.get('adjusted_score', '-'))
                        contrib = data.get('contribution', '-')
                        contrib_cn = str(round(float(contrib) * 100, 1)) + '%' if contrib not in ['-', ''] else '-'
                        post_count = data.get('news_count', data.get('post_count', ''))
                        count_info = f"(条数:{post_count})" if post_count else ""
                        msg = f"【{source_cn}】{count_info}分数:{score:.3f} 贡献:{contrib_cn}"
                        _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 风险管理：发送风险评分详情
        elif ('risk' in agent_name_lower and 'management' in agent_name_lower) or '风险' in agent_name:
            risk_indicators = reasoning_data.get('风险指标', reasoning_data.get('risk_indicators', {}))
            if isinstance(risk_indicators, dict):
                vol = risk_indicators.get('波动率', risk_indicators.get('volatility', '-'))
                var = risk_indicators.get('95%风险价值(VaR)', risk_indicators.get('var_95', '-'))
                dd = risk_indicators.get('最大回撤', risk_indicators.get('max_drawdown', '-'))
                vol_cn = str(round(float(vol)*100, 2)) + '%' if vol not in ['-', ''] else vol
                msg = f"波动率:{vol_cn} VaR:{var} 最大回撤:{dd}"
                _send_sse_event('agent_log', agent_name, level='info', message=msg)

            # 发送完整风险评分
            risk_score = reasoning_data.get('风险评分', reasoning_data.get('风险评分', '-'))
            threshold = reasoning_data.get('动态阈值', reasoning_data.get('动态阈值', '-'))
            action = reasoning_data.get('交易建议', reasoning_data.get('交易行动', '-'))
            if risk_score != '-':
                _send_sse_event('agent_log', agent_name, level='info', message=f"风险评分:{risk_score} 阈值:{threshold} 建议:{action}")

            # 发送决策逻辑
            decision_logic = reasoning_data.get('决策逻辑', '')
            if decision_logic:
                logic_lines = str(decision_logic).split('；')
                for line in logic_lines[:4]:
                    if line.strip():
                        _send_sse_event('agent_log', agent_name, level='info', message=f"决策:{line.strip()[:100]}")

        # 辩论室：发送多空置信度和决策逻辑
        elif 'debate' in agent_name_lower or '辩论' in agent_name:
            bull_conf = reasoning_data.get('bull_confidence') or reasoning_data.get('看多置信度', '-')
            bear_conf = reasoning_data.get('bear_confidence') or reasoning_data.get('看空置信度', '-')
            signal = reasoning_data.get('signal') or reasoning_data.get('最终信号', '-')
            confidence = reasoning_data.get('confidence') or reasoning_data.get('置信度', '-')
            signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
            msg = f"看多置信度:{bull_conf} 看空置信度:{bear_conf} 最终信号:{signal_cn} 置信度:{confidence}"
            _send_sse_event('agent_log', agent_name, level='info', message=msg)

            # 发送看多/看空论点列表
            bull_points = reasoning_data.get('看多论点', [])
            bear_points = reasoning_data.get('看空论点', [])
            if bull_points:
                _send_sse_event('agent_log', agent_name, level='info', message=f"看多论点({len(bull_points)}条):")
                for i, point in enumerate(bull_points[:3], 1):
                    _send_sse_event('agent_log', agent_name, level='info', message=f"  {i}. {str(point)[:80]}")
            if bear_points:
                _send_sse_event('agent_log', agent_name, level='info', message=f"看空论点({len(bear_points)}条):")
                for i, point in enumerate(bear_points[:3], 1):
                    _send_sse_event('agent_log', agent_name, level='info', message=f"  {i}. {str(point)[:80]}")

            # 发送决策逻辑
            reasoning_text = reasoning_data.get('决策逻辑') or reasoning_data.get('llm_analysis') or reasoning_data.get('reasoning', '')
            if reasoning_text and isinstance(reasoning_text, str) and len(reasoning_text) > 0:
                if len(reasoning_text) > 300:
                    reasoning_text = reasoning_text[:300] + "..."
                _send_sse_event('agent_log', agent_name, level='info', message=f"决策逻辑: {reasoning_text}")

        # 多头/空头研究员：发送论点统计
        elif 'researcher' in agent_name_lower or '研究员' in agent_name:
            count = (reasoning_data.get('bullish_count') or reasoning_data.get('bearish_count')
                     or reasoning_data.get('看涨信号数') or reasoning_data.get('看跌信号数')
                     or reasoning_data.get('论点数量', '-'))
            signal = (reasoning_data.get('signal') or reasoning_data.get('perspective')
                      or reasoning_data.get('最终信号', '-'))
            confidence = reasoning_data.get('confidence') or reasoning_data.get('置信度', '-')
            signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
            msg = f"信号:{signal_cn} 置信度:{confidence} 统计:{count}"
            _send_sse_event('agent_log', agent_name, level='info', message=msg)

            # 发送看多/看空论点列表
            thesis_points = reasoning_data.get('thesis_points', [])
            if thesis_points:
                _send_sse_event('agent_log', agent_name, level='info', message=f"看多论点列表:")
                for i, point in enumerate(thesis_points[:5], 1):
                    _send_sse_event('agent_log', agent_name, level='info', message=f"  {i}. {str(point)[:80]}")

            risk_points = reasoning_data.get('risk_points', [])
            if risk_points:
                _send_sse_event('agent_log', agent_name, level='info', message=f"看空风险列表:")
                for i, point in enumerate(risk_points[:5], 1):
                    _send_sse_event('agent_log', agent_name, level='info', message=f"  {i}. {str(point)[:80]}")

        # 行业周期：发送周期、阶段和决策逻辑
        elif 'industry' in agent_name_lower or 'cycle' in agent_name_lower or '行业' in agent_name:
            if '行业' in agent_name or '行业周期' in agent_name:
                industry = reasoning_data.get('行业', reasoning_data.get('industry', '-'))
                cycle_type = reasoning_data.get('周期类型', reasoning_data.get('cycle_type_cn', reasoning_data.get('cycle_type', '-')))
                phase = reasoning_data.get('周期阶段', reasoning_data.get('phase', '-'))
                signal = reasoning_data.get('信号', reasoning_data.get('signal', reasoning_data.get('最终信号', '-')))
                confidence = reasoning_data.get('置信度', reasoning_data.get('confidence', '-'))
                signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                _send_sse_event('agent_log', agent_name, level='info', message=f"行业周期: {industry} | {cycle_type} | {phase} | 信号{signal_cn} | 置信度{confidence}")
                # 发送决策逻辑
                logic = reasoning_data.get('决策逻辑', reasoning_data.get('decision_logic', ''))
                if logic:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"决策逻辑: {logic}")
                strategy = reasoning_data.get('投资策略', reasoning_data.get('投资建议', reasoning_data.get('reason', '')))
                if strategy:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"投资策略: {strategy}")

        # 机构持仓：发送各维度详情和决策逻辑
        elif 'institutional' in agent_name_lower or '机构' in agent_name:
            if '机构' in agent_name or '机构持仓' in agent_name:
                signal = reasoning_data.get('最终信号', reasoning_data.get('signal', '-'))
                confidence = reasoning_data.get('置信度', reasoning_data.get('confidence', '-'))
                signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                _send_sse_event('agent_log', agent_name, level='info', message=f"机构持仓综合: {signal_cn} | 置信度{confidence}")
                # 各维度详情
                for key, label in [('北向资金', '北向'), ('主力资金', '主力'), ('资金流向', '资金流'), ('融资融券', '两融')]:
                    val = reasoning_data.get(key, '')
                    if val:
                        _send_sse_event('agent_log', agent_name, level='info', message=f"{label}: {val}")
                # 决策逻辑
                logic = reasoning_data.get('决策逻辑', reasoning_data.get('decision_logic', reasoning_data.get('决策预览', '')))
                if logic:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"决策逻辑: {logic}")

        # 预期差
        elif 'expectation' in agent_name_lower or 'diff' in agent_name_lower or '预期' in agent_name:
            if '预期' in agent_name or '预期差' in agent_name:
                signal = reasoning_data.get('最终信号', reasoning_data.get('signal', '-'))
                confidence = reasoning_data.get('置信度', reasoning_data.get('confidence', '-'))
                earnings_signal = reasoning_data.get('盈利预测', '-')
                diff_signal = reasoning_data.get('预期差', '-')
                signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
                msg = f"预期差分析：信号:{signal_cn} 置信度:{confidence} 盈利预测:{earnings_signal} 预期差:{diff_signal}"
                _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 宏观分析
        elif 'macro' in agent_name_lower and 'analyst' in agent_name_lower:
            env = reasoning_data.get('macro_environment', '-')
            impact = reasoning_data.get('impact_on_stock', '-')
            signal = reasoning_data.get('signal', '-')
            confidence = reasoning_data.get('confidence', '-')
            signal_cn = {'bullish': '看涨', 'bearish': '看跌', 'neutral': '中性'}.get(signal, signal)
            env_cn = {'favorable': '有利', 'unfavorable': '不利', 'neutral': '中性'}.get(env, env)
            impact_cn = {'positive': '正面', 'negative': '负面', 'neutral': '中性'}.get(impact, impact)
            msg = f"宏观环境:{env_cn} 对股票影响:{impact_cn} 信号:{signal_cn} 置信度:{confidence}"
            _send_sse_event('agent_log', agent_name, level='info', message=msg)

            # 发送关键因素
            key_factors = reasoning_data.get('key_factors', [])
            if key_factors:
                factors_msg = "关键因素:" + " | ".join(str(f)[:30] for f in key_factors[:5])
                _send_sse_event('agent_log', agent_name, level='info', message=factors_msg)

            # 发送决策逻辑
            decision_logic = reasoning_data.get('decision_logic', '')
            if decision_logic:
                logic_lines = str(decision_logic).split('\n')
                for line in logic_lines[:6]:  # 限制行数避免过长
                    if line.strip():
                        _send_sse_event('agent_log', agent_name, level='info', message=f"决策逻辑:{line.strip()[:100]}")

            # 发送决策结果
            decision_result = reasoning_data.get('decision_result', {})
            if decision_result:
                action = decision_result.get('action', '')
                risk_warning = decision_result.get('risk_warning', '')
                recommendation = decision_result.get('recommendation', '')
                if recommendation:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"决策建议:{recommendation}")
                if action:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"操作建议:{action}")
                if risk_warning:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"风险提示:{risk_warning}")

        # 宏观新闻
        elif 'macro' in agent_name_lower and 'news' in agent_name_lower:
            reasoning = reasoning_data.get('reasoning', reasoning_data.get('summary', '-'))
            if isinstance(reasoning, str) and len(reasoning) > 100:
                reasoning = reasoning[:100] + "..."
            msg = f"宏观新闻摘要:{reasoning}"
            _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 市场数据
        elif 'market' in agent_name_lower and 'data' in agent_name_lower:
            prices = reasoning_data.get('prices', [])
            industry = reasoning_data.get('industry', '-')
            financial_metrics = reasoning_data.get('financial_metrics', {})
            has_metrics = bool(financial_metrics and any(v not in (None, 0, "", [], {}) for v in financial_metrics.values()) if isinstance(financial_metrics, dict) else False)
            financial_statements = reasoning_data.get('financial_line_items', {})
            has_statements = bool(financial_statements and len(financial_statements) > 0)
            price_count = len(prices) if isinstance(prices, list) else '-'
            msg = f"价格:{price_count}条 行业:{industry} 财务指标:{'✓' if has_metrics else '✗'} 报表:{'✓' if has_statements else '✗'}"
            _send_sse_event('agent_log', agent_name, level='info', message=msg)

        # 通用回退：对未匹配特定处理器的数据，发送关键字段作为agent_log
        else:
            for key, value in reasoning_data.items():
                if isinstance(value, dict):
                    _send_sse_event('agent_log', agent_name, level='info',
                                    message=f"【{key}】{json.dumps(value, ensure_ascii=False)[:200]}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        _send_sse_event('agent_log', agent_name, level='info',
                                        message=f"【{key}】共{len(value)}项: {str(value[0])[:100]}")
                elif isinstance(value, str) and len(value) > 150:
                    _send_sse_event('agent_log', agent_name, level='info',
                                    message=f"【{key}】{value[:150]}...")
                elif value is not None:
                    _send_sse_event('agent_log', agent_name, level='info', message=f"【{key}】{value}")

    except Exception as e:
        logger.debug(f"发送关键推理日志失败: {e}")
