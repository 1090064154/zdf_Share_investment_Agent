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

def _send_sse_event(event_type: str, agent_name: str, status: str = None, message: str = None, data: dict = None):
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


def show_agent_reasoning(output, agent_name):
    """Display agent's analysis results."""
    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    # logger.info(f"{'='*20} {agent_name} Analysis Details {'='*20}")

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        logger.info(json.dumps(serializable_output, indent=2, ensure_ascii=False))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)
