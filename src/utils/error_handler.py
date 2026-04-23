"""
统一错误处理和容错机制

提供resilient_agent装饰器，为所有Agent添加统一的异常边界处理，
确保单个Agent失败不会导致整个workflow中断。
"""

import json
import traceback
from functools import wraps
from typing import Any, Callable, Dict
from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger

logger = setup_logger('error_handler')


def create_fallback_result(agent_name: str, error: Exception) -> Dict[str, Any]:
    """
    创建标准化的fallback结果
    
    Args:
        agent_name: Agent名称
        error: 捕获的异常
        
    Returns:
        包含错误信息的标准化结果字典
    """
    error_msg = f"{type(error).__name__}: {str(error)}"
    logger.error(f"Agent {agent_name} 执行失败: {error_msg}")
    logger.debug(traceback.format_exc())
    
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "error": error_msg,
        "fallback": True,
        "reasoning": f"{agent_name} 执行失败，使用保守的中性信号。错误: {error_msg}"
    }


def create_fallback_message(agent_name: str, error: Exception) -> HumanMessage:
    """
    创建fallback消息对象
    
    Args:
        agent_name: Agent名称
        error: 捕获的异常
        
    Returns:
        HumanMessage对象，包含错误信息
    """
    fallback_result = create_fallback_result(agent_name, error)
    return HumanMessage(
        content=json.dumps(fallback_result, ensure_ascii=False),
        name=agent_name
    )


def resilient_agent(agent_func: Callable = None, *, critical: bool = False):
    """
    Agent容错装饰器
    
    为Agent函数添加统一的异常处理，确保：
    1. 单个Agent失败不会中断整个workflow
    2. 返回标准化的fallback结果
    3. 记录详细的错误日志
    
    Args:
        agent_func: 被装饰的Agent函数
        critical: 是否为关键Agent。如果为True，失败时会重新抛出异常
        
    Usage:
        @resilient_agent
        def technical_analyst_agent(state: AgentState):
            ...
        
        @resilient_agent(critical=True)
        def portfolio_management_agent(state: AgentState):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent_name = func.__name__
            try:
                # 正常执行Agent
                result = func(*args, **kwargs)
                
                # 验证返回结果的基本结构
                if result is None:
                    raise ValueError(f"Agent {agent_name} 返回了None")
                
                if not isinstance(result, dict):
                    raise TypeError(f"Agent {agent_name} 返回类型错误: {type(result)}")
                
                if "messages" not in result:
                    raise KeyError(f"Agent {agent_name} 返回结果缺少'messages'字段")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Agent {agent_name} 执行异常: {e}", exc_info=True)
                
                # 关键Agent失败时重新抛出异常
                if critical:
                    raise
                
                # 非关键Agent返回fallback结果
                state = args[0] if args else kwargs.get('state', {})
                
                # 构造fallback返回
                fallback_message = create_fallback_message(agent_name, e)
                
                return {
                    "messages": state.get("messages", []) + [fallback_message],
                    "data": state.get("data", {}),
                    "metadata": {
                        **state.get("metadata", {}),
                        f"{agent_name}_error": str(e),
                        f"{agent_name}_fallback": True
                    }
                }
        
        # 保留原始函数的属性
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._is_resilient = True
        wrapper._is_critical = critical
        
        return wrapper
    
    # 支持无参数调用 @resilient_agent
    if agent_func is not None:
        return decorator(agent_func)
    
    return decorator


def validate_agent_input(state: Dict[str, Any], required_fields: list) -> bool:
    """
    验证Agent输入是否包含必需字段
    
    Args:
        state: Agent状态
        required_fields: 必需的字段列表
        
    Returns:
        验证是否通过
    """
    missing_fields = []
    for field in required_fields:
        if field not in state.get("data", {}):
            missing_fields.append(field)
    
    if missing_fields:
        logger.warning(f"输入数据缺少必需字段: {missing_fields}")
        return False
    
    return True


def safe_json_parse(content: str, default: dict = None) -> dict:
    """
    安全的JSON解析，失败时返回默认值
    
    Args:
        content: JSON字符串
        default: 解析失败时的默认值
        
    Returns:
        解析后的字典或默认值
    """
    if default is None:
        default = {}
    
    if not content:
        return default
    
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON解析失败: {e}, 内容: {content[:100]}")
        return default


class ErrorHandler:
    """
    错误处理器类
    
    提供更细粒度的错误处理策略
    """
    
    def __init__(self, agent_name: str, max_retries: int = 0):
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.retry_count = 0
    
    def handle_error(self, error: Exception, context: dict = None) -> dict:
        """
        处理错误并返回恢复策略
        
        Args:
            error: 捕获的异常
            context: 错误上下文信息
            
        Returns:
            恢复策略字典
        """
        self.retry_count += 1
        
        error_info = {
            "agent": self.agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "can_retry": self.retry_count < self.max_retries,
            "context": context or {}
        }
        
        logger.error(f"ErrorHandler [{self.agent_name}]: {error_info}")
        
        return error_info
    
    def should_retry(self) -> bool:
        """判断是否应该重试"""
        return self.retry_count < self.max_retries
    
    def reset(self):
        """重置重试计数器"""
        self.retry_count = 0
