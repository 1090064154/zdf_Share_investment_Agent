"""日志钩子模块 - 在Agent执行时发送SSE事件"""
import sys
from datetime import datetime
from typing import Dict, Any

# 全局SSE管理器引用
_sse_manager = None

def set_sse_manager(sse_manager):
    """设置SSE管理器"""
    global _sse_manager
    _sse_manager = sse_manager

def send_agent_event(run_id: str, event_type: str, agent_name: str, **kwargs):
    """发送Agent事件到SSE"""
    if not _sse_manager:
        return
    
    event = {
        "type": event_type,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    try:
        _sse_manager.broadcast_sync(run_id, event)
    except Exception as e:
        print(f"发送SSE事件失败: {e}", file=sys.stderr)

def agent_started(run_id: str, agent_name: str, message: str = ""):
    """通知Agent开始执行"""
    send_agent_event(run_id, "agent_start", agent_name, message=message)

def agent_log(run_id: str, agent_name: str, level: str, message: str):
    """发送Agent日志"""
    send_agent_event(run_id, "agent_log", agent_name, level=level, message=message)

def agent_completed(run_id: str, agent_name: str, signal: str = None, confidence: float = 0.0, message: str = "", **kwargs):
    """通知Agent完成"""
    send_agent_event(run_id, "agent_complete", agent_name, 
                    status="success", signal=signal, confidence=confidence, message=message, **kwargs)

def agent_failed(run_id: str, agent_name: str, error: str):
    """通知Agent失败"""
    send_agent_event(run_id, "agent_complete", agent_name, status="failed", error=error)

def task_complete(run_id: str, result: Dict[str, Any]):
    """通知任务完成"""
    send_agent_event(run_id, "task_complete", "system", 
                    status="completed", result=result,
                    action=result.get("action"),
                    confidence=result.get("confidence", 0.0))

def task_error(run_id: str, error: str):
    """通知任务失败"""
    send_agent_event(run_id, "task_error", "system", error=error)
