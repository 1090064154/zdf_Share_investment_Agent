"""异步任务管理器"""
import asyncio
import logging
import uuid
from typing import Dict, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .sse_manager import sse_manager, SSEEvent

logger = logging.getLogger("task_manager")

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentState:
    """Agent状态"""
    name: str
    status: AgentStatus = AgentStatus.PENDING
    signal: Optional[str] = None  # bullish, bearish, neutral
    confidence: float = 0.0
    message: str = ""
    logs: list = field(default_factory=list)

@dataclass
class TaskState:
    """任务状态"""
    id: str
    ticker: str
    status: TaskStatus = TaskStatus.PENDING
    agents: Dict[str, AgentState] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class TaskManager:
    """异步任务管理器"""

    # 9个基础Agent
    BASE_AGENTS = [
        "technical_analyst_agent",
        "fundamentals_agent",
        "sentiment_agent",
        "valuation_agent",
        "industry_cycle_agent",
        "institutional_agent",
        "expectation_diff_agent",
        "macro_news_agent",
        "macro_analyst_agent",
    ]

    # Level 2 综合Agent
    RESEARCHER_AGENTS = ["researcher_bull_agent", "researcher_bear_agent"]

    # Level 3 辩论Agent
    DEBATE_AGENTS = ["debate_room_agent"]

    # Level 4 最终决策Agent
    FINAL_AGENTS = ["risk_management_agent", "portfolio_management_agent"]

    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self._sse = sse_manager

    def generate_run_id(self) -> str:
        """生成唯一运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}-{short_uuid}"

    async def create_task(
        self,
        ticker: str,
        params: Dict[str, Any]
    ) -> str:
        """创建新任务"""
        run_id = self.generate_run_id()

        # 初始化所有Agent状态
        agents = {}
        for name in self.BASE_AGENTS + self.RESEARCHER_AGENTS + self.DEBATE_AGENTS + self.FINAL_AGENTS:
            agents[name] = AgentState(name=name)

        task = TaskState(
            id=run_id,
            ticker=ticker,
            params=params,
            agents=agents
        )
        self.tasks[run_id] = task

        logger.info(f"创建任务 {run_id}，股票 {ticker}")
        return run_id

    async def start_task(self, run_id: str):
        """开始任务"""
        task = self.tasks.get(run_id)
        if not task:
            raise ValueError(f"任务 {run_id} 不存在")

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        await self._broadcast(run_id, {
            "type": SSEEvent.SYSTEM_STATUS,
            "agent": "system",
            "status": "running",
            "message": f"任务开始执行，股票 {task.ticker}"
        })

    async def agent_started(self, run_id: str, agent_name: str, message: str = ""):
        """通知Agent开始执行"""
        task = self.tasks.get(run_id)
        if not task:
            return

        if agent_name in task.agents:
            task.agents[agent_name].status = AgentStatus.RUNNING
            task.agents[agent_name].message = message or f"开始执行 {agent_name}"

        await self._broadcast(run_id, {
            "type": SSEEvent.AGENT_START,
            "agent": agent_name,
            "message": message or f"开始执行 {agent_name}"
        })

    async def agent_log(
        self,
        run_id: str,
        agent_name: str,
        level: str,
        message: str
    ):
        """添加Agent日志"""
        task = self.tasks.get(run_id)
        if not task:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }

        if agent_name in task.agents:
            task.agents[agent_name].logs.append(log_entry)

        await self._broadcast(run_id, {
            "type": SSEEvent.AGENT_LOG,
            "agent": agent_name,
            "level": level,
            "message": message,
            "timestamp": log_entry["timestamp"]
        })

    async def agent_completed(
        self,
        run_id: str,
        agent_name: str,
        signal: Optional[str] = None,
        confidence: float = 0.0,
        message: str = ""
    ):
        """通知Agent完成"""
        task = self.tasks.get(run_id)
        if not task:
            return

        if agent_name in task.agents:
            task.agents[agent_name].status = AgentStatus.COMPLETED
            task.agents[agent_name].signal = signal
            task.agents[agent_name].confidence = confidence
            task.agents[agent_name].message = message

        await self._broadcast(run_id, {
            "type": SSEEvent.AGENT_COMPLETE,
            "agent": agent_name,
            "status": "success",
            "signal": signal,
            "confidence": confidence,
            "message": message
        })

    async def agent_failed(
        self,
        run_id: str,
        agent_name: str,
        error: str
    ):
        """通知Agent失败"""
        task = self.tasks.get(run_id)
        if not task:
            return

        if agent_name in task.agents:
            task.agents[agent_name].status = AgentStatus.FAILED
            task.agents[agent_name].message = error

        await self._broadcast(run_id, {
            "type": SSEEvent.AGENT_COMPLETE,
            "agent": agent_name,
            "status": "failed",
            "error": error
        })

    async def task_completed(self, run_id: str, result: Dict):
        """任务完成"""
        task = self.tasks.get(run_id)
        if not task:
            return

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result

        final_action = result.get("action", "unknown")
        confidence = result.get("confidence", 0.0)

        await self._broadcast(run_id, {
            "type": SSEEvent.TASK_COMPLETE,
            "run_id": run_id,
            "status": "completed",
            "action": final_action,
            "confidence": confidence,
            "result": result
        })

    async def task_failed(self, run_id: str, error: str):
        """任务失败"""
        task = self.tasks.get(run_id)
        if not task:
            return

        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error = error

        await self._broadcast(run_id, {
            "type": SSEEvent.TASK_ERROR,
            "run_id": run_id,
            "status": "failed",
            "error": error
        })

    async def get_task_state(self, run_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self.tasks.get(run_id)
        if not task:
            return None

        return {
            "id": task.id,
            "ticker": task.ticker,
            "status": task.status.value,
            "agents": {
                name: {
                    "status": agent.status.value,
                    "signal": agent.signal,
                    "confidence": agent.confidence,
                    "message": agent.message,
                    "log_count": len(agent.logs)
                }
                for name, agent in task.agents.items()
            },
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result
        }

    async def _broadcast(self, run_id: str, event: dict):
        """广播事件"""
        await self._sse.broadcast(run_id, event)

# 全局任务管理器实例
task_manager = TaskManager()