from abc import ABC, abstractmethod
from typing import List, Optional, Set

from backend.schemas import LLMInteractionLog, AgentExecutionLog


class BaseLogStorage(ABC):
    """Abstract base class for LLM interaction log storage."""

    @abstractmethod
    def add_log(self, log: LLMInteractionLog) -> None:
        """Adds a new log entry to the storage."""
        pass

    @abstractmethod
    def get_logs(
        self,
        agent_name: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LLMInteractionLog]:
        """Retrieves logs, optionally filtering by agent name or run ID."""
        pass

    @abstractmethod
    def add_agent_log(self, log: AgentExecutionLog) -> None:
        """添加Agent执行日志"""
        pass

    @abstractmethod
    def get_agent_logs(
        self,
        agent_name: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AgentExecutionLog]:
        """获取Agent执行日志，可按agent名称或run ID过滤"""
        pass

    @abstractmethod
    def get_unique_run_ids(self) -> List[str]:
        """获取所有唯一的运行ID列表"""
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """删除指定 run_id 的所有日志"""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """清空所有日志"""
        pass
