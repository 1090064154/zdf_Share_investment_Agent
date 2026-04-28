"""SSE (Server-Sent Events) 连接管理器"""
import asyncio
import json
import logging
from typing import Dict, AsyncGenerator, Optional
from datetime import datetime

logger = logging.getLogger("sse_manager")

class SSEEvent:
    """SSE事件类型定义"""
    SYSTEM_STATUS = "system_status"       # 系统状态更新
    AGENT_START = "agent_start"           # Agent开始执行
    AGENT_LOG = "agent_log"               # Agent日志
    AGENT_COMPLETE = "agent_complete"     # Agent完成
    TASK_COMPLETE = "task_complete"      # 任务完成
    TASK_ERROR = "task_error"             # 任务错误

class SSEManager:
    """Server-Sent Events 管理器"""

    def __init__(self):
        # {(run_id, client_id): Queue}
        self._connections: Dict[tuple, asyncio.Queue] = {}
        self._client_counter = 0

    def _generate_client_id(self) -> str:
        """生成唯一客户端ID"""
        self._client_counter += 1
        return f"client_{self._client_counter}"

    async def subscribe(self, run_id: str) -> tuple[AsyncGenerator, str]:
        """订阅任务更新

        Returns:
            AsyncGenerator: 异步生成器，用于生成SSE事件流
            str: 客户端ID，用于取消订阅
        """
        queue = asyncio.Queue()
        client_id = self._generate_client_id()
        self._connections[(run_id, client_id)] = queue
        logger.info(f"客户端 {client_id} 订阅了任务 {run_id}")

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    try:
                        # 等待事件，超时后发送心跳
                        event = await asyncio.wait_for(queue.get(), timeout=30)
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        # 发送心跳
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
            except asyncio.CancelledError:
                logger.info(f"客户端 {client_id} 断开连接")
            finally:
                self._disconnect(run_id, client_id)

        return event_generator(), client_id

    def _disconnect(self, run_id: str, client_id: str):
        """断开连接"""
        key = (run_id, client_id)
        if key in self._connections:
            del self._connections[key]
            logger.info(f"客户端 {client_id} 已断开")

    async def broadcast(self, run_id: str, event: dict):
        """广播事件到所有订阅者"""
        disconnected = []
        for key, queue in self._connections.items():
            if key[0] == run_id:
                try:
                    await queue.put(event)
                except Exception as e:
                    logger.error(f"广播失败: {e}")
                    disconnected.append(key)

        # 清理断开的连接
        for key in disconnected:
            if key in self._connections:
                del self._connections[key]

    async def send_to_client(self, run_id: str, client_id: str, event: dict):
        """发送事件到指定客户端"""
        key = (run_id, client_id)
        if key in self._connections:
            await self._connections[key].put(event)

# 全局SSE管理器实例
sse_manager = SSEManager()