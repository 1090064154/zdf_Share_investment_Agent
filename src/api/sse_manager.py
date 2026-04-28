"""SSE (Server-Sent Events) 连接管理器"""
import asyncio
import json
import logging
from typing import Dict, AsyncGenerator
from datetime import datetime
from queue import Empty, Queue as SyncQueue

logger = logging.getLogger("sse_manager")

class SSEEvent:
    SYSTEM_STATUS = "system_status"
    AGENT_START = "agent_start"
    AGENT_LOG = "agent_log"
    AGENT_COMPLETE = "agent_complete"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"

class SSEManager:
    """Server-Sent Events 管理器 - 线程安全版本"""

    def __init__(self):
        self._connections: Dict[tuple, asyncio.Queue] = {}
        self._sync_queues: Dict[str, SyncQueue] = {}
        self._pump_tasks: Dict[str, asyncio.Task] = {}
        self._client_counter = 0
        self._loop = None

    def set_loop(self, loop):
        """设置事件循环（由主线程调用）"""
        self._loop = loop

    def _generate_client_id(self) -> str:
        self._client_counter += 1
        return f"client_{self._client_counter}"

    def create_sync_queue(self, run_id: str) -> SyncQueue:
        """创建同步队列（供工作线程使用）"""
        if run_id in self._sync_queues:
            self._ensure_pump_task(run_id)
            return self._sync_queues[run_id]
        q = SyncQueue()
        self._sync_queues[run_id] = q
        self._ensure_pump_task(run_id)
        return q

    def _ensure_pump_task(self, run_id: str):
        """确保每个任务只有一个后台泵送协程。"""
        if not self._loop:
            return

        task = self._pump_tasks.get(run_id)
        if task and not task.done():
            return

        def _start():
            existing = self._pump_tasks.get(run_id)
            if existing and not existing.done():
                return
            self._pump_tasks[run_id] = asyncio.create_task(self.pump_sync_queue(run_id))

        self._loop.call_soon_threadsafe(_start)

    async def subscribe(self, run_id: str) -> tuple[AsyncGenerator, str]:
        """订阅任务更新"""
        self.set_loop(asyncio.get_running_loop())
        self.create_sync_queue(run_id)

        queue = asyncio.Queue()
        client_id = self._generate_client_id()
        self._connections[(run_id, client_id)] = queue
        logger.info(f"客户端 {client_id} 订阅了任务 {run_id}")

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30)
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
            except asyncio.CancelledError:
                logger.info(f"客户端 {client_id} 断开连接")
            finally:
                self._disconnect(run_id, client_id)

        return event_generator(), client_id

    def _disconnect(self, run_id: str, client_id: str):
        key = (run_id, client_id)
        if key in self._connections:
            del self._connections[key]

    async def broadcast(self, run_id: str, event: dict):
        """异步广播（主线程中使用）"""
        disconnected = []
        for key, queue in self._connections.items():
            if key[0] == run_id:
                try:
                    await queue.put(event)
                except Exception as e:
                    logger.error(f"广播失败: {e}")
                    disconnected.append(key)
        for key in disconnected:
            if key in self._connections:
                del self._connections[key]

    def broadcast_sync(self, run_id: str, event: dict):
        """同步广播（工作线程中使用，自动转递到主事件循环）"""
        queue = self.create_sync_queue(run_id)
        queue.put(event)

    async def pump_sync_queue(self, run_id: str):
        """将同步队列中的事件泵送到异步队列"""
        if run_id not in self._sync_queues:
            return
        sync_q = self._sync_queues[run_id]
        while True:
            try:
                event = await asyncio.to_thread(sync_q.get, True, 0.5)
                await self.broadcast(run_id, event)
            except Empty:
                if run_id not in self._sync_queues:
                    break
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SSE pump failed for run_id=%s", run_id)
                await asyncio.sleep(0.1)

    def cleanup_run(self, run_id: str):
        """清理任务相关的 SSE 资源。"""
        self._sync_queues.pop(run_id, None)
        task = self._pump_tasks.pop(run_id, None)
        if task and not task.done():
            task.cancel()
        stale_connections = [key for key in self._connections if key[0] == run_id]
        for key in stale_connections:
            self._connections.pop(key, None)

sse_manager = SSEManager()
