"""API路由模块"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .task_manager import task_manager, TaskStatus
from .sse_manager import sse_manager
from .file_storage import file_storage

logger = logging.getLogger("api_routes")

router = APIRouter(prefix="/api", tags=["API"])

# ==================== 任务管理 ====================

@router.post("/run")
async def create_run(request: dict):
    """创建新任务

    POST /api/run
    {
        "ticker": "002714",
        "initial_capital": 100000,
        "initial_position": 0,
        "num_of_news": 5,
        "show_reasoning": true
    }
    """
    ticker = request.get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    # 创建任务
    run_id = await task_manager.create_task(
        ticker=ticker,
        params={
            "initial_capital": request.get("initial_capital", 100000),
            "initial_position": request.get("initial_position", 0),
            "num_of_news": request.get("num_of_news", 5),
            "show_reasoning": request.get("show_reasoning", True)
        }
    )

    # 保存到文件存储
    file_storage.add_run(run_id, ticker, request)

    return {
        "run_id": run_id,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }


@router.get("/run/{run_id}")
async def get_run(run_id: str):
    """获取任务状态"""
    state = await task_manager.get_task_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return state


@router.post("/run/{run_id}/start")
async def start_run(run_id: str):
    """开始执行任务"""
    task = task_manager.tasks.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if task.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Run {run_id} is not pending")

    # 在后台启动任务
    asyncio.create_task(_execute_workflow(run_id))

    return {"run_id": run_id, "status": "starting"}


async def _execute_workflow(run_id: str):
    """执行工作流（在后台运行）"""
    from backend.services.analysis import execute_stock_analysis
    from backend.models.api_models import StockAnalysisRequest

    task = task_manager.tasks[run_id]
    params = task.params

    # 构建请求
    request = StockAnalysisRequest(
        ticker=task.ticker,
        initial_capital=params.get("initial_capital", 100000),
        initial_position=params.get("initial_position", 0),
        show_reasoning=params.get("show_reasoning", True),
        num_of_news=params.get("num_of_news", 5)
    )

    # 更新文件存储状态
    file_storage.update_run(run_id, status="running")

    try:
        await task_manager.start_task(run_id)

        # 执行分析（这会调用现有的LangGraph工作流）
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: execute_stock_analysis(request, run_id)
        )

        await task_manager.task_completed(run_id, result)

        # 更新文件存储
        duration = None
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()

        file_storage.update_run(
            run_id,
            status="completed",
            action=result.get("action"),
            confidence=result.get("confidence", 0.0),
            completed_at=datetime.now().isoformat(),
            duration_seconds=duration
        )
        file_storage.save_result(run_id, result)

    except Exception as e:
        logger.error(f"任务 {run_id} 执行失败: {e}")
        await task_manager.task_failed(run_id, str(e))
        file_storage.update_run(run_id, status="failed")


@router.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE实时日志流

    GET /api/run/{run_id}/stream

    Event Types:
    - system_status: 系统状态更新
    - agent_start: Agent开始执行
    - agent_log: Agent日志
    - agent_complete: Agent完成
    - task_complete: 任务完成
    - task_error: 任务错误
    - heartbeat: 心跳保活
    """
    # 验证任务存在
    task = task_manager.tasks.get(run_id)
    if not task:
        # 返回404事件
        async def not_found_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Run not found\"}}\n\n"

        return StreamingResponse(
            not_found_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    # 订阅SSE事件
    event_generator, client_id = await sse_manager.subscribe(run_id)

    async def sse_response():
        try:
            async for event in event_generator:
                yield event
        except asyncio.CancelledError:
            logger.info(f"SSE连接取消: {run_id}/{client_id}")

    return StreamingResponse(
        sse_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/runs")
async def list_runs(page: int = 1, limit: int = 20):
    """获取历史任务列表

    GET /api/runs?page=1&limit=20
    """
    return file_storage.get_all_runs(page=page, limit=limit)


@router.get("/run/{run_id}/logs/{agent_name}")
async def get_agent_logs(run_id: str, agent_name: str):
    """获取Agent日志"""
    logs = file_storage.get_agent_log(run_id, agent_name)
    return {"run_id": run_id, "agent": agent_name, "logs": logs}


@router.get("/run/{run_id}/result")
async def get_run_result(run_id: str):
    """获取任务结果"""
    result = file_storage.get_result(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result for {run_id} not found")
    return result


@router.post("/run/{run_id}/cancel")
async def cancel_run(run_id: str):
    """取消任务（暂不支持）"""
    raise HTTPException(status_code=501, detail="Cancellation not yet supported")