"""API路由模块"""
import asyncio
import json
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .task_manager import task_manager, TaskStatus
from .sse_manager import sse_manager
from .file_storage import file_storage
from .log_hook import set_sse_manager

logger = logging.getLogger("api_routes")

router = APIRouter(prefix="/api", tags=["API"])

# 初始化日志钩子的SSE管理器
set_sse_manager(sse_manager)

# ==================== 任务管理 ====================

@router.post("/run")
async def create_run(request: dict):
    """创建新任务

    POST /api/run
    {
        "ticker": "002714",
        "investment_horizon": "medium",
        "initial_capital": 100000,
        "initial_position": 0,
        "num_of_news": 5,
        "show_reasoning": true,
        "start_date": "2025-01-01",
        "end_date": "2025-04-20"
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
            "show_reasoning": request.get("show_reasoning", True),
            "investment_horizon": request.get("investment_horizon", "medium"),
            "start_date": request.get("start_date"),
            "end_date": request.get("end_date"),
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
    import json
    from pathlib import Path
    
    print(f">>> get_run 被调用: {run_id}", flush=True)
    
    # 检查任务管理器
    state = await task_manager.get_task_state(run_id)
    print(f">>> task_manager state: {state}", flush=True)
    
    # 检查文件存储中的结果
    run_dir = Path("src/data/runs") / run_id
    result_file = run_dir / "result.json"
    index_file = Path("src/data/runs") / "index.json"
    
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
            print(f">>> 从文件读取到结果: {result.get('action')}", flush=True)
            if state:
                state['result'] = result
            else:
                state = {
                    "id": run_id,
                    "status": "completed",
                    "result": result
                }
        except Exception as e:
            print(f">>> 读取结果文件失败: {e}", flush=True)
    
    # 检查索引文件状态
    if index_file.exists():
        try:
            data = json.loads(index_file.read_text())
            for run in data.get("runs", []):
                if run.get("run_id") == run_id:
                    print(f">>> 索引文件状态: {run.get('status')}", flush=True)
                    if not state:
                        state = {"id": run_id, "status": run.get("status")}
                    break
        except Exception as e:
            print(f">>> 读取索引失败: {e}", flush=True)
    
    if not state:
        print(f">>> 找不到任务 {run_id}", flush=True)
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return state


@router.post("/run/{run_id}/start")
async def start_run(run_id: str):
    """开始执行任务"""
    print(f">>> start_run 被调用: {run_id}", flush=True)
    
    import logging
    logger = logging.getLogger("api_routes")
    logger.info(f"start_run called: {run_id}")
    
    task = task_manager.tasks.get(run_id)
    if not task:
        print(f">>> 任务 {run_id} 不存在", flush=True)
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if task.status != TaskStatus.PENDING:
        print(f">>> 任务 {run_id} 状态错误: {task.status}", flush=True)
        raise HTTPException(status_code=400, detail=f"Run {run_id} is not pending")

    print(f">>> 准备启动任务 {run_id}", flush=True)
    
    # 等待1秒让SSE连接建立
    print(f">>> 等待SSE连接建立...", flush=True)
    await asyncio.sleep(1)
    
    # 在后台启动任务
    task_handle = asyncio.create_task(_execute_workflow(run_id))
    print(f">>> async task created: {task_handle}", flush=True)

    return {"run_id": run_id, "status": "starting"}


async def _execute_workflow(run_id: str):
    """执行工作流（在后台运行）"""
    import logging
    import asyncio
    from backend.services.analysis import execute_stock_analysis
    from backend.models.api_models import StockAnalysisRequest

    print(f">>> _execute_workflow 开始: {run_id}", flush=True)

    # 确保同步队列存在（在 SSE 连接之前就开始收集事件）
    if run_id not in sse_manager._sync_queues:
        sync_queue = sse_manager.create_sync_queue(run_id)
        print(f">>> 创建同步队列: {run_id}", flush=True)
    
    task = task_manager.tasks.get(run_id)
    if not task:
        print(f">>> 任务 {run_id} 不存在", flush=True)
        return
        
    params = task.params
    print(f">>> ticker={task.ticker}", flush=True)

    # 构建请求
    request = StockAnalysisRequest(
        ticker=task.ticker,
        initial_capital=params.get("initial_capital", 100000),
        initial_position=params.get("initial_position", 0),
        show_reasoning=params.get("show_reasoning", True),
        num_of_news=params.get("num_of_news", 5),
        investment_horizon=params.get("investment_horizon", "medium"),
        start_date=params.get("start_date"),
        end_date=params.get("end_date")
    )

    file_storage.update_run(run_id, status="running")

    # 设置当前 run_id 用于 SSE 事件发送
    from src.agents.state import set_current_run_id
    set_current_run_id(run_id)
    print(f">>> 已设置 run_id: {run_id}", flush=True)

    try:
        # 发送系统开始事件
        print(f">>> 发送SSE开始事件", flush=True)
        sse_manager.broadcast_sync(run_id, {
            "type": "system_status",
            "agent": "system",
            "status": "running",
            "message": f"开始分析股票 {task.ticker}"
        })
        print(f">>> SSE开始事件已发送", flush=True)

        # 在线程中执行同步分析，但不能阻塞 FastAPI 事件循环；
        # 否则 SSE 无法持续推送，前端会一直卡在“连接中”。
        print(f">>> 开始执行 execute_stock_analysis", flush=True)
        result = await asyncio.to_thread(execute_stock_analysis, request, run_id)
        
        # 解析结果 - 工作流返回的是字符串格式的JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                print(f">>> 解析结果失败: {e}", flush=True)
                result = {"action": "hold", "quantity": 0, "confidence": 0.0, "reasoning": "解析结果失败"}
        
        print(f">>> execute_stock_analysis 完成, action={result.get('action')}", flush=True)

        # 发送任务完成事件
        print(f">>> 发送 task_complete 事件", flush=True)
        sse_manager.broadcast_sync(run_id, {
            "type": "task_complete",
            "run_id": run_id,
            "status": "completed",
            "action": result.get("action", "hold"),
            "confidence": result.get("confidence", 0.0),
            "result": result
        })
        print(f">>> task_complete 已发送", flush=True)

        await task_manager.task_completed(run_id, result)

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
        logger.info(f"工作流 {run_id} 完成")

    except Exception as e:
        logger.error(f"任务 {run_id} 执行失败: {e}")
        import traceback
        traceback.print_exc()
        await task_manager.task_failed(run_id, str(e))
        file_storage.update_run(run_id, status="failed")
    finally:
        # 给前端一点时间接收最后一条事件，再释放队列资源。
        await asyncio.sleep(5)
        sse_manager.cleanup_run(run_id)


@router.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE实时日志流"""
    # 订阅事件；同步线程事件会由 sse_manager 内部泵送到订阅者
    event_generator, client_id = await sse_manager.subscribe(run_id)

    async def sse_response():
        try:
            # 先发送初始消息
            yield f"data: {json.dumps({'type': 'system_status', 'message': '已连接，等待数据...'})}\n\n"

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
