"""
运行历史相关路由模块

此模块提供与工作流运行历史相关的API端点
"""

from fastapi import APIRouter, Query
from pathlib import Path
from typing import Dict, List

from ..models.api_models import ApiResponse, RunInfo
from ..state import api_state

# 创建路由器
router = APIRouter(prefix="/api/runs", tags=["Runs"])


@router.get("/", response_model=List[RunInfo])
async def list_runs(limit: int = Query(10, ge=1, le=100)):
    """获取运行历史列表 (基于内存状态)

    此接口查询内存中的 api_state 对象，返回其中记录的运行摘要信息。
    适合查看近期运行或正在进行的运行。
    注意：内存状态在服务重启后会丢失。
    """
    runs = api_state.get_all_runs()
    # 按开始时间倒序排序，并限制数量
    runs.sort(key=lambda x: x.start_time, reverse=True)
    return runs[:limit]


@router.get("/{run_id}", response_model=ApiResponse[RunInfo])
async def get_run_info(run_id: str):
    """获取指定运行的信息 (基于内存状态)

    此接口查询内存中的 api_state 对象，返回特定 run_id 的运行摘要信息。
    注意：内存状态在服务重启后会丢失。
    """
    run = api_state.get_run(run_id)
    if not run:
        return ApiResponse(
            success=False,
            message=f"运行 '{run_id}' 不存在",
            data=None
        )
    return ApiResponse(data=run)


@router.delete("/{run_id}")
async def delete_run(run_id: str):
    """删除指定运行记录"""
    from src.api.file_storage import file_storage
    api_state.delete_run(run_id)
    try:
        runs_file = Path("src/data/runs/index.json")
        if runs_file.exists():
            import json
            with open(runs_file, 'r') as f:
                data = json.load(f)
            data['runs'] = [r for r in data.get('runs', []) if r['run_id'] != run_id]
            with open(runs_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return ApiResponse(success=True, message=f"已删除运行 {run_id}")


@router.delete("/")
async def clear_all_runs():
    """清空所有运行记录"""
    api_state.clear_all()
    try:
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        runs_file = Path(base_dir) / "src" / "data" / "runs" / "index.json"
        if runs_file.exists():
            import json
            with open(runs_file, 'w') as f:
                json.dump({"runs": []}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Clear error: {e}")
    return ApiResponse(success=True, message="已清空所有运行记录")
