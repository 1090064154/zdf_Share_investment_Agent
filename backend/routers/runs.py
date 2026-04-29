from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path as FilePath
import json

from backend.schemas import RunSummary, AgentSummary, AgentDetail, WorkflowFlow
from backend.storage.base import BaseLogStorage
from backend.dependencies import get_log_storage

# 创建API路由
router = APIRouter(
    prefix="/runs",
    tags=["Workflow Runs"]
)


def load_run_from_filesystem(run_id: str) -> Optional[dict]:
    """从文件系统加载运行数据"""
    base_dir = FilePath(__file__).parent.parent.parent
    result_file = base_dir / "src" / "data" / "runs" / run_id / "result.json"
    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


@router.get("/", response_model=List[RunSummary])
async def list_runs(
    limit: int = Query(10, ge=1, le=100, description="要返回的最大运行数"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """获取最近运行的列表 (基于日志存储)

    此接口查询 BaseLogStorage (当前为内存实现 InMemoryLogStorage)
    中的 AgentExecutionLog 记录，返回最近完成的工作流运行摘要。
    注意：基于内存的日志在服务重启后会丢失。

    TODO: 通过依赖注入 BaseLogStorage 的不同实现 (如数据库存储)，
          可以轻松切换到底层存储，无需修改此接口代码。
    """
    try:
        # 获取所有运行ID
        run_ids = storage.get_unique_run_ids()

        # 为每个运行ID构建摘要
        results = []
        for run_id in run_ids[:limit]:  # 限制返回数量
            # 获取该运行的所有Agent日志
            agent_logs = storage.get_agent_logs(run_id=run_id)
            if not agent_logs:
                continue

            # 计算开始和结束时间
            start_time = min(log.timestamp_start for log in agent_logs)
            end_time = max(log.timestamp_end for log in agent_logs)

            # 收集执行的Agent列表
            agents = sorted(set(log.agent_name for log in agent_logs))

            # 创建摘要对象
            summary = RunSummary(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                agents_executed=agents,
                status="completed"  # 默认状态，可以根据需要确定
            )
            results.append(summary)

        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取运行列表失败: {str(e)}"
        )


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: str = Path(..., description="要获取的运行ID"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """获取特定运行的概述信息 (基于日志存储)

    查询 BaseLogStorage (当前为内存实现 InMemoryLogStorage) 中
    特定 run_id 的 AgentExecutionLog 记录，返回该运行的摘要信息。
    注意：基于内存的日志在服务重启后会丢失。

    TODO: 通过依赖注入 BaseLogStorage 的不同实现 (如数据库存储)，
          可以轻松切换到底层存储，无需修改此接口代码。
    """
    try:
        # 获取该运行的所有Agent日志
        agent_logs = storage.get_agent_logs(run_id=run_id)
        if not agent_logs:
            raise HTTPException(
                status_code=404,
                detail=f"未找到ID为 {run_id} 的运行"
            )

        # 计算开始和结束时间
        start_time = min(log.timestamp_start for log in agent_logs)
        end_time = max(log.timestamp_end for log in agent_logs)

        # 收集执行的Agent列表
        agents = sorted(set(log.agent_name for log in agent_logs))

        # 创建摘要对象
        return RunSummary(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            agents_executed=agents,
            status="completed"  # 默认状态，可以根据需要确定
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取运行信息失败: {str(e)}"
        )


@router.get("/{run_id}/agents", response_model=List[AgentSummary])
async def get_run_agents(
    run_id: str = Path(..., description="要获取Agent的运行ID"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """获取特定运行中所有Agent的执行情况 (基于日志存储)

    查询 BaseLogStorage (当前为内存实现 InMemoryLogStorage) 中
    特定 run_id 的所有 AgentExecutionLog 记录。
    注意：基于内存的日志在服务重启后会丢失。

    TODO: 通过依赖注入 BaseLogStorage 的不同实现 (如数据库存储)，
          可以轻松切换到底层存储，无需修改此接口代码。
    """
    try:
        # 获取该运行的所有Agent日志
        agent_logs = storage.get_agent_logs(run_id=run_id)
        if agent_logs:
            # 转换为AgentSummary对象
            results = []
            for log in agent_logs:
                summary = AgentSummary(
                    agent_name=log.agent_name,
                    start_time=log.timestamp_start,
                    end_time=log.timestamp_end,
                    execution_time_seconds=(
                        log.timestamp_end - log.timestamp_start).total_seconds(),
                    status="completed"
                )
                results.append(summary)
            results.sort(key=lambda x: x.start_time)
            return results

        # 回退：从文件系统读取
        fs_data = load_run_from_filesystem(run_id)
        if fs_data and "agent_signals" in fs_data:
            results = []
            for i, agent in enumerate(fs_data["agent_signals"]):
                summary = AgentSummary(
                    agent_name=agent.get("agent_name", f"Agent_{i}"),
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    execution_time_seconds=0,
                    status="completed"
                )
                results.append(summary)
            return results

        raise HTTPException(
            status_code=404,
            detail=f"未找到ID为 {run_id} 的运行"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取运行Agent信息失败: {str(e)}"
        )


@router.get("/{run_id}/agents/{agent_name}", response_model=AgentDetail)
async def get_agent_detail(
    run_id: str = Path(..., description="运行ID"),
    agent_name: str = Path(..., description="Agent名称"),
    include_states: bool = Query(True, description="是否包含输入/输出状态"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """获取特定运行中特定Agent的详细执行情况 (基于日志存储)

    查询 BaseLogStorage (当前为内存实现 InMemoryLogStorage) 中
    特定 run_id 和 agent_name 的 AgentExecutionLog 及关联的 LLMInteractionLog。
    注意：基于内存的日志在服务重启后会丢失。

    TODO: 通过依赖注入 BaseLogStorage 的不同实现 (如数据库存储)，
          可以轻松切换到底层存储，无需修改此接口代码。
    """
    try:
        # 获取特定Agent的日志
        agent_logs = storage.get_agent_logs(
            run_id=run_id, agent_name=agent_name)
        if not agent_logs:
            raise HTTPException(
                status_code=404,
                detail=f"在运行 {run_id} 中未找到Agent {agent_name}"
            )

        # 获取相关的LLM交互记录
        llm_logs = storage.get_logs(run_id=run_id, agent_name=agent_name)
        llm_interaction_ids = [str(i) for i in range(
            len(llm_logs))] if llm_logs else []

        # 构建详细信息
        log = agent_logs[0]  # 应该只有一个匹配的日志
        result = AgentDetail(
            agent_name=log.agent_name,
            start_time=log.timestamp_start,
            end_time=log.timestamp_end,
            execution_time_seconds=(
                log.timestamp_end - log.timestamp_start).total_seconds(),
            status="completed",
            llm_interactions=llm_interaction_ids
        )

        # 添加状态和推理信息（如果需要）
        if include_states:
            result.input_state = log.input_state
            result.output_state = log.output_state
            result.reasoning = log.reasoning_details

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取Agent详情失败: {str(e)}"
        )


@router.get("/{run_id}/flow", response_model=WorkflowFlow)
async def get_workflow_flow(
    run_id: str = Path(..., description="运行ID"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """获取特定运行的完整工作流程和数据流 (基于日志存储)

    查询 BaseLogStorage (当前为内存实现 InMemoryLogStorage) 中
    特定 run_id 的所有 AgentExecutionLog，构建工作流图。
    注意：基于内存的日志在服务重启后会丢失。

    TODO: 通过依赖注入 BaseLogStorage 的不同实现 (如数据库存储)，
          可以轻松切换到底层存储，无需修改此接口代码。
    """
    try:
        # 获取该运行的所有Agent日志
        agent_logs = storage.get_agent_logs(run_id=run_id)

        if agent_logs:
            # 计算开始和结束时间
            start_time = min(log.timestamp_start for log in agent_logs)
            end_time = max(log.timestamp_end for log in agent_logs)

            # 构建Agent摘要
            agents = {}
            for log in agent_logs:
                agents[log.agent_name] = AgentSummary(
                    agent_name=log.agent_name,
                    start_time=log.timestamp_start,
                    end_time=log.timestamp_end,
                    execution_time_seconds=(
                        log.timestamp_end - log.timestamp_start).total_seconds(),
                    status="completed"
                )

            # 构建状态转换列表
            agent_logs_sorted = sorted(agent_logs, key=lambda x: x.timestamp_start)
            state_transitions = []

            for i, log in enumerate(agent_logs_sorted):
                transition = {
                    "from_agent": "start" if i == 0 else agent_logs_sorted[i-1].agent_name,
                    "to_agent": log.agent_name,
                    "state_size": len(str(log.input_state)) if log.input_state else 0,
                    "timestamp": log.timestamp_start.isoformat()
                }
                state_transitions.append(transition)

            # 添加最后一个转换到结束
            if agent_logs_sorted:
                state_transitions.append({
                    "from_agent": agent_logs_sorted[-1].agent_name,
                    "to_agent": "end",
                    "state_size": len(str(agent_logs_sorted[-1].output_state)) if agent_logs_sorted[-1].output_state else 0,
                    "timestamp": agent_logs_sorted[-1].timestamp_end.isoformat()
                })

            # 尝试提取最终决策
            final_decision = None
            if agent_logs_sorted:
                last_log = agent_logs_sorted[-1]
                if last_log.output_state and isinstance(last_log.output_state, dict):
                    messages = last_log.output_state.get("messages", [])
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        if isinstance(last_message, dict) and "content" in last_message:
                            final_decision = last_message["content"]

            fs_result = load_run_from_filesystem(run_id)
            return WorkflowFlow(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                agents=agents,
                state_transitions=state_transitions,
                final_decision=final_decision,
                result_data=fs_result
            )

        # 回退：从文件系统读取
        fs_data = load_run_from_filesystem(run_id)
        if fs_data:
            now = datetime.now()
            agents = {}
            if "agent_signals" in fs_data:
                for i, agent in enumerate(fs_data["agent_signals"]):
                    agents[agent.get("agent_name", f"Agent_{i}")] = AgentSummary(
                        agent_name=agent.get("agent_name", f"Agent_{i}"),
                        start_time=now,
                        end_time=now,
                        execution_time_seconds=0,
                        status="completed"
                    )

            # 构建简单的状态转换
            agent_names = list(agents.keys())
            state_transitions = []
            for i, name in enumerate(agent_names):
                state_transitions.append({
                    "from_agent": "start" if i == 0 else agent_names[i-1],
                    "to_agent": name,
                    "state_size": 0,
                    "timestamp": now.isoformat()
                })
            if agent_names:
                state_transitions.append({
                    "from_agent": agent_names[-1],
                    "to_agent": "end",
                    "state_size": 0,
                    "timestamp": now.isoformat()
                })

            final_decision = json.dumps({
                "action": fs_data.get("action"),
                "confidence": fs_data.get("confidence"),
                "reasoning": fs_data.get("reasoning")
            }) if fs_data.get("action") else None

            return WorkflowFlow(
                run_id=run_id,
                start_time=now,
                end_time=now,
                agents=agents,
                state_transitions=state_transitions,
                final_decision=final_decision,
                result_data=fs_data
            )

        raise HTTPException(
            status_code=404,
            detail=f"未找到ID为 {run_id} 的运行"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取工作流程失败: {str(e)}"
        )


@router.delete("/")
async def clear_all_runs(
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """清空所有运行记录"""
    try:
        storage.clear_all()
        return {"message": "已清空所有运行记录"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清空运行记录失败: {str(e)}"
        )


@router.delete("/{run_id}")
async def delete_run(
    run_id: str = Path(..., description="要删除的运行ID"),
    storage: BaseLogStorage = Depends(get_log_storage)
):
    """删除指定 run_id 的运行记录"""
    print(f"[DELETE] 收到删除请求: {run_id}")
    try:
        success = storage.delete_run(run_id)
        print(f"[DELETE] 删除结果: {run_id}, success: {success}")
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"未找到ID为 {run_id} 的运行"
            )
        return {"message": f"已删除运行记录 {run_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除运行记录失败: {str(e)}"
        )
