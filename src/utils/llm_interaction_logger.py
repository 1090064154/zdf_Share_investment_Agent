import functools
import io
import sys
import logging
import math
from contextvars import ContextVar
from typing import Any, Callable, List, Optional, Dict, Tuple
from datetime import datetime, UTC

from backend.schemas import LLMInteractionLog, AgentExecutionLog
from backend.storage.base import BaseLogStorage
from src.agents.state import AgentState, normalize_agent_name
from src.utils.serialization import serialize_agent_state

try:
    from src.api.log_hook import agent_started, agent_log, agent_completed, agent_failed
    HAS_LOG_HOOK = True
except ImportError:
    HAS_LOG_HOOK = False

# --- Context Variables ---
# These variables hold state specific to the current execution context (e.g., a single agent run within a workflow).

# Holds the BaseLogStorage instance for the current run. Initialized in main.py.
log_storage_context: ContextVar[Optional[BaseLogStorage]] = ContextVar(
    "log_storage_context", default=None
)

# Holds the name of the agent currently being executed. Set by the decorator.
current_agent_name_context: ContextVar[Optional[str]] = ContextVar(
    "current_agent_name_context", default=None
)

# Holds the unique ID for the entire workflow run. Set in main.py and passed via state.
current_run_id_context: ContextVar[Optional[str]] = ContextVar(
    "current_run_id_context", default=None
)


# --- Output Capture Utility ---

class OutputCapture:
    """捕获标准输出和日志的工具类。
    为保持前端简洁，关闭实时日志推送，只通过agent_start/agent_complete事件推送关键信息。
    """

    class _LiveStream(io.TextIOBase):
        """将输出写到原始流和缓存，不发送SSE日志（避免杂乱）"""

        def __init__(self, mirror, outputs, run_id=None, agent_name=None):
            self.mirror = mirror
            self.outputs = outputs

        def write(self, text):
            if not text:
                return 0

            self.outputs.append(text)

            if self.mirror:
                self.mirror.write(text)
                self.mirror.flush()

            return len(text)

        def flush(self):
            if self.mirror:
                self.mirror.flush()

    def __init__(self, run_id: Optional[str] = None, agent_name: Optional[str] = None):
        self.outputs = []
        self.stdout_buffer = io.StringIO()
        self.old_stdout = None
        self.log_handler = None
        self.old_log_level = None
        self.run_id = run_id
        self.agent_name = agent_name
        self.live_stream = None

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.live_stream = self._LiveStream(
            mirror=self.old_stdout,
            outputs=self.outputs,
            run_id=self.run_id,
            agent_name=self.agent_name
        )
        sys.stdout = self.live_stream

        self.log_handler = logging.StreamHandler(self.live_stream)
        self.log_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        self.old_log_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.log_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

        root_logger = logging.getLogger()
        root_logger.removeHandler(self.log_handler)
        root_logger.setLevel(self.old_log_level)


# --- Wrapper for LLM Calls ---

def wrap_llm_call(original_llm_func: Callable) -> Callable:
    """Wraps an LLM call function (like get_chat_completion) to log interactions.

    Reads context variables set by the agent decorator to get agent name,
    run ID, and the storage instance.

    Args:
        original_llm_func: The original function that makes the LLM call.

    Returns:
        A wrapped function that logs the interaction before returning the original result.
    """

    @functools.wraps(original_llm_func)
    def wrapper(*args, **kwargs) -> Any:
        # Retrieve context information
        storage = log_storage_context.get()
        agent_name = current_agent_name_context.get()
        run_id = current_run_id_context.get()

        # Proceed with the original call even if context is missing, but don't log
        if not storage or not agent_name:
            # Maybe log a warning here if desired
            return original_llm_func(*args, **kwargs)

        # Assume the first argument is usually the list of messages or prompt
        # This might need adjustment if the wrapped function signature varies
        request_data = args[0] if args else kwargs.get(
            'messages', kwargs)  # Adapt based on common usage

        # Execute the original LLM call
        response_data = original_llm_func(*args, **kwargs)

        # Create and store the log entry
        log_entry = LLMInteractionLog(
            agent_name=agent_name,
            run_id=run_id,  # run_id can be None if not set
            request_data=request_data,  # Consider serializing complex objects if needed
            response_data=response_data,  # Consider serializing complex objects if needed
            # Explicit timestamp in case storage adds its own
            timestamp=datetime.now(UTC)
        )
        storage.add_log(log_entry)

        return response_data

    return wrapper


def _normalize_confidence_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        try:
            numeric = float(cleaned)
            return numeric / 100.0 if numeric > 1 else numeric
        except ValueError:
            return 0.0
    try:
        numeric = float(value)
        return numeric / 100.0 if numeric > 1 else numeric
    except (TypeError, ValueError):
        return 0.0


def _normalize_signal_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "positive": "bullish",
        "negative": "bearish",
        "buy": "bullish",
        "sell": "bearish",
        "hold": "neutral",
        "reduce": "bearish",
    }
    return mapping.get(normalized, normalized or None)


def _compact_payload(value: Any, depth: int = 0) -> Any:
    import math
    if depth > 3:
        return "..."
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        items = list(value.items())[:20]
        compacted = {str(k): _compact_payload(v, depth + 1) for k, v in items}
        if len(value) > 20:
            compacted["_truncated"] = f"+{len(value) - 20} more keys"
        return compacted
    if isinstance(value, list):
        compacted = [_compact_payload(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            compacted.append(f"... {len(value) - 12} more items")
        return compacted
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1200] + "...(truncated)"
    return value


def _build_agent_summary(agent_name: str, latest_payload: Dict[str, Any], reasoning: Any, serialized_input: Optional[Dict[str, Any]], serialized_output: Optional[Dict[str, Any]]) -> str:
    input_data = (serialized_input or {}).get("data", {})
    output_data = (serialized_output or {}).get("data", {})

    if agent_name == "market_data_agent":
        ticker = input_data.get("ticker") or output_data.get("ticker") or ""
        prices = output_data.get("prices", [])
        industry = output_data.get("industry") or "未知行业"
        start_date = output_data.get("start_date") or input_data.get("start_date") or "未提供"
        end_date = output_data.get("end_date") or input_data.get("end_date") or "未提供"
        metrics_ready = bool(output_data.get("financial_metrics"))
        return f"{ticker} 在 {start_date} 到 {end_date} 区间共整理 {len(prices)} 条价格数据，行业为 {industry}，财务指标{'已获取' if metrics_ready else '未获取'}。"

    if agent_name == "technical_analyst_agent":
        strategies = latest_payload.get("strategy_signals", {})
        picked = []
        for key in ("trend_following", "mean_reversion", "momentum"):
            item = strategies.get(key, {})
            if item:
                picked.append(f"{key}={item.get('signal', 'neutral')}")
        return f"技术面最终判断为 {latest_payload.get('signal', 'neutral')}，主要子策略结论：{', '.join(picked) or '暂无'}。"

    if agent_name == "fundamentals_agent":
        reason = latest_payload.get("reasoning", {}) if isinstance(latest_payload, dict) else {}
        profitability = reason.get("profitability_signal", {}).get("signal", "neutral")
        growth = reason.get("growth_signal", {}).get("signal", "neutral")
        health = reason.get("financial_health_signal", {}).get("signal", "neutral")
        return f"基本面综合判断为 {latest_payload.get('signal', 'neutral')}，其中盈利={profitability}、成长={growth}、财务健康={health}。"

    if agent_name == "risk_management_agent":
        action = latest_payload.get("交易行动", "hold")
        score = latest_payload.get("风险评分", "N/A")
        max_pos = latest_payload.get("最大持仓规模", "N/A")
        return f"风险评分 {score}/10，建议动作为 {action}，允许的最大持仓约 {max_pos} 股。"

    if agent_name == "portfolio_management_agent":
        action = latest_payload.get("action", "hold")
        quantity = latest_payload.get("quantity", 0)
        confidence = _normalize_confidence_value(latest_payload.get("confidence"))
        agent_signals = latest_payload.get("agent_signals", [])
        return f"最终决策为 {action} {quantity} 股，决策把握度约 {round(confidence * 100)}%，综合参考了 {len(agent_signals)} 个模块信号。"

    if agent_name == "macro_news_agent":
        if isinstance(reasoning, str):
            return reasoning[:140] + ("..." if len(reasoning) > 140 else "")

    if isinstance(reasoning, dict):
        for key in ("reasoning", "summary", "推理", "details"):
            if key in reasoning and reasoning[key]:
                return str(reasoning[key])[:160]

    if isinstance(latest_payload, dict):
        for key in ("reasoning", "summary", "details", "raw_content"):
            if key in latest_payload and latest_payload[key]:
                return str(latest_payload[key])[:160]

    return "该节点已完成，但没有产出适合直接展示的摘要说明。"


def extract_agent_completion_payload(
    result_state: Optional[AgentState],
    serialized_input: Optional[Dict[str, Any]] = None,
    serialized_output: Optional[Dict[str, Any]] = None,
    agent_name: str = "",
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """从 Agent 输出中提取前端展示所需的结构化信息。"""
    signal = None
    confidence = 0.0
    reasoning = None
    latest_payload = {}

    if result_state:
        messages = result_state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                try:
                    import json
                    latest_payload = json.loads(last_msg.content)
                except Exception:
                    latest_payload = {"raw_content": getattr(last_msg, "content", "")}

        metadata = result_state.get("metadata", {})
        reasoning = metadata.get("agent_reasoning")

    if isinstance(latest_payload, dict):
        signal = _normalize_signal_value(
            latest_payload.get("signal")
            or latest_payload.get("impact_on_stock")
            or latest_payload.get("action")
            or latest_payload.get("交易行动")
            or latest_payload.get("macro_environment")
        )
        confidence = _normalize_confidence_value(
            latest_payload.get("confidence")
            or latest_payload.get("置信度")
        )

    if (not signal or confidence == 0.0) and isinstance(reasoning, dict):
        signal = signal or _normalize_signal_value(
            reasoning.get("signal")
            or reasoning.get("impact_on_stock")
            or reasoning.get("action")
            or reasoning.get("交易行动")
            or reasoning.get("macro_environment")
        )
        if confidence == 0.0:
            confidence = _normalize_confidence_value(
                reasoning.get("confidence")
                or reasoning.get("置信度")
            )

    if confidence == 0.0 and isinstance(serialized_output, dict):
        maybe_reasoning = serialized_output.get("metadata", {}).get("agent_reasoning", {})
        if isinstance(maybe_reasoning, dict):
            confidence = _normalize_confidence_value(
                maybe_reasoning.get("confidence")
                or maybe_reasoning.get("置信度")
            )
            signal = signal or _normalize_signal_value(
                maybe_reasoning.get("signal")
                or maybe_reasoning.get("impact_on_stock")
                or maybe_reasoning.get("action")
                or maybe_reasoning.get("交易行动")
                or maybe_reasoning.get("macro_environment")
            )

    details = {
        "input": _compact_payload((serialized_input or {}).get("data", serialized_input or {})),
        "output": _compact_payload((serialized_output or {}).get("data", serialized_output or {})),
        "reasoning": _compact_payload(reasoning if reasoning is not None else latest_payload),
        "result": _compact_payload(latest_payload),
        "summary": _build_agent_summary(
            agent_name=agent_name,
            latest_payload=latest_payload if isinstance(latest_payload, dict) else {},
            reasoning=reasoning,
            serialized_input=serialized_input,
            serialized_output=serialized_output,
        ),
    }

    return signal, confidence, details


# --- Decorator for Agent Functions ---

def log_agent_execution(agent_name: str):
    """Decorator for agent functions to set logging context variables.

    Retrieves the run_id from the agent state's metadata.

    Args:
        agent_name: The name of the agent being decorated.
    """

    def decorator(agent_func: Callable[[AgentState], AgentState]):
        @functools.wraps(agent_func)
        def wrapper(state: AgentState) -> AgentState:
            # Retrieve run_id from state metadata (set in main.py)
            run_id = state.get("metadata", {}).get("run_id")
            storage = log_storage_context.get()

            # 设置上下文变量
            agent_token = current_agent_name_context.set(agent_name)
            run_id_token = current_run_id_context.set(run_id)

            # 捕获开始时间和输入状态
            timestamp_start = datetime.now(UTC)
            serialized_input = serialize_agent_state(state)

            # 准备输出捕获
            output_capture = OutputCapture(run_id=run_id, agent_name=agent_name)
            result_state = None
            error = None

            try:
                # 发送Agent开始事件
                if HAS_LOG_HOOK and run_id:
                    agent_started(run_id, agent_name, f"开始执行 {agent_name}")

                # 使用输出捕获器
                with output_capture:
                    # 执行原始Agent函数
                    result_state = agent_func(state)

                # 成功执行，记录日志
                timestamp_end = datetime.now(UTC)
                terminal_outputs = ["".join(output_capture.outputs)] if output_capture.outputs else []

                serialized_output = serialize_agent_state(result_state) if result_state else {}
                normalized_agent_name = normalize_agent_name(agent_name)
                signal, confidence, details = extract_agent_completion_payload(
                    result_state=result_state,
                    serialized_input=serialized_input,
                    serialized_output=serialized_output,
                    agent_name=normalized_agent_name,
                )

                # 发送Agent完成事件
                if HAS_LOG_HOOK and run_id:
                    agent_completed(
                        run_id,
                        normalized_agent_name,
                        signal,
                        confidence,
                        f"{normalized_agent_name} 执行完成",
                        details=details,
                    )
                    # 发送终端输出作为日志（更多关键行）
                    if terminal_outputs:
                        full_output = "".join(terminal_outputs)
                        lines = full_output.split("\n")
                        # 放宽过滤条件，捕捉更多有用的日志
                        key_lines = [l.strip() for l in lines if l.strip() and (
                            "开始" in l or "完成" in l or "获取" in l or "✅" in l or "❌" in l or 
                            "错误" in l or "警告" in l or "正在" in l or "分析" in l or "处理" in l or
                            "调用" in l or "成功" in l or "失败" in l or "返回" in l or "数据" in l
                        )]
                        for line in key_lines[:15]:  # 最多15条
                            agent_log(run_id, normalized_agent_name, "info", line)

                if storage and result_state:
                    # 提取推理详情（如果有）
                    reasoning_details = None
                    if result_state.get("metadata", {}).get("show_reasoning", False):
                        if "agent_reasoning" in result_state.get("metadata", {}):
                            reasoning_details = result_state["metadata"]["agent_reasoning"]

                    # 创建日志条目
                    log_entry = AgentExecutionLog(
                        agent_name=agent_name,
                        run_id=run_id,
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        input_state=serialized_input,
                        output_state=serialized_output,
                        reasoning_details=reasoning_details,
                        terminal_outputs=terminal_outputs
                    )

                    # 存储日志
                    storage.add_agent_log(log_entry)
            except Exception as e:
                # 记录错误
                error = str(e)
                # 重新抛出异常，让上层处理
                raise
            finally:
                # 清理上下文变量
                current_agent_name_context.reset(agent_token)
                current_run_id_context.reset(run_id_token)

                # 如果出现错误但存储可用，记录错误日志
                if error and storage:
                    timestamp_end = datetime.now(UTC)
                    log_entry = AgentExecutionLog(
                        agent_name=agent_name,
                        run_id=run_id,
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        input_state=serialized_input,
                        output_state={"error": error},
                        reasoning_details=None,
                        terminal_outputs=output_capture.outputs
                    )
                    storage.add_agent_log(log_entry)

            return result_state
        return wrapper
    return decorator

# Helper to set the global storage instance (called from main.py)


def set_global_log_storage(storage: BaseLogStorage):
    log_storage_context.set(storage)
