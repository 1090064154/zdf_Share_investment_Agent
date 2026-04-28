"""JSON文件存储系统"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger("file_storage")

# 存储根目录
STORAGE_ROOT = Path("src/data/runs")

@dataclass
class RunIndex:
    """运行索引"""
    run_id: str
    ticker: str
    status: str
    action: Optional[str] = None
    confidence: float = 0.0
    created_at: str = ""
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

class FileStorage:
    """JSON文件存储"""

    def __init__(self, storage_root: Path = STORAGE_ROOT):
        self.storage_root = storage_root
        self._ensure_storage_root()

    def _ensure_storage_root(self):
        """确保存储目录存在"""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._ensure_index_file()

    def _ensure_index_file(self):
        """确保索引文件存在"""
        index_file = self.storage_root / "index.json"
        if not index_file.exists():
            self._write_json(index_file, {"runs": []})

    def _run_dir(self, run_id: str) -> Path:
        """获取运行目录"""
        return self.storage_root / run_id

    def _ensure_run_dir(self, run_id: str) -> Path:
        """确保运行目录存在"""
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_json(self, path: Path, data: dict):
        """写入JSON文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _read_json(self, path: Path) -> dict:
        """读取JSON文件"""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================== 运行索引 ====================

    def add_run(self, run_id: str, ticker: str, params: Dict) -> RunIndex:
        """添加新运行到索引"""
        index_file = self.storage_root / "index.json"
        data = self._read_json(index_file)

        run_index = RunIndex(
            run_id=run_id,
            ticker=ticker,
            status="pending",
            created_at=datetime.now().isoformat()
        )

        data["runs"].insert(0, asdict(run_index))  # 最新在前
        self._write_json(index_file, data)

        logger.info(f"添加运行索引: {run_id}")
        return run_index

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        action: Optional[str] = None,
        confidence: Optional[float] = None,
        completed_at: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ):
        """更新运行索引"""
        index_file = self.storage_root / "index.json"
        data = self._read_json(index_file)

        for run in data["runs"]:
            if run["run_id"] == run_id:
                if status is not None:
                    run["status"] = status
                if action is not None:
                    run["action"] = action
                if confidence is not None:
                    run["confidence"] = confidence
                if completed_at is not None:
                    run["completed_at"] = completed_at
                if duration_seconds is not None:
                    run["duration_seconds"] = duration_seconds
                break

        self._write_json(index_file, data)

    def get_run_index(self, run_id: str) -> Optional[Dict]:
        """获取运行索引"""
        index_file = self.storage_root / "index.json"
        data = self._read_json(index_file)

        for run in data["runs"]:
            if run["run_id"] == run_id:
                return run
        return None

    def get_all_runs(self, page: int = 1, limit: int = 20) -> Dict:
        """获取所有运行列表"""
        index_file = self.storage_root / "index.json"
        data = self._read_json(index_file)

        runs = data.get("runs", [])
        total = len(runs)
        start = (page - 1) * limit
        end = start + limit

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "items": runs[start:end]
        }

    # ==================== 运行详情 ====================

    def save_metadata(self, run_id: str, metadata: Dict):
        """保存运行元数据"""
        run_dir = self._ensure_run_dir(run_id)
        metadata_file = run_dir / "metadata.json"
        self._write_json(metadata_file, metadata)

    def get_metadata(self, run_id: str) -> Optional[Dict]:
        """获取运行元数据"""
        metadata_file = self._run_dir(run_id) / "metadata.json"
        return self._read_json(metadata_file)

    def save_result(self, run_id: str, result: Dict):
        """保存运行结果"""
        run_dir = self._ensure_run_dir(run_id)
        result_file = run_dir / "result.json"
        self._write_json(result_file, result)

    def get_result(self, run_id: str) -> Optional[Dict]:
        """获取运行结果"""
        result_file = self._run_dir(run_id) / "result.json"
        return self._read_json(result_file)

    def save_messages(self, run_id: str, messages: List[Dict]):
        """保存Agent消息"""
        run_dir = self._ensure_run_dir(run_id)
        messages_file = run_dir / "messages.json"
        self._write_json(messages_file, {"messages": messages})

    def get_messages(self, run_id: str) -> List[Dict]:
        """获取Agent消息"""
        messages_file = self._run_dir(run_id) / "messages.json"
        data = self._read_json(messages_file)
        return data.get("messages", [])

    # ==================== Agent日志 ====================

    def save_agent_log(self, run_id: str, agent_name: str, logs: List[Dict]):
        """保存Agent日志"""
        run_dir = self._ensure_run_dir(run_id)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / f"{agent_name}.json"
        self._write_json(log_file, {"logs": logs})

    def get_agent_log(self, run_id: str, agent_name: str) -> List[Dict]:
        """获取Agent日志"""
        log_file = self._run_dir(run_id) / "logs" / f"{agent_name}.json"
        data = self._read_json(log_file)
        return data.get("logs", [])

    def append_agent_log(self, run_id: str, agent_name: str, log_entry: Dict):
        """追加Agent日志"""
        logs = self.get_agent_log(run_id, agent_name)
        logs.append(log_entry)
        self.save_agent_log(run_id, agent_name, logs)

    def save_system_log(self, run_id: str, logs: List[Dict]):
        """保存系统日志"""
        run_dir = self._ensure_run_dir(run_id)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / "system.json"
        self._write_json(log_file, {"logs": logs})

    def get_system_log(self, run_id: str) -> List[Dict]:
        """获取系统日志"""
        log_file = self._run_dir(run_id) / "logs" / "system.json"
        data = self._read_json(log_file)
        return data.get("logs", [])

# 全局存储实例
file_storage = FileStorage()