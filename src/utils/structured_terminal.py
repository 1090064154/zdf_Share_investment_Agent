"""
结构化终端输出模块

此模块提供了一个简单但灵活的系统，用于收集和格式化agent数据，
然后在工作流结束时以美观、结构化的格式一次性展示。

完全独立于后端，只负责终端输出的格式化。
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logging_config import setup_logger

# 设置日志记录器
logger = setup_logger('structured_terminal')

# 格式化符号
SYMBOLS = {
    "border": "═",
    "header_left": "╔",
    "header_right": "╗",
    "footer_left": "╚",
    "footer_right": "╝",
    "separator": "─",
    "vertical": "║",
    "tree_branch": "├─",
    "tree_last": "└─",
    "section_prefix": "● ",
    "bullet": "• ",
}

# 状态图标
STATUS_ICONS = {
    "bearish": "📉",
    "bullish": "📈",
    "neutral": "◽",
    "hold": "⏸️",
    "buy": "🛒",
    "sell": "💰",
    "completed": "✅",
    "in_progress": "🔄",
    "error": "❌",
    "warning": "⚠️",
}

# Agent图标和名称映射
AGENT_MAP = {
    "market_data_agent": {"icon": "📊", "name": "市场数据"},
    "technical_analyst_agent": {"icon": "📈", "name": "技术"},
    "fundamentals_agent": {"icon": "📝", "name": "基本面"},
    "sentiment_agent": {"icon": "🔍", "name": "情感"},
    "valuation_agent": {"icon": "💰", "name": "估值"},
    "researcher_bull_agent": {"icon": "🐂", "name": "多方研究"},
    "researcher_bear_agent": {"icon": "🐻", "name": "空方研究"},
    "debate_room_agent": {"icon": "🗣️", "name": "辩论室"},
    "risk_management_agent": {"icon": "⚠️", "name": "风险管理"},
    "macro_analyst_agent": {"icon": "🌍", "name": "针对所选股宏观"},
    "macro_news_agent": {"icon": "📰", "name": "宏观新闻"},
    "portfolio_management_agent": {"icon": "📂", "name": "投资组合管理"}
}

# 字段名中英文映射
FIELD_NAME_MAP = {
    # 通用字段
    "signal": "信号",
    "confidence": "置信度",
    "reasoning": "推理",
    "action": "行动",
    "quantity": "数量",

    # 风险管理字段
    "max_position_size": "最大持仓规模",
    "risk_score": "风险评分",
    "trading_action": "交易行动",
    "risk_metrics": "风险指标",
    "volatility": "波动率",
    "value_at_risk_95": "95%风险价值",
    "max_drawdown": "最大回撤",
    "market_risk_score": "市场风险评分",
    "stress_test_results": "压力测试结果",
    "debate_analysis": "辩论分析",
    "bull_confidence": "多方置信度",
    "bear_confidence": "空方置信度",
    "debate_confidence": "辩论置信度",
    "debate_signal": "辩论信号",
    "potential_loss": "潜在损失",
    "portfolio_impact": "组合影响",

    # 压力测试场景
    "market_crash": "市场崩盘",
    "moderate_decline": "中度下跌",
    "slight_decline": "轻度下跌",

    # 技术分析字段
    "strategy_signals": "策略信号",
    "trend_following": "趋势跟踪",
    "mean_reversion": "均值回归",
    "momentum": "动量",
    "volatility": "波动性",
    "statistical_arbitrage": "统计套利",
    "metrics": "指标",
    "adx": "ADX",
    "trend_strength": "趋势强度",
    "z_score": "Z分数",
    "price_vs_bb": "价格vs布林带",
    "rsi_14": "RSI(14)",
    "rsi_28": "RSI(28)",
    "momentum_1m": "1月动量",
    "momentum_3m": "3月动量",
    "momentum_6m": "6月动量",
    "volume_momentum": "成交量动量",
    "historical_volatility": "历史波动率",
    "volatility_regime": "波动率区间",
    "volatility_z_score": "波动率Z分数",
    "atr_ratio": "ATR比率",
    "hurst_exponent": "赫斯特指数",
    "skewness": "偏度",
    "kurtosis": "峰度",

    # 基本面分析字段
    "profitability_signal": "盈利能力信号",
    "growth_signal": "增长信号",
    "financial_health_signal": "财务健康信号",
    "price_ratios_signal": "价格比率信号",
    "fallback": "回退",
    "details": "详情",

    # 估值分析字段
    "dcf_analysis": "DCF分析",
    "owner_earnings_analysis": "所有者收益分析",

    # 辩论室字段
    "confidence_diff": "置信度差异",
    "llm_score": "LLM评分",
    "llm_analysis": "LLM分析",
    "llm_reasoning": "LLM推理",
    "mixed_confidence_diff": "混合置信度差异",
    "debate_summary": "辩论摘要",

    # 投资组合管理字段
    "agent_signals": "各分析师信号",
    "agent_name": "分析师名称",
    # Agent名称中英映射
    "technical_analysis": "技术分析",
    "fundamental_analysis": "基本面分析",
    "sentiment_analysis": "情绪分析",
    "valuation_analysis": "估值分析",
    "risk_management": "风险管理",
    "macro_analysis": "宏观分析",
    "macro_news_analysis": "宏观新闻分析",
    "selected_stock_macro_analysis": "所选股宏观分析",
    "market_wide_news_summary(沪深300指数)": "大盘新闻摘要(沪深300)",

    # 宏观分析字段
    "macro_environment": "宏观环境",
    "impact_on_stock": "对股票影响",
    "key_factors": "关键因素",
    "positive": "积极",
    "negative": "消极",
    "neutral": "中性",
    "cautious_weak": "谨慎偏弱",

    # 财务指标缩写映射
    "ROE": "净资产收益率",
    "Net Margin": "净利率",
    "Op Margin": "营业利润率",
    "Revenue Growth": "营收增长率",
    "Earnings Growth": "盈利增长率",
    "D/E": "负债权益比",
    "P/E": "市盈率",
    "P/B": "市净率",
    "P/S": "市销率",
    "Current Ratio": "流动比率",
    "Free Cash Flow": "自由现金流",
    "EPS": "每股收益",
    "Intrinsic Value": "内在价值",
    "Market Cap": "市值",
    "Owner Earnings Value": "所有者收益价值",
    "Gap": "差距",
    "DCF": "现金流折现",
    "ADX": "平均趋向指数",
    "RSI": "相对强弱指数",
    "Z-score": "Z分数",
    "ATR": "平均真实波幅",

    # 研究员字段
    "perspective": "观点",
    "thesis_points": "论点",

    # 信号值映射
    "bullish": "看多",
    "bearish": "看空",
    "hold": "持有",
    "buy": "买入",
    "sell": "卖出",
}

# Agent显示顺序
AGENT_ORDER = [
    "market_data_agent",
    "technical_analyst_agent",
    "fundamentals_agent",
    "sentiment_agent",
    "valuation_agent",
    "researcher_bull_agent",
    "researcher_bear_agent",
    "debate_room_agent",
    "risk_management_agent",
    "macro_analyst_agent",
    "macro_news_agent",
    "portfolio_management_agent"
]


class StructuredTerminalOutput:
    """结构化终端输出类"""

    def __init__(self):
        """初始化"""
        self.data = {}
        self.metadata = {}

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value

    def add_agent_data(self, agent_name: str, data: Any) -> None:
        """添加agent数据"""
        self.data[agent_name] = data

    def _format_value(self, value: Any, key: str = "") -> str:
        """格式化单个值"""
        if isinstance(value, bool):
            return "✅" if value else "❌"
        elif isinstance(value, (int, float)):
            # 对置信度字段进行特殊处理
            if key.lower() == "confidence":
                # 智能判断：大于1认为是百分比，否则是小数
                conf_value = value if value > 1 else value * 100
                return f"{conf_value:.0f}%"
            # 对百分比值进行特殊处理
            if -1 <= value <= 1 and isinstance(value, float):
                return f"{value:.2%}"
            return str(value)
        elif value is None:
            return "N/A"
        elif isinstance(value, str):
            # 对置信度字段进行特殊处理
            if key.lower() == "confidence":
                try:
                    conf_str = value.strip().replace("%", "")
                    conf_num = float(conf_str)
                    # 如果字符串值大于1，认为是百分比；否则是小数
                    conf_value = conf_num if conf_num > 1 else conf_num * 100
                    return f"{conf_value:.0f}%"
                except (ValueError, TypeError):
                    return value
            # 尝试将字符串值转换为中文（如 neutral -> 中性）
            # 先尝试精确匹配，再尝试小写匹配
            if value in FIELD_NAME_MAP:
                return FIELD_NAME_MAP[value]
            lower_value = value.lower()
            if lower_value in FIELD_NAME_MAP:
                return FIELD_NAME_MAP[lower_value]
            return value
        else:
            return str(value)

    def _format_dict_as_tree(self, data: Dict[str, Any], indent: int = 0) -> List[str]:
        """将字典格式化为树形结构"""
        result = []
        items = list(data.items())

        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            prefix = SYMBOLS["tree_last"] if is_last else SYMBOLS["tree_branch"]
            indent_str = "  " * indent

            # 将英文key转换为中文显示
            display_key = FIELD_NAME_MAP.get(key, key)

            if isinstance(value, dict) and value:
                result.append(f"{indent_str}{prefix} {display_key}:")
                result.extend(self._format_dict_as_tree(value, indent + 1))
            elif isinstance(value, list) and value:
                result.append(f"{indent_str}{prefix} {display_key}:")
                for j, item in enumerate(value):
                    sub_is_last = j == len(value) - 1
                    sub_prefix = SYMBOLS["tree_last"] if sub_is_last else SYMBOLS["tree_branch"]
                    if isinstance(item, dict):
                        result.append(
                            f"{indent_str}  {sub_prefix} Agent {j+1}:")
                        result.extend(
                            ["  " + line for line in self._format_dict_as_tree(item, indent + 2)])
                    else:
                        result.append(f"{indent_str}  {sub_prefix} {item}")
            else:
                formatted_value = self._format_value(value, key)
                result.append(f"{indent_str}{prefix} {display_key}: {formatted_value}")

        return result

    def _format_agent_section(self, agent_name: str, data: Any) -> List[str]:
        """格式化agent部分"""
        result = []

        # 获取agent信息
        agent_info = AGENT_MAP.get(
            agent_name, {"icon": "🔄", "name": agent_name})
        icon = agent_info["icon"]
        display_name = agent_info["name"]

        # 创建标题
        width = 80
        title = f"{icon} {display_name}分析"
        result.append(
            f"{SYMBOLS['header_left']}{SYMBOLS['border'] * ((width - len(title) - 2) // 2)} {title} {SYMBOLS['border'] * ((width - len(title) - 2) // 2)}{SYMBOLS['header_right']}")

        # 添加内容
        if isinstance(data, dict):
            # 特殊处理portfolio_management_agent
            if agent_name == "portfolio_management_agent":
                # 尝试提取action和confidence
                if "action" in data:
                    action = data.get("action", "")
                    action_icon = STATUS_ICONS.get(action.lower(), "")
                    result.append(
                        f"{SYMBOLS['vertical']} 交易行动: {action_icon} {action.upper() if action else ''}")

                if "quantity" in data:
                    quantity = data.get("quantity", 0)
                    result.append(f"{SYMBOLS['vertical']} 交易数量: {quantity}")

                if "confidence" in data:
                    conf = data.get("confidence", 0)
                    # 格式化置信度（处理多种格式）
                    try:
                        if isinstance(conf, (int, float)):
                            # 如果是数字，判断范围：大于1则认为是百分比，否则是小数
                            conf_value = conf if conf > 1 else conf * 100
                        elif isinstance(conf, str):
                            conf_str = conf.strip().replace("%", "")
                            conf_num = float(conf_str)
                            # 如果字符串值大于1，认为是百分比；否则是小数
                            conf_value = conf_num if conf_num > 1 else conf_num * 100
                        else:
                            conf_value = 0
                        conf_value = max(0, min(100, conf_value))  # 限制在0-100之间
                    except (ValueError, TypeError):
                        conf_value = 0
                    conf_str = f"{conf_value:.0f}%"
                    result.append(f"{SYMBOLS['vertical']} 决策信心: {conf_str}")

                # 显示各个Agent的信号
                if "agent_signals" in data:
                    result.append(
                        f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}各分析师意见:")

                    for signal_info in data["agent_signals"]:
                        agent = signal_info.get("agent_name", signal_info.get("agent", ""))
                        signal = signal_info.get("signal", "")
                        conf = signal_info.get("confidence", 1.0)

                        # 跳过空信号
                        if not agent or not signal:
                            continue

                        # 获取信号图标
                        signal_icon = STATUS_ICONS.get(signal.lower(), "")

                        # 格式化置信度（处理多种格式）
                        try:
                            if isinstance(conf, (int, float)):
                                # 如果是数字，判断范围：大于1则认为是百分比，否则是小数
                                conf_value = conf if conf > 1 else conf * 100
                            elif isinstance(conf, str):
                                conf_str = str(conf).strip().replace("%", "")
                                conf_num = float(conf_str)
                                # 如果字符串值大于1，认为是百分比；否则是小数
                                conf_value = conf_num if conf_num > 1 else conf_num * 100
                            else:
                                conf_value = 0
                            conf_value = max(0, min(100, conf_value))  # 限制在0-100之间
                        except (ValueError, TypeError):
                            conf_value = 0
                        conf_str = f"{conf_value:.0f}%"

                        # 转换agent名称和信号值为中文
                        agent_cn = FIELD_NAME_MAP.get(agent, agent)
                        signal_cn = FIELD_NAME_MAP.get(signal.lower(), signal)

                        result.append(
                            f"{SYMBOLS['vertical']}   • {agent_cn}: {signal_icon} {signal_cn} (置信度: {conf_str})")

                # 决策理由
                if "reasoning" in data:
                    reasoning = data["reasoning"]
                    result.append(
                        f"{SYMBOLS['vertical']} {SYMBOLS['section_prefix']}决策理由:")
                    if isinstance(reasoning, str):
                        # 将长文本拆分为多行，每行不超过width-4个字符
                        for i in range(0, len(reasoning), width-4):
                            line = reasoning[i:i+width-4]
                            result.append(f"{SYMBOLS['vertical']}   {line}")
            else:
                # 标准处理其他agent
                # 提取信号和置信度（如果有）
                if "signal" in data:
                    signal = data.get("signal", "")
                    # 转换信号值为中文
                    signal_cn = FIELD_NAME_MAP.get(signal.lower(), signal)
                    signal_icon = STATUS_ICONS.get(signal.lower(), "")
                    result.append(
                        f"{SYMBOLS['vertical']} 信号: {signal_icon} {signal_cn}")

                if "confidence" in data:
                    conf = data.get("confidence", "")
                    # 格式化置信度（处理多种格式）
                    try:
                        if isinstance(conf, (int, float)):
                            # 如果是数字，判断范围：大于1则认为是百分比，否则是小数
                            conf_value = conf if conf > 1 else conf * 100
                        elif isinstance(conf, str):
                            conf_str = conf.strip().replace("%", "")
                            conf_num = float(conf_str)
                            # 如果字符串值大于1，认为是百分比；否则是小数
                            conf_value = conf_num if conf_num > 1 else conf_num * 100
                        else:
                            conf_value = 0
                        conf_value = max(0, min(100, conf_value))  # 限制在0-100之间
                    except (ValueError, TypeError):
                        conf_value = 0
                    conf_str = f"{conf_value:.0f}%"
                    result.append(f"{SYMBOLS['vertical']} 置信度: {conf_str}")

            # 添加其他数据
            tree_lines = self._format_dict_as_tree(data)
            for line in tree_lines:
                result.append(f"{SYMBOLS['vertical']} {line}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                prefix = SYMBOLS["tree_last"] if i == len(
                    data) - 1 else SYMBOLS["tree_branch"]
                result.append(f"{SYMBOLS['vertical']} {prefix} {item}")
        else:
            result.append(f"{SYMBOLS['vertical']} {data}")

        # 添加底部
        result.append(
            f"{SYMBOLS['footer_left']}{SYMBOLS['border'] * (width - 2)}{SYMBOLS['footer_right']}")

        return result

    def generate_output(self) -> str:
        """生成格式化输出"""
        width = 80
        result = []

        # 添加标题
        ticker = self.metadata.get("ticker", "未知")
        title = f"股票代码 {ticker} 投资分析报告"
        result.append(SYMBOLS["border"] * width)
        result.append(f"{title:^{width}}")
        result.append(SYMBOLS["border"] * width)

        # 添加日期范围（如果有）
        if "start_date" in self.metadata and "end_date" in self.metadata:
            date_range = f"分析区间: {self.metadata['start_date']} 至 {self.metadata['end_date']}"
            result.append(f"{date_range:^{width}}")
            result.append("")

        # 按顺序添加每个agent的输出
        for agent_name in AGENT_ORDER:
            if agent_name in self.data:
                result.extend(self._format_agent_section(
                    agent_name, self.data[agent_name]))
                result.append("")  # 添加空行

        # 添加结束分隔线
        result.append(SYMBOLS["border"] * width)

        return "\n".join(result)

    def print_output(self) -> None:
        """打印格式化输出"""
        output = self.generate_output()
        # 使用INFO级别记录，确保在控制台可见
        logger.info("\n" + output)


# 创建全局实例
terminal = StructuredTerminalOutput()


def extract_agent_data(state: Dict[str, Any], agent_name: str) -> Any:
    """
    从状态中提取指定agent的数据

    Args:
        state: 工作流状态
        agent_name: agent名称

    Returns:
        提取的agent数据
    """
    # 特殊处理portfolio_management_agent
    if agent_name == "portfolio_management_agent":
        # 尝试从最后一条消息中获取数据
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
            # 尝试解析JSON
            if isinstance(content, str):
                try:
                    # 如果是JSON字符串，尝试解析
                    if content.strip().startswith('{') and content.strip().endswith('}'):
                        return json.loads(content)
                    # 如果是JSON字符串包含在其他文本中，尝试提取并解析
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end+1]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    # 如果解析失败，返回原始内容
                    return {"message": content}
            return {"message": content}

    # 首先尝试从metadata中的all_agent_reasoning获取
    metadata = state.get("metadata", {})
    all_reasoning = metadata.get("all_agent_reasoning", {})

    # 查找匹配的agent数据
    for name, data in all_reasoning.items():
        if agent_name in name:
            return data

    # 如果在all_agent_reasoning中找不到，尝试从agent_reasoning获取
    if agent_name == metadata.get("current_agent_name") and "agent_reasoning" in metadata:
        return metadata["agent_reasoning"]

    # 尝试从messages中获取
    messages = state.get("messages", [])
    for message in messages:
        if hasattr(message, "name") and message.name and agent_name in message.name:
            # 尝试解析消息内容
            try:
                if hasattr(message, "content"):
                    content = message.content
                    # 尝试解析JSON
                    if isinstance(content, str) and (content.startswith('{') or content.startswith('[')):
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    return content
            except Exception:
                pass

    # 如果都找不到，返回None
    return None


def process_final_state(state: Dict[str, Any]) -> None:
    """
    处理最终状态，提取所有agent的数据

    Args:
        state: 工作流的最终状态
    """
    # 提取元数据
    data = state.get("data", {})

    # 设置元数据
    terminal.set_metadata("ticker", data.get("ticker", "未知"))
    if "start_date" in data and "end_date" in data:
        terminal.set_metadata("start_date", data["start_date"])
        terminal.set_metadata("end_date", data["end_date"])

    # 提取每个agent的数据
    for agent_name in AGENT_ORDER:
        agent_data = extract_agent_data(state, agent_name)
        if agent_data:
            terminal.add_agent_data(agent_name, agent_data)


def print_structured_output(state: Dict[str, Any]) -> None:
    """
    处理最终状态并打印结构化输出

    Args:
        state: 工作流的最终状态
    """
    try:
        # 处理最终状态
        process_final_state(state)

        # 打印输出
        terminal.print_output()
    except Exception as e:
        logger.error(f"生成结构化输出时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
