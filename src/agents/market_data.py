from langchain_core.messages import HumanMessage
from src.tools.openrouter_config import get_chat_completion
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history, get_industry
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent

from datetime import datetime, timedelta
import pandas as pd
import json

# 设置日志记录
logger = setup_logger('market_data_agent')


def _has_meaningful_records(value) -> bool:
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, dict):
        return any(v not in (None, 0, "", [], {}) for v in value.values())
    if isinstance(value, list):
        return any(_has_meaningful_records(item) for item in value)
    return bool(value)


@resilient_agent(critical=True)
@agent_endpoint("market_data", "市场数据收集，负责获取股价历史、财务指标和市场信息")
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    show_workflow_status("市场数据Agent")
    logger.info("="*50)
    logger.info("📊 [MARKET_DATA] 开始收集市场数据")
    logger.info("="*50)

    show_reasoning = state["metadata"]["show_reasoning"]
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data["end_date"] or yesterday.strftime('%Y-%m-%d')

    # Ensure end_date is not in the future
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    if end_date_obj > yesterday:
        end_date = yesterday.strftime('%Y-%m-%d')
        end_date_obj = yesterday

    if not data["start_date"]:
        # Calculate 1 year before end_date
        start_date = end_date_obj - timedelta(days=365)  # 默认获取一年的数据
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Get all required data
    ticker = data["ticker"]

    show_agent_reasoning({"股票": ticker, "数据区间": f"{start_date} ~ {end_date}"}, "市场数据Agent")

    # 获取价格数据并验证
    prices_df = get_price_history(ticker, start_date, end_date)
    if prices_df is None or prices_df.empty:
        show_agent_reasoning({"错误": f"无法获取{ticker}的价格数据"}, "市场数据Agent")
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])
    else:
        latest_price = prices_df.iloc[-1]['close'] if len(prices_df) > 0 else 0
        show_agent_reasoning({"价格记录": f"{len(prices_df)}条", "最新价": f"{latest_price:.2f}"}, "市场数据Agent")

    # 获取财务指标
    try:
        financial_metrics_result = get_financial_metrics(ticker)
        if isinstance(financial_metrics_result, list) and len(financial_metrics_result) > 0:
            financial_metrics = financial_metrics_result[0]
            has_metrics = True
        else:
            financial_metrics = {}
            has_metrics = False
        market_cap_val = financial_metrics.get('market_cap', 0) if financial_metrics else 0
        metrics_status = "已获取" if has_metrics else "未获取"
        show_agent_reasoning({"财务指标": metrics_status, "市值": f"{market_cap_val/1e8:.1f}亿" if market_cap_val else "无数据"}, "市场数据Agent")
    except Exception as e:
        financial_metrics = {}
        show_agent_reasoning({"财务指标": "获取异常", "错误": str(e)[:30]}, "市场数据Agent")

    # 获取财务报表
    try:
        financial_line_items = get_financial_statements(ticker)
        has_statements = bool(financial_line_items and len(financial_line_items) > 0)
        statements_status = "已获取" if has_statements else "未获取"
        show_agent_reasoning({"财务报表": statements_status, "报告期数": len(financial_line_items) if has_statements else 0}, "市场数据Agent")
    except Exception as e:
        financial_line_items = {}
        show_agent_reasoning({"财务报表": "获取异常"}, "市场数据Agent")

    # 获取市场数据
    try:
        market_data = get_market_data(ticker)
    except Exception as e:
        market_data = {"market_cap": 0}

    # 获取行业信息
    try:
        industry = get_industry(ticker)
        show_agent_reasoning({"行业": industry}, "市场数据Agent")
    except Exception as e:
        industry = ""

    # 确保数据格式正确
    if not isinstance(prices_df, pd.DataFrame):
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])

    # 转换价格数据为字典格式
    prices_dict = prices_df.to_dict('records')

    # 保存推理信息到metadata供API使用
    market_data_summary = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            "price_history": len(prices_dict) > 0,
            "financial_metrics": _has_meaningful_records(financial_metrics),
            "financial_statements": _has_meaningful_records(financial_line_items),
            "market_data": _has_meaningful_records(market_data)
        },
        "summary": f"为{ticker}收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息"
    }

    show_agent_reasoning({
        "价格记录": f"{len(prices_dict)}条",
        "财务指标": "✓" if _has_meaningful_records(financial_metrics) else "✗",
        "财务报表": "✓" if _has_meaningful_records(financial_line_items) else "✗",
        "行业": industry
    }, "市场数据Agent")

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    show_workflow_status("市场数据Agent", "completed")

    # 构建结构化输出
    latest_price = prices_df.iloc[-1]['close'] if len(prices_df) > 0 else 0
    message_content = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "prices_count": len(prices_dict),
        "latest_price": round(float(latest_price), 2) if latest_price else None,
        "industry": industry,
        "market_cap": market_data.get("market_cap", 0),
        "has_financial_metrics": _has_meaningful_records(financial_metrics),
        "has_financial_statements": _has_meaningful_records(financial_line_items),
        "summary": f"为{ticker}收集了{start_date}至{end_date}的数据：{len(prices_dict)}条价格记录、行业{industry}、市值{market_data.get('market_cap', 0)/1e8:.1f}亿"
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="market_data_agent",
    )

    return {
        "messages": messages + [message],
        "data": {
            **data,
            "prices": prices_dict,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "financial_line_items": financial_line_items,
            "market_cap": market_data.get("market_cap", 0),
            "market_data": market_data,
            "industry": industry,
        },
        "metadata": state["metadata"],
    }
