from langchain_core.messages import HumanMessage
from src.tools.openrouter_config import get_chat_completion
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history, get_industry
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction

from datetime import datetime, timedelta
import pandas as pd

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

    logger.info(f"  📈 股票代码: {ticker}")
    logger.info(f"  📅 数据区间: {start_date} ~ {end_date}")

    # 获取价格数据并验证
    logger.info("  [1/4] 获取价格历史...")
    prices_df = get_price_history(ticker, start_date, end_date)
    if prices_df is None or prices_df.empty:
        logger.warning(f"  ⚠️ 无法获取{ticker}的价格数据，将使用空数据继续")
        prices_df = pd.DataFrame(
            columns=['close', 'open', 'high', 'low', 'volume'])
    else:
        logger.info(f"  ✅ 获取到 {len(prices_df)} 条价格记录")

    # 获取财务指标
    logger.info("  [2/4] 获取财务指标...")
    try:
        financial_metrics = get_financial_metrics(ticker)
        logger.info(f"  ✅ 财务指标获取成功")
    except Exception as e:
        logger.error(f"  ❌ 获取财务指标失败: {str(e)}")
        financial_metrics = {}

    # 获取财务报表
    logger.info("  [3/4] 获取财务报表...")
    try:
        financial_line_items = get_financial_statements(ticker)
        logger.info(f"  ✅ 财务报表获取成功")
    except Exception as e:
        logger.error(f"  ❌ 获取财务报表失败: {str(e)}")
        financial_line_items = {}

    # 获取市场数据
    logger.info("  [4/4] 获取市场数据...")
    try:
        market_data = get_market_data(ticker)
        logger.info(f"  ✅ 市场数据获取成功")
    except Exception as e:
        logger.error(f"  ❌ 获取市场数据失败: {str(e)}")
        market_data = {"market_cap": 0}

    # 获取行业信息
    logger.info("  [5/5] 获取行业信息...")
    try:
        industry = get_industry(ticker)
        if industry:
            logger.info(f"  ✅ 行业信息获取成功: {industry}")
        else:
            logger.warning(f"  ⚠️ 无法获取行业信息")
    except Exception as e:
        logger.error(f"  ❌ 获取行业信息失败: {str(e)}")
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

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    return {
        "messages": messages,
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
