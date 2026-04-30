from langchain_core.messages import HumanMessage
from src.tools.openrouter_config import get_chat_completion
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.tools.api import get_financial_metrics, get_financial_statements, get_market_data, get_price_history, get_industry, get_stock_name
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
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
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
        start_date = end_date_obj - timedelta(days=365)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Get all required data
    ticker = data["ticker"]

    show_agent_reasoning({"股票": ticker, "数据区间": f"{start_date} ~ {end_date}"}, "市场数据Agent")

    # ========== 优化1: 并行获取多个数据源 ==========
    logger.info("🚀 并行获取多个数据源...")
    
    def fetch_price_data():
        return get_price_history(ticker, start_date, end_date)
    
    def fetch_financial_metrics():
        try:
            result = get_financial_metrics(ticker)
            if isinstance(result, list) and len(result) > 0:
                return result[0], True
            return {}, False
        except Exception as e:
            logger.warning(f"财务指标获取失败: {e}")
            return {}, False
    
    def fetch_financial_statements():
        try:
            result = get_financial_statements(ticker)
            return result if result and len(result) > 0 else []
        except Exception as e:
            logger.warning(f"财务报表获取失败: {e}")
            return []
    
    def fetch_market_data():
        try:
            return get_market_data(ticker)
        except Exception as e:
            logger.warning(f"市场数据获取失败: {e}")
            return {"market_cap": 0}
    
    def fetch_industry():
        try:
            return get_industry(ticker)
        except Exception as e:
            logger.warning(f"行业获取失败: {e}")
            return ""
    
    def fetch_stock_name():
        try:
            return get_stock_name(ticker)
        except Exception as e:
            logger.warning(f"股票名称获取失败: {e}")
            return ""
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_price = executor.submit(fetch_price_data)
        future_financial_metrics = executor.submit(fetch_financial_metrics)
        future_financial_statements = executor.submit(fetch_financial_statements)
        future_market_data = executor.submit(fetch_market_data)
        future_industry = executor.submit(fetch_industry)
        future_stock_name = executor.submit(fetch_stock_name)
        
        # 等待所有结果
        prices_df = future_price.result()
        financial_metrics, has_metrics = future_financial_metrics.result()
        financial_line_items = future_financial_statements.result()
        market_data = future_market_data.result()
        industry = future_industry.result()
        stock_name = future_stock_name.result()
    
    logger.info("✅ 并行获取完成")

    # ========== 优化2: 数据有效性校验 ==========
    def validate_data_quality():
        """校验数据质量，返回有效数据标识"""
        quality = {
            "price_valid": False,
            "financial_valid": False,
            "market_cap_valid": False,
            "industry_valid": False
        }
        
        # 价格有效性
        if prices_df is not None and not prices_df.empty and len(prices_df) > 30:
            quality["price_valid"] = True
            # 计算52周高低点
            if 'high' in prices_df.columns and 'low' in prices_df.columns:
                prices_df['52w_high'] = prices_df['high'].tail(252).max()
                prices_df['52w_low'] = prices_df['low'].tail(252).min()
        
        # 财务有效性：市值>0 或 其他关键指标有效
        if financial_metrics:
            market_cap = financial_metrics.get('market_cap', 0)
            if market_cap and market_cap > 0:
                quality["financial_valid"] = True
                quality["market_cap_valid"] = True
        
        # 行业有效性
        if industry and isinstance(industry, str) and len(industry) > 0:
            quality["industry_valid"] = True
            
        return quality
    
    data_quality = validate_data_quality()

    # 确保数据格式正确
    if prices_df is None or prices_df.empty:
        show_agent_reasoning({"错误": f"无法获取{ticker}的价格数据"}, "market_dataAgent")
        prices_df = pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])
    
    # 计算实时指标（如果价格数据有效）
    latest_price = 0
    price_change_pct = 0
    week_52_high = None
    week_52_low = None
    current_position = None
    
    if data_quality["price_valid"] and len(prices_df) > 0:
        latest_price = prices_df.iloc[-1]['close']
        
        # 计算涨跌幅
        if len(prices_df) >= 2:
            prev_price = prices_df.iloc[-2]['close']
            if prev_price > 0:
                price_change_pct = ((latest_price - prev_price) / prev_price) * 100
        
        # 52周高低点
        week_52_high = prices_df['52w_high'].iloc[0] if '52w_high' in prices_df.columns and len(prices_df) > 0 else None
        week_52_low = prices_df['52w_low'].iloc[0] if '52w_low' in prices_df.columns and len(prices_df) > 0 else None
        
        # 计算当前价格位置
        if week_52_high is not None and week_52_low is not None and week_52_high > week_52_low:
            current_position = ((latest_price - week_52_low) / (week_52_high - week_52_low)) * 100

    # 显示价格信息
    show_agent_reasoning({
        "价格记录": f"{len(prices_df)}条",
        "最新价": f"{latest_price:.2f}",
        "涨跌幅": f"{price_change_pct:+.2f}%" if latest_price > 0 else "无",
        "52周高位": f"{week_52_high:.2f}" if week_52_high else "-",
        "52周低位": f"{week_52_low:.2f}" if week_52_low else "-",
        "价格位置": f"{current_position:.1f}%" if current_position else "-"
    }, "Market Data Agent")

    # 显示财务信息
    market_cap_val = financial_metrics.get('market_cap', 0) if financial_metrics else 0
    metrics_status = "✓有效" if data_quality["financial_valid"] else "✗无效"
    show_agent_reasoning({
        "财务指标": metrics_status,
        "市值": f"{market_cap_val/1e8:.1f}亿" if market_cap_val else "无数据"
    }, "Market Data Agent")

    # 显示财务报表
    has_statements = bool(financial_line_items and len(financial_line_items) > 0)
    statements_status = "✓有效" if has_statements else "✗无效"
    show_agent_reasoning({"财务报表": statements_status}, "Market Data Agent")

    # 显示行业
    show_agent_reasoning({"行业": industry if data_quality["industry_valid"] else "未获取"}, "Market Data Agent")

    # 转换价格数据为字典格式
    prices_dict = prices_df.to_dict('records')

    # 保存推理信息到metadata供API使用（复用stock_name）
    market_data_summary = {
        "ticker": ticker,
        "stock_name": stock_name,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            "price_history": data_quality["price_valid"],
            "financial_metrics": data_quality["financial_valid"],
            "financial_statements": has_statements,
            "market_data": data_quality["market_cap_valid"]
        },
        "data_quality": data_quality,
        "has_financial_metrics": data_quality["financial_valid"],
        "has_financial_statements": has_statements,
        "realtime": {
            "latest_price": round(float(latest_price), 2) if latest_price else None,
            "change_pct": round(price_change_pct, 2),
            "week_52_high": round(float(week_52_high), 2) if week_52_high else None,
            "week_52_low": round(float(week_52_low), 2) if week_52_low else None,
            "current_position": round(current_position, 1) if current_position else None
        },
        "summary": f"{stock_name}({ticker})收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息"
    }

    show_agent_reasoning({
        "股票": f"{stock_name}({ticker})" if stock_name else ticker,
        "数据质量": f"价格{'✓' if data_quality['price_valid'] else '✗'} 财务{'✓' if data_quality['financial_valid'] else '✗'} 行业{'✓' if data_quality['industry_valid'] else '✗'}",
        "has_financial_metrics": data_quality["financial_valid"],
        "has_financial_statements": has_statements
    }, "Market Data Agent")

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    show_workflow_complete(
        "Market Data Agent",
        signal="neutral",
        confidence=1.0,
        details=market_data_summary,
        message=f"市场数据收集完成：{len(prices_dict)}条价格，行业{industry}，数据质量{'良好' if data_quality['price_valid'] else '部分缺失'}"
    )

    # 构建结构化输出（复用stock_name）
    message_content = {
        "ticker": ticker,
        "stock_name": stock_name,
        "start_date": start_date,
        "end_date": end_date,
        "prices_count": len(prices_dict),
        "latest_price": round(float(latest_price), 2) if latest_price else None,
        "change_pct": round(price_change_pct, 2),
        "week_52_high": round(float(week_52_high), 2) if week_52_high else None,
        "week_52_low": round(float(week_52_low), 2) if week_52_low else None,
        "current_position": round(current_position, 1) if current_position else None,
        "industry": industry,
        "market_cap": market_data.get("market_cap", 0),
        "has_financial_metrics": data_quality["financial_valid"],
        "has_financial_statements": has_statements,
        "data_quality": data_quality,
        "summary": f"{stock_name}({ticker})收集了{start_date}至{end_date}的数据：{len(prices_dict)}条价格记录、行业{industry}、市值{market_data.get('market_cap', 0)/1e8:.1f}亿"
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
            "realtime": {
                "latest_price": round(float(latest_price), 2) if latest_price else None,
                "change_pct": round(price_change_pct, 2),
                "week_52_high": round(float(week_52_high), 2) if week_52_high else None,
                "week_52_low": round(float(week_52_low), 2) if week_52_low else None,
                "current_position": round(current_position, 1) if current_position else None
            }
        },
        "metadata": state["metadata"],
    }
