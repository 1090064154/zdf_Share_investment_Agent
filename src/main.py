import sys
import argparse
import json
import uuid  # Import uuid for run IDs
import threading  # Import threading for background task
import uvicorn  # Import uvicorn to run FastAPI
import traceback

from datetime import datetime, timedelta
# Removed START as it's implicit with set_entry_point
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import pandas as pd
import akshare as ak

# --- Agent Imports ---
from src.agents.valuation import valuation_agent
from src.agents.state import AgentState
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.market_data import market_data_agent
from src.agents.fundamentals import fundamentals_agent
from src.agents.researcher_bull import researcher_bull_agent
from src.agents.researcher_bear import researcher_bear_agent
from src.agents.debate_room import debate_room_agent
from src.agents.macro_analyst import macro_analyst_agent
from src.agents.macro_news_agent import macro_news_agent
# [NEW] 新增3个Agent
from src.agents.industry_cycle import industry_cycle_agent
from src.agents.institutional import institutional_agent
from src.agents.expectation_diff import expectation_diff_agent

# --- Logging and Backend Imports ---
from src.utils.output_logger import OutputLogger
from src.tools.openrouter_config import get_chat_completion
from src.utils.llm_interaction_logger import (
    log_agent_execution,
    set_global_log_storage
)
from backend.dependencies import get_log_storage
from backend.main import app as fastapi_app
from src.utils.logging_config import setup_logger

# --- Import Summary Report Generator ---
try:
    from src.utils.summary_report import print_summary_report
    from src.utils.agent_collector import store_final_state, get_enhanced_final_state
    HAS_SUMMARY_REPORT = True
except ImportError:
    HAS_SUMMARY_REPORT = False

# --- Import Structured Terminal Output ---
try:
    from src.utils.structured_terminal import print_structured_output
    HAS_STRUCTURED_OUTPUT = True
except ImportError:
    HAS_STRUCTURED_OUTPUT = False

# --- Initialize Logging ---
log_storage = get_log_storage()
set_global_log_storage(log_storage)
sys.stdout = OutputLogger()
logger = setup_logger('main_workflow')

# --- Run the Hedge Fund Workflow ---


def run_hedge_fund(run_id: str, ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, num_of_news: int = 5, show_summary: bool = False):
    print(f"\n{'='*60}")
    print(f"🚀 开始执行工作流")
    print(f"  Run ID: {run_id}")
    print(f"  股票: {ticker}")
    print(f"  区间: {start_date} ~ {end_date}")
    print(f"  现金: {portfolio.get('cash', 0):.2f} 元")
    print(f"  持仓: {portfolio.get('stock', 0)} 股")
    print(f"{'='*60}\n")
    logger.info("STEP 3: 启动 LangGraph 工作流")
    logger.info(f"  run_id={run_id}, ticker={ticker}, period={start_date}~{end_date}")
    try:
        from backend.state import api_state
        api_state.current_run_id = run_id
        print(f"--- API状态已更新 Run ID: {run_id} ---")
    except Exception as e:
        print(f"Note: Could not update API state: {str(e)}")

    initial_state = {
        "messages": [],  # 初始消息为空
        "data": {
            "ticker": ticker,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "num_of_news": num_of_news,
        },
        "metadata": {
            "show_reasoning": show_reasoning,
            "run_id": run_id,
            "show_summary": show_summary,
        }
    }

    try:
        from backend.utils.context_managers import workflow_run
        with workflow_run(run_id):
            logger.info("Invoking compiled workflow graph for run_id=%s", run_id)
            final_state = app.invoke(initial_state)
            logger.info(
                "Workflow graph completed for run_id=%s with %s messages",
                run_id,
                len(final_state.get("messages", [])),
            )
            print(f"\n{'='*60}")
            print(f"✅ 工作流执行完成 Run ID: {run_id}")
            print(f"  共执行了 {len(final_state.get('messages', []))} 个 Agent 节点")
            print(f"{'='*60}\n")

            if HAS_SUMMARY_REPORT and show_summary:
                store_final_state(final_state)
                enhanced_state = get_enhanced_final_state()
                print_summary_report(enhanced_state)

            if HAS_STRUCTURED_OUTPUT and show_reasoning:
                print_structured_output(final_state)
    except ImportError:
        logger.info(
            "workflow_run context manager unavailable; invoking workflow directly for run_id=%s",
            run_id,
        )
        final_state = app.invoke(initial_state)
        logger.info(
            "Workflow graph completed for run_id=%s with %s messages",
            run_id,
            len(final_state.get("messages", [])),
        )
        print(f"--- Finished Workflow Run ID: {run_id} ---")

        if HAS_SUMMARY_REPORT and show_summary:
            store_final_state(final_state)
            enhanced_state = get_enhanced_final_state()
            print_summary_report(enhanced_state)

        if HAS_STRUCTURED_OUTPUT and show_reasoning:
            print_structured_output(final_state)
        try:
            api_state.complete_run(run_id, "completed")
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Workflow execution failed for run_id=%s: %s", run_id, exc)
        print(f"--- 工作流执行失败 Run ID: {run_id} ---")
        print(traceback.format_exc())
        raise

    messages = final_state.get("messages", [])
    if not messages:
        logger.error("Workflow completed but no messages in final state")
        return json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.0,
            "reasoning": "工作流执行完成但无输出消息"
        })

    return messages[-1].content


# --- Define the Workflow Graph ---
# [OPTIMIZED] 按照计划文档4.1节优化工作流
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("valuation_agent", valuation_agent)
# [NEW] 新增3个Agent
workflow.add_node("industry_cycle_agent", industry_cycle_agent)
workflow.add_node("institutional_agent", institutional_agent)
workflow.add_node("expectation_diff_agent", expectation_diff_agent)
# 原有Agent
workflow.add_node("macro_news_agent", macro_news_agent)
workflow.add_node("researcher_bull_agent", researcher_bull_agent)
workflow.add_node("researcher_bear_agent", researcher_bear_agent)
workflow.add_node("debate_room_agent", debate_room_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("macro_analyst_agent", macro_analyst_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Set entry point
workflow.set_entry_point("market_data_agent")

# Level 1: market_data -> 4个基础分析Agent (并行)
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")

# Level 2: 4个基础分析 -> 3个新增Agent (并行)
# 按照计划: 技术/基本面/估值 -> industry_cycle/institutional/expectation_diff
# 这里简化为4个基础分析都完成后进入新增Agent
workflow.add_edge(
    ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent"],
    "industry_cycle_agent",
)
workflow.add_edge(
    ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent"],
    "institutional_agent",
)
workflow.add_edge(
    ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent"],
    "expectation_diff_agent",
)

# Level 3: 新增Agent + sentiment -> macro_analyst (汇合后分析宏观)
# 按照计划: industry_cycle/institutional/expectation_diff -> sentiment -> macro
# 实际简化: 3个新增Agent完成后进入sentiment
workflow.add_edge(["industry_cycle_agent", "institutional_agent", "expectation_diff_agent"], "sentiment_agent")

# sentiment -> macro_news (并行)
workflow.add_edge("sentiment_agent", "macro_news_agent")

# Level 4: macro_news -> researchers (并行)
workflow.add_edge("macro_news_agent", "researcher_bull_agent")
workflow.add_edge("macro_news_agent", "researcher_bear_agent")

# Level 5: researchers -> debate_room
workflow.add_edge(["researcher_bull_agent", "researcher_bear_agent"], "debate_room_agent")

# Level 6: debate -> risk -> macro_analyst
workflow.add_edge("debate_room_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "macro_analyst_agent")

# Level 7: macro_analyst -> portfolio_manager (最终决策)
workflow.add_edge("macro_analyst_agent", "portfolio_management_agent")

# Final node
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# --- FastAPI Background Task ---


def run_fastapi():
    print("--- 正在后台启动 FastAPI 服务器 (端口 8000) ---")
    try:
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_config=None)
    except Exception as exc:
        logger.exception("Background FastAPI server failed to start: %s", exc)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD). Defaults to 1 year before end date')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD). Defaults to yesterday')
    parser.add_argument('--show-reasoning', action='store_true',
                        help='Show reasoning from each agent')
    parser.add_argument('--num-of-news', type=int, default=20,
                        help='Number of news articles to analyze for sentiment (default: 20)')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial cash amount (default: 100,000)')
    parser.add_argument('--initial-position', type=int,
                        default=0, help='Initial stock position (default: 0)')
    parser.add_argument('--summary', action='store_true',
                        help='Show beautiful summary report at the end')
    parser.add_argument('--start-backend', action='store_true',
                        help='Start the FastAPI backend server in the background')
    args = parser.parse_args()
    logger.info("="*60)
    logger.info("STEP 0: 命令行参数解析完成")
    logger.info(f"  股票代码: {args.ticker}")
    logger.info(f"  开始日期: {args.start_date or '默认一年前'}")
    logger.info(f"  结束日期: {args.end_date or '昨天'}")
    logger.info(f"  初始资金: {args.initial_capital}")
    logger.info(f"  初始持仓: {args.initial_position}")
    logger.info(f"  新闻数量: {args.num_of_news}")
    logger.info(f"  显示推理: {args.show_reasoning}")
    logger.info("="*60)

    # Validate stock ticker format
    ticker = args.ticker.strip()
    if not ticker:
        raise ValueError("Stock ticker cannot be empty")

    # A-share stock code validation
    # Shanghai: 600000-603999, 688000-688999
    # Shenzhen: 000001-003999, 300000-300999
    if not ticker.isdigit():
        raise ValueError(f"Invalid stock ticker '{ticker}': must be numeric")

    if len(ticker) != 6:
        raise ValueError(f"Invalid stock ticker '{ticker}': must be 6 digits")

    ticker_int = int(ticker)
    is_valid = False
    exchange = "unknown"

    # Shanghai Stock Exchange
    if 600000 <= ticker_int <= 603999:
        is_valid = True
        exchange = "SSE"
    elif 688000 <= ticker_int <= 688999:
        is_valid = True
        exchange = "SSE STAR"
    # Shenzhen Stock Exchange
    elif 1 <= ticker_int <= 3999:
        is_valid = True
        exchange = "SZSE Main"
    elif 300000 <= ticker_int <= 300999:
        is_valid = True
    elif 1 <= ticker_int <= 3999:
        is_valid = True
        exchange = "SZSE ChiNext"

    if not is_valid:
        raise ValueError(f"Invalid A-share ticker '{ticker}': must be in valid range (600000-603999, 688000-688999, 000001-003999, 300000-300999)")

    logger.info(f"Validated ticker: {ticker} ({exchange} exchange)")

    if args.start_backend:
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday if not args.end_date else min(
        datetime.strptime(args.end_date, '%Y-%m-%d'), yesterday)
    if not args.start_date:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")
    if args.num_of_news < 1:
        raise ValueError("Number of news articles must be at least 1")
    if args.num_of_news > 100:
        raise ValueError("Number of news articles cannot exceed 100")

    logger.info("="*60)
    logger.info("STEP 1: 计算日期范围")
    logger.info(f"  结束日期: {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  开始日期: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"  分析天数: {(end_date - start_date).days} 天")
    logger.info("="*60)

    logger.info("="*60)
    logger.info("STEP 2: 初始化投资组合")
    logger.info(f"  现金: {args.initial_capital:.2f} 元")
    logger.info(f"  持仓: {args.initial_position} 股")
    logger.info("="*60)

    portfolio = {"cash": args.initial_capital, "stock": args.initial_position}
    main_run_id = str(uuid.uuid4())
    result = run_hedge_fund(
        run_id=main_run_id,
        ticker=args.ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news,
        show_summary=args.summary
    )

    # 解析并打印最终决策
    try:
        import json
        decision = json.loads(result)
        action_cn = {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(decision.get("action", "hold"), "持有")
        print(f"\n{'='*60}")
        print(f"🏁 最终投资决策")
        print(f"{'='*60}")
        print(f"  股票: {args.ticker}")
        print(f"  行动: {action_cn} ({decision.get('action', 'hold')})")
        print(f"  数量: {decision.get('quantity', 0)} 股")
        print(f"  置信度: {float(decision.get('confidence', 0))*100:.0f}%")
        print(f"{'='*60}")
    except:
        print("\n最终投资决策:")
        print(result)
