import os
import json
from datetime import datetime
import akshare as ak
from src.utils.logging_config import setup_logger
# from langgraph.graph import AgentState # Changed import
# Added for alignment
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from typing import Dict, Any, List
from src.utils.api_utils import agent_endpoint  # Added for alignment
from src.tools.openrouter_config import get_chat_completion
from src.tools.news_crawler import get_stock_news
from langchain_core.messages import HumanMessage  # Added import
from src.utils.error_handler import resilient_agent

# LLM Prompt for analyzing full news data
LLM_PROMPT_MACRO_ANALYSIS = """你是一名资深的A股市场宏观分析师。请根据以下提供的沪深300指数（代码：000300）当日的**全部新闻数据**，进行深入分析并生成一份专业的宏观总结报告。

报告应包含以下几个方面：
1.  **市场情绪解读**：整体评估当前市场情绪（如：乐观、谨慎、悲观），并简述判断依据。
2.  **热点板块识别**：找出新闻中反映出的1-3个主要热点板块或主题，并说明其驱动因素。
3.  **潜在风险提示**：揭示新闻中可能隐藏的1-2个宏观层面或市场层面的潜在风险点。
4.  **政策影响分析**：如果新闻提及重要政策变动，请分析其可能对市场产生的短期和长期影响。
5.  **综合展望**：基于以上分析，对短期市场走势给出一个简明扼要的展望。

请确保分析客观、逻辑清晰，语言专业。直接返回分析报告内容，不要包含任何额外说明或客套话。

**当日新闻数据如下：**
{news_data_json_string}
"""

# 初始化 logger
logger = setup_logger('macro_news_agent')


def _is_usable_cached_summary(summary: str) -> bool:
    if not summary:
        return False
    invalid_markers = [
        "发生错误",
        "expecting value",
        "llm分析未能返回有效结果",
        "执行出错",
    ]
    lowered = summary.lower()
    return not any(marker.lower() in lowered for marker in invalid_markers)


def _resolve_analysis_date(state: AgentState) -> str:
    end_date = state.get("data", {}).get("end_date")
    if end_date:
        return end_date
    return datetime.now().strftime("%Y-%m-%d")


@resilient_agent
@agent_endpoint("macro_news_agent", "获取沪深300全量新闻并进行宏观分析，为投资决策提供市场层面的宏观环境评估")
def macro_news_agent(state: AgentState) -> Dict[str, Any]:
    """
    获取沪深300全量新闻，调用LLM进行宏观分析，并保存结果。
    该Agent独立运行，不依赖特定上游数据，结果注入AgentState。
    """
    agent_name = "macro_news_agent"
    show_workflow_status(f"{agent_name}: --- 正在执行宏观新闻 Agent ---")
    logger.info("="*50)
    logger.info("📰 [MACRO_NEWS] 开始宏观新闻分析")
    logger.info("="*50)
    symbol = "000300"  # 沪深300指数
    news_list_for_llm: List[Dict[str, str]] = []
    summary = "今日宏观新闻摘要暂不可用。"  # Default fallback summary
    retrieved_news_count = 0
    from_cache = False  # Flag to indicate if summary was loaded from cache

    analysis_date = _resolve_analysis_date(state)
    output_file_path = os.path.join("src", "data", "macro_summary.json")

    # Attempt to load from cache first
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                all_summaries = json.load(f)
            if analysis_date in all_summaries and _is_usable_cached_summary(all_summaries[analysis_date].get("summary_content", "")):
                cached_data = all_summaries[analysis_date]
                summary = cached_data["summary_content"]
                retrieved_news_count = cached_data.get(
                    "retrieved_news_count", 0)  # Get cached news count
                from_cache = True
                show_workflow_status(
                    f"{agent_name}: 从缓存加载 {analysis_date} 的宏观新闻总结。")
                show_agent_reasoning(
                    f"Loaded macro summary for {analysis_date} from cache. News count: {retrieved_news_count}", agent_name)
            elif analysis_date in all_summaries:
                logger.warning("Skipping unusable cached macro summary for %s", analysis_date)
        except json.JSONDecodeError:
            show_agent_reasoning(
                f"JSONDecodeError for {output_file_path} when trying to load cache. Will fetch fresh data.", agent_name)
            all_summaries = {}  # Reset if file is corrupt
        except Exception as e:
            show_agent_reasoning(
                f"Error loading cache from {output_file_path}: {str(e)}. Will fetch fresh data.", agent_name)
            all_summaries = {}  # Reset on other errors

    if not from_cache:
        show_workflow_status(f"{agent_name}: 缓存中未找到今日总结或缓存无效，开始获取实时新闻。")
        try:
            show_workflow_status(
                f"{agent_name}: 正在获取 {symbol} 的新闻")
            news_list_for_llm = get_stock_news(symbol, max_news=100, date=analysis_date)
            if not news_list_for_llm:
                message = f"未获取到 {symbol} 的新闻数据。"
                show_workflow_status(f"{agent_name}: {message}")
                show_agent_reasoning(
                    f"未获取到 {symbol} 的新闻。将使用无数据摘要继续。", agent_name)
                summary = "今日未获取到相关宏观新闻数据。"
            else:
                retrieved_news_count = len(news_list_for_llm)
                message = f"成功获取到 {symbol} 的 {retrieved_news_count} 条新闻数据。"
                show_workflow_status(f"{agent_name}: {message}")
                show_agent_reasoning(
                    f"成功获取 {retrieved_news_count} 条 {symbol} 的新闻。正在准备LLM分析。", agent_name)

                news_data_json_string = json.dumps(
                    news_list_for_llm, ensure_ascii=False, indent=2)
                prompt_filled = LLM_PROMPT_MACRO_ANALYSIS.format(
                    news_data_json_string=news_data_json_string)

                show_workflow_status(
                    f"{agent_name}: 正在调用LLM进行分析。")
                llm_response = get_chat_completion(
                    messages=[{"role": "user", "content": prompt_filled}],
                    max_retries=1,
                    initial_retry_delay=0.5,
                )
                summary = llm_response.strip() if llm_response else "LLM分析未能返回有效结果。"
                if _is_usable_cached_summary(summary):
                    show_workflow_status(f"{agent_name}: LLM宏观分析结果获取成功.")
                    show_agent_reasoning(
                        f"LLM分析完成。摘要(前100字符): {summary[:100]}...", agent_name)
                else:
                    show_workflow_status(f"{agent_name}: LLM未返回可用总结，使用中性摘要。")
                    summary = "今日宏观新闻数据已获取，但自动总结服务暂不可用，建议结合原始新闻人工复核。"

        except Exception as e:
            show_workflow_status(f"{agent_name}: 新闻源不可用，使用中性摘要。")
            show_agent_reasoning(
                f"启用宏观新闻获取备用方案: {str(e)}", agent_name)
            summary = "今日宏观新闻数据暂不可用，已跳过该模块并使用中性摘要。"

    # 保存总结到JSON文件 (only if not from cache and successful, or if updating existing)
    if not from_cache:  # Also save if summary was updated, even if initially from cache but e.g. re-analyzed
        show_workflow_status(
            f"{agent_name}: 正在保存摘要到 {output_file_path}")

        # Ensure all_summaries is initialized if cache loading failed or file didn't exist
        if not os.path.exists(output_file_path) or 'all_summaries' not in locals():
            all_summaries = {}
            # if file exists but all_summaries wasn't set (e.g. decode error)
            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, 'r', encoding='utf-8') as f:
                        all_summaries = json.load(f)
                except json.JSONDecodeError:
                    all_summaries = {}  # If still error, start fresh

        os.makedirs(os.path.dirname(output_file_path),
                    exist_ok=True)  # Ensure directory exists

        if _is_usable_cached_summary(summary):
            current_summary_details = {
                "summary_content": summary,
                "retrieved_news_count": retrieved_news_count,
                "last_updated": datetime.now().isoformat()
            }
            all_summaries[analysis_date] = current_summary_details

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_summaries, f, ensure_ascii=False, indent=4)
            show_workflow_status(
                f"{agent_name}: 宏观新闻总结已保存到: {output_file_path}")
        except Exception as e:
            show_workflow_status(f"{agent_name}: 保存宏观新闻总结文件失败: {e}")
            show_agent_reasoning(
                f"Failed to save summary to {output_file_path}: {str(e)}", agent_name)

    show_workflow_status(agent_name, "completed")

    new_message_content = f"宏观新闻Agent分析 {analysis_date} (是否从缓存加载={from_cache}):\\n{summary}"
    new_message = HumanMessage(content=new_message_content, name=agent_name)

    show_agent_reasoning({
        "最终结论": summary[:150] + "..." if len(summary) > 150 else summary,
        "获取新闻数": f"{retrieved_news_count}条",
        "数据来源": "缓存" if from_cache else "实时获取"
    }, agent_name)

    if not from_cache and news_list_for_llm and len(news_list_for_llm) > 0:
        for i, news in enumerate(news_list_for_llm[:10], 1):
            title = news.get('title', '无标题')[:60]
            date = news.get('date', '')
            show_agent_reasoning({
                f"新闻{i}": title + (f" ({date})" if date else "")
            }, agent_name)
        if len(news_list_for_llm) > 10:
            show_agent_reasoning({
                f"...还有{len(news_list_for_llm)-10}条": "未展示"
            }, agent_name)

    agent_details_for_metadata = {
        "summary_generated_on": analysis_date,
        "news_count_for_summary": retrieved_news_count,
        "llm_summary_preview": summary[:150] + "..." if len(summary) > 150 else summary,
        "loaded_from_cache": from_cache
    }

    logger.info("────────────────────────────────────────────────────────")
    logger.info("✅ 宏观新闻分析完成:")
    logger.info(f"  📰 新闻数量: {retrieved_news_count}")
    logger.info(f"  📊 缓存加载: {from_cache}")
    logger.info(f"  📝 摘要预览: {summary[:100] if summary else 'N/A'}...")
    logger.info("────────────────────────────────────────────────────────")

    return {
        "messages": [new_message],
        "data": {**state["data"], "macro_news_analysis_result": summary},
        "metadata": {
            **state["metadata"],
            f"{agent_name}_details": agent_details_for_metadata
        }
    }
