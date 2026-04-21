from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import get_stock_news, get_news_sentiment
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
from datetime import datetime, timedelta

# 设置日志记录
logger = setup_logger('sentiment_agent')


@agent_endpoint("sentiment", "情感分析师，分析市场新闻和社交媒体情绪")
def sentiment_agent(state: AgentState):
    """Responsible for sentiment analysis"""
    show_workflow_status("情绪分析师")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"正在分析股票: {symbol}")
    # 从命令行参数获取新闻数量，默认为20条
    num_of_news = data.get("num_of_news", 20)

    # 获取 end_date 并传递给 get_stock_news
    end_date = data.get("end_date")  # 从 run_hedge_fund 传递来的 end_date

    # 获取新闻数据并分析情感，添加 date 参数
    try:
        news_list = get_stock_news(symbol, max_news=num_of_news, date=end_date)
    except Exception as exc:
        logger.exception("获取新闻失败，情感分析将回退为空新闻集: %s", exc)
        news_list = []

    # 过滤7天内的新闻（只对有publish_time字段的新闻进行过滤）
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_news = []
    for news in news_list:
        if 'publish_time' in news:
            try:
                news_date = datetime.strptime(
                    news['publish_time'], '%Y-%m-%d %H:%M:%S')
                if news_date > cutoff_date:
                    recent_news.append(news)
            except ValueError:
                # 如果时间格式无法解析，默认包含这条新闻
                recent_news.append(news)
        else:
            # 如果没有publish_time字段，默认包含这条新闻
            recent_news.append(news)

    try:
        sentiment_score = get_news_sentiment(recent_news, num_of_news=num_of_news)
    except Exception as exc:
        logger.exception("新闻情感分析失败，回退为中性: %s", exc)
        sentiment_score = 0.0

    # 根据情感分数生成交易信号和置信度
    if not recent_news:
        signal = "neutral"
        confidence = "0%"
    elif sentiment_score >= 0.5:
        signal = "bullish"
        # 置信度基于情感强度，范围 50%-100%
        conf_value = 50 + (sentiment_score - 0.5) * 100  # 0.5->50%, 1.0->100%
        confidence = f"{round(conf_value)}%"
    elif sentiment_score <= -0.5:
        signal = "bearish"
        # 置信度基于情感强度，范围 50%-100%
        conf_value = 50 + (abs(sentiment_score) - 0.5) * 100  # -0.5->50%, -1.0->100%
        confidence = f"{round(conf_value)}%"
    else:
        signal = "neutral"
        # 中性信号时，置信度基于新闻数量和情感分数偏离中性的程度
        # 新闻越多，置信度越高；情感越接近0，对"中性"判断越确定
        news_factor = min(len(recent_news) / 10, 1.0) * 30  # 最多30%
        deviation_factor = (0.5 - abs(sentiment_score)) * 100  # 越接近0越高，最多50%
        conf_value = min(20 + news_factor + deviation_factor, 80)  # 基础20%，最高80%
        confidence = f"{round(conf_value)}%"

    # 生成分析结果
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": f"基于{len(recent_news)}篇近期新闻，情绪分数: {sentiment_score:.2f}"
    }

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "情绪分析Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    show_workflow_status("情绪分析师", "completed")
    # logger.info(
    # f"--- DEBUG: sentiment_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "sentiment_analysis": sentiment_score
        },
        "metadata": state["metadata"],
    }
