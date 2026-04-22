from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.news_crawler import get_stock_news, get_news_sentiment
from src.tools.guba_crawler import get_guba_sentiment
from src.tools.quant_sentiment import get_quant_sentiment
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
from datetime import datetime, timedelta

# 设置日志记录
logger = setup_logger('sentiment_agent')


def _resolve_news_window_end(end_date: str | None) -> datetime:
    if not end_date:
        return datetime.now()
    return datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)


def _get_north_money_data(ticker: str, days: int = 5) -> dict:
    """
    [OPTIMIZED] 获取北向资金数据
    注意：需要真实的数据源接口，此处为模拟实现
    """
    try:
        import akshare as ak
        # 尝试使用akshare获取北向资金数据
        # 注意：akshare的接口可能需要单独安装
        # 这里使用try-except包裹，如果失败则返回空数据
        try:
            # 北向资金持股数据
            north_df = ak.stock_hsgt_individual_em(symbol=ticker)
            if north_df is not None and len(north_df) > 0:
                # 获取最近几天的资金流向
                recent = north_df.head(days)
                net_flows = recent['资金流向'].sum() if '资金流向' in recent.columns else 0
                avg_flow = net_flows / days if days > 0 else 0

                # 判断方向
                if avg_flow > 0:
                    signal = 'bullish'
                    confidence = min(0.3 + abs(avg_flow) / 1e8 * 0.1, 0.8)
                elif avg_flow < 0:
                    signal = 'bearish'
                    confidence = min(0.3 + abs(avg_flow) / 1e8 * 0.1, 0.8)
                else:
                    signal = 'neutral'
                    confidence = 0.3

                return {
                    'signal': signal,
                    'confidence': confidence,
                    'net_flow': avg_flow,
                    'reason': f"北向资金{avg_flow/1e8:.2f}亿"
                }
        except Exception as e:
            logger.debug(f"北向资金数据获取失败: {e}")

    except ImportError:
        pass

    # 返回空数据
    return {
        'signal': 'neutral',
        'confidence': 0,
        'net_flow': 0,
        'reason': '无北向资金数据'
    }


def _reverse_guba_signal(guba_score: float, post_count: int) -> dict:
    """
    [OPTIMIZED] 股吧情绪反向使用
    散户买入往往是卖点，散户卖出往往是买点
    """
    if post_count == 0:
        return {
            'reversed_signal': 0,
            'confidence_boost': 0,
            'reason': '无股吧数据'
        }

    # 极端值检测
    if guba_score < -0.5:  # 极度悲观
        reversed_signal = -guba_score * 1.5  # 反向
        confidence_boost = 0.2
        reason = "股吧极度悲观，可能见底"
    elif guba_score > 0.5:  # 极度乐观
        reversed_signal = -guba_score * 1.5  # 反向
        confidence_boost = 0.2
        reason = "股吧极度乐观，可能见顶"
    else:
        # 非极端值使用原信号但降低权重
        reversed_signal = guba_score * 0.5
        confidence_boost = 0
        reason = "股吧情绪处于中性区间"

    # 热度加成：热度越高，反向信号越可靠
    if post_count > 100:
        confidence_boost += 0.1

    return {
        'reversed_signal': reversed_signal,
        'confidence_boost': confidence_boost,
        'reason': reason
    }


def _calculate_combined_sentiment(
    news_score: float,
    guba_result: dict,
    quant_result: dict,
    north_result: dict,
    news_count: int
) -> dict:
    """
    [OPTIMIZED] 计算综合情绪分数，整合新闻、股吧、量化指标、北向资金
    权重分配:
    - 新闻情绪: 35%
    - 股吧情绪: 15% (反向使用)
    - 量化指标: 35%
    - 北向资金: 15%

    Returns:
        {
            'combined_score': 综合分数,
            'signal': 交易信号,
            'confidence': 置信度,
            'components': 各数据源详情,
            'weights': 权重分配
        }
    """
    logger.info("[综合情绪] 开始计算综合情绪分数")
    logger.info(f"[综合情绪] 输入: news_score={news_score:.3f}, guba_score={guba_result.get('score', 0):.3f}, quant_score={quant_result.get('score', 0):.3f}, north_confidence={north_result.get('confidence', 0):.3f}")

    # [OPTIMIZED] 数据源权重调整
    weights = {
        'news': 0.35,   # 新闻保持
        'guba': 0.15,  # 股吧降低+反向
        'quant': 0.35,  # 量化提升
        'north': 0.15   # 新增北向资金
    }

    # 新闻情绪分数 (-1 到 1)
    news_normalized = news_score

    # [OPTIMIZED] 股吧情绪反向使用
    guba_score = guba_result.get('score', 0)
    post_count = guba_result.get('post_count', 0)
    guba_reverse = _reverse_guba_signal(guba_score, post_count)
    guba_adjusted = guba_reverse['reversed_signal']
    logger.info(f"[综合情绪] 股吧情绪: 原始={guba_score:.3f}, 反向调整={guba_reverse['confidence_boost']:.3f}, 调整后={guba_adjusted:.3f}, 原因={guba_reverse['reason']}")

    # 量化情绪分数
    quant_score = quant_result.get('score', 0)

    # [OPTIMIZED] 北向资金分数转换（0-1范围）
    north_score = 0.5
    if north_result.get('signal') == 'bullish':
        north_score = 0.5 + north_result.get('confidence', 0) * 0.5
    elif north_result.get('signal') == 'bearish':
        north_score = 0.5 - north_result.get('confidence', 0) * 0.5

    logger.info(f"[综合情绪] 北向资金: {north_result.get('reason', 'N/A')}")

    # 量化情绪分数
    quant_score = quant_result.get('score', 0)

    # 计算加权综合分数
    combined_score = (
        news_normalized * weights['news'] +
        guba_adjusted * weights['guba'] +
        quant_score * weights['quant']
    )
    logger.info(f"[综合情绪] 初始综合分数: {combined_score:.3f}")

    # [OPTIMIZED] 根据数据可用性动态调整权重，包含北向资金
    actual_weights = weights.copy()
    has_north_data = north_result.get('confidence', 0) > 0

    if news_count == 0:
        # 无新闻时，提高其他数据源权重
        actual_weights['news'] = 0
        if has_north_data:
            actual_weights['guba'] = 0.30
            actual_weights['quant'] = 0.35
            actual_weights['north'] = 0.35
            combined_score = guba_adjusted * 0.30 + quant_score * 0.35 + north_score * 0.35
            logger.info(f"[综合情绪] 无新闻数据，调整权重: guba=30%, quant=35%, north=35%")
        else:
            actual_weights['guba'] = 0.40
            actual_weights['quant'] = 0.60
            actual_weights['north'] = 0
            combined_score = guba_adjusted * 0.40 + quant_score * 0.60
            logger.info(f"[综合情绪] 无新闻数据且无北向，调整权重: guba=40%, quant=60%")

    if guba_result.get('post_count', 0) == 0:
        # 无股吧数据时，提高新闻和量化/北向权重
        actual_weights['guba'] = 0
        if has_north_data:
            actual_weights['news'] = 0.40
            actual_weights['quant'] = 0.30
            actual_weights['north'] = 0.30
            combined_score = news_normalized * 0.40 + quant_score * 0.30 + north_score * 0.30
            logger.info(f"[综合情绪] 无股吧数据，调整权重: news=40%, quant=30%, north=30%")
        else:
            actual_weights['news'] = 0.60
            actual_weights['quant'] = 0.40
            actual_weights['north'] = 0
            combined_score = news_normalized * 0.60 + quant_score * 0.40
            logger.info(f"[综合情绪] 无股吧数据且无北向，调整权重: news=60%, quant=40%")

    if not has_north_data:
        # 无北向数据时
        actual_weights['north'] = 0
        # 重新分配权重给其他数据源
        total_other = actual_weights['news'] + actual_weights['guba'] + actual_weights['quant']
        if total_other > 0:
            scale = sum(weights.values()) / total_other
            actual_weights['news'] *= scale
            actual_weights['guba'] *= scale
            actual_weights['quant'] *= scale

    # 重新归一化权重
    total_weight = sum(actual_weights.values())
    if total_weight > 0:
        combined_score = combined_score / total_weight * sum(weights.values())

    logger.info(f"[综合情绪] 最终权重: news={actual_weights['news']:.2f}, guba={actual_weights['guba']:.2f}, quant={actual_weights['quant']:.2f}, north={actual_weights['north']:.2f}")

    # 生成交易信号
    if combined_score >= 0.3:
        signal = "bullish"
    elif combined_score <= -0.3:
        signal = "bearish"
    else:
        signal = "neutral"
    logger.info(f"[综合情绪] 交易信号判定: combined_score={combined_score:.3f} -> signal={signal}")

    # 计算置信度
    confidence = _calculate_confidence(
        combined_score,
        news_count,
        guba_result,
        quant_result,
        actual_weights
    )

    # 构建各数据源详情
    components = {
        'news': {
            'score': round(news_score, 3),
            'weight': round(actual_weights['news'], 2),
            'count': news_count,
            'contribution': round(news_normalized * actual_weights['news'], 3)
        },
        'guba': {
            'score': round(guba_score, 3),
            'reverse_signal': round(guba_reverse.get('reversed_signal', 0), 3),
            'adjusted_score': round(guba_adjusted, 3),
            'weight': round(actual_weights['guba'], 2),
            'post_count': guba_result.get('post_count', 0),
            'bull_ratio': round(guba_result.get('bull_ratio', 0), 2),
            'bear_ratio': round(guba_result.get('bear_ratio', 0), 2),
            'heat': round(guba_result.get('heat', 0), 2),
            'contribution': round(guba_adjusted * actual_weights['guba'], 3),
            'reverse_reason': guba_reverse.get('reason', '')
        },
        'quant': {
            'score': round(quant_score, 3),
            'weight': round(actual_weights['quant'], 2),
            'rating': quant_result.get('rating', {}).get('current_rating', 50),
            'rating_trend': quant_result.get('rating', {}).get('rating_trend', 'stable'),
            'institution': quant_result.get('institution', {}).get('current_participation', 50),
            'focus': quant_result.get('focus', {}).get('current_focus', 50),
            'contribution': round(quant_score * actual_weights['quant'], 3)
        },
        'north_money': {
            'signal': north_result.get('signal', 'neutral'),
            'score': round(north_score, 3),
            'weight': round(actual_weights['north'], 2),
            'confidence': round(north_result.get('confidence', 0), 2),
            'net_flow': north_result.get('net_flow', 0),
            'reason': north_result.get('reason', ''),
            'contribution': round(north_score * actual_weights['north'], 3)
        }
    }

    logger.info(f"[综合情绪] 各数据源贡献: news={components['news']['contribution']:.3f}, guba={components['guba']['contribution']:.3f}, quant={components['quant']['contribution']:.3f}, north={components['north_money']['contribution']:.3f}")
    logger.info(f"[综合情绪] 最终结果: combined_score={combined_score:.3f}, signal={signal}, confidence={confidence}")

    return {
        'combined_score': round(combined_score, 3),
        'signal': signal,
        'confidence': confidence,
        'components': components,
        'weights': actual_weights
    }


def _calculate_confidence(
    combined_score: float,
    news_count: int,
    guba_result: dict,
    quant_result: dict,
    weights: dict
) -> str:
    """
    计算置信度

    考虑因素:
    1. 数据源数量和质量
    2. 各数据源的一致性
    3. 情绪强度
    """
    logger.debug(f"[置信度] 开始计算: combined_score={combined_score:.3f}")

    # 基础置信度
    base_confidence = 30

    # 数据源数量加分
    data_source_score = 0
    if news_count > 0:
        data_source_score += 15
    if news_count >= 10:
        data_source_score += 10
    if guba_result.get('post_count', 0) > 0:
        data_source_score += 15
    if guba_result.get('post_count', 0) >= 20:
        data_source_score += 5
    if quant_result.get('score', 0) != 0:
        data_source_score += 15
    logger.debug(f"[置信度] 数据源加分: {data_source_score}")

    # 情绪强度加分
    intensity_score = abs(combined_score) * 30
    logger.debug(f"[置信度] 情绪强度加分: {intensity_score:.2f}")

    # 一致性加分 (各数据源方向一致时加分)
    news_sign = combined_score  # 使用综合分数作为参考
    guba_sign = guba_result.get('score', 0)
    quant_sign = quant_result.get('score', 0)

    consistency_score = 0
    if news_count > 0 and guba_result.get('post_count', 0) > 0:
        if (news_sign > 0 and guba_sign > 0) or (news_sign < 0 and guba_sign < 0):
            consistency_score += 10
    if news_count > 0 and quant_result.get('score', 0) != 0:
        if (news_sign > 0 and quant_sign > 0) or (news_sign < 0 and quant_sign < 0):
            consistency_score += 10
    logger.debug(f"[置信度] 一致性加分: {consistency_score}")

    # 计算最终置信度
    final_confidence = min(95, base_confidence + data_source_score + intensity_score + consistency_score)
    logger.info(f"[置信度] 最终置信度: {final_confidence:.0f}% (基础={base_confidence}, 数据源={data_source_score}, 强度={intensity_score:.2f}, 一致性={consistency_score})")

    return f"{round(final_confidence)}%"


def _format_reasoning(result: dict, news_count: int) -> str:
    """
    格式化推理说明，展示各数据源比例
    """
    components = result['components']
    weights = result['weights']

    # 构建推理说明
    reasoning_parts = []

    # 综合结果
    reasoning_parts.append(
        f"综合情绪分数: {result['combined_score']:.2f} → 信号: {result['signal']} (置信度: {result['confidence']})"
    )

    # 数据源权重说明
    reasoning_parts.append("\n【数据源权重分配】")
    for source, weight in weights.items():
        if weight > 0:
            reasoning_parts.append(f"  • {source}: {weight*100:.0f}%")

    # 各数据源详情
    reasoning_parts.append("\n【各数据源分析】")

    # 新闻情绪
    news_comp = components['news']
    if news_comp['count'] > 0:
        reasoning_parts.append(
            f"  新闻情绪: 分数={news_comp['score']:.2f}, "
            f"权重={news_comp['weight']*100:.0f}%, "
            f"贡献={news_comp['contribution']:.2f}, "
            f"新闻数={news_comp['count']}"
        )
    else:
        reasoning_parts.append("  新闻情绪: 无数据")

    # 股吧情绪
    guba_comp = components['guba']
    if guba_comp['post_count'] > 0:
        reasoning_parts.append(
            f"  股吧情绪: 分数={guba_comp['score']:.2f}, "
            f"反向信号={guba_comp['reverse_signal']:.2f}, "
            f"调整后={guba_comp['adjusted_score']:.2f}, "
            f"权重={guba_comp['weight']*100:.0f}%, "
            f"贡献={guba_comp['contribution']:.2f}"
        )
        reasoning_parts.append(
            f"    (帖子数={guba_comp['post_count']}, "
            f"看多={guba_comp['bull_ratio']*100:.0f}%, "
            f"看空={guba_comp['bear_ratio']*100:.0f}%, "
            f"热度={guba_comp['heat']:.1f})"
        )
    else:
        reasoning_parts.append("  股吧情绪: 无数据")

    # 量化指标
    quant_comp = components['quant']
    if quant_comp['score'] != 0:
        reasoning_parts.append(
            f"  量化指标: 分数={quant_comp['score']:.2f}, "
            f"权重={quant_comp['weight']*100:.0f}%, "
            f"贡献={quant_comp['contribution']:.2f}"
        )
        reasoning_parts.append(
            f"    (评分={quant_comp['rating']:.1f} 趋势={quant_comp['rating_trend']}, "
            f"机构参与度={quant_comp['institution']:.1f}, "
            f"关注度={quant_comp['focus']:.1f})"
        )
    else:
        reasoning_parts.append("  量化指标: 无数据")

    return "\n".join(reasoning_parts)


@agent_endpoint("sentiment", "情感分析师，综合分析新闻、股吧、量化情绪指标")
def sentiment_agent(state: AgentState):
    """Responsible for sentiment analysis - 整合多数据源"""
    logger.info("=" * 50)
    logger.info("[情绪分析Agent] 开始执行")
    logger.info("=" * 50)

    show_workflow_status("情绪分析师")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"[情绪分析Agent] 股票代码: {symbol}")

    # 从命令行参数获取新闻数量，默认为20条
    num_of_news = data.get("num_of_news", 20)
    logger.info(f"[情绪分析Agent] 新闻数量上限: {num_of_news}")

    # 获取 end_date 并传递给 get_stock_news
    end_date = data.get("end_date")
    if end_date:
        logger.info(f"[情绪分析Agent] 截止日期: {end_date}")

    # ========== 1. 获取新闻数据并分析情感 ==========
    logger.info("-" * 30)
    logger.info("[步骤1] 获取新闻数据")
    logger.info("-" * 30)

    try:
        news_list = get_stock_news(symbol, max_news=num_of_news, date=end_date)
        logger.info(f"[新闻] 获取到 {len(news_list)} 条新闻")
    except Exception as exc:
        logger.exception("[新闻] 获取失败: %s", exc)
        news_list = []

    # 过滤7天内的新闻
    reference_time = _resolve_news_window_end(end_date)
    cutoff_date = reference_time - timedelta(days=7)
    logger.info(f"[新闻] 时间窗口: {cutoff_date.strftime('%Y-%m-%d')} ~ {reference_time.strftime('%Y-%m-%d')}")

    recent_news = []
    for news in news_list:
        if 'publish_time' in news:
            try:
                news_date = datetime.strptime(
                    news['publish_time'], '%Y-%m-%d %H:%M:%S')
                if cutoff_date < news_date <= reference_time:
                    recent_news.append(news)
            except ValueError:
                recent_news.append(news)
        else:
            recent_news.append(news)

    logger.info(f"[新闻] 过滤后近期新闻: {len(recent_news)} 条")

    try:
        logger.info("[新闻情绪] 开始分析...")
        news_sentiment_score = get_news_sentiment(recent_news, num_of_news=num_of_news)
        logger.info(f"[新闻情绪] 分析完成: score={news_sentiment_score:.3f}")
    except Exception as exc:
        logger.exception("[新闻情绪] 分析失败: %s", exc)
        news_sentiment_score = 0.0

    # ========== 2. 获取股吧情绪 ==========
    logger.info("-" * 30)
    logger.info("[步骤2] 获取股吧情绪")
    logger.info("-" * 30)

    try:
        guba_result = get_guba_sentiment(symbol, max_posts=30)
        logger.info(f"[股吧情绪] 获取完成: score={guba_result.get('score', 0):.3f}, posts={guba_result.get('post_count', 0)}")
    except Exception as exc:
        logger.exception("[股吧情绪] 获取失败: %s", exc)
        guba_result = {'score': 0.0, 'post_count': 0, 'reverse_signal': 0.0}

    # ========== 3. 获取量化情绪指标 ==========
    logger.info("-" * 30)
    logger.info("[步骤3] 获取量化情绪指标")
    logger.info("-" * 30)

    try:
        quant_result = get_quant_sentiment(symbol)
        logger.info(f"[量化情绪] 获取完成: score={quant_result.get('score', 0):.3f}")
    except Exception as exc:
        logger.exception("[量化情绪] 获取失败: %s", exc)
        quant_result = {'score': 0.0}

    # ========== 4. [OPTIMIZED] 获取北向资金数据 ==========
    logger.info("-" * 30)
    logger.info("[步骤4] 获取北向资金数据")
    logger.info("-" * 30)

    try:
        north_result = _get_north_money_data(symbol)
        logger.info(f"[北向资金] 获取完成: {north_result.get('reason', 'N/A')}")
    except Exception as exc:
        logger.exception("[北向资金] 获取失败: %s", exc)
        north_result = {'signal': 'neutral', 'confidence': 0, 'net_flow': 0, 'reason': '数据获取失败'}

    # ========== 5. 计算综合情绪分数 ==========
    logger.info("-" * 30)
    logger.info("[步骤5] 计算综合情绪分数")
    logger.info("-" * 30)

    combined_result = _calculate_combined_sentiment(
        news_sentiment_score,
        guba_result,
        quant_result,
        north_result,
        len(recent_news)
    )

    # 格式化推理说明
    reasoning = _format_reasoning(combined_result, len(recent_news))

    # 生成分析结果
    message_content = {
        "signal": combined_result['signal'],
        "confidence": combined_result['confidence'],
        "reasoning": reasoning,
        "combined_score": combined_result['combined_score'],
        "components": combined_result['components'],
        "weights": combined_result['weights']
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

    logger.info("=" * 50)
    logger.info(f"[情绪分析Agent] 执行完成")
    logger.info(f"[情绪分析Agent] 最终结果: signal={combined_result['signal']}, confidence={combined_result['confidence']}, score={combined_result['combined_score']:.3f}")
    logger.info("=" * 50)

    return {
        "messages": [message],
        "data": {
            **data,
            "sentiment_analysis": combined_result['combined_score'],
            "sentiment_details": {
                "news_score": news_sentiment_score,
                "guba_score": guba_result.get('score', 0),
                "quant_score": quant_result.get('score', 0),
                "weights": combined_result['weights']
            }
        },
        "metadata": state["metadata"],
    }