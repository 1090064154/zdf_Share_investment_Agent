from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.tools.news_crawler import get_stock_news
from src.utils.logging_config import setup_logger
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent
import json
from datetime import datetime, timedelta
from src.tools.openrouter_config import get_chat_completion
from concurrent.futures import ThreadPoolExecutor

# 设置日志记录
logger = setup_logger('macro_analyst_agent')


# [NEW] 获取宏观数据指标
def _get_macro_indicators() -> dict:
    """
    获取关键宏观指标：GDP、CPI、PMI、利率等
    """
    indicators = {'gdp': None, 'cpi': None, 'pmi': None, 'interest_rate': None}
    
    try:
        import akshare as ak
        try:
            # GDP数据
            gdp_df = ak.macro_china_gdp()
            if gdp_df is not None and len(gdp_df) > 0:
                indicators['gdp'] = float(gdp_df.iloc[0].get('gdp', 0) or 0)
        except:
            pass
        
        try:
            # CPI数据
            cpi_df = ak.macro_china_cpi()
            if cpi_df is not None and len(cpi_df) > 0:
                indicators['cpi'] = float(cpi_df.iloc[0].get('cpi', 0) or 0)
        except:
            pass
        
        try:
            # PMI数据
            pmi_df = ak.cn_pmi()
            if pmi_df is not None and len(pmi_df) > 0:
                indicators['pmi'] = float(pmi_df.iloc[0].get('pmi', 50) or 50)
        except:
            pass
            
    except ImportError:
        pass
    
    return indicators


# [NEW] 筛选宏观相关新闻
def _filter_macro_relevant_news(news_list: list) -> list:
    """
    筛选与宏观相关的新闻
    """
    macro_keywords = [
        '宏观', '经济', 'GDP', 'CPI', 'PPI', 'PMI', '降准', '加息', '降息',
        '财政', '货币', '政策', '国务院', '央行', '证监会', '财政部',
        'GDP', '增长', '数据', '经济指标', '通胀', '流动性', '利率',
        '实体经济', '消费', '投资', '出口', '进口', '贸易', '关税',
        '房地产', '基建', '制造业', 'A股', '股市', '大盘', '指数'
    ]
    
    relevant_news = []
    for news in news_list:
        title = news.get('title', '') + news.get('content', '')
        if any(kw in title for kw in macro_keywords):
            relevant_news.append(news)
    
    return relevant_news[:50]  # 保留最多50条宏观相关新闻


# [NEW] 并行获取多源新闻
def _get_multi_source_news(symbol: str, end_date: str = None) -> list:
    """
    获取目标股票+宏观指数的新闻
    """
    all_news = []
    seen_titles = set()
    
    def fetch_news(ticker):
        try:
            return get_stock_news(ticker, max_news=30, date=end_date) or []
        except:
            return []
    
    # 并行获取
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_stock = executor.submit(fetch_news, symbol)
        future_hs300 = executor.submit(fetch_news, "000300")
        future_sh = executor.submit(fetch_news, "000001")
        future_sz = executor.submit(fetch_news, "399001")
        
        results = [
            future_stock.result(),
            future_hs300.result(),
            future_sh.result(),
            future_sz.result()
        ]
    
    # 合并去重
    for news_list in results:
        for news in news_list:
            title = news.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                all_news.append(news)
    
    # 筛选宏观相关
    return _filter_macro_relevant_news(all_news)


def _resolve_news_window_end(end_date: str | None) -> datetime:
    if not end_date:
        return datetime.now()
    return datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)


def _normalize_news_item(news: dict) -> dict:
    return {
        "title": str(news.get("title", "")).strip() or "未命名新闻",
        "source": str(news.get("source", "未知来源")).strip() or "未知来源",
        "publish_time": str(news.get("publish_time", "未知时间")).strip() or "未知时间",
        "content": str(news.get("content", "")).strip() or str(news.get("title", "")).strip() or "无正文内容",
    }


@resilient_agent
@agent_endpoint("macro_analyst", "宏观分析师，分析宏观经济环境对目标股票的影响")
def macro_analyst_agent(state: AgentState):
    """Responsible for macro analysis"""
    show_workflow_status("宏观分析师")
    logger.info("="*50)
    logger.info("🌍 [MACRO_ANALYST] 开始宏观分析")
    logger.info("="*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    logger.info(f"  股票代码: {symbol}")

    # 获取 end_date
    end_date = data.get("end_date")

    # [NEW] 获取宏观数据指标
    logger.info("  获取宏观数据指标...")
    macro_indicators = _get_macro_indicators()
    logger.info(f"  宏观指标: GDP={macro_indicators.get('gdp')}%, CPI={macro_indicators.get('cpi')}%, PMI={macro_indicators.get('pmi')}")

    # [NEW] 并行获取多源新闻
    logger.info("  获取多源新闻...")
    news_list = _get_multi_source_news(symbol, end_date)

    # 过滤七天前的新闻
    reference_time = _resolve_news_window_end(end_date)
    cutoff_date = reference_time - timedelta(days=7)
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

    logger.info(f"获取到 {len(recent_news)} 条宏观相关新闻")

    # 如果没有获取到新闻，返回默认结果
    if not recent_news:
        logger.warning(f"未获取到 {symbol} 的最近新闻，无法进行宏观分析")
        message_content = {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "未获取到最近新闻，无法进行宏观分析"
        }
    else:
        # 获取宏观分析结果
        macro_analysis = get_macro_news_analysis(recent_news)
        message_content = macro_analysis

    # 提取并规范化字段
    raw_env = message_content.get('macro_environment', 'neutral')
    raw_impact = message_content.get('impact_on_stock', 'neutral')
    key_factors = message_content.get('key_factors', [])
    reasoning = message_content.get('reasoning', '')

    # 兼容 LLM 可能返回的各种值
    env_map = {'favorable': '有利', 'positive': '有利', 'unfavorable': '不利', 'negative': '不利', 'neutral': '中性'}
    impact_map = {'positive': '正面', 'favorable': '正面', 'negative': '负面', 'unfavorable': '负面', 'neutral': '中性'}
    env_cn = env_map.get(raw_env, raw_env)
    impact_cn = impact_map.get(raw_impact, raw_impact)

    # 派生 signal
    if raw_impact in ('positive', 'favorable'):
        signal = 'bullish'
    elif raw_impact in ('negative', 'unfavorable'):
        signal = 'bearish'
    else:
        signal = 'neutral'
    signal_cn = {'bullish': '看多', 'bearish': '看空', 'neutral': '中性'}.get(signal, signal)

    # [NEW] 估算置信度（基于关键因素数量、宏观指标有效性、新闻数量）
    base_confidence = 0.4
    
    # 新闻数量加分
    if len(recent_news) > 20:
        base_confidence += 0.15
    elif len(recent_news) > 10:
        base_confidence += 0.1
    
    # 宏观指标加分
    valid_indicators = sum(1 for v in macro_indicators.values() if v is not None)
    base_confidence += valid_indicators * 0.05
    
    # 关键因素加分
    base_confidence += min(len(key_factors) * 0.08, 0.2)
    
    # 推理详细程度加分
    if len(reasoning) > 300:
        base_confidence += 0.1
    elif len(reasoning) > 150:
        base_confidence += 0.05
    
    confidence = min(base_confidence, 0.9)
    
    # [NEW] 引用行业周期分析
    industry_cycle_analysis = data.get("industry_cycle_analysis", {})
    industry = data.get("industry", "")
    cycle_phase = industry_cycle_analysis.get('phase', '未知')
    cycle_signal = industry_cycle_analysis.get('signal', 'neutral')
    cycle_type = industry_cycle_analysis.get('cycle_type_cn', industry_cycle_analysis.get('cycle_type', '未知'))

    # 构建决策逻辑（详细）
    factor_text = "；".join(key_factors[:5]) if key_factors else "无明显关键因素"

    # 生成详细决策逻辑
    env_score = {"有利": "+1", "中性": "0", "不利": "-1"}.get(env_cn, "0")
    impact_score = {"正面": "+1", "中性": "0", "负面": "-1"}.get(impact_cn, "0")
    
    # 行业周期信息
    cycle_text = f"，该行业处于{cycle_type}{cycle_phase}" if cycle_type != '未知' else ""
    
    # 行业周期与宏观的交互判断
    if cycle_signal == 'bullish' and raw_env in ('positive', 'favorable'):
        combined_assessment = "宏观有利+行业上行，双重利好"
        final_recommendation = "积极做多"
    elif cycle_signal == 'bearish' and raw_env in ('negative', 'unfavorable'):
        combined_assessment = "宏观利空+行业下行，风险叠加"
        final_recommendation = "建议减仓或观望"
    elif cycle_signal == 'bullish' and raw_env in ('negative', 'unfavorable'):
        combined_assessment = "行业逆势上行，可关注结构性机会"
        final_recommendation = "关注行业龙头"
    elif cycle_signal == 'bearish' and raw_env in ('positive', 'favorable'):
        combined_assessment = "宏观有利但行业下行，注意风格切换"
        final_recommendation = "可考虑行业切换"
    else:
        combined_assessment = f"宏观{env_cn}，行业{cycle_signal}"
        final_recommendation = "保持中性，关注方向"

    decision_logic = f"""【宏观环境评估】{env_cn}（得分：{env_score}）
- 环境判断依据：{reasoning[:200] + '...' if reasoning and len(reasoning) > 200 else reasoning or '基于新闻数据分析'}

【行业周期参考】{industry}属于{cycle_type}，当前{cycle_phase}{cycle_text}

【对个股影响评估】{impact_cn}（得分：{impact_score}）
- 影响路径：宏观环境变化 → 行业周期传导 → 个股基本面影响
- 关键因素包括：{factor_text}

【综合决策逻辑】
1. 宏观环境：当前{env_cn}，{('流动性收紧，风险偏好下降' if env_cn == '不利' else '经济运行平稳' if env_cn == '中性' else '政策支持力度加大，经济活跃度提升')}
2. 行业周期：{industry}行业当前{cycle_phase}，{('下游需求强劲' if cycle_signal == 'bullish' else '下游需求承压' if cycle_signal == 'bearish' else '需求平稳')}
3. 综合判断：{combined_assessment}
4. 最终结论：{final_recommendation}"""

    # 决策结果
    decision_result = {
        "recommendation": signal_cn,
        "environment_assessment": env_cn,
        "stock_impact": impact_cn,
        "confidence": round(confidence, 4),
        "key_factors": key_factors,
        "action": {
            "bullish": "可考虑逢低布局或加仓",
            "bearish": "建议谨慎，控制仓位或对冲",
            "neutral": "保持现有仓位，等待更多信号"
        }.get(signal, "观望为主"),
        "risk_warning": {
            "bullish": "注意市场波动，避免追高",
            "bearish": "关注超跌反弹机会",
            "neutral": "保持中性思路，等待方向明确"
        }.get(signal, "控制仓位")
    }

    # 重建完整的 message_content
    message_content = {
        "signal": signal,
        "confidence": round(confidence, 4),
        "signal_cn": signal_cn,
        "macro_environment": raw_env,
        "macro_environment_cn": env_cn,
        "impact_on_stock": raw_impact,
        "impact_on_stock_cn": impact_cn,
        "key_factors": key_factors,
        "reasoning": reasoning,
        "decision_logic": decision_logic,
        "decision_result": decision_result,
        "macro_indicators": macro_indicators,
        "news_count": len(recent_news),
        "industry": industry,
        "industry_cycle": {
            "cycle_type": cycle_type,
            "phase": cycle_phase,
            "signal": cycle_signal
        },
        "summary": f"宏观分析{signal_cn}（置信度{confidence*100:.0f}%），环境{env_cn}，{industry}行业{cycle_phase}，对个股影响{impact_cn}"
    }

    # 如果需要显示推理过程
    if show_reasoning:
        show_agent_reasoning({
            "macro_environment": raw_env,
            "impact_on_stock": raw_impact,
            "signal": signal,
            "confidence": round(confidence, 4),
            "关键因素": key_factors if key_factors else ["无明显关键因素"],
            "decision_logic": decision_logic,
            "decision_result": decision_result,
            "reasoning": reasoning,
            "最终结论": f"宏观环境{env_cn}，对股票影响{impact_cn}",
            "信号": signal_cn,
            "置信度": f"{confidence*100:.0f}%",
        }, "宏观分析师")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="macro_analyst_agent",
    )

    # 发送完整的完成事件，包含信号、置信度和详情
    show_workflow_complete(
        "宏观分析师",
        signal=signal,
        confidence=confidence,
        details=message_content,
        message=f"宏观分析完成，信号:{signal_cn}，置信度:{confidence*100:.0f}%"
    )
    logger.info("────────────────────────────────────────────────────────")
    logger.info("✅ 宏观分析完成:")
    logger.info(f"  🌍 宏观环境: {message_content.get('macro_environment')}")
    logger.info(f"  📊 对个股影响: {message_content.get('impact_on_stock')}")
    logger.info(f"  🔑 关键因素: {len(message_content.get('key_factors', []))} 个")
    logger.info("────────────────────────────────────────────────────────")

    return {
        "messages": state["messages"] + [message],
        "data": {
            **data,
            "macro_analysis": message_content
        },
        "metadata": state["metadata"],
    }


def get_macro_news_analysis(news_list: list) -> dict:
    """分析宏观经济新闻对股票的影响

    Args:
        news_list (list): 新闻列表

    Returns:
        dict: 宏观分析结果，包含环境评估、对股票的影响、关键因素和详细推理
    """
    if not news_list:
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": "没有足够的新闻数据进行宏观分析"
        }

    # 检查缓存
    import os
    cache_file = "src/data/macro_analysis_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # 生成新闻内容的唯一标识
    normalized_news_list = [_normalize_news_item(news) for news in news_list]
    news_key = "|".join([
        f"{news['title']}|{news['publish_time']}"
        for news in normalized_news_list[:20]  # 使用前20条新闻作为标识
    ])

    # 检查缓存
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if news_key in cache:
                    logger.info("使用缓存的宏观分析结果")
                    return cache[news_key]
        except Exception as e:
            logger.error(f"读取宏观分析缓存出错: {e}")
            cache = {}
    else:
        logger.info("未找到宏观分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """你是一位专业的宏观经济分析师，专注于分析宏观经济环境对A股个股的影响。
        请分析提供的新闻，从宏观角度评估当前经济环境，并分析这些宏观因素对目标股票的潜在影响。
        
        请关注以下宏观因素：
        1. 货币政策：利率、准备金率、公开市场操作等
        2. 财政政策：政府支出、税收政策、补贴等
        3. 产业政策：行业规划、监管政策、环保要求等
        4. 国际环境：全球经济形势、贸易关系、地缘政治等
        5. 市场情绪：投资者信心、市场流动性、风险偏好等
        
        你的分析应该包括：
        1. 宏观环境评估：积极(positive)、中性(neutral)或消极(negative)
        2. 对目标股票的影响：利好(positive)、中性(neutral)或利空(negative)
        3. 关键影响因素：列出3-5个最重要的宏观因素
        4. 详细推理：解释为什么这些因素会影响目标股票
        
        请确保你的分析：
        1. 基于事实和数据，而非猜测
        2. 考虑行业特性和公司特点
        3. 关注中长期影响，而非短期波动
        4. 提供具体、可操作的见解"""
    }

    # 准备新闻内容（减少到10条以避免超时）
    news_content = "\n\n".join([
        f"标题：{news['title']}\n"
        f"来源：{news['source']}\n"
        f"时间：{news['publish_time']}\n"
        f"内容：{news['content']}"
        # 减少到10条新闻进行分析，避免LLM超时
        for news in normalized_news_list[:10]
    ])

    user_message = {
        "role": "user",
        "content": f"请分析以下新闻，评估当前宏观经济环境及其对相关A股上市公司的影响：\n\n{news_content}\n\n请以JSON格式返回结果，包含以下字段：macro_environment（宏观环境：positive/neutral/negative）、impact_on_stock（对股票影响：positive/neutral/negative）、key_factors（关键因素数组）、reasoning（详细推理）。"
    }

    try:
        # 获取LLM分析结果
        logger.info("正在调用LLM进行宏观分析...")
        logger.info(f"新闻内容长度: {len(news_content)} 字符")
        logger.info(f"准备发送消息到LLM...")

        result = get_chat_completion(
            [system_message, user_message],
            max_retries=1,
            initial_retry_delay=0.5,
        )

        logger.info(f"LLM调用完成，结果类型: {type(result)}")
        if result is None:
            logger.warning("LLM分析暂不可用，宏观分析回退为中性结果")
            return {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": [],
                "reasoning": "LLM分析失败，无法获取宏观分析结果"
            }
        logger.info(f"LLM返回结果长度: {len(result) if result else 0} 字符")

        # 解析JSON结果
        try:
            # 尝试直接解析
            analysis_result = json.loads(result.strip())
            logger.info("成功解析LLM返回的JSON结果")
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group(1).strip())
                    logger.info("成功从代码块中提取并解析JSON结果")
                except:
                    # 如果仍然失败，返回默认结果
                    logger.error("无法解析代码块中的JSON结果")
                    return {
                        "macro_environment": "neutral",
                        "impact_on_stock": "neutral",
                        "key_factors": [],
                        "reasoning": "无法解析LLM返回的JSON结果"
                    }
            else:
                # 如果没有找到JSON，返回默认结果
                logger.error("LLM未返回有效的JSON格式结果")
                return {
                    "macro_environment": "neutral",
                    "impact_on_stock": "neutral",
                    "key_factors": [],
                    "reasoning": "LLM未返回有效的JSON格式结果"
                }

        # 缓存结果
        cache[news_key] = analysis_result
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info("宏观分析结果已缓存")
        except Exception as e:
            logger.error(f"写入宏观分析缓存出错: {e}")

        return analysis_result

    except Exception as e:
        logger.error(f"宏观分析出错: {e}")
        return {
            "macro_environment": "neutral",
            "impact_on_stock": "neutral",
            "key_factors": [],
            "reasoning": f"分析过程中出错: {str(e)}"
        }
