import os
import sys
import json
from datetime import datetime, timedelta
import time
import pandas as pd
from urllib.parse import urlparse
from src.tools.openrouter_config import get_chat_completion, logger as api_logger

# 导入新的搜索模块
try:
    from src.crawler.search import google_search_sync, SearchOptions
except ImportError:
    print("警告: 无法导入新的搜索模块，将回退到 akshare")
    google_search_sync = None
    SearchOptions = None

# 保留 akshare 作为备用
try:
    import akshare as ak
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("警告: akshare 不可用")
    ak = None


def resolve_stock_name(symbol: str) -> str:
    """尽力解析股票名称，失败时返回空字符串。"""
    if ak is None:
        return ""

    # 常见指数或特殊代码的固定映射
    static_names = {
        "000300": "沪深300",
        "399001": "深证成指",
        "000001": "上证指数",
    }
    if symbol in static_names:
        return static_names[symbol]

    # 优先从实时行情表里取证券名称，避免依赖东方财富个股信息接口
    try:
        spot_df = ak.stock_zh_a_spot()
        if spot_df is not None and not spot_df.empty:
            stock_code = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
            code_col = "代码" if "代码" in spot_df.columns else None
            name_col = next((col for col in ["名称", "简称"] if col in spot_df.columns), None)
            if code_col and name_col:
                matched = spot_df[spot_df[code_col] == stock_code]
                if not matched.empty:
                    return str(matched.iloc[0][name_col]).strip()
    except Exception:
        pass

    return ""


def build_search_query(symbol: str, date: str = None, stock_name: str = "") -> str:
    """
    构建针对股票新闻的 Google 搜索查询

    Args:
        symbol: 股票代码，如 "300059"
        stock_name: 股票名称，如 "北方稀土"
        date: 截止日期，格式 "YYYY-MM-DD"

    Returns:
        构建好的搜索查询字符串
    """
    identity_terms = [f'"{symbol}"', symbol]
    if stock_name:
        identity_terms = [f'"{stock_name}"', f'"{stock_name}" "{symbol}"', symbol]

    # 基础查询：优先股票名称 + 代码 + 财经关键词
    base_query = f"({' OR '.join(identity_terms)}) (股票 OR 个股 OR 公司) (新闻 OR 公告 OR 财经)"

    # 添加时间限制（搜索指定日期之前的新闻）
    if date:
        try:
            # 解析日期并计算一周前的日期作为开始时间
            end_date = datetime.strptime(date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=7)  # 搜索过去一周的新闻

            # Google 搜索时间语法：after:YYYY-MM-DD before:YYYY-MM-DD
            base_query += f" after:{start_date.strftime('%Y-%m-%d')} before:{date}"
        except ValueError:
            print(f"日期格式错误: {date}，忽略时间限制")

    # 限制新闻网站 - 只选择主要的财经网站
    news_sites = [
        "site:sina.com.cn",
        "site:finance.sina.com.cn",
        "site:cj.sina.com.cn",
        "site:163.com",
        "site:cnstock.com",
        "site:hexun.com"
    ]

    # 添加网站限制
    query = f"{base_query} ({' OR '.join(news_sites)})"

    return query


def extract_domain(url: str) -> str:
    """从 URL 提取域名作为新闻来源"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "未知来源"


def convert_search_results_to_news_format(search_results, symbol: str) -> list:
    """
    将搜索结果转换为现有新闻格式

    Args:
        search_results: Google 搜索结果
        symbol: 股票代码

    Returns:
        符合现有格式的新闻列表
    """
    news_list = []

    for result in search_results:
        # 过滤掉明显不相关的结果
        if any(keyword in result.title.lower() for keyword in ['招聘', '求职', '广告', '登录', '注册']):
            continue

        # 尝试从snippet中提取时间信息
        publish_time = None
        if result.snippet:
            # 查找常见的时间模式
            import re
            time_patterns = [
                r'(\d{1,2}天前)',
                r'(\d{1,2}小时前)',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{2}-\d{2})'
            ]

            for pattern in time_patterns:
                match = re.search(pattern, result.snippet)
                if match:
                    time_str = match.group(1)
                    try:
                        # 处理相对时间
                        if '天前' in time_str:
                            days = int(time_str.replace('天前', ''))
                            publish_date = datetime.now() - timedelta(days=days)
                            publish_time = publish_date.strftime(
                                '%Y-%m-%d %H:%M:%S')
                        elif '小时前' in time_str:
                            hours = int(time_str.replace('小时前', ''))
                            publish_date = datetime.now() - timedelta(hours=hours)
                            publish_time = publish_date.strftime(
                                '%Y-%m-%d %H:%M:%S')
                        # YYYY-MM-DD格式
                        elif '-' in time_str and len(time_str) == 10:
                            publish_time = f"{time_str} 00:00:00"
                        break
                    except:
                        continue

        news_item = {
            "title": result.title,
            "content": result.snippet or result.title,
            "source": extract_domain(result.link),
            "url": result.link,
            "keyword": symbol,
            "search_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 搜索时间
        }

        # 只有当能提取到发布时间时才添加，否则不包含这个字段
        if publish_time:
            news_item["publish_time"] = publish_time

        news_list.append(news_item)

    return news_list


def get_stock_news_via_sina(symbol: str, stock_name: str = "", max_news: int = 10, date: str = None) -> list:
    """使用新浪搜索抓取新闻，作为 Google 搜索的备用来源。"""
    keyword = stock_name or symbol
    search_query = f"{keyword} {symbol}".strip() if stock_name else symbol
    base_url = "https://search.sina.com.cn/"

    try:
        params = {
            "q": search_query,
            "c": "news",
            "sort": "time",
            "range": "all",
            "num": max_news * 2,
        }
        response = requests.get(
            base_url,
            params=params,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.sina.com.cn/",
            },
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        news_list = []
        for item in soup.select("div.box-result, div.result, div.r-info")[:max_news * 3]:
            link = item.select_one("h2 a, h3 a, a")
            if link is None:
                continue

            title = link.get_text(" ", strip=True)
            url = link.get("href", "").strip()
            snippet_node = item.select_one("p, .content, .box-result-content")
            meta_node = item.select_one(".fgray_time, .from, .news-from")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else title
            meta_text = meta_node.get_text(" ", strip=True) if meta_node else ""

            publish_time = ""
            import re
            time_match = re.search(r"(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?)", meta_text)
            if time_match:
                publish_time = time_match.group(1)
                if len(publish_time) == 10:
                    publish_time += " 00:00:00"
                elif len(publish_time) == 16:
                    publish_time += ":00"

            source = extract_domain(url) or "finance.sina.com.cn"
            news_item = {
                "title": title,
                "content": snippet,
                "source": source,
                "url": url,
                "keyword": keyword,
            }
            if publish_time:
                news_item["publish_time"] = publish_time
            news_list.append(news_item)

        print(f"通过新浪搜索获取到{len(news_list)}条新闻")
        return news_list[:max_news]
    except Exception as e:
        print(f"新浪搜索获取新闻时出错: {e}")
        return []


def get_stock_news(symbol: str, max_news: int = 10, date: str = None) -> list:
    """获取并处理个股新闻

    Args:
        symbol (str): 股票代码，如 "300059"
        max_news (int, optional): 获取的新闻条数，默认为10条。最大支持100条。
        date (str, optional): 截止日期，格式 "YYYY-MM-DD"，用于限制获取新闻的时间范围，
                             获取该日期及之前的新闻。如果不指定，则使用当前日期。

    Returns:
        list: 新闻列表，每条新闻包含标题、内容、发布时间等信息。
              新闻来源通过智能搜索引擎获取，包含各大财经网站的相关报道。
    """

    # 限制最大新闻条数
    max_news = min(max_news, 100)

    def is_valid_news_item(news: dict) -> bool:
        title = str(news.get("title", "")).strip()
        content = str(news.get("content", "")).strip()
        if not title:
            return False
        invalid_markers = ["搜索失败", "无法完成搜索", "timeout", "page.goto"]
        combined = f"{title}\n{content}".lower()
        return not any(marker.lower() in combined for marker in invalid_markers)

    # 获取当前日期或使用指定日期
    cache_date = date if date else datetime.now().strftime("%Y-%m-%d")

    # 构建新闻文件路径
    news_dir = os.path.join("src", "data", "stock_news")
    print(f"新闻保存目录: {news_dir}")

    # 确保目录存在
    try:
        os.makedirs(news_dir, exist_ok=True)
        print(f"成功创建或确认目录存在: {news_dir}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        return []

    # 缓存文件名包含日期信息
    news_file = os.path.join(news_dir, f"{symbol}_news_{cache_date}.json")
    print(f"新闻文件路径: {news_file}")

    # 检查缓存是否存在且有效
    cached_news = []
    cache_valid = False

    if os.path.exists(news_file):
        try:
            # 检查缓存文件的修改时间（时效性检查）
            file_mtime = os.path.getmtime(news_file)
            current_time = time.time()
            # 缓存有效期：当天的缓存在当天有效，历史日期的缓存始终有效
            if date:  # 如果指定了历史日期，缓存始终有效
                cache_valid = True
            else:  # 如果是当天数据，检查是否在同一天创建
                cache_date_obj = datetime.fromtimestamp(file_mtime).date()
                today = datetime.now().date()
                cache_valid = cache_date_obj == today

            if cache_valid:
                with open(news_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_news = data.get("news", [])

                    if len(cached_news) >= max_news:
                        print(
                            f"使用缓存的新闻数据: {news_file} (缓存数量: {len(cached_news)})")
                        return cached_news[:max_news]
                    else:
                        print(
                            f"缓存的新闻数量({len(cached_news)})不足，需要获取更多新闻({max_news}条)")
            else:
                print(f"缓存文件已过期，将重新获取新闻")

        except Exception as e:
            print(f"读取缓存文件失败: {e}")
            cached_news = []

    print(f'开始获取{symbol}的新闻数据...')
    stock_name = resolve_stock_name(symbol)
    if stock_name:
        print(f"已解析股票名称: {stock_name}")
    else:
        print(f"未解析到股票名称，将仅使用代码 {symbol} 搜索新闻")

    # 计算需要获取的新闻数量
    need_more_news = max_news - len(cached_news)
    fetch_count = max(need_more_news, max_news)  # 至少获取请求的数量

    # 优先尝试使用新的 Google 搜索方法
    new_news_list = []
    if google_search_sync and SearchOptions:
        try:
            print("使用 Google 搜索获取新闻...")

            # 构建搜索查询
            search_query = build_search_query(symbol, date, stock_name=stock_name)
            print(f"搜索查询: {search_query}")

            # 执行搜索
            search_options = SearchOptions(
                limit=fetch_count * 2,  # 获取更多结果以便过滤
                timeout=30000,
                locale="zh-CN"
            )

            search_response = google_search_sync(search_query, search_options)

            if search_response.results:
                # 转换搜索结果为新闻格式
                new_news_list = convert_search_results_to_news_format(
                    search_response.results, symbol)

                print(f"通过 Google 搜索成功获取到{len(new_news_list)}条新闻")
            else:
                print("Google 搜索未返回有效结果，尝试回退到 akshare")

        except Exception as e:
            print(f"Google 搜索获取新闻时出错: {e}，回退到 akshare")

    # 如果 Google 搜索失败，回退到新浪搜索
    if not new_news_list:
        print("使用新浪搜索获取新闻...")
        new_news_list = get_stock_news_via_sina(
            symbol,
            stock_name=stock_name,
            max_news=fetch_count,
            date=date,
        )

    # 合并缓存和新获取的新闻，去重
    if cached_news and new_news_list:
        # 创建已有新闻的标题集合用于去重
        existing_titles = {news['title'] for news in cached_news}

        # 过滤掉重复的新闻
        unique_new_news = [
            news for news in new_news_list
            if news['title'] not in existing_titles and is_valid_news_item(news)
        ]

        # 合并新闻列表
        combined_news = cached_news + unique_new_news
        print(
            f"合并缓存新闻({len(cached_news)}条)和新获取新闻({len(unique_new_news)}条)，总计{len(combined_news)}条")
    else:
        combined_news = [
            news for news in (new_news_list or cached_news)
            if is_valid_news_item(news)
        ]

    # 按发布时间排序（如果有发布时间信息）
    try:
        combined_news.sort(key=lambda x: x.get(
            "publish_time", ""), reverse=True)
    except:
        pass  # 如果排序失败，保持原顺序

    # 只保留指定条数的新闻
    final_news_list = [news for news in combined_news if is_valid_news_item(news)][:max_news]

    # 保存到文件（只有当获取到新数据时才保存）
    if new_news_list or not cache_valid:
        try:
            save_data = {
                "date": cache_date,
                "method": "online_search" if new_news_list and google_search_sync else "sina_search",
                "query": build_search_query(symbol, date, stock_name=stock_name) if new_news_list and google_search_sync else None,
                "news": combined_news,  # 保存所有新闻，不只是返回的部分
                "cached_count": len(cached_news),
                "new_count": len(new_news_list),
                "total_count": len(combined_news),
                "last_updated": datetime.now().isoformat()
            }
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"成功保存{len(combined_news)}条新闻到文件: {news_file}")
        except Exception as e:
            print(f"保存新闻数据到文件时出错: {e}")

    return final_news_list


def get_news_sentiment(news_list: list, num_of_news: int = 5) -> float:
    """分析新闻情感得分

    Args:
        news_list (list): 新闻列表
        num_of_news (int): 用于分析的新闻数量，默认为5条

    Returns:
        float: 情感得分，范围[-1, 1]，-1最消极，1最积极
    """
    filtered_news = []
    for news in news_list:
        title = str(news.get("title", "")).strip()
        content = str(news.get("content", "")).strip()
        combined = f"{title}\n{content}".lower()
        if not title:
            continue
        if any(marker in combined for marker in ["搜索失败", "无法完成搜索", "timeout", "page.goto"]):
            continue
        filtered_news.append(news)

    news_list = filtered_news

    if not news_list:
        return 0.0

    # # 获取项目根目录
    # project_root = os.path.dirname(os.path.dirname(
    #     os.path.dirname(os.path.abspath(__file__))))

    # 检查是否有缓存的情感分析结果
    # 检查是否有缓存的情感分析结果
    cache_file = "src/data/sentiment_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # 生成新闻内容的唯一标识
    news_key = "|".join([
        f"{news.get('title', '')}|{news.get('content', '')[:100]}|{news.get('publish_time', '')}"
        for news in news_list[:num_of_news]
    ])

    # 检查缓存
    if os.path.exists(cache_file):
        print("发现情感分析缓存文件")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if news_key in cache:
                    cached_value = cache[news_key]
                    if isinstance(cached_value, (int, float)) and -1.0 <= cached_value <= 1.0:
                        print("使用缓存的情感分析结果")
                        return cached_value
                    print("发现无效的情感分析缓存，重新计算")
                print("未找到匹配的情感分析缓存")
        except Exception as e:
            print(f"读取情感分析缓存出错: {e}")
            cache = {}
    else:
        print("未找到情感分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """你是一个专业的A股市场分析师，擅长解读新闻对股票走势的影响。你需要分析一组新闻的情感倾向，并给出一个介于-1到1之间的分数：
        - 1表示极其积极（例如：重大利好消息、超预期业绩、行业政策支持）
        - 0.5到0.9表示积极（例如：业绩增长、新项目落地、获得订单）
        - 0.1到0.4表示轻微积极（例如：小额合同签订、日常经营正常）
        - 0表示中性（例如：日常公告、人事变动、无重大影响的新闻）
        - -0.1到-0.4表示轻微消极（例如：小额诉讼、非核心业务亏损）
        - -0.5到-0.9表示消极（例如：业绩下滑、重要客户流失、行业政策收紧）
        - -1表示极其消极（例如：重大违规、核心业务严重亏损、被监管处罚）

        分析时重点关注：
        1. 业绩相关：财报、业绩预告、营收利润等
        2. 政策影响：行业政策、监管政策、地方政策等
        3. 市场表现：市场份额、竞争态势、商业模式等
        4. 资本运作：并购重组、股权激励、定增配股等
        5. 风险事件：诉讼仲裁、处罚、债务等
        6. 行业地位：技术创新、专利、市占率等
        7. 舆论环境：媒体评价、社会影响等

        请确保分析：
        1. 新闻的真实性和可靠性
        2. 新闻的时效性和影响范围
        3. 对公司基本面的实际影响
        4. A股市场的特殊反应规律"""
    }

    # 准备新闻内容
    news_content = "\n\n".join([
        f"标题：{news.get('title', '')}\n"
        f"来源：{news.get('source', '')}\n"
        f"时间：{news.get('publish_time', '未知')}\n"
        f"内容：{news.get('content', '')}"
        for news in news_list[:num_of_news]  # 使用指定数量的新闻
    ])

    user_message = {
        "role": "user",
        "content": f"请分析以下A股上市公司相关新闻的情感倾向：\n\n{news_content}\n\n请直接返回一个数字，范围是-1到1，无需解释。"
    }

    try:
        # 获取LLM分析结果
        result = get_chat_completion(
            [system_message, user_message],
            max_retries=1,
            initial_retry_delay=0.5,
        )
        if result is None:
            print("Error: PI error occurred, LLM returned None")
            return 0.0

        # 提取数字结果
        try:
            sentiment_score = float(result.strip())
        except ValueError as e:
            print(f"Error parsing sentiment score: {e}")
            print(f"Raw result: {result}")
            return 0.0

        # 确保分数在-1到1之间
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # 缓存结果
        cache[news_key] = sentiment_score
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")

        return sentiment_score

    except Exception as e:
        print(f"Error analyzing news sentiment: {e}")
        return 0.0  # 出错时返回中性分数
