"""
东方财富股吧数据获取模块

功能：
1. 获取股吧帖子列表
2. 获取帖子详情内容
3. 分析帖子情绪
"""

import os
import json
import time
import math
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from src.tools.openrouter_config import get_chat_completion
from src.utils.logging_config import setup_logger

logger = setup_logger('guba_crawler')

# 缓存目录
CACHE_DIR = "src/data/guba_cache"


def _ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(symbol: str, cache_type: str) -> str:
    """获取缓存文件路径"""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"{symbol}_{cache_type}_{today}.json")


def _load_cache(symbol: str, cache_type: str) -> dict | None:
    """加载缓存数据"""
    cache_path = _get_cache_path(symbol, cache_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"[缓存] 命中股吧缓存: {cache_path}")
                return cached_data
        except Exception as e:
            logger.warning(f"[缓存] 读取失败: {e}")
    return None


def _save_cache(symbol: str, cache_type: str, data: dict):
    """保存缓存数据"""
    _ensure_cache_dir()
    cache_path = _get_cache_path(symbol, cache_type)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[缓存] 保存股吧数据: {cache_path}")
    except Exception as e:
        logger.warning(f"[缓存] 保存失败: {e}")


def get_guba_posts(symbol: str, max_posts: int = 30, page: int = 1) -> list[dict]:
    """
    获取东方财富股吧帖子列表

    Args:
        symbol: 股票代码，如 "300059"
        max_posts: 最大获取帖子数
        page: 页码

    Returns:
        帖子列表，每个帖子包含:
        - title: 标题
        - author: 作者
        - reads: 阅读数
        - comments: 评论数
        - url: 链接
        - post_time: 发布时间（如果能解析）
    """
    logger.info(f"[股吧] 开始获取帖子列表: symbol={symbol}, max_posts={max_posts}, page={page}")

    # 检查缓存
    cache_key = f"posts_p{page}"
    cached = _load_cache(symbol, cache_key)
    if cached and len(cached.get('posts', [])) >= max_posts:
        logger.info(f"[股吧] 使用缓存数据，返回 {min(max_posts, len(cached['posts']))} 条帖子")
        return cached['posts'][:max_posts]

    posts = []
    url = f'https://guba.eastmoney.com/list,{symbol},99_{page}.html'

    try:
        logger.info(f"[股吧] 请求URL: {url}")
        start_time = time.time()
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        elapsed = time.time() - start_time
        logger.info(f"[股吧] 请求完成: 状态码={resp.status_code}, 耗时={elapsed:.2f}s")

        if resp.status_code != 200:
            logger.error(f"[股吧] 请求失败，状态码: {resp.status_code}")
            return posts

        soup = BeautifulSoup(resp.text, 'html.parser')
        items = soup.select('.listitem')
        logger.info(f"[股吧] 解析页面，找到 {len(items)} 个帖子元素")

        for item in items[:max_posts]:
            try:
                # 标题
                title_elem = item.select_one('.title a')
                title = title_elem.text.strip() if title_elem else ''
                if not title:
                    continue

                # 链接
                link = title_elem.get('href', '')
                if link and not link.startswith('http'):
                    if link.startswith('//'):
                        link = 'https:' + link
                    else:
                        link = 'https://guba.eastmoney.com' + link

                # 作者
                author_elem = item.select_one('.author')
                author = author_elem.text.strip() if author_elem else ''

                # 阅读数
                read_elem = item.select_one('.read')
                reads_str = read_elem.text.strip() if read_elem else '0'
                reads = int(reads_str) if reads_str.isdigit() else 0

                # 评论数
                comment_elem = item.select_one('.reply')
                comments_str = comment_elem.text.strip() if comment_elem else '0'
                comments = int(comments_str) if comments_str.isdigit() else 0

                posts.append({
                    'title': title,
                    'author': author,
                    'reads': reads,
                    'comments': comments,
                    'url': link,
                    'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            except Exception as e:
                logger.warning(f"[股吧] 解析单条帖子失败: {e}")
                continue

        # 保存缓存
        if posts:
            _save_cache(symbol, cache_key, {'posts': posts, 'fetch_time': datetime.now().isoformat()})

        logger.info(f"[股吧] 成功获取 {len(posts)} 条帖子")

    except Exception as e:
        logger.error(f"[股吧] 获取帖子列表失败: {e}")

    return posts


def get_guba_post_content(post_url: str) -> str:
    """
    获取帖子详情内容

    Args:
        post_url: 帖子链接

    Returns:
        帖子正文内容
    """
    logger.debug(f"[股吧] 获取帖子详情: {post_url}")

    try:
        start_time = time.time()
        resp = requests.get(post_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        elapsed = time.time() - start_time
        logger.debug(f"[股吧] 帖子详情请求完成: 状态码={resp.status_code}, 耗时={elapsed:.2f}s")

        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 查找内容区域
        content_divs = soup.find_all('div', class_=lambda x: x and ('content' in x.lower() or 'text' in x.lower()) if x else False)

        for div in content_divs:
            text = div.get_text(strip=True)
            # 过滤掉太短或明显是导航的内容
            if len(text) > 50 and '股吧首页' not in text:
                # 清理文本
                text = text.split('郑重声明')[0]  # 去掉声明部分
                return text[:1000]  # 限制长度

        return ""

    except Exception as e:
        logger.warning(f"[股吧] 获取帖子内容失败: {e}")
        return ""


def analyze_guba_sentiment(posts: list[dict], max_analyze: int = 20) -> dict:
    """
    分析股吧帖子情绪

    Args:
        posts: 帖子列表
        max_analyze: 最多分析的帖子数

    Returns:
        情绪分析结果:
        - score: 情绪分数 (-1 到 1)
        - bull_ratio: 看多比例
        - bear_ratio: 看空比例
        - heat: 热度指标
        - divergence: 分歧度
        - reverse_signal: 反向信号
    """
    logger.info(f"[股吧情绪] 开始分析，帖子数={len(posts)}, 分析上限={max_analyze}")

    if not posts:
        logger.warning("[股吧情绪] 无帖子数据，返回默认结果")
        return {
            'score': 0.0,
            'bull_ratio': 0.0,
            'bear_ratio': 0.0,
            'heat': 0.0,
            'divergence': 0.0,
            'reverse_signal': 0.0,
            'post_count': 0
        }

    # 准备分析内容
    titles_to_analyze = [p['title'] for p in posts[:max_analyze]]
    logger.debug(f"[股吧情绪] 待分析标题: {titles_to_analyze[:5]}...")

    # 使用LLM分析情绪
    system_message = {
        "role": "system",
        "content": """你是一个专业的A股市场情绪分析师。分析股吧帖子的整体情绪倾向。

请分析以下股吧帖子标题，返回JSON格式结果：
{
    "score": 情绪分数(-1到1，-1极度悲观，1极度乐观，0中性),
    "bull_count": 看多帖子数量,
    "bear_count": 看空帖子数量,
    "neutral_count": 中性帖子数量
}

判断标准：
- 看多(bullish): 看涨、买入、利好、上涨、突破、加仓等
- 看空(bearish): 看跌、卖出、利空、下跌、止损、减仓等
- 中性(neutral): 讨论技术、提问、复盘、无明显倾向

注意：股吧情绪往往有反向指标作用，极端乐观可能意味着风险，极端悲观可能意味着机会。"""
    }

    user_message = {
        "role": "user",
        "content": f"请分析以下股吧帖子标题的情绪：\n\n" + "\n".join(f"{i+1}. {title}" for i, title in enumerate(titles_to_analyze))
    }

    try:
        logger.info("[股吧情绪] 调用LLM分析...")
        start_time = time.time()
        result = get_chat_completion([system_message, user_message], max_retries=1)
        elapsed = time.time() - start_time
        logger.info(f"[股吧情绪] LLM分析完成，耗时={elapsed:.2f}s")

        if result is None:
            logger.error("[股吧情绪] LLM返回None")
            return _default_guba_result(len(posts))

        # 解析JSON结果
        import re
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get('score', 0))
            bull_count = int(data.get('bull_count', 0))
            bear_count = int(data.get('bear_count', 0))
            neutral_count = int(data.get('neutral_count', 0))
            logger.info(f"[股吧情绪] LLM结果: score={score:.2f}, 看多={bull_count}, 看空={bear_count}, 中性={neutral_count}")
        else:
            # 尝试直接解析数字
            try:
                score = float(result.strip())
                bull_count = sum(1 for t in titles_to_analyze if any(k in t for k in ['涨', '多', '买', '利好', '突破']))
                bear_count = sum(1 for t in titles_to_analyze if any(k in t for k in ['跌', '空', '卖', '利空', '止损']))
                neutral_count = len(titles_to_analyze) - bull_count - bear_count
                logger.info(f"[股吧情绪] 关键词统计: 看多={bull_count}, 看空={bear_count}, 中性={neutral_count}")
            except:
                logger.error("[股吧情绪] 解析LLM结果失败")
                return _default_guba_result(len(posts))

        # 确保分数在范围内
        score = max(-1.0, min(1.0, score))

        # 计算比例
        total = bull_count + bear_count + neutral_count
        bull_ratio = bull_count / total if total > 0 else 0
        bear_ratio = bear_count / total if total > 0 else 0

        # 计算热度 (基于阅读数和评论数)
        total_reads = sum(p.get('reads', 0) for p in posts)
        total_comments = sum(p.get('comments', 0) for p in posts)
        heat = min(10, total_reads / 10000 + total_comments / 100)
        logger.info(f"[股吧情绪] 热度计算: 总阅读={total_reads}, 总评论={total_comments}, 热度={heat:.2f}")

        # 计算分歧度
        divergence = min(1.0, abs(bull_ratio - bear_ratio) * 2) if bull_ratio + bear_ratio > 0 else 0

        # 反向信号检测
        reverse_signal = 0.0
        if score > 0.6:  # 极度乐观
            reverse_signal = -0.2
            logger.info(f"[股吧情绪] 检测到极度乐观，触发反向信号: {reverse_signal}")
        elif score < -0.6:  # 极度悲观
            reverse_signal = 0.2
            logger.info(f"[股吧情绪] 检测到极度悲观，触发反向信号: {reverse_signal}")

        result = {
            'score': score,
            'bull_ratio': bull_ratio,
            'bear_ratio': bear_ratio,
            'heat': heat,
            'divergence': divergence,
            'reverse_signal': reverse_signal,
            'post_count': len(posts),
            'bull_count': bull_count,
            'bear_count': bear_count,
            'neutral_count': neutral_count
        }

        logger.info(f"[股吧情绪] 分析完成: score={score:.2f}, bull_ratio={bull_ratio:.2f}, bear_ratio={bear_ratio:.2f}")
        return result

    except Exception as e:
        logger.error(f"[股吧情绪] 分析失败: {e}")
        return _default_guba_result(len(posts))


def _default_guba_result(post_count: int) -> dict:
    """返回默认的股吧情绪结果"""
    return {
        'score': 0.0,
        'bull_ratio': 0.0,
        'bear_ratio': 0.0,
        'heat': 0.0,
        'divergence': 0.0,
        'reverse_signal': 0.0,
        'post_count': post_count
    }


def get_guba_sentiment(symbol: str, max_posts: int = 30) -> dict:
    """
    获取股吧情绪分析结果（主入口函数）

    Args:
        symbol: 股票代码
        max_posts: 最大获取帖子数

    Returns:
        情绪分析结果
    """
    logger.info(f"========== [股吧情绪分析] 开始 ==========")
    logger.info(f"[股吧情绪] 股票代码: {symbol}, 最大帖子数: {max_posts}")

    # 获取帖子列表
    posts = get_guba_posts(symbol, max_posts=max_posts)

    if not posts:
        logger.warning(f"[股吧情绪] 未获取到帖子，返回默认结果")
        return _default_guba_result(0)

    # 分析情绪
    result = analyze_guba_sentiment(posts, max_analyze=min(20, len(posts)))

    logger.info(f"========== [股吧情绪分析] 完成 ==========")
    logger.info(f"[股吧情绪] 最终结果: score={result['score']:.2f}, posts={result['post_count']}")

    return result


if __name__ == "__main__":
    # 测试
    result = get_guba_sentiment("300059")
    print(json.dumps(result, ensure_ascii=False, indent=2))