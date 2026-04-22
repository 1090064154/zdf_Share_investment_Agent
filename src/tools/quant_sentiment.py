"""
量化情绪指标模块

使用akshare获取东方财富的情绪量化指标：
1. 历史评分 - stock_comment_detail_zhpj_lspf_em
2. 散户关注度 - stock_comment_detail_scrd_focus_em
3. 机构参与度 - stock_comment_detail_zlkp_jgcyd_em
"""

import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logging_config import setup_logger

logger = setup_logger('quant_sentiment')

# 缓存目录
CACHE_DIR = "src/data/quant_sentiment_cache"


def _ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(symbol: str) -> str:
    """获取缓存文件路径"""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"{symbol}_{today}.json")


def _load_cache(symbol: str) -> dict | None:
    """加载缓存数据"""
    cache_path = _get_cache_path(symbol)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"[缓存] 命中量化情绪缓存: {cache_path}")
                return cached_data
        except Exception as e:
            logger.warning(f"[缓存] 读取失败: {e}")
    return None


def _save_cache(symbol: str, data: dict):
    """保存缓存数据"""
    _ensure_cache_dir()
    cache_path = _get_cache_path(symbol)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[缓存] 保存量化情绪数据: {cache_path}")
    except Exception as e:
        logger.warning(f"[缓存] 保存失败: {e}")


def get_historical_rating(symbol: str) -> dict:
    """
    获取历史评分数据

    Args:
        symbol: 股票代码

    Returns:
        {
            'current_rating': 当前评分,
            'rating_change': 评分变化,
            'avg_rating': 平均评分,
            'rating_trend': 趋势 ('up', 'down', 'stable'),
            'data': 历史数据列表
        }
    """
    logger.info(f"[量化-评分] 开始获取历史评分: symbol={symbol}")

    try:
        import akshare as ak
        start_time = time.time()
        df = ak.stock_comment_detail_zhpj_lspf_em(symbol=symbol)
        elapsed = time.time() - start_time
        logger.info(f"[量化-评分] API调用完成，耗时={elapsed:.2f}s, 数据行数={len(df) if df is not None else 0}")

        if df is None or df.empty:
            logger.warning(f"[量化-评分] 无数据返回")
            return {'current_rating': 50, 'rating_change': 0, 'rating_trend': 'stable', 'data': []}

        # 转换数据
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': str(row['交易日']),
                'rating': float(row['评分'])
            })

        # 计算当前评分和变化
        current_rating = data[-1]['rating'] if data else 50
        prev_rating = data[-2]['rating'] if len(data) > 1 else current_rating
        rating_change = current_rating - prev_rating

        # 计算平均评分
        avg_rating = sum(d['rating'] for d in data) / len(data) if data else 50

        # 判断趋势
        if len(data) >= 5:
            recent_avg = sum(d['rating'] for d in data[-5:]) / 5
            older_avg = sum(d['rating'] for d in data[-10:-5]) / 5 if len(data) >= 10 else recent_avg
            if recent_avg > older_avg + 2:
                rating_trend = 'up'
            elif recent_avg < older_avg - 2:
                rating_trend = 'down'
            else:
                rating_trend = 'stable'
        else:
            rating_trend = 'stable'

        result = {
            'current_rating': round(current_rating, 2),
            'rating_change': round(rating_change, 2),
            'avg_rating': round(avg_rating, 2),
            'rating_trend': rating_trend,
            'data': data[-10:]  # 只返回最近10天
        }

        logger.info(f"[量化-评分] 结果: 当前评分={current_rating:.2f}, 变化={rating_change:.2f}, 趋势={rating_trend}")
        return result

    except Exception as e:
        logger.error(f"[量化-评分] 获取失败: {e}")
        return {'current_rating': 50, 'rating_change': 0, 'rating_trend': 'stable', 'data': []}


def get_user_focus(symbol: str) -> dict:
    """
    获取散户关注度数据

    Args:
        symbol: 股票代码

    Returns:
        {
            'current_focus': 当前关注度,
            'focus_change': 关注度变化,
            'avg_focus': 平均关注度,
            'data': 历史数据列表
        }
    """
    logger.info(f"[量化-关注度] 开始获取散户关注度: symbol={symbol}")

    try:
        import akshare as ak
        start_time = time.time()
        df = ak.stock_comment_detail_scrd_focus_em(symbol=symbol)
        elapsed = time.time() - start_time
        logger.info(f"[量化-关注度] API调用完成，耗时={elapsed:.2f}s, 数据行数={len(df) if df is not None else 0}")

        if df is None or df.empty:
            logger.warning(f"[量化-关注度] 无数据返回")
            return {'current_focus': 50, 'focus_change': 0, 'data': []}

        # 转换数据
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': str(row['交易日']),
                'focus': float(row['用户关注指数'])
            })

        # 计算当前关注度和变化
        current_focus = data[-1]['focus'] if data else 50
        prev_focus = data[-2]['focus'] if len(data) > 1 else current_focus
        focus_change = current_focus - prev_focus

        # 计算平均关注度
        avg_focus = sum(d['focus'] for d in data) / len(data) if data else 50

        result = {
            'current_focus': round(current_focus, 2),
            'focus_change': round(focus_change, 2),
            'avg_focus': round(avg_focus, 2),
            'data': data[-10:]
        }

        logger.info(f"[量化-关注度] 结果: 当前关注度={current_focus:.2f}, 变化={focus_change:.2f}")
        return result

    except Exception as e:
        logger.error(f"[量化-关注度] 获取失败: {e}")
        return {'current_focus': 50, 'focus_change': 0, 'data': []}


def get_institution_participation(symbol: str) -> dict:
    """
    获取机构参与度数据

    Args:
        symbol: 股票代码

    Returns:
        {
            'current_participation': 当前参与度,
            'participation_change': 参与度变化,
            'avg_participation': 平均参与度,
            'data': 历史数据列表
        }
    """
    logger.info(f"[量化-机构] 开始获取机构参与度: symbol={symbol}")

    try:
        import akshare as ak
        start_time = time.time()
        df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=symbol)
        elapsed = time.time() - start_time
        logger.info(f"[量化-机构] API调用完成，耗时={elapsed:.2f}s, 数据行数={len(df) if df is not None else 0}")

        if df is None or df.empty:
            logger.warning(f"[量化-机构] 无数据返回")
            return {'current_participation': 50, 'participation_change': 0, 'data': []}

        # 转换数据
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': str(row['交易日']),
                'participation': float(row['机构参与度'])
            })

        # 计算当前参与度和变化
        current_participation = data[-1]['participation'] if data else 50
        prev_participation = data[-2]['participation'] if len(data) > 1 else current_participation
        participation_change = current_participation - prev_participation

        # 计算平均参与度
        avg_participation = sum(d['participation'] for d in data) / len(data) if data else 50

        result = {
            'current_participation': round(current_participation, 2),
            'participation_change': round(participation_change, 2),
            'avg_participation': round(avg_participation, 2),
            'data': data[-10:]
        }

        logger.info(f"[量化-机构] 结果: 当前参与度={current_participation:.2f}, 变化={participation_change:.2f}")
        return result

    except Exception as e:
        logger.error(f"[量化-机构] 获取失败: {e}")
        return {'current_participation': 50, 'participation_change': 0, 'data': []}


def calculate_quant_sentiment_score(rating_data: dict, focus_data: dict, institution_data: dict) -> dict:
    """
    计算量化情绪综合分数

    Args:
        rating_data: 历史评分数据
        focus_data: 散户关注度数据
        institution_data: 机构参与度数据

    Returns:
        {
            'score': 综合分数 (-1 到 1),
            'rating_normalized': 评分归一化分数,
            'institution_normalized': 机构参与度归一化分数,
            'focus_signal': 关注度信号
        }
    """
    logger.info("[量化-综合] 开始计算综合分数")

    # 评分归一化 (0-100 -> -1 到 1)
    # 评分 > 60 为积极，< 40 为消极
    rating = rating_data.get('current_rating', 50)
    rating_normalized = (rating - 50) / 50  # 50为中性点
    rating_normalized = max(-1, min(1, rating_normalized))
    logger.info(f"[量化-综合] 评分归一化: {rating:.2f} -> {rating_normalized:.3f}")

    # 机构参与度归一化 (高参与度通常意味着更专业的关注)
    participation = institution_data.get('current_participation', 50)
    # 机构参与度高，说明股票受到专业投资者关注，通常是正面信号
    institution_normalized = (participation - 50) / 50
    institution_normalized = max(-1, min(1, institution_normalized))
    logger.info(f"[量化-综合] 机构参与度归一化: {participation:.2f} -> {institution_normalized:.3f}")

    # 关注度信号 (关注度极高可能是反向信号)
    focus = focus_data.get('current_focus', 50)
    focus_avg = focus_data.get('avg_focus', 50)

    # 关注度突然升高可能是风险信号（散户过度关注）
    if focus > focus_avg + 5:
        focus_signal = -0.1  # 轻微负面
        logger.info(f"[量化-综合] 关注度信号: 关注度({focus:.2f})高于均值({focus_avg:.2f}), 触发轻微负面信号")
    elif focus < focus_avg - 5:
        focus_signal = 0.1  # 轻微正面（关注度下降可能意味着底部）
        logger.info(f"[量化-综合] 关注度信号: 关注度({focus:.2f})低于均值({focus_avg:.2f}), 触发轻微正面信号")
    else:
        focus_signal = 0

    # 综合分数计算
    # 权重: 评分 60%, 机构参与度 30%, 关注度 10%
    score = (
        rating_normalized * 0.6 +
        institution_normalized * 0.3 +
        focus_signal * 0.1
    )

    logger.info(f"[量化-综合] 综合分数计算: 评分*0.6({rating_normalized*0.6:.3f}) + 机构*0.3({institution_normalized*0.3:.3f}) + 关注度*0.1({focus_signal*0.1:.3f}) = {score:.3f}")

    return {
        'score': round(score, 3),
        'rating_normalized': round(rating_normalized, 3),
        'institution_normalized': round(institution_normalized, 3),
        'focus_signal': round(focus_signal, 3),
        'rating_trend': rating_data.get('rating_trend', 'stable')
    }


def get_quant_sentiment(symbol: str) -> dict:
    """
    获取量化情绪指标（主入口函数）

    Args:
        symbol: 股票代码

    Returns:
        量化情绪分析结果
    """
    logger.info(f"========== [量化情绪分析] 开始 ==========")
    logger.info(f"[量化情绪] 股票代码: {symbol}")

    # 检查缓存
    cached = _load_cache(symbol)
    if cached:
        logger.info(f"[量化情绪] 使用缓存数据")
        logger.info(f"========== [量化情绪分析] 完成(缓存) ==========")
        return cached

    # 获取各项数据
    rating_data = get_historical_rating(symbol)
    focus_data = get_user_focus(symbol)
    institution_data = get_institution_participation(symbol)

    # 计算综合分数
    score_data = calculate_quant_sentiment_score(rating_data, focus_data, institution_data)

    # 组合结果
    result = {
        'score': score_data['score'],
        'rating': rating_data,
        'focus': focus_data,
        'institution': institution_data,
        'score_details': score_data,
        'fetch_time': datetime.now().isoformat()
    }

    # 保存缓存
    _save_cache(symbol, result)

    logger.info(f"========== [量化情绪分析] 完成 ==========")
    logger.info(f"[量化情绪] 最终结果: score={result['score']:.3f}")

    return result


if __name__ == "__main__":
    # 测试
    result = get_quant_sentiment("300059")
    print(json.dumps(result, ensure_ascii=False, indent=2))