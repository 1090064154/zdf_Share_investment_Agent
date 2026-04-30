import math
from typing import Dict
from src.utils.logging_config import setup_logger

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status, show_workflow_complete
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.utils.error_handler import resilient_agent

import json
import pandas as pd
import numpy as np

from src.tools.api import prices_to_df

# 初始化 logger (必须在任何日志调用之前)
logger = setup_logger('technical_analyst_agent')

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy未安装，支撑阻力位计算将使用简化方法")


# 策略名称中英文映射
STRATEGY_NAME_CN = {
    "trend_following": "趋势跟踪",
    "mean_reversion": "均值回归",
    "momentum": "动量策略",
    "volatility": "波动率分析",
    "statistical_arbitrage": "统计套利",
    "kdj": "KDJ指标"
}

# 技术指标权重配置
TECHNICAL_WEIGHTS = {
    "macd": 0.25,
    "rsi": 0.20,
    "bollinger": 0.15,
    "obv": 0.10,
    "kdj": 0.20,
    "volume_change": 0.10
}


def _build_fallback_analysis(reason: str) -> dict:
    return {
        "signal": "neutral",
        "confidence": 0.0,
        "strategy_signals": {
            "trend_following": {"signal": "neutral", "confidence": 0.0, "metrics": {}},
            "mean_reversion": {"signal": "neutral", "confidence": 0.0, "metrics": {}},
            "momentum": {"signal": "neutral", "confidence": 0.0, "metrics": {}},
            "volatility": {"signal": "neutral", "confidence": 0.0, "metrics": {}},
            "statistical_arbitrage": {"signal": "neutral", "confidence": 0.0, "metrics": {}}
        },
        "reasoning": {
            "fallback": {
                "signal": "neutral",
                "details": reason
            }
        }
    }


##### Technical Analyst #####
@resilient_agent
@agent_endpoint("technical_analyst", "技术分析师，提供基于价格走势、指标和技术模式的交易信号")
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    show_workflow_status("技术分析师")

    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)

    latest_price = prices_df['close'].iloc[-1] if len(prices_df) > 0 else 0
    show_agent_reasoning({"数据点": f"{len(prices_df)}条", "最新价": f"{latest_price:.2f}"}, "技术分析师")

    if prices_df.empty or len(prices_df) < 60:
        reason = (
            f"Insufficient price history for technical analysis: "
            f"received {len(prices_df)} rows, need at least 60."
        )
        analysis_report = _build_fallback_analysis(reason)
        message = HumanMessage(
            content=json.dumps(analysis_report, ensure_ascii=False),
            name="technical_analyst_agent",
        )
        if show_reasoning:
            show_agent_reasoning(analysis_report, "技术分析师")
            state["metadata"]["agent_reasoning"] = analysis_report
        show_workflow_complete(
            "技术分析师",
            signal=analysis_report.get('signal'),
            confidence=0,
            details=analysis_report,
            message="技术分析不可用：数据不足"
        )
        return {
            "messages": [message],
            "data": data,
            "metadata": state["metadata"],
        }

    # Initialize confidence variable
    confidence = 0.0

    # Calculate indicators
    macd_line, signal_line = calculate_macd(prices_df)
    macd_val = macd_line.iloc[-1] if len(macd_line) > 0 else 0
    signal_val = signal_line.iloc[-1] if len(signal_line) > 0 else 0

    rsi = calculate_rsi(prices_df)
    rsi_val = rsi.iloc[-1] if len(rsi) > 0 else 50

    upper_band, lower_band = calculate_bollinger_bands(prices_df)

    obv = calculate_obv(prices_df)

    # [NEW] KDJ指标
    kdj = calculate_kdj(prices_df)
    kdj_k = kdj['k'].iloc[-1] if len(kdj) > 0 else 50
    kdj_d = kdj['d'].iloc[-1] if len(kdj) > 0 else 50
    kdj_j = kdj['j'].iloc[-1] if len(kdj) > 0 else 50

    # [NEW] 成交量变化率
    volume_ma5 = prices_df['volume'].rolling(5).mean()
    volume_ma20 = prices_df['volume'].rolling(20).mean()
    volume_change_rate = ((volume_ma5.iloc[-1] - volume_ma20.iloc[-1]) / volume_ma20.iloc[-1]) if volume_ma20.iloc[-1] > 0 else 0

    # [NEW] 支撑阻力位
    try:
        support_resistance = calculate_support_resistance(prices_df)
    except Exception as e:
        logger.warning(f"支撑阻力位计算失败: {e}")
        support_resistance = {'nearest_resistance': None, 'nearest_support': None, 'resistance_distance_pct': None, 'support_distance_pct': None}

    show_agent_reasoning({
        "MACD": f"{macd_val:.4f}", 
        "RSI": f"{rsi_val:.2f}",
        "KDJ": f"K:{kdj_k:.1f} D:{kdj_d:.1f} J:{kdj_j:.1f}",
        "量价变化": f"{volume_change_rate*100:+.1f}%"
    }, "技术分析师")

    # Generate individual signals with weighted approach
    signal_scores = {}

    # MACD signal
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            signal_scores['macd'] = 1
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            signal_scores['macd'] = -1
        else:
            signal_scores['macd'] = 0
    else:
        signal_scores['macd'] = 0

    # RSI signal
    if rsi_val < 30:
        signal_scores['rsi'] = 1
    elif rsi_val > 70:
        signal_scores['rsi'] = -1
    else:
        signal_scores['rsi'] = 0

    # Bollinger Bands signal
    current_price = prices_df['close'].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signal_scores['bollinger'] = 1
    elif current_price > upper_band.iloc[-1]:
        signal_scores['bollinger'] = -1
    else:
        signal_scores['bollinger'] = 0

    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signal_scores['obv'] = 1
    elif obv_slope < 0:
        signal_scores['obv'] = -1
    else:
        signal_scores['obv'] = 0

    # [NEW] KDJ signal
    if kdj_k < 20 or kdj_j < 0:
        signal_scores['kdj'] = 1
    elif kdj_k > 80 or kdj_j > 100:
        signal_scores['kdj'] = -1
    else:
        signal_scores['kdj'] = 0

    # [NEW] Volume change signal
    if volume_change_rate > 0.3:
        signal_scores['volume_change'] = 1
    elif volume_change_rate < -0.3:
        signal_scores['volume_change'] = -1
    else:
        signal_scores['volume_change'] = 0

    # Weighted signal combination
    total_weight = sum(TECHNICAL_WEIGHTS.values())
    weighted_score = sum(signal_scores.get(key, 0) * TECHNICAL_WEIGHTS.get(key, 0) for key in TECHNICAL_WEIGHTS.keys())
    
    if weighted_score > 0.2:
        overall_signal = 'bullish'
    elif weighted_score < -0.2:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'

    confidence = min(abs(weighted_score) / total_weight * 2, 1.0)

    # 兼容性：保留旧的signals列表用于展示
    signals = []
    for key in ['macd', 'rsi', 'bollinger', 'obv', 'kdj', 'volume_change']:
        val = signal_scores.get(key, 0)
        signals.append('bullish' if val == 1 else ('bearish' if val == -1 else 'neutral'))

    # Reasoning collection
    reasoning = {
        "MACD": {"signal": signals[0], "details": f"MACD交叉{'金叉' if signals[0]=='bullish' else '死叉' if signals[0]=='bearish' else '中性'}"},
        "RSI": {"signal": signals[1], "details": f"RSI={rsi_val:.2f}，{'超卖' if signals[1]=='bullish' else '超买' if signals[1]=='bearish' else '中性'}"},
        "Bollinger": {"signal": signals[2], "details": f"价格{'低于下轨' if signals[2]=='bullish' else '高于上轨' if signals[2]=='bearish' else '带内'}"},
        "OBV": {"signal": signals[3], "details": f"OBV斜率{'正' if signals[3]=='bullish' else '负' if signals[3]=='bearish' else '平'}"},
        "KDJ": {"signal": signals[4], "details": f"K={kdj_k:.1f} D={kdj_d:.1f} J={kdj_j:.1f}，{'超卖' if signals[4]=='bullish' else '超买' if signals[4]=='bearish' else '中性'}"},
        "Volume": {"signal": signals[5], "details": f"成交量变化率{volume_change_rate*100:+.1f}%，{'放量' if signals[5]=='bullish' else '缩量' if signals[5]=='bearish' else '持平'}"}
    }

    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 4),
        "reasoning": {
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"]
        }
    }

    # 1. Trend Following Strategy
    trend_signals = calculate_trend_signals(prices_df)
    logger.info(f"📈 趋势跟踪: {trend_signals['signal']} 置信度:{trend_signals['confidence']:.2%}")
    show_agent_reasoning({"strategy_signals": {"trend_following": {"signal": trend_signals['signal'], "confidence": f"{trend_signals['confidence']:.0%}"}}}, "技术分析师")

    # 2. Mean Reversion Strategy
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)
    logger.info(f"📊 均值回归: {mean_reversion_signals['signal']} 置信度:{mean_reversion_signals['confidence']:.2%}")
    show_agent_reasoning({"strategy_signals": {"mean_reversion": {"signal": mean_reversion_signals['signal'], "confidence": f"{mean_reversion_signals['confidence']:.0%}"}}}, "技术分析师")

    # 3. Momentum Strategy
    momentum_signals = calculate_momentum_signals(prices_df)
    logger.info(f"📉 动量策略: {momentum_signals['signal']} 置信度:{momentum_signals['confidence']:.2%}")
    show_agent_reasoning({"strategy_signals": {"momentum": {"signal": momentum_signals['signal'], "confidence": f"{momentum_signals['confidence']:.0%}"}}}, "技术分析师")

    # 4. Volatility Strategy
    volatility_signals = calculate_volatility_signals(prices_df)
    logger.info(f"🌊 波动率策略: {volatility_signals['signal']} 置信度:{volatility_signals['confidence']:.2%}")
    show_agent_reasoning({"strategy_signals": {"volatility": {"signal": volatility_signals['signal'], "confidence": f"{volatility_signals['confidence']:.0%}"}}}, "技术分析师")

    # 5. Statistical Arbitrage Signals
    stat_arb_signals = calculate_stat_arb_signals(prices_df)
    logger.info(f"📐 统计套利: {stat_arb_signals['signal']} 置信度:{stat_arb_signals['confidence']:.2%}")
    show_agent_reasoning({"strategy_signals": {"stat_arb": {"signal": stat_arb_signals['signal'], "confidence": f"{stat_arb_signals['confidence']:.0%}"}}}, "技术分析师")

    # Combine all signals using a weighted ensemble approach
    # [OPTIMIZED] A股特性：强化趋势(45%)，削弱均值回归(5%)，动量保持(30%)
    strategy_weights = {
        'trend': 0.45,        # A股牛短熊长，强化趋势
        'mean_reversion': 0.05, # A股均值回归效果差，削弱
        'momentum': 0.30,     # 动量保持
        'volatility': 0.15,     # 波动性保持
        'stat_arb': 0.05       # 统计套利保持
    }

    combined_signal = weighted_signal_combination({
        'trend': trend_signals,
        'mean_reversion': mean_reversion_signals,
        'momentum': momentum_signals,
        'volatility': volatility_signals,
        'stat_arb': stat_arb_signals
    }, strategy_weights)

    # Generate detailed analysis report
    analysis_report = {
        "signal": combined_signal['signal'],
        "confidence": combined_signal['confidence'],
        "technical_indicators": {
            "macd": {"value": float(macd_val), "signal": signals[0]},
            "rsi": {"value": float(rsi_val), "signal": signals[1]},
            "bollinger": {"upper": float(upper_band.iloc[-1]) if len(upper_band) > 0 else None, 
                        "lower": float(lower_band.iloc[-1]) if len(lower_band) > 0 else None,
                        "signal": signals[2]},
            "obv_slope": float(obv_slope) if not pd.isna(obv_slope) else 0,
            "kdj": {"k": float(kdj_k), "d": float(kdj_d), "j": float(kdj_j), "signal": signals[4]},
            "volume_change_rate": float(volume_change_rate) if not pd.isna(volume_change_rate) else 0
        },
        "support_resistance": support_resistance,
        "reasoning": reasoning,
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals['signal'],
                "confidence": trend_signals['confidence'],
                "metrics": normalize_pandas(trend_signals['metrics'])
            },
            "mean_reversion": {
                "signal": mean_reversion_signals['signal'],
                "confidence": mean_reversion_signals['confidence'],
                "metrics": normalize_pandas(mean_reversion_signals['metrics'])
            },
            "momentum": {
                "signal": momentum_signals['signal'],
                "confidence": momentum_signals['confidence'],
                "metrics": normalize_pandas(momentum_signals['metrics'])
            },
            "volatility": {
                "signal": volatility_signals['signal'],
                "confidence": volatility_signals['confidence'],
                "metrics": normalize_pandas(volatility_signals['metrics'])
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals['signal'],
                "confidence": stat_arb_signals['confidence'],
                "metrics": normalize_pandas(stat_arb_signals['metrics'])
            }
        }
    }

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(analysis_report, ensure_ascii=False),
        name="technical_analyst_agent",
    )

    if show_reasoning:
        show_agent_reasoning(analysis_report, "技术分析师")
        # 保存推理信息到state的metadata供API使用
        state["metadata"]["agent_reasoning"] = analysis_report

    # 发送最终分析结果摘要到前端
    final_summary = {
        "最终信号": analysis_report.get('signal'),
        "置信度": analysis_report.get('confidence'),
        "策略详情": {k: v.get('signal') for k, v in analysis_report.get('strategy_signals', {}).items()}
    }
    show_agent_reasoning(final_summary, "技术分析师")

    show_workflow_complete(
        "技术分析师",
        signal=analysis_report.get('signal'),
        confidence=combined_signal.get('confidence', 0),
        details=analysis_report,
        message=f"技术分析完成，信号:{analysis_report.get('signal')}，置信度:{analysis_report.get('confidence')}"
    )

    return {
        "messages": [message],
        "data": {
            **data,
            "prices": prices[-100:] if len(prices) > 0 else []
        },
        "metadata": state["metadata"],
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    [OPTIMIZED] 添加A股特色：量价配合、均线多头排列
    """
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    ema_120 = calculate_ema(prices_df, 120)

    adx = calculate_adx(prices_df, 14)

    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    ma5_above_ma10 = ema_8.iloc[-1] > ema_21.iloc[-1]
    ma10_above_ma20 = ema_21.iloc[-1] > ema_55.iloc[-1]
    ma20_above_ma60 = ema_55.iloc[-1] > ema_120.iloc[-1] if len(ema_120) > 0 else False

    volume_ma20 = prices_df['volume'].rolling(20).mean()
    current_price = prices_df['close'].iloc[-1]
    prev_price = prices_df['close'].iloc[-2]
    price_up = current_price > prev_price
    volume_confirmation = (prices_df['volume'].iloc[-1] / volume_ma20.iloc[-1]) > 1.2 if volume_ma20.iloc[-1] > 0 else False

    volume_change_rate = ((prices_df['volume'].iloc[-1] - prices_df['volume'].iloc[-5]) / 
                          prices_df['volume'].iloc[-5]) if prices_df['volume'].iloc[-5] > 0 else 0

    # Combine signals with confidence weighting
    trend_strength = adx['adx'].iloc[-1] / 100.0

    # 基础信号判定
    if ma5_above_ma10 and ma10_above_ma20:
        base_signal = 'bullish'
        base_confidence = min(trend_strength * 1.2, 1.0)  # 均线多头强化
    elif not ma5_above_ma10 and not ma10_above_ma20:
        base_signal = 'bearish'
        base_confidence = min(trend_strength * 1.2, 1.0)
    else:
        base_signal = 'neutral'
        base_confidence = 0.5

    # [OPTIMIZED] 量价确认加成
    if price_up and volume_confirmation:
        if base_signal == 'bullish':
            base_confidence = min(base_confidence * 1.1, 1.0)  # 上涨且放量，确认趋势
    elif not price_up and not volume_confirmation:
        if base_signal == 'bearish':
            base_confidence = min(base_confidence * 1.1, 1.0)

    ma_alignment = 'bullish' if (ma5_above_ma10 and ma10_above_ma20) else ('bearish' if (not ma5_above_ma10 and not ma10_above_ma20) else 'neutral')

    return {
        'signal': base_signal,
        'confidence': base_confidence,
        'metrics': {
            'adx': float(adx['adx'].iloc[-1]) if len(adx) > 0 else 0,
            'trend_strength': float(trend_strength),
            'volume_confirmation': float(volume_confirmation) if volume_confirmation else 0,
            'volume_change_rate': float(volume_change_rate) if not pd.isna(volume_change_rate) else 0,
            'ma_alignment': ma_alignment,
            'ema_8': float(ema_8.iloc[-1]) if len(ema_8) > 0 else 0,
            'ema_21': float(ema_21.iloc[-1]) if len(ema_21) > 0 else 0,
            'ema_55': float(ema_55.iloc[-1]) if len(ema_55) > 0 else 0
        }
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df['close'].rolling(window=50).mean()
    std_50 = prices_df['close'].rolling(window=50).std()
    z_score = (prices_df['close'] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    extreme_z_score = abs(z_score.iloc[-1]) > 2
    price_vs_bb = (prices_df['close'].iloc[-1] - bb_lower.iloc[-1]
                   ) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = 'bullish'
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = 'bearish'
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'z_score': float(z_score.iloc[-1]),
            'price_vs_bb': float(price_vs_bb),
            'rsi_14': float(rsi_14.iloc[-1]),
            'rsi_28': float(rsi_28.iloc[-1])
        }
    }


def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy with conservative settings
    """
    # Price momentum with adjusted min_periods
    returns = prices_df['close'].pct_change()
    mom_1m = returns.rolling(21, min_periods=5).sum()  # 短期动量允许较少数据点
    mom_3m = returns.rolling(63, min_periods=42).sum()  # 中期动量要求更多数据点
    mom_6m = returns.rolling(126, min_periods=63).sum()  # 长期动量保持严格要求

    # Volume momentum
    volume_ma = prices_df['volume'].rolling(21, min_periods=10).mean()
    volume_momentum = prices_df['volume'] / volume_ma

    # 处理NaN值
    mom_1m = mom_1m.fillna(0)  # 短期动量可以用0填充
    mom_3m = mom_3m.fillna(mom_1m)  # 中期动量可以用短期动量填充
    mom_6m = mom_6m.fillna(mom_3m)  # 长期动量可以用中期动量填充

    # Calculate momentum score with more weight on longer timeframes
    momentum_score = (
        0.2 * mom_1m +  # 降低短期权重
        0.3 * mom_3m +
        0.5 * mom_6m    # 增加长期权重
    ).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = 'bullish'
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = 'bearish'
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'momentum_1m': float(mom_1m.iloc[-1]),
            'momentum_3m': float(mom_3m.iloc[-1]),
            'momentum_6m': float(mom_6m.iloc[-1]),
            'volume_momentum': float(volume_momentum.iloc[-1])
        }
    }


def calculate_volatility_signals(prices_df):
    """
    Optimized volatility calculation with shorter lookback periods
    """
    returns = prices_df['close'].pct_change()

    # 使用更短的周期和最小周期要求计算历史波动率
    hist_vol = returns.rolling(21, min_periods=10).std() * math.sqrt(252)

    # 使用更短的周期计算波动率均值，并允许更少的数据点
    vol_ma = hist_vol.rolling(42, min_periods=21).mean()
    vol_regime = hist_vol / vol_ma

    # 使用更灵活的标准差计算
    vol_std = hist_vol.rolling(42, min_periods=21).std()
    vol_z_score = (hist_vol - vol_ma) / vol_std.replace(0, np.nan)

    # ATR计算优化
    atr = calculate_atr(prices_df, period=14, min_periods=7)
    atr_ratio = atr / prices_df['close']

    # 如果关键指标为NaN，使用替代值而不是直接返回中性信号
    if pd.isna(vol_regime.iloc[-1]):
        vol_regime.iloc[-1] = 1.0  # 假设处于正常波动率区间
    if pd.isna(vol_z_score.iloc[-1]):
        vol_z_score.iloc[-1] = 0.0  # 假设处于均值位置

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = 'bullish'  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = 'bearish'  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'historical_volatility': float(hist_vol.iloc[-1]),
            'volatility_regime': float(current_vol_regime),
            'volatility_z_score': float(vol_z),
            'atr_ratio': float(atr_ratio.iloc[-1])
        }
    }


def calculate_stat_arb_signals(prices_df):
    """
    Optimized statistical arbitrage signals with shorter lookback periods
    """
    # Calculate price distribution statistics
    returns = prices_df['close'].pct_change()

    # 使用更短的周期计算偏度和峰度
    skew = returns.rolling(42, min_periods=21).skew()
    kurt = returns.rolling(42, min_periods=21).kurt()

    # 优化Hurst指数计算
    hurst = calculate_hurst_exponent(prices_df['close'], max_lag=10)

    # 处理NaN值
    if pd.isna(skew.iloc[-1]):
        skew.iloc[-1] = 0.0  # 假设正态分布
    if pd.isna(kurt.iloc[-1]):
        kurt.iloc[-1] = 3.0  # 假设正态分布

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = 'bullish'
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = 'bearish'
        confidence = (0.5 - hurst) * 2
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'hurst_exponent': float(hurst),
            'skewness': float(skew.iloc[-1]),
            'kurtosis': float(kurt.iloc[-1])
        }
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal['signal']]
        weight = weights[strategy]
        confidence = signal['confidence']

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = 'bullish'
    elif final_score < -0.2:
        signal = 'bearish'
    else:
        signal = 'neutral'

    return {
        'signal': signal,
        'confidence': abs(final_score)
    }


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_kdj(prices_df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """
    Calculate KDJ indicator (Stochastic)
    """
    low_min = prices_df['low'].rolling(window=period).min()
    high_max = prices_df['high'].rolling(window=period).max()
    
    rsv = (prices_df['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return pd.DataFrame({'k': k, 'd': d, 'j': j})


def calculate_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df['close'].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # Calculate ADX
    df['+di'] = 100 * (df['plus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['-di'] = 100 * (df['minus_dm'].ewm(span=period).mean() /
                       df['tr'].ewm(span=period).mean())
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].ewm(span=period).mean()

    return df[['adx', '+di', '-di']]


def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud indicators

    Args:
        df: DataFrame with OHLC data

    Returns:
        Dictionary containing Ichimoku components
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = df['close'].shift(-26)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def calculate_atr(df: pd.DataFrame, period: int = 14, min_periods: int = 7) -> pd.Series:
    """
    Optimized ATR calculation with minimum periods parameter

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
        min_periods: Minimum number of periods required

    Returns:
        pd.Series: ATR values
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period, min_periods=min_periods).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 10) -> float:
    """
    Optimized Hurst exponent calculation with shorter lookback and better error handling

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation (reduced from 20 to 10)

    Returns:
        float: Hurst exponent
    """
    try:
        # 使用对数收益率而不是价格
        returns = np.log(price_series / price_series.shift(1)).dropna()

        # 如果数据不足，返回0.5（随机游走）
        if len(returns) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        # 使用更稳定的计算方法
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag])))
               for lag in lags]

        # 添加小的常数避免log(0)
        tau = [max(1e-8, t) for t in tau]

        # 使用对数回归计算Hurst指数
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        h = reg[0]

        # 限制Hurst指数在合理范围内
        return max(0.0, min(1.0, h))

    except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
        # 如果计算失败，返回0.5表示随机游走
        return 0.5


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']


def calculate_support_resistance(prices_df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Calculate support and resistance levels
    """
    recent = prices_df.tail(lookback)
    current_price = prices_df['close'].iloc[-1]
    
    if not SCIPY_AVAILABLE:
        high_max = recent['high'].max()
        low_min = recent['low'].min()
        return {
            'resistance_levels': [float(high_max)],
            'support_levels': [float(low_min)],
            'nearest_resistance': float(high_max),
            'nearest_support': float(low_min),
            'resistance_distance_pct': ((high_max - current_price) / current_price * 100),
            'support_distance_pct': ((current_price - low_min) / current_price * 100)
        }
    
    high_prices = recent['high'].values
    low_prices = recent['low'].values
    
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(high_prices, distance=5)
    troughs, _ = find_peaks(-low_prices, distance=5)
    
    resistance_levels = high_prices[peaks].tolist() if len(peaks) > 0 else []
    support_levels = low_prices[troughs].tolist() if len(troughs) > 0 else []
    
    resistance_levels.sort(reverse=True)
    support_levels.sort()
    
    nearest_resistance = next((r for r in resistance_levels if r > current_price), None)
    nearest_support = next((s for s in support_levels if s < current_price), None)
    
    return {
        'resistance_levels': resistance_levels[:3],
        'support_levels': support_levels[:3],
        'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
        'nearest_support': float(nearest_support) if nearest_support else None,
        'resistance_distance_pct': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
        'support_distance_pct': ((current_price - nearest_support) / current_price * 100) if nearest_support else None
    }
