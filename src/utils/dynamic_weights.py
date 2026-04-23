"""
动态权重配置模块

根据市场环境、股票类型等因素动态调整各Agent信号权重，提升决策适应性。
"""

from typing import Dict, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger('dynamic_weights')


# 基础权重配置（中性市场）
BASE_WEIGHTS = {
    'technical': 0.15,
    'fundamentals': 0.20,
    'sentiment': 0.05,
    'valuation': 0.25,
    'risk': 0.10,
    'macro': 0.10,
    'debate': 0.15,
}

# 牛市权重配置
BULL_MARKET_WEIGHTS = {
    'technical': 0.25,      # 技术面更重要
    'fundamentals': 0.15,
    'sentiment': 0.20,      # 情绪面重要
    'valuation': 0.15,
    'risk': 0.10,
    'macro': 0.10,
    'debate': 0.05,
}

# 熊市权重配置
BEAR_MARKET_WEIGHTS = {
    'technical': 0.10,
    'fundamentals': 0.25,   # 基本面更重要
    'sentiment': 0.05,
    'valuation': 0.20,      # 估值重要
    'risk': 0.25,           # 风险管理最重要
    'macro': 0.10,
    'debate': 0.05,
}

# 周期股权重配置
CYCLICAL_STOCK_WEIGHTS = {
    'technical': 0.15,
    'fundamentals': 0.10,
    'sentiment': 0.05,
    'valuation': 0.30,      # 估值最重要
    'risk': 0.10,
    'macro': 0.25,          # 宏观环境重要
    'debate': 0.05,
}

# 成长股权重配置
GROWTH_STOCK_WEIGHTS = {
    'technical': 0.20,
    'fundamentals': 0.25,   # 基本面重要
    'sentiment': 0.10,
    'valuation': 0.15,
    'risk': 0.10,
    'macro': 0.10,
    'debate': 0.10,
}

# 防御板块权重配置
DEFENSIVE_STOCK_WEIGHTS = {
    'technical': 0.10,
    'fundamentals': 0.20,
    'sentiment': 0.05,
    'valuation': 0.20,
    'risk': 0.15,
    'macro': 0.15,
    'debate': 0.15,
}


def get_market_regime_weights(regime: str = 'neutral') -> Dict[str, float]:
    """
    根据市场状态获取权重配置
    
    Args:
        regime: 市场状态 ('bull', 'bear', 'neutral')
        
    Returns:
        权重字典
    """
    weights_map = {
        'bull': BULL_MARKET_WEIGHTS,
        'bear': BEAR_MARKET_WEIGHTS,
        'neutral': BASE_WEIGHTS,
    }
    
    return weights_map.get(regime, BASE_WEIGHTS).copy()


def get_stock_type_weights(stock_type: str) -> Dict[str, float]:
    """
    根据股票类型获取权重配置
    
    Args:
        stock_type: 股票类型 ('cyclical', 'growth', 'defensive', 'blue_chip')
        
    Returns:
        权重字典
    """
    weights_map = {
        'cyclical': CYCLICAL_STOCK_WEIGHTS,
        'growth': GROWTH_STOCK_WEIGHTS,
        'defensive': DEFENSIVE_STOCK_WEIGHTS,
        'blue_chip': BASE_WEIGHTS,  # 蓝筹股使用基础权重
    }
    
    return weights_map.get(stock_type, BASE_WEIGHTS).copy()


def merge_weights(base_weights: Dict[str, float], adjustment_weights: Dict[str, float], 
                  adjustment_factor: float = 0.3) -> Dict[str, float]:
    """
    合并基础权重和调整权重
    
    Args:
        base_weights: 基础权重
        adjustment_weights: 调整权重
        adjustment_factor: 调整因子 (0-1)，控制调整幅度
        
    Returns:
        合并后的权重
    """
    merged = {}
    for key in base_weights:
        base = base_weights.get(key, 0)
        adjustment = adjustment_weights.get(key, 0)
        merged[key] = base * (1 - adjustment_factor) + adjustment * adjustment_factor
    
    # 归一化
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}
    
    return merged


def calculate_dynamic_weights(market_regime: str = 'neutral', 
                             stock_type: str = 'other',
                             custom_adjustments: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    计算动态权重
    
    Args:
        market_regime: 市场状态
        stock_type: 股票类型
        custom_adjustments: 自定义调整
        
    Returns:
        动态权重字典
    """
    # 1. 获取市场状态权重
    regime_weights = get_market_regime_weights(market_regime)
    
    # 2. 获取股票类型权重
    type_weights = get_stock_type_weights(stock_type)
    
    # 3. 合并权重（股票类型占30%权重）
    final_weights = merge_weights(regime_weights, type_weights, adjustment_factor=0.3)
    
    # 4. 应用自定义调整
    if custom_adjustments:
        final_weights = merge_weights(final_weights, custom_adjustments, adjustment_factor=0.2)
    
    logger.info(f"🎯 [DYNAMIC_WEIGHTS] 市场={market_regime}, 类型={stock_type}")
    logger.info(f"   最终权重: {', '.join([f'{k}={v:.2f}' for k, v in sorted(final_weights.items(), key=lambda x: -x[1])])}")
    
    return final_weights


# 便捷函数
def get_agent_weights(market_regime: str = 'neutral', stock_type: str = 'other') -> Dict[str, float]:
    """获取Agent权重配置的便捷函数"""
    return calculate_dynamic_weights(market_regime, stock_type)
