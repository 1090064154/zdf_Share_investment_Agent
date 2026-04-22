from typing import Dict, List, Optional
import numpy as np

class DynamicWeights:
    """动态权重计算器"""

    @staticmethod
    def detect_market_state(prices: List[float]) -> str:
        """
        根据价格序列检测市场状态
        
        Args:
            prices: 沪深300指数的历史价格
            
        Returns:
            "bull_market" | "bear_market" | "震荡市"
        """
        if prices is None or len(prices) < 20:
            return "震荡市"
        
        try:
            prices_arr = np.array(prices)
            
            # 计算20日均线
            ma20 = np.mean(prices_arr[-20:])
            current_price = prices_arr[-1]
            
            # 计算月度变化
            monthly_return = (current_price - prices_arr[-20]) / prices_arr[-20] if len(prices_arr) >= 20 else 0
            
            if monthly_return > 0.05:  # 5%以上上涨
                return "bull_market"
            elif monthly_return < -0.05:  # 5%以上下跌
                return "bear_market"
            else:
                return "震荡市"
        except (ZeroDivisionError, IndexError, TypeError):
            return "震荡市"

    @staticmethod
    def calculate(market_state: str, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        根据市场状态计算动态权重
        
        Args:
            market_state: "bull_market" | "bear_market" | "震荡市"
            base_weights: 如果提供,以此为基础调整(可选,已废弃)
            
        Returns:
            各维度权重字典
        """
        DEFAULT_WEIGHTS = {
            "bull_market": {
                "valuation": 0.30,
                "technical": 0.35,
                "fundamentals": 0.10,
                "macro": 0.05,
                "sentiment": 0.05,
                "debate": 0.05,
                "risk": 0.10,
            },
            "bear_market": {
                "valuation": 0.30,
                "fundamentals": 0.35,
                "technical": 0.10,
                "macro": 0.05,
                "sentiment": 0.05,
                "debate": 0.05,
                "risk": 0.10,
            },
            "震荡市": {
                "valuation": 0.25,
                "technical": 0.15,
                "fundamentals": 0.20,
                "macro": 0.10,
                "sentiment": 0.10,
                "debate": 0.10,
                "risk": 0.10,
            },
        }
        
        return DEFAULT_WEIGHTS.get(market_state, DEFAULT_WEIGHTS["震荡市"])

    @staticmethod
    def apply_stock_type_adjustment(weights: Dict[str, float], stock_type: str) -> Dict[str, float]:
        """
        根据股票类型调整权重
        
        Args:
            weights: 基础权重
            stock_type: "蓝筹股" | "题材股" | "周期股" | "成长股"
            
        Returns:
            调整后的权重
        """
        adjustments = {
            "蓝筹股": {"valuation": 1.2, "fundamentals": 1.1},
            "题材股": {"sentiment": 1.3, "technical": 1.2},
            "周期股": {"macro": 1.2, "technical": 1.1},
            "成长股": {"fundamentals": 1.2, "valuation": 1.1},
        }
        
        adj = adjustments.get(stock_type, {})
        result = weights.copy()
        for key, factor in adj.items():
            if key in result:
                result[key] = min(0.5, result[key] * factor)  # 最高0.5
        
        return result


def detect_market_state(prices: List[float]) -> str:
    """便捷函数"""
    return DynamicWeights.detect_market_state(prices)


def calculate(market_state: str) -> Dict[str, float]:
    """便捷函数"""
    return DynamicWeights.calculate(market_state)