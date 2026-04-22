from typing import Union, Dict, List, Optional

class ConfidenceNormalizer:
    """置信度标准化工具"""

    @staticmethod
    def parse(value: Union[str, float, int]) -> float:
        """
        解析置信度值为浮点数
        
        Args:
            value: "75%" 或 0.75
            
        Returns:
            0.0-1.0的浮点数
        """
        if isinstance(value, str):
            value = value.strip().replace("%", "")
            try:
                return float(value) / 100.0
            except ValueError:
                return 0.0
        elif isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        return 0.0

    @staticmethod
    def normalize(confidence: float, data_quality_score: float = 1.0) -> float:
        """
        基于数据质量标准化置信度
        
        Args:
            confidence: agent原始置信度
            data_quality_score: 数据质量评分 (0-1)
            
        Returns:
            标准化后的置信度
        """
        confidence = max(0.0, min(1.0, confidence))
        
        # 数据质量低时, 降低置信度上限
        if data_quality_score <= 0.5:
            # 数据质量越低，置信度上限越低
            max_conf = 0.3 + data_quality_score * 0.4  # 0.5->0.5, 0.3->0.42
            return min(confidence, max_conf)
        
        return confidence

    @staticmethod
    def fallback_confidence(agent_name: str) -> float:
        """
        Fallback时的默认置信度
        
        Args:
            agent_name: agent名称
            
        Returns:
            默认置信度
        """
        fallbacks = {
            "technical": 0.30,
            "fundamentals": 0.40,
            "sentiment": 0.25,
            "valuation": 0.35,
            "macro": 0.20,
            "debate": 0.40,
            "risk": 0.60,
            "researcher_bull": 0.25,
            "researcher_bear": 0.30,
        }
        return fallbacks.get(agent_name, 0.30)

    @staticmethod
    def weighted_average(signals: List[Dict], weights: Dict[str, float]) -> tuple:
        """
        计算加权平均信号
        
        Args:
            signals: [{"signal": "bullish", "confidence": 0.75, "agent": "technical"}, ...]
            weights: {"technical": 0.15, ...}
            
        Returns:
            (最终信号, 置信度)
        """
        signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in signals:
            signal = item.get("signal", "neutral")
            confidence = item.get("confidence", 0.3)
            agent_key = item.get("agent", "unknown")
            
            weight = weights.get(agent_key, 0.1)
            weighted_sum += signal_values.get(signal, 0) * confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return "neutral", 0.0
        
        final_score = weighted_sum / total_weight
        
        if final_score > 0.2:
            return "bullish", min(abs(final_score), 0.9)
        elif final_score < -0.2:
            return "bearish", min(abs(final_score), 0.9)
        else:
            return "neutral", 0.5


def normalize(confidence: float, data_quality_score: float = 1.0) -> float:
    """便捷函数"""
    return ConfidenceNormalizer.normalize(confidence, data_quality_score)


def fallback_confidence(agent_name: str) -> float:
    """便捷函数"""
    return ConfidenceNormalizer.fallback_confidence(agent_name)