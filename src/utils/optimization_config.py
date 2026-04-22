import os
from pathlib import Path
from typing import Dict, Any, Optional

class OptimizationConfig:
    """优化配置单例类"""
    _instance: Optional['OptimizationConfig'] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> 'OptimizationConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config is not None:
            return self._config
        
        if config_path is None:
            config_path = os.environ.get('OPTIMIZATION_CONFIG')
        
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "optimization.yaml"
        
        if not Path(config_path).exists():
            return self._default_config()
        
        import yaml
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        
        return self._config

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "optimization": {
                "enable_veto_power": True,
                "enable_dynamic_weights": True,
                "enable_confidence_normalizer": True,
            },
            "weights": {
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
            },
            "confidence_fallback": {
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
        }

    @property
    def enable_veto_power(self) -> bool:
        """是否启用一票否决权"""
        if self._config is None:
            self.load()
        return self._config.get("optimization", {}).get("enable_veto_power", True)

    @property
    def enable_dynamic_weights(self) -> bool:
        """是否启用动态权重"""
        if self._config is None:
            self.load()
        return self._config.get("optimization", {}).get("enable_dynamic_weights", True)

    @property
    def enable_confidence_normalizer(self) -> bool:
        """是否启用置信度标准化"""
        if self._config is None:
            self.load()
        return self._config.get("optimization", {}).get("enable_confidence_normalizer", True)

    def get_weights(self, market_state: str) -> Dict[str, float]:
        """获取指定市场状态的权重"""
        if self._config is None:
            self.load()
        return self._config.get("weights", {}).get(market_state, {})

    def get_confidence_fallback(self, agent_name: str) -> float:
        """获取指定agent的置信度fallback"""
        if self._config is None:
            self.load()
        return self._config.get("confidence_fallback", {}).get(agent_name, 0.30)


def get_config() -> OptimizationConfig:
    """获取配置单例"""
    config = OptimizationConfig()
    config.load()
    return config