"""
规则化决策引擎
按照计划文档3.7节实现
"""
from typing import Dict, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger('decision_engine')

DEFAULT_WEIGHTS = {
    'valuation': 0.25,
    'fundamentals': 0.20,
    'technical': 0.15,
    'industry_cycle': 0.10,
    'institutional': 0.10,
    'sentiment': 0.05,
    'macro': 0.05,
    'risk': 0.10
}


class DecisionEngine:
    """
    规则化决策引擎
    """

    def __init__(self, config=None):
        if config is not None:
            # Handle OptimizationConfig object
            if hasattr(config, 'get_agent_weights'):
                self.weights = config.get_agent_weights()
            else:
                # Handle dictionary
                self.weights = config.get('agent_weights', DEFAULT_WEIGHTS)
        else:
            self.weights = DEFAULT_WEIGHTS

    def make_decision(self, signals: Dict, risk_score: float, risk_action: str, macro_factor: float, portfolio: Dict) -> Dict:
        """
        决策入口
        """
        logger.info(f"DecisionEngine: 开始决策, signals={list(signals.keys())}, risk_score={risk_score}, risk_action={risk_action}, macro_factor={macro_factor}")

        # Step 1: 风险过滤
        if risk_score >= 7:
            return self._risk_filter_decision(risk_score, portfolio)

        if risk_action in ['sell', 'reduce', '减仓', '清仓']:
            return self._risk_action_decision(risk_action, portfolio)

        if macro_factor < 0.7:
            return self._macro_decision(macro_factor, portfolio)

        # Step 2: 信号加权
        weighted_signals = self._calculate_weighted_signals(signals)

        # Step 3: 决策规则
        return self._apply_decision_rules(weighted_signals, risk_score, macro_factor, portfolio)

    def _risk_filter_decision(self, risk_score: float, portfolio: Dict) -> Dict:
        """
        风险过滤决策
        """
        logger.warning(f"风险过滤: risk_score={risk_score} >= 7, 强制持有")
        return {
            'action': 'hold',
            'quantity': portfolio.get('stock', 0),
            'confidence': 0.3,
            'reason': f'风险评分{risk_score:.0f} >= 7，强制持有'
        }

    def _risk_action_decision(self, risk_action: str, portfolio: Dict) -> Dict:
        """
        风险行动决策
        """
        current_position = portfolio.get('stock', 0)

        if risk_action in ['sell', '清仓']:
            if current_position > 0:
                logger.warning(f"风险行动: 卖出, quantity={current_position}")
                return {
                    'action': 'sell',
                    'quantity': current_position,
                    'confidence': 0.8,
                    'reason': '风险管理建议卖出'
                }
            else:
                logger.warning(f"风险行动: 无持仓可卖，阻止买入")
                return {
                    'action': 'hold',
                    'quantity': 0,
                    'confidence': 0.5,
                    'reason': '风险管理建议卖出但无持仓，阻止买入'
                }
        elif risk_action in ['reduce', '减仓']:
            if current_position > 0:
                reduce_quantity = int(current_position * 0.5)
                logger.warning(f"风险行动: 减仓50%, quantity={reduce_quantity}")
                return {
                    'action': 'sell',
                    'quantity': reduce_quantity,
                    'confidence': 0.6,
                    'reason': '风险管理建议减仓50%'
                }

        return {
            'action': 'hold',
            'quantity': 0,
            'confidence': 0.5,
            'reason': '风险���理无明确建议'
        }

    def _macro_decision(self, macro_factor: float, portfolio: Dict) -> Dict:
        """
        宏观决策
        """
        logger.warning(f"宏观决策: macro_factor={macro_factor:.2f} < 0.7, 建议观望")
        return {
            'action': 'hold',
            'quantity': portfolio.get('stock', 0),
            'confidence': 0.4,
            'reason': f'宏观环境较差(系数{macro_factor:.2f})，建议观望'
        }

    def _calculate_weighted_signals(self, signals: Dict) -> Dict:
        """
        计算加权信号
        """
        weighted = {}

        for agent_name, weight in self.weights.items():
            if agent_name in signals:
                signal = signals[agent_name]
                signal_value = {'bullish': 1, 'neutral': 0, 'bearish': -1}.get(signal.get('signal', 'neutral'), 0)
                confidence = signal.get('confidence', 0.5)

                weighted[agent_name] = {
                    'value': signal_value * weight * confidence,
                    'raw_signal': signal.get('signal', 'neutral'),
                    'confidence': confidence
                }

        logger.debug(f"加权信号: {[(k, v['value'], v['raw_signal']) for k, v in weighted.items()]}")
        return weighted

    def _apply_decision_rules(self, weighted_signals: Dict, risk_score: float, macro_factor: float, portfolio: Dict) -> Dict:
        """
        应用决策规则
        """
        if not weighted_signals:
            return {
                'action': 'hold',
                'quantity': portfolio.get('stock', 0),
                'confidence': 0.3,
                'reason': '无有效信号，观望'
            }

        # 计算总分
        total_score = sum(w['value'] for w in weighted_signals.values())

        # 统计信号
        bullish_count = sum(1 for w in weighted_signals.values() if w['raw_signal'] == 'bullish')
        bearish_count = sum(1 for w in weighted_signals.values() if w['raw_signal'] == 'bearish')
        neutral_count = sum(1 for w in weighted_signals.values() if w['raw_signal'] == 'neutral')

        logger.info(f"决策规则: total_score={total_score:.3f}, bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}")

        # 决策规则
        # 规则1：多数看多且总分>0.3
        if bullish_count >= 4 and total_score > 0.3 and risk_score < 5:
            action = 'buy'
            quantity = self._calculate_buy_quantity(portfolio, risk_score, macro_factor)
            confidence = min(abs(total_score) + 0.3, 0.9)
            logger.info(f"决策: 规则1触发 -> buy, quantity={quantity}")

        # 规则2：多数看空或总分<-0.3
        elif bearish_count >= 3 or total_score < -0.3:
            action = 'sell'
            quantity = portfolio.get('stock', 0)
            confidence = min(abs(total_score) + 0.3, 0.9)
            logger.info(f"决策: 规则2触发 -> sell, quantity={quantity}")

        # 规则3：看多但风险偏高
        elif bullish_count >= 3 and risk_score >= 5:
            action = 'hold'
            quantity = portfolio.get('stock', 0)
            confidence = 0.4
            reason = '看多但风险偏高，建议观望'
            logger.info(f"决策: 规则3触发 -> hold (看多但风险偏高)")

        # 规则4：中性或均衡
        else:
            action = 'hold'
            quantity = portfolio.get('stock', 0)
            confidence = 0.3
            reason = '信号均衡，观望为主'
            logger.info(f"决策: 规则4触发 -> hold (信号均衡)")

        return {
            'action': action,
            'quantity': quantity,
            'confidence': confidence,
            'reason': self._generate_reason(
                weighted_signals, total_score, action,
                bullish_count, bearish_count, neutral_count, risk_score
            )
        }

    def _calculate_buy_quantity(self, portfolio: Dict, risk_score: float, macro_factor: float) -> int:
        """
        计算买入数量
        """
        cash = portfolio.get('cash', 0)
        current_price = portfolio.get('current_price', 0)

        if current_price <= 0:
            return 0

        risk_adjustment = 1.0 - (risk_score / 20)
        macro_adjustment = macro_factor

        base_quantity = cash / current_price
        adjusted_quantity = base_quantity * risk_adjustment * macro_adjustment

        return int(adjusted_quantity)

    def _generate_reason(self, weighted_signals: Dict, total_score: float, action: str, 
                         bullish_count: int = 0, bearish_count: int = 0, 
                         neutral_count: int = 0, risk_score: float = 0) -> str:
        """
        生成详细决策理由
        """
        reasons = []
        
        reasons.append(f"看多{bullish_count}个、看空{bearish_count}个、中性{neutral_count}个")
        
        reasons.append(f"加权总分: {total_score:.3f}")
        
        positive = [(k, v['value'], v['raw_signal'], v['confidence']) for k, v in weighted_signals.items() if v['value'] > 0]
        positive.sort(key=lambda x: x[1], reverse=True)
        
        negative = [(k, v['value'], v['raw_signal'], v['confidence']) for k, v in weighted_signals.items() if v['value'] < 0]
        negative.sort(key=lambda x: x[1])
        
        if positive:
            reasons.append("正向因素:")
            for name, val, signal, conf in positive[:4]:
                reasons.append(f"  - {name}({signal}): {val:.3f}(置信度{conf:.0%})")
        
        if negative:
            reasons.append("负向因素:")
            for name, val, signal, conf in negative[:4]:
                reasons.append(f"  - {name}({signal}): {val:.3f}(置信度{conf:.0%})")
        
        reasons.append(f"风险评分: {risk_score:.0f}/10")
        
        if action == 'buy':
            if bullish_count >= 4 and total_score > 0.3:
                reasons.append("触发规则: 多数看多(>=4)且总分>0.3，建议买入")
            else:
                reasons.append(f"触发规则: 综合评估，建议买入")
        elif action == 'sell':
            if bearish_count >= 3 or total_score < -0.3:
                reasons.append("触发规则: 多数看空(>=3)或总分<-0.3，建议卖出")
            else:
                reasons.append(f"触发规则: 综合评估，建议卖出")
        else:
            if bullish_count >= 3 and risk_score >= 5:
                reasons.append("触发规则: 看多但风险偏高，观望")
            else:
                reasons.append("触发规则: 信号均衡或不符合买入条件，观望")

        return '; '.join(reasons)


def create_decision_engine(config: Optional[Dict] = None) -> DecisionEngine:
    """便捷工厂函数"""
    return DecisionEngine(config)