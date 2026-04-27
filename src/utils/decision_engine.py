"""
规则化决策引擎
按照计划文档3.7节实现
"""
from typing import Dict, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger('decision_engine')

INVESTMENT_HORIZON_WEIGHTS = {
    'short': {  # 1-5天：短线交易
        'technical': 0.30,
        'sentiment': 0.25,
        'fundamentals': 0.10,
        'valuation': 0.10,
        'institutional': 0.10,
        'industry_cycle': 0.05,
        'macro': 0.05,
        'risk': 0.05
    },
    'medium': {  # 1-3个月：中线投资
        'technical': 0.20,
        'sentiment': 0.15,
        'fundamentals': 0.20,
        'valuation': 0.15,
        'institutional': 0.10,
        'industry_cycle': 0.10,
        'macro': 0.05,
        'risk': 0.05
    },
    'long': {  # 6个月+：长线投资
        'fundamentals': 0.25,
        'valuation': 0.20,
        'industry_cycle': 0.15,
        'macro': 0.15,
        'technical': 0.05,
        'sentiment': 0.05,
        'institutional': 0.10,
        'risk': 0.05
    }
}

DEFAULT_WEIGHTS = INVESTMENT_HORIZON_WEIGHTS['medium']


class DecisionEngine:
    """
    规则化决策引擎
    """

    def __init__(self, config=None, investment_horizon: str = 'medium'):
        """
        初始化决策引擎
        
        Args:
            config: 配置对象（包含agent_weights）
            investment_horizon: 持仓周期 ('short'|'medium'|'long')，默认'medium'
        """
        self.investment_horizon = investment_horizon
        
        if config is not None:
            if hasattr(config, 'get_agent_weights'):
                self.weights = config.get_agent_weights()
            else:
                self.weights = config.get('agent_weights', 
                    INVESTMENT_HORIZON_WEIGHTS.get(investment_horizon, DEFAULT_WEIGHTS))
        else:
            self.weights = INVESTMENT_HORIZON_WEIGHTS.get(investment_horizon, DEFAULT_WEIGHTS)
        
        logger.info(f"决策引擎初始化: 持仓周期={investment_horizon}, 权重={self.weights}")

    def make_decision(self, signals: Dict, risk_score: float, risk_action: str, portfolio: Dict, dynamic_threshold: float = 7.0) -> Dict:
        """
        决策入口

        Args:
            risk_score: 风险评分 0-10
            risk_action: 风险管理建议的交易行动
            portfolio: 投资组合
            dynamic_threshold: 动态风险阈值（替代固定的7分）

        Note: macro因子已通过9维度信号中的macro信号加权计算，不再单独检查
        """
        logger.info(f"DecisionEngine: risk_score={risk_score}, threshold={dynamic_threshold}, risk_action={risk_action}")

        # Step 1: 风险过滤（使用动态阈值）
        if risk_score >= 9:
            return self._risk_filter_decision(risk_score, portfolio, "极端风险")

        if risk_score >= dynamic_threshold:
            return self._risk_filter_decision(risk_score, portfolio, f"高风险(>{dynamic_threshold})")

        # Step 2: 风控强制行动
        if risk_action in ['sell', '清仓']:
            return self._risk_action_decision(risk_action, portfolio)
        elif risk_action in ['reduce', '减仓']:
            return self._risk_action_decision(risk_action, portfolio, risk_score)

        # Step 3: 信号加权
        weighted_signals = self._calculate_weighted_signals(signals)

        # Step 4: 决策规则
        return self._apply_decision_rules(weighted_signals, risk_score, portfolio)

    def _risk_filter_decision(self, risk_score: float, portfolio: Dict, reason: str = "") -> Dict:
        """
        风险过滤决策
        """
        reason_str = reason if reason else "高风险"
        logger.warning(f"风险过滤: {reason_str}, 强制持有")
        return {
            'action': 'hold',
            'quantity': portfolio.get('stock', 0),
            'confidence': 0.3,
            'reason': f'风险评分{risk_score:.0f}，{reason_str}，强制持有'
        }

    def _risk_action_decision(self, risk_action: str, portfolio: Dict, risk_score: float = 5.0) -> Dict:
        """
        风险行动决策
        """
        current_position = portfolio.get('stock', 0)

        if risk_action in ['sell', '清仓']:
            if current_position > 0:
                logger.warning(f"风险行动: 卖出全部, quantity={current_position}")
                return {
                    'action': 'sell',
                    'quantity': current_position,
                    'confidence': 0.8,
                    'reason': '风险管理建议卖出'
                }
            else:
                logger.warning(f"风险行动: 无持仓可卖，观望")
                return {
                    'action': 'hold',
                    'quantity': 0,
                    'confidence': 0.5,
                    'reason': '风险管理建议卖出但无持仓，观望'
                }
        elif risk_action in ['reduce', '减仓']:
            if current_position > 0:
                # 动态减仓比例：风险越高，减仓越多
                if risk_score >= 8:
                    reduce_ratio = 0.3 + (risk_score - 8) * 0.4 / 2  # 50-70%
                else:
                    reduce_ratio = 0.3 + (risk_score - 7) * 0.2  # 30-50%

                reduce_ratio = min(reduce_ratio, 0.7)
                reduce_quantity = int(current_position * reduce_ratio)
                logger.warning(f"风险行动: 减仓{reduce_ratio:.0%}, quantity={reduce_quantity}")
                return {
                    'action': 'sell',
                    'quantity': reduce_quantity,
                    'confidence': 0.6,
                    'reason': f'风险管理建议减仓{reduce_ratio:.0%}'
                }

        return {
            'action': 'hold',
            'quantity': 0,
            'confidence': 0.5,
            'reason': '风险管理无明确建议'
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

    def _apply_decision_rules(self, weighted_signals: Dict, risk_score: float, portfolio: Dict) -> Dict:
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
            quantity = self._calculate_buy_quantity(portfolio, risk_score)
            confidence = min(abs(total_score) + 0.3, 0.9)
            logger.info(f"决策: 规则1触发 -> buy, quantity={quantity}")

        # 规则2：多数看空或总分<-0.3
        elif bearish_count >= 3 or total_score < -0.3:
            action = 'sell'
            quantity = int(portfolio.get('stock', 0) * 0.5)  # 分批卖出，先卖50%
            confidence = min(abs(total_score) + 0.3, 0.9)
            logger.info(f"决策: 规则2触发 -> sell, quantity={quantity} (分批卖出)")

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

    def _calculate_buy_quantity(self, portfolio: Dict, risk_score: float) -> int:
        """
        计算买入数量

        Note: macro因子已通过9维度信号加权计算，不再单独调整
        """
        cash = portfolio.get('cash', 0)
        current_price = portfolio.get('current_price', 0)

        if current_price <= 0:
            return 0

        # 根据风险评分调整仓位：风险越高，买入越少
        risk_adjustment = 1.0 - (risk_score / 20)

        base_quantity = cash / current_price
        adjusted_quantity = base_quantity * risk_adjustment

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


def create_decision_engine(config: Optional[Dict] = None, investment_horizon: str = 'medium') -> DecisionEngine:
    """便捷工厂函数
    
    Args:
        config: 配置对象
        investment_horizon: 持仓周期 ('short'|'medium'|'long')
    """
    return DecisionEngine(config, investment_horizon)