"""
统一决策验证框架

解决portfolio_manager中决策规则冲突问题，实现清晰的决策优先级和验证流程。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum
from src.utils.logging_config import setup_logger

logger = setup_logger('decision_validator')


class DecisionPriority(IntEnum):
    """决策优先级枚举 - 数值越大优先级越高"""
    TACTICAL_ADJUSTMENT = 60      # 战术调整（仓位微调）
    STRATEGIC_DECISION = 70       # 战略决策（DecisionEngine/LLM）
    REGULATORY_COMPLIANCE = 80    # 合规要求
    RISK_CONSTRAINT = 90          # 风险管理强制规则
    CRITICAL_VETO = 100           # 系统性风险一票否决


@dataclass
class ValidationResult:
    """验证结果"""
    action: str                    # buy/sell/hold
    quantity: int                  # 交易数量
    confidence: float              # 置信度
    reason: str                    # 决策理由
    priority: DecisionPriority     # 决策优先级
    overridden: bool = False       # 是否被覆盖
    original_action: str = ""      # 原始决策（如果被覆盖）


@dataclass
class RiskConstraint:
    """风险约束规则"""
    name: str
    check_func: callable
    description: str


class DecisionValidator:
    """
    统一决策验证器
    
    职责：
    1. 执行一票否决规则（最高优先级）
    2. 应用风险管理约束
    3. 验证资金和持仓限制
    4. 确保决策一致性
    """
    
    def __init__(self):
        self.veto_rules: List[callable] = []
        self.risk_constraints: List[RiskConstraint] = []
        self.validation_rules: List[callable] = []
        
        # 初始化默认规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认验证规则"""
        
        # === 一票否决规则（Critical Veto）===
        self.veto_rules.append(self._systemic_risk_veto)
        
        # === 风险管理约束（Risk Constraints）===
        self.risk_constraints.extend([
            RiskConstraint(
                name="high_risk_hold",
                check_func=self._check_high_risk_hold,
                description="风险评分>=7时强制持有"
            ),
            RiskConstraint(
                name="risk_management_directive",
                check_func=self._check_risk_management_directive,
                description="遵循风险管理师的交易指令"
            ),
        ])
        
        # === 基础验证规则（Validation Rules）===
        self.validation_rules.extend([
            self._validate_cash_sufficiency,
            self._validate_position_availability,
            self._validate_max_position_limit,
        ])
    
    def validate(self, decision: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """
        验证并可能修正交易决策
        
        Args:
            decision: 原始决策字典 {action, quantity, confidence, reasoning}
            context: 上下文信息 {
                risk_score, 
                risk_action, 
                max_position,
                portfolio: {cash, stock},
                current_price,
                ...
            }
            
        Returns:
            ValidationResult: 验证后的决策结果
        """
        original_action = decision.get("action", "hold")
        original_quantity = decision.get("quantity", 0)
        original_confidence = decision.get("confidence", 0.5)
        
        logger.info(f"🔍 [DECISION_VALIDATOR] 开始验证决策: {original_action} {original_quantity}股")
        
        # 步骤1: 检查一票否决规则
        veto_result = self._check_veto_rules(decision, context)
        if veto_result:
            logger.warning(f"⛔ 触发一票否决: {veto_result.reason}")
            return veto_result
        
        # 步骤2: 应用风险管理约束
        constraint_result = self._apply_risk_constraints(decision, context)
        if constraint_result:
            logger.warning(f"⚠️  触发风险约束: {constraint_result.reason}")
            return constraint_result
        
        # 步骤3: 执行基础验证规则
        validated_decision = decision.copy()
        for rule in self.validation_rules:
            validation_result = rule(validated_decision, context)
            if validation_result:
                validated_decision.update(validation_result)
                logger.info(f"✅ 应用验证规则: {rule.__name__}")
        
        # 步骤4: 返回最终决策
        final_result = ValidationResult(
            action=validated_decision.get("action", original_action),
            quantity=validated_decision.get("quantity", original_quantity),
            confidence=validated_decision.get("confidence", original_confidence),
            reason=validated_decision.get("reasoning", "通过所有验证"),
            priority=DecisionPriority.STRATEGIC_DECISION,
            overridden=(validated_decision.get("action") != original_action)
        )
        
        if final_result.overridden:
            logger.info(f"🔄 决策被修正: {original_action} -> {final_result.action}")
        else:
            logger.info(f"✅ 决策通过验证: {final_result.action}")
        
        return final_result
    
    def _check_veto_rules(self, decision: Dict, context: Dict) -> Optional[ValidationResult]:
        """检查一票否决规则"""
        for rule in self.veto_rules:
            result = rule(decision, context)
            if result:
                return result
        return None
    
    def _apply_risk_constraints(self, decision: Dict, context: Dict) -> Optional[ValidationResult]:
        """应用风险管理约束"""
        for constraint in self.risk_constraints:
            result = constraint.check_func(decision, context)
            if result:
                return ValidationResult(
                    action=result["action"],
                    quantity=result["quantity"],
                    confidence=decision.get("confidence", 0.5),
                    reason=f"风险约束 [{constraint.name}]: {result['reason']}",
                    priority=DecisionPriority.RISK_CONSTRAINT,
                    overridden=True,
                    original_action=decision.get("action", "hold")
                )
        return None
    
    # ==================== 一票否决规则 ====================
    
    def _systemic_risk_veto(self, decision: Dict, context: Dict) -> Optional[ValidationResult]:
        """
        系统性风险一票否决
        例如：市场崩盘、流动性危机等极端情况
        """
        # 预留接口，未来可添加更多系统性风险检测
        market_crash_detected = context.get("market_crash_detected", False)
        if market_crash_detected:
            return ValidationResult(
                action="hold",
                quantity=0,
                confidence=1.0,
                reason="检测到系统性风险，强制持有观望",
                priority=DecisionPriority.CRITICAL_VETO
            )
        return None
    
    # ==================== 风险管理约束 ====================
    
    def _check_high_risk_hold(self, decision: Dict, context: Dict) -> Optional[Dict]:
        """
        高风险强制持有规则
        风险评分 >= 7 时，无论其他信号如何，必须持有
        """
        risk_score = context.get("risk_score", 0)
        if risk_score >= 7:
            current_stock = context.get("portfolio", {}).get("stock", 0)
            return {
                "action": "hold",
                "quantity": current_stock,
                "reason": f"风险评分{risk_score:.0f}/10 >= 7，强制执行持有"
            }
        return None
    
    def _check_risk_management_directive(self, decision: Dict, context: Dict) -> Optional[Dict]:
        """
        风险管理师指令优先规则
        如果风险管理建议sell/reduce，且有持仓，则执行卖出
        如果无持仓，则阻止新买入（但不强制卖出）
        """
        risk_action = context.get("risk_action", "hold")
        enable_veto = context.get("enable_veto_power", False)
        portfolio = context.get("portfolio", {})
        current_position = portfolio.get("stock", 0)
        
        if risk_action in ["sell", "reduce", "减仓", "清仓"]:
            if current_position > 0:
                # 有持仓时执行卖出
                sell_quantity = min(decision.get("quantity", current_position), current_position)
                return {
                    "action": "sell",
                    "quantity": sell_quantity,
                    "reason": f"风险管理建议{risk_action}，有持仓执行卖出"
                }
            else:
                # 无持仓时，一票否决阻止买入
                if enable_veto and decision.get("action") == "buy":
                    return {
                        "action": "hold",
                        "quantity": 0,
                        "reason": f"风控一票否决：风险管理建议{risk_action}，无持仓可卖，阻止新买入"
                    }
        return None
    
    # ==================== 基础验证规则 ====================
    
    def _validate_cash_sufficiency(self, decision: Dict, context: Dict) -> Optional[Dict]:
        """验证现金充足性"""
        if decision.get("action") != "buy":
            return None
        
        current_price = context.get("current_price", 0)
        if current_price <= 0:
            return None
        
        required_cash = decision.get("quantity", 0) * current_price
        available_cash = context.get("portfolio", {}).get("cash", 0)
        
        if required_cash > available_cash:
            max_shares = int(available_cash / current_price)
            logger.warning(f"💰 现金不足：需要{required_cash:.2f}元，现有{available_cash:.2f}元")
            
            if max_shares <= 0:
                return {
                    "action": "hold",
                    "quantity": 0,
                    "reasoning": f"现金不足，无法买入（需要{required_cash:.2f}元，现有{available_cash:.2f}元）"
                }
            else:
                return {
                    "action": "buy",
                    "quantity": max_shares,
                    "reasoning": f"现金不足，调整为{max_shares}股（原计划{decision['quantity']}股）"
                }
        return None
    
    def _validate_position_availability(self, decision: Dict, context: Dict) -> Optional[Dict]:
        """验证持仓充足性"""
        if decision.get("action") != "sell":
            return None
        
        available_stock = context.get("portfolio", {}).get("stock", 0)
        requested_quantity = decision.get("quantity", 0)
        
        if requested_quantity > available_stock:
            logger.warning(f"📉 持仓不足：需要卖出{requested_quantity}股，现有{available_stock}股")
            
            if available_stock <= 0:
                return {
                    "action": "hold",
                    "quantity": 0,
                    "reasoning": f"无持仓可卖"
                }
            else:
                return {
                    "action": "sell",
                    "quantity": available_stock,
                    "reasoning": f"持仓不足，调整为{available_stock}股（原计划{requested_quantity}股）"
                }
        return None
    
    def _validate_max_position_limit(self, decision: Dict, context: Dict) -> Optional[Dict]:
        """验证最大持仓限制"""
        if decision.get("action") != "buy":
            return None
        
        max_position = context.get("max_position", 0)
        if max_position <= 0:
            return None
        
        current_position = context.get("portfolio", {}).get("stock", 0)
        requested_quantity = decision.get("quantity", 0)
        potential_position = current_position + requested_quantity
        
        if potential_position > max_position:
            allowed_quantity = int(max_position - current_position)
            logger.warning(f"🚫 超过最大持仓限制：当前{current_position}股，买入后{potential_position}股，上限{max_position}股")
            
            if allowed_quantity <= 0:
                return {
                    "action": "hold",
                    "quantity": 0,
                    "reasoning": f"已达最大持仓限制{max_position}股"
                }
            else:
                return {
                    "action": "buy",
                    "quantity": allowed_quantity,
                    "reasoning": f"超过最大持仓限制，调整为{allowed_quantity}股（原计划{requested_quantity}股）"
                }
        return None
    
    def add_veto_rule(self, rule: callable):
        """添加自定义一票否决规则"""
        self.veto_rules.append(rule)
        logger.info(f"已添加一票否决规则: {rule.__name__}")
    
    def add_risk_constraint(self, name: str, check_func: callable, description: str):
        """添加自定义风险约束"""
        self.risk_constraints.append(RiskConstraint(
            name=name,
            check_func=check_func,
            description=description
        ))
        logger.info(f"已添加风险约束: {name}")


def create_decision_validator() -> DecisionValidator:
    """创建决策验证器实例"""
    return DecisionValidator()
