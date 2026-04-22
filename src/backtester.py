"""
回测验证框架
验证策略有效性，计算因子IC值和分组回测
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os


class BacktestConfig:
    """回测配置"""
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000,
                 commission: float = 0.0003,
                 slippage: float = 0.001):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission  # 手续费
        self.slippage = slippage  # 滑点


class BacktestResult:
    """回测结果"""
    def __init__(self):
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.returns = []

    def to_dict(self) -> dict:
        return {
            'total_trades': len(self.trades),
            'final_equity': self.equity_curve[-1] if self.equity_curve else 0,
            'total_return': self.calculate_total_return(),
            'annual_return': self.calculate_annual_return(),
            'sharpe_ratio': self.calculate_sharpe(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate()
        }

    def calculate_total_return(self) -> float:
        if not self.equity_curve or self.equity_curve[0] == 0:
            return 0
        return (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]

    def calculate_annual_return(self) -> float:
        total_ret = self.calculate_total_return()
        if not self.equity_curve:
            return 0
        days = len(self.equity_curve)
        years = days / 252
        if years > 0:
            return (1 + total_ret) ** (1 / years) - 1
        return 0

    def calculate_sharpe(self) -> float:
        if not self.returns or len(self.returns) < 2:
            return 0
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return mean_return / std_return * np.sqrt(252)

    def calculate_max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        equity = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        return np.min(drawdown)

    def calculate_win_rate(self) -> float:
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        return wins / len(self.trades)


class FactorValidator:
    """因子有效性验证"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_ic(self, factor_values: pd.Series, future_returns: pd.Series) -> float:
        """
        计算IC值（信息系数）
        """
        # 去除NaN
        valid_mask = factor_values.notna() & future_returns.notna()
        if valid_mask.sum() < 10:
            return 0

        ic = np.corrcoef(factor_values[valid_mask], future_returns[valid_mask])[0, 1]
        return ic if not np.isnan(ic) else 0

    def calculate_ic_series(self, factor_data: pd.DataFrame, forward_days: int = 20) -> pd.Series:
        """
        计算IC时间序列
        """
        ic_series = []

        for date in factor_data.index:
            if date not in factor_data.index:
                continue

            # 获取因子值
            factor_values = factor_data.loc[date]

            # 获取未来收益
            future_date_idx = factor_data.index.get_loc(date) + forward_days
            if future_date_idx >= len(factor_data):
                continue

            future_date = factor_data.index[future_date_idx]
            if future_date not in factor_data.index:
                continue

            # 简化：使用当天收益率作为未来收益代理
            future_returns = factor_data['return'].loc[date] if 'return' in factor_data.columns else 0

            ic = self.calculate_ic(factor_values, future_returns)
            ic_series.append({'date': date, 'ic': ic})

        return pd.DataFrame(ic_series).set_index('date')['ic']

    def validate_factor(self, factor_name: str, factor_data: pd.DataFrame) -> dict:
        """
        验证单个因子有效性
        """
        ic_series = self.calculate_ic_series(factor_data)

        if len(ic_series) == 0:
            return {
                'factor_name': factor_name,
                'status': 'insufficient_data',
                'mean_ic': 0,
                'ic_std': 0,
                'ic_ir': 0,
                'positive_ic_ratio': 0
            }

        mean_ic = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        positive_ic_ratio = (ic_series > 0).sum() / len(ic_series)

        # 判断因子有效性
        if abs(mean_ic) > 0.05 and ic_ir > 0.5:
            status = 'excellent'
        elif abs(mean_ic) > 0.03 and ic_ir > 0.3:
            status = 'good'
        elif abs(mean_ic) > 0.01:
            status = 'acceptable'
        else:
            status = 'weak'

        return {
            'factor_name': factor_name,
            'status': status,
            'mean_ic': float(mean_ic),
            'ic_std': float(ic_std),
            'ic_ir': float(ic_ir),
            'positive_ic_ratio': float(positive_ic_ratio),
            'ic_series': ic_series.to_dict()
        }


class Backtester:
    """策略回测"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.result = BacktestResult()
        self.factor_validator = FactorValidator(config)

    def run_simple_backtest(self,
                           signals: pd.DataFrame,
                           prices: pd.DataFrame) -> BacktestResult:
        """
        简单回测
        signals: 包含signal, confidence列的DataFrame，index为日期
        prices: 包含close列的DataFrame
        """
        cash = self.config.initial_capital
        position = 0
        entry_price = 0

        for date in signals.index:
            if date not in prices.index:
                continue

            current_price = prices.loc[date, 'close']
            signal = signals.loc[date, 'signal'] if 'signal' in signals.columns else 'hold'
            confidence = signals.loc[date, 'confidence'] if 'confidence' in signals.columns else 0.5

            # 交易逻辑
            if signal == 'buy' and position == 0:
                # 买入
                shares = int(cash / current_price / 100) * 100  # 整手
                if shares > 0:
                    cost = shares * current_price * (1 + self.config.commission + self.config.slippage)
                    if cost <= cash:
                        position = shares
                        entry_price = cost / shares
                        cash -= cost
                        self.result.trades.append({
                            'date': date,
                            'action': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'cost': cost
                        })

            elif signal == 'sell' and position > 0:
                # 卖出
                proceeds = position * current_price * (1 - self.config.commission - self.config.slippage)
                profit = proceeds - position * entry_price
                cash += proceeds
                self.result.trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': position,
                    'proceeds': proceeds,
                    'profit': profit
                })
                position = 0
                entry_price = 0

            # 更新权益
            equity = cash + position * current_price
            self.result.equity_curve.append(equity)

            # 计算日收益率
            if len(self.result.equity_curve) > 1:
                daily_return = (equity - self.result.equity_curve[-2]) / self.result.equity_curve[-2]
                self.result.returns.append(daily_return)

        # 最后一天如果还有持仓，按收盘价平仓
        if position > 0 and date in prices.index:
            final_price = prices.loc[date, 'close']
            proceeds = position * final_price * (1 - self.config.commission)
            profit = proceeds - position * entry_price
            cash += proceeds
            self.result.trades.append({
                'date': date,
                'action': 'close',
                'price': final_price,
                'shares': position,
                'profit': profit
            })

        return self.result


def run_factor_validation(factor_data: pd.DataFrame) -> List[dict]:
    """
    运行因子有效性验证
    """
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2025-12-31'
    )

    validator = FactorValidator(config)

    results = []

    # 验证每个因子
    factor_columns = [col for col in factor_data.columns if col != 'return']
    for factor in factor_columns:
        result = validator.validate_factor(factor, factor_data)
        results.append(result)

    return results


# 测试代码
if __name__ == '__main__':
    # 模拟数据测试
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    test_data = pd.DataFrame({
        'return': np.random.randn(len(dates)) * 0.02,
        'pe_factor': np.random.randn(len(dates)),
        'pb_factor': np.random.randn(len(dates)),
        'momentum': np.random.randn(len(dates))
    }, index=dates)

    results = run_factor_validation(test_data)

    print("因子验证结果:")
    for r in results:
        print(f"  {r['factor_name']}: IC={r['mean_ic']:.4f}, IR={r['ic_ir']:.4f}, 状态={r['status']}")
