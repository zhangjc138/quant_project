#!/usr/bin/env python3
"""
策略对比模块

提供多策略对比分析功能
支持回测指标对比、可视化对比、策略排名
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class StrategyResult:
    """策略回测结果"""
    name: str                    # 策略名称
    total_return: float          # 总收益率
    annual_return: float         # 年化收益率
    sharpe_ratio: float          # 夏普比率
    max_drawdown: float          # 最大回撤
    win_rate: float              # 胜率
    profit_factor: float         # 盈亏比
    avg_win_rate: float          # 平均盈利
    avg_loss_rate: float         # 平均亏损
    trade_count: int             # 交易次数
    avg_holding_days: float      # 平均持仓天数
    volatility: float            # 波动率
    sortino_ratio: float         # 索提诺比率
    calmar_ratio: float          # 卡玛比率
    
    # 扩展指标
    monthly_returns: Dict[str, float] = field(default_factory=dict)  # 月度收益
    equity_curve: List[float] = field(default_factory=list)         # 资金曲线
    drawdown_curve: List[float] = field(default_factory=list)      # 回撤曲线
    trade_log: List[Dict] = field(default_factory=list)             # 交易日志
    
    # 策略参数
    params: Dict = field(default_factory=dict)                      # 策略参数


class StrategyComparator:
    """策略对比器"""
    
    def __init__(self, initial_capital: float = 100000):
        """
        初始化策略对比器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.results: List[StrategyResult] = []
    
    def add_result(self, result: StrategyResult):
        """添加策略回测结果"""
        self.results.append(result)
    
    def compare(self) -> pd.DataFrame:
        """
        对比所有策略
        
        Returns:
            DataFrame: 对比结果表格
        """
        if not self.results:
            return pd.DataFrame()
        
        # 构建对比数据
        comparison_data = []
        
        for result in self.results:
            comparison_data.append({
                '策略名称': result.name,
                '总收益率': f"{result.total_return*100:.2f}%",
                '年化收益率': f"{result.annual_return*100:.2f}%",
                '夏普比率': f"{result.sharpe_ratio:.2f}",
                '最大回撤': f"{result.max_drawdown*100:.2f}%",
                '胜率': f"{result.win_rate*100:.1f}%",
                '盈亏比': f"{result.profit_factor:.2f}",
                '交易次数': result.trade_count,
                '波动率': f"{result.volatility*100:.2f}%",
                '索提诺比率': f"{result.sortino_ratio:.2f}",
                '卡玛比率': f"{result.calmar_ratio:.2f}",
            })
        
        return pd.DataFrame(comparison_data)
    
    def rank_strategies(self, metric: str = 'sharpe_ratio') -> List[Tuple[str, float]]:
        """
        策略排名
        
        Args:
            metric: 排名依据指标
        
        Returns:
            List: [(策略名, 分数), ...] 降序排列
        """
        ranking = []
        
        for result in self.results:
            value = getattr(result, metric, 0)
            ranking.append((result.name, value))
        
        # 降序排列
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Optional[StrategyResult]:
        """
        获取最佳策略
        
        Args:
            metric: 评估指标
        
        Returns:
            StrategyResult: 最佳策略结果
        """
        ranking = self.rank_strategies(metric)
        
        if not ranking:
            return None
        
        best_name = ranking[0][0]
        
        for result in self.results:
            if result.name == best_name:
                return result
        
        return None
    
    def print_comparison(self):
        """打印策略对比结果"""
        if not self.results:
            print("没有策略数据可供对比")
            return
        
        print("\n" + "="*100)
        print("📊 策略对比分析报告")
        print("="*100)
        
        # 对比表格
        comparison = self.compare()
        
        print("\n【综合对比表】")
        print("-"*100)
        print(comparison.to_string(index=False))
        
        # 最佳策略
        print("\n" + "-"*100)
        best = self.get_best_strategy()
        if best:
            print(f"\n🏆 最佳策略（综合）: {best.name}")
            print(f"   - 总收益率: {best.total_return*100:.2f}%")
            print(f"   - 夏普比率: {best.sharpe_ratio:.2f}")
            print(f"   - 最大回撤: {best.max_drawdown*100:.2f}%")
        
        # 各项最佳
        print("\n【各维度最佳】")
        print("-"*50)
        
        metrics = [
            ('总收益率', 'total_return'),
            ('夏普比率', 'sharpe_ratio'),
            ('最大回撤', 'max_drawdown'),
            ('胜率', 'win_rate'),
            ('盈亏比', 'profit_factor'),
        ]
        
        for name, metric in metrics:
            best = self.get_best_strategy(metric)
            if best:
                value = getattr(best, metric)
                if '率' in name or '回撤' in name:
                    print(f"  {name}: {best.name} ({value*100:.2f}%)")
                else:
                    print(f"  {name}: {best.name} ({value:.2f})")
        
        print("="*100)
    
    def generate_html_report(self) -> str:
        """
        生成HTML对比报告
        
        Returns:
            str: HTML报告内容
        """
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>策略对比报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #6366f1; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #333; padding: 12px; text-align: center; }
        th { background: #6366f1; color: white; }
        tr:nth-child(even) { background: #16213e; }
        tr:hover { background: #1f4068; }
        .best { background: #22c55e !important; color: white; }
        .metric-card { display: inline-block; background: #16213e; padding: 20px; margin: 10px; border-radius: 10px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #6366f1; }
        .metric-label { color: #aaa; }
    </style>
</head>
<body>
    <h1>📊 策略对比报告</h1>
    <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    
    <h2>🏆 最佳策略</h2>
"""
        
        # 最佳策略信息
        best = self.get_best_strategy()
        if best:
            html += f"""
    <div class="metric-card">
        <div class="metric-value">{best.total_return*100:.2f}%</div>
        <div class="metric-label">总收益率</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{best.sharpe_ratio:.2f}</div>
        <div class="metric-label">夏普比率</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{best.max_drawdown*100:.2f}%</div>
        <div class="metric-label">最大回撤</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{best.win_rate*100:.1f}%</div>
        <div class="metric-label">胜率</div>
    </div>
"""
        
        # 对比表格
        html += """
    <h2>📈 策略对比表</h2>
    <table>
        <tr>
            <th>策略名称</th>
            <th>总收益率</th>
            <th>年化收益率</th>
            <th>夏普比率</th>
            <th>最大回撤</th>
            <th>胜率</th>
            <th>盈亏比</th>
            <th>交易次数</th>
        </tr>
"""
        
        for result in self.results:
            is_best = result.name == best.name if best else False
            row_class = 'class="best"' if is_best else ''
            
            html += f"""
        <tr {row_class}>
            <td>{result.name}</td>
            <td>{result.total_return*100:.2f}%</td>
            <td>{result.annual_return*100:.2f}%</td>
            <td>{result.sharpe_ratio:.2f}</td>
            <td>{result.max_drawdown*100:.2f}%</td>
            <td>{result.win_rate*100:.1f}%</td>
            <td>{result.profit_factor:.2f}</td>
            <td>{result.trade_count}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        return html
    
    def export_json(self) -> str:
        """
        导出JSON格式结果
        
        Returns:
            str: JSON字符串
        """
        data = {
            'generated_at': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'strategies': []
        }
        
        for result in self.results:
            data['strategies'].append({
                'name': result.name,
                'metrics': {
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'trade_count': result.trade_count,
                },
                'params': result.params
            })
        
        return json.dumps(data, ensure_ascii=False, indent=2)


def create_sample_results() -> List[StrategyResult]:
    """
    创建示例策略结果（用于演示）
    
    Returns:
        List[StrategyResult]: 示例结果
    """
    results = []
    
    # 策略1: MA20角度策略
    results.append(StrategyResult(
        name="MA20角度策略",
        total_return=0.285,
        annual_return=0.342,
        sharpe_ratio=1.85,
        max_drawdown=0.085,
        win_rate=0.623,
        profit_factor=2.15,
        avg_win_rate=0.085,
        avg_loss_rate=0.042,
        trade_count=156,
        avg_holding_days=8.5,
        volatility=0.185,
        sortino_ratio=2.45,
        calmar_ratio=4.02,
        params={'ma_period': 20, 'angle_threshold': 3.0}
    ))
    
    # 策略2: RSI策略
    results.append(StrategyResult(
        name="RSI均值回归策略",
        total_return=0.198,
        annual_return=0.235,
        sharpe_ratio=1.42,
        max_drawdown=0.062,
        win_rate=0.585,
        profit_factor=1.85,
        avg_win_rate=0.065,
        avg_loss_rate=0.038,
        trade_count=89,
        avg_holding_days=5.2,
        volatility=0.142,
        sortino_ratio=1.95,
        calmar_ratio=3.79,
        params={'rsi_period': 14, 'oversold': 30, 'overbought': 70}
    ))
    
    # 策略3: MACD策略
    results.append(StrategyResult(
        name="MACD趋势策略",
        total_return=0.245,
        annual_return=0.298,
        sharpe_ratio=1.68,
        max_drawdown=0.095,
        win_rate=0.542,
        profit_factor=1.92,
        avg_win_rate=0.092,
        avg_loss_rate=0.048,
        trade_count=112,
        avg_holding_days=12.3,
        volatility=0.168,
        sortino_ratio=2.15,
        calmar_ratio=3.14,
        params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    ))
    
    # 策略4: 多因子策略
    results.append(StrategyResult(
        name="MA20+RSI组合策略",
        total_return=0.352,
        annual_return=0.425,
        sharpe_ratio=2.15,
        max_drawdown=0.072,
        win_rate=0.678,
        profit_factor=2.45,
        avg_win_rate=0.078,
        avg_loss_rate=0.035,
        trade_count=98,
        avg_holding_days=6.8,
        volatility=0.158,
        sortino_ratio=2.85,
        calmar_ratio=5.90,
        params={'ma_period': 20, 'angle': 3.0, 'rsi_period': 14}
    ))
    
    return results


if __name__ == "__main__":
    # 测试策略对比
    print("策略对比模块测试")
    
    # 创建对比器
    comparator = StrategyComparator(initial_capital=100000)
    
    # 添加示例结果
    results = create_sample_results()
    for result in results:
        comparator.add_result(result)
    
    # 打印对比报告
    comparator.print_comparison()
    
    # 导出HTML
    html_report = comparator.generate_html_report()
    
    with open('strategy_comparison.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print("\nHTML报告已生成: strategy_comparison.html")
    
    # 导出JSON
    json_report = comparator.export_json()
    
    with open('strategy_comparison.json', 'w', encoding='utf-8') as f:
        f.write(json_report)
    
    print("JSON报告已生成: strategy_comparison.json")
