#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能选股脚本 - 付费版专属功能
结合基本面筛选和技术面评分的智能选股工具

功能:
- 基本面筛选（PE、PB、市值）
- 技术面评分（趋势、动量、波动率、RSI、MACD）
- ML辅助预测（可选）
- 导出精选股票列表
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 导入付费版模块
try:
    from scoring_system import ScoringSystem, ScoreResult, SignalLevel, print_score_result
    PREMIUM_FEATURES = True
except ImportError as e:
    PREMIUM_FEATURES = False
    print(f"⚠️ 付费版模块未导入，基本面筛选功能受限: {e}")

try:
    from ml_selector import MLSelector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML模块未导入，ML功能不可用")


@dataclass
class StockPick:
    """精选股票数据类"""
    symbol: str           # 股票代码
    name: str            # 股票名称
    score: float         # 综合评分
    signal: str          # 信号类型
    tech_score: float    # 技术面评分
    pe: Optional[float] # 市盈率
    pb: Optional[float] # 市净率
    market_cap: Optional[float]  # 市值(亿)
    reason: str          # 推荐理由
    features: Dict      # 关键特征


class SmartStockPicker:
    """
    智能选股器
    
    综合基本面和技术面选股
    """
    
    # 基本面筛选参数
    DEFAULT_FUNDAMENTAL_FILTERS = {
        'pe_min': 0,           # 最低PE
        'pe_max': 50,          # 最高PE
        'pb_min': 0,           # 最低PB
        'pb_max': 5,           # 最高PB
        'market_cap_min': 50,  # 最低市值(亿)
        'market_cap_max': 5000, # 最高市值(亿)
        'exclude_st': True,     # 排除ST
        'exclude_new': True,    # 排除新股(上市不满60日)
    }
    
    def __init__(self, fundamental_filters: Optional[Dict] = None):
        """
        初始化智能选股器
        
        Args:
            fundamental_filters: 基本面筛选参数
        """
        self.filters = fundamental_filters or self.DEFAULT_FUNDAMENTAL_FILTERS.copy()
        self.scoring_system = ScoringSystem()
        self.ml_selector = None
        self.use_ml = False
        
        # 缓存
        self.stock_data_cache = {}
        self.fundamental_cache = {}
    
    def enable_ml(self, model_type: str = 'random_forest'):
        """
        启用机器学习辅助
        
        Args:
            model_type: 模型类型
        """
        if PREMIUM_FEATURES:
            self.ml_selector = MLSelector(model_type)
            self.use_ml = True
            print(f"✅ ML辅助功能已启用 ({model_type})")
        else:
            print("⚠️ 付费版模块未安装，无法启用ML功能")
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载单只股票数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            OHLCV DataFrame
        """
        if symbol in self.stock_data_cache:
            return self.stock_data_cache[symbol]
        
        # 尝试从本地数据加载
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        for ext in ['.csv', '.parquet']:
            file_path = os.path.join(data_dir, f"{symbol}{ext}")
            if os.path.exists(file_path):
                if ext == '.csv':
                    df = pd.read_csv(file_path, parse_dates=['date'])
                else:
                    df = pd.read_parquet(file_path)
                df.set_index('date', inplace=True)
                self.stock_data_cache[symbol] = df
                return df
        
        # 尝试从 akshare 获取
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date="2024-01-01", adjust="qfq")
            if df is not None and len(df) > 0:
                df.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high', 
                    '最低': 'low', '收盘': 'close', '成交量': 'volume',
                    '成交额': 'amount', '振幅': 'amplitude', 
                    '涨跌幅': 'pct_change', '涨跌额': 'change'
                }, inplace=True)
                df.set_index('date', inplace=True)
                self.stock_data_cache[symbol] = df
                return df
        except Exception as e:
            pass
        
        return None
    
    def load_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """
        加载基本面数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            基本面数据字典
        """
        if symbol in self.fundamental_cache:
            return self.fundamental_cache[symbol]
        
        try:
            import akshare as ak
            # 获取PE
            stock_pe = ak.stock_pe(symbol=symbol)
            if stock_pe is not None:
                pe = float(stock_pe.iloc[0]['动态市盈率-动态'])
            else:
                pe = None
            
            # 获取PB
            stock_pb = ak.stock_pb(symbol=symbol)
            if stock_pb is not None:
                pb = float(stock_pb.iloc[0]['市净率'])
            else:
                pb = None
            
            # 获取市值
            stock_market_cap = ak.stock_market_cap(symbol=symbol)
            if stock_market_cap is not None:
                market_cap = float(stock_market_cap.iloc[0]['市值'])
            else:
                market_cap = None
            
            result = {'pe': pe, 'pb': pb, 'market_cap': market_cap}
            self.fundamental_cache[symbol] = result
            return result
        except Exception as e:
            return None
    
    def check_fundamental_filter(self, symbol: str, data: Dict) -> Tuple[bool, str]:
        """
        检查基本面筛选条件
        
        Args:
            symbol: 股票代码
            data: 基本面数据
            
        Returns:
            (是否通过, 原因)
        """
        filters = self.filters
        
        # 排除ST
        if filters['exclude_st'] and symbol.startswith(('ST', '*ST', 'ST')):
            return False, "ST股票"
        
        # PE筛选
        pe = data.get('pe')
        if pe is not None:
            if pe < filters['pe_min']:
                return False, f"PE过低({pe:.1f})"
            if pe > filters['pe_max']:
                return False, f"PE过高({pe:.1f})"
        
        # PB筛选
        pb = data.get('pb')
        if pb is not None:
            if pb < filters['pb_min']:
                return False, f"PB过低({pb:.1f})"
            if pb > filters['pb_max']:
                return False, f"PB过高({pb:.1f})"
        
        # 市值筛选
        market_cap = data.get('market_cap')
        if market_cap is not None:
            if market_cap < filters['market_cap_min']:
                return False, f"市值过小({market_cap:.0f}亿)"
            if market_cap > filters['market_cap_max']:
                return False, f"市值过大({market_cap:.0f}亿)"
        
        return True, "基本面合格"
    
    def analyze_stock(self, symbol: str, name: str = "") -> Optional[StockPick]:
        """
        分析单只股票
        
        Args:
            symbol: 股票代码
            name: 股票名称
            
        Returns:
            StockPick 或 None
        """
        # 加载数据
        df = self.load_stock_data(symbol)
        if df is None or len(df) < 60:
            return None
        
        # 基本面筛选
        fund_data = self.load_fundamental_data(symbol)
        if fund_data is None:
            fund_data = {'pe': None, 'pb': None, 'market_cap': None}
        
        passed, reason = self.check_fundamental_filter(symbol, fund_data)
        if not passed:
            return None
        
        # 技术面评分
        score_result = self.scoring_system.calculate(df)
        
        # ML预测
        ml_signal = None
        ml_confidence = 0
        if self.use_ml and self.ml_selector:
            try:
                ml_result = self.ml_selector.predict(df)
                ml_signal = ml_result['signal']
                ml_confidence = ml_result['confidence']
            except Exception:
                pass
        
        # 综合评分
        tech_score = score_result.total_score
        
        # ML加成
        if ml_signal == 'BUY' and ml_confidence > 0.6:
            tech_score = min(tech_score * 1.1, 100)
            reason += ", ML买入信号"
        elif ml_signal == 'SELL' and ml_confidence > 0.6:
            tech_score = max(tech_score * 0.9, 0)
            reason += ", ML卖出信号"
        
        # 生成推荐理由
        reasons = [reason]
        if score_result.trend_score >= 20:
            reasons.append("趋势强劲")
        if score_result.momentum_score >= 20:
            reasons.append("动量充足")
        if score_result.rsi_score >= 15 and score_result.details.get('rsi', 50) < 70:
            reasons.append("RSI位置良好")
        if score_result.macd_score >= 12:
            reasons.append("MACD金叉")
        
        final_reason = " | ".join(reasons)
        
        return StockPick(
            symbol=symbol,
            name=name or symbol,
            score=tech_score,
            signal=score_result.signal.value,
            tech_score=tech_score,
            pe=fund_data.get('pe'),
            pb=fund_data.get('pb'),
            market_cap=fund_data.get('market_cap'),
            reason=final_reason,
            features={
                'ma20_angle': score_result.details.get('ma20_angle', 0),
                'momentum_5': score_result.details.get('momentum_5', 0),
                'rsi': score_result.details.get('rsi', 50),
                'macd_hist': score_result.details.get('macd_histogram', 0),
                'volume_ratio': score_result.details.get('volume_ratio', 1),
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence
            }
        )
    
    def scan_market(self, symbols: List[str], 
                     max_workers: int = 10) -> List[StockPick]:
        """
        扫描市场选股
        
        Args:
            symbols: 股票代码列表
            max_workers: 并行线程数
            
        Returns:
            精选股票列表
        """
        results = []
        
        def analyze(symbol: str) -> Optional[StockPick]:
            return self.analyze_stock(symbol)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze, s): s for s in symbols}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    pass
        
        # 按评分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def get_top_picks(self, symbols: List[str], 
                       top_n: int = 20,
                       min_score: float = 50) -> List[StockPick]:
        """
        获取精选股票
        
        Args:
            symbols: 股票代码列表
            top_n: 返回数量
            min_score: 最低评分
            
        Returns:
            精选股票列表
        """
        all_picks = self.scan_market(symbols)
        
        # 过滤低分
        filtered = [p for p in all_picks if p.score >= min_score]
        
        return filtered[:top_n]
    
    def train_ml_model(self, symbols: List[str]):
        """
        使用多只股票数据训练ML模型
        
        Args:
            symbols: 用于训练的股票代码列表
        """
        if not self.use_ml or not self.ml_selector:
            print("请先调用 enable_ml() 启用ML功能")
            return
        
        # 收集训练数据
        all_data = []
        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if df is not None and len(df) >= 100:
                all_data.append(df)
        
        if len(all_data) < 5:
            print(f"训练数据不足，需要至少5只股票，已收集: {len(all_data)}")
            return
        
        # 合并训练
        combined_data = pd.concat(all_data, ignore_index=True)
        self.ml_selector.train(combined_data, verbose=True)
    
    def export_results(self, picks: List[StockPick], 
                        filepath: str = None,
                        format: str = 'csv') -> str:
        """
        导出结果
        
        Args:
            picks: 精选股票列表
            filepath: 文件路径
            format: 格式 ('csv' 或 'json')
            
        Returns:
            导出文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"smart_picks_{timestamp}"
        
        if format == 'csv':
            df = pd.DataFrame([asdict(p) for p in picks])
            filepath = f"{filepath}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        elif format == 'json':
            data = {
                'export_time': datetime.now().isoformat(),
                'total_picks': len(picks),
                'picks': [asdict(p) for p in picks]
            }
            filepath = f"{filepath}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已导出至: {filepath}")
        return filepath
    
    def print_results(self, picks: List[StockPick]):
        """打印选股结果"""
        print(f"\n{'='*80}")
        print(f"🎯 智能选股结果 | 共 {len(picks)} 只")
        print(f"{'='*80}")
        
        if not picks:
            print("未找到符合条件的股票")
            return
        
        # 按评分分组
        strong_buy = [p for p in picks if p.score >= 80]
        buy = [p for p in picks if 60 <= p.score < 80]
        hold = [p for p in picks if 40 <= p.score < 60]
        
        print(f"\n🟢 强力买入 ({len(strong_buy)}只):")
        for p in strong_buy[:5]:
            print(f"   {p.symbol} | {p.score:.1f}分 | PE:{p.pe or 'N/A'} | {p.reason[:30]}")
        
        print(f"\n🟢 买入 ({len(buy)}只):")
        for p in buy[:5]:
            print(f"   {p.symbol} | {p.score:.1f}分 | PE:{p.pe or 'N/A'} | {p.reason[:30]}")
        
        print(f"\n🟡 持有 ({len(hold)}只):")
        for p in hold[:5]:
            print(f"   {p.symbol} | {p.score:.1f}分 | PE:{p.pe or 'N/A'} | {p.reason[:30]}")
        
        # 完整列表
        print(f"\n📊 完整列表:")
        print(f"{'代码':<10} {'评分':>6} {'信号':>8} {'PE':>8} {'PB':>6} {'市值(亿)':>10} {'推荐理由'}")
        print("-" * 80)
        for p in picks:
            pe_str = f"{p.pe:.1f}" if p.pe else "N/A"
            pb_str = f"{p.pb:.2f}" if p.pb else "N/A"
            cap_str = f"{p.market_cap:.0f}" if p.market_cap else "N/A"
            print(f"{p.symbol:<10} {p.score:>6.1f} {p.signal:>8} {pe_str:>8} {pb_str:>6} {cap_str:>10} {p.reason[:20]}")


# 预定义股票池
A_SHARE_POOL = {
    # 蓝筹股
    '600519': '贵州茅台', '600036': '招商银行', '601398': '工商银行',
    '601857': '中国石油', '601288': '农业银行', '601988': '中国银行',
    '600016': '民生银行', '600000': '浦发银行', '601166': '兴业银行',
    # 科技股
    '600703': '三安光电', '000063': '中兴通讯', '002475': '立讯精密',
    '002475': '歌尔股份', '000725': '京东方A', '002456': '欧菲光',
    # 消费股
    '000858': '五粮液', '000568': '泸州老窖', '603288': '海天味业',
    '000651': '格力电器', '000333': '美的集团', '002304': '洋河股份',
    # 医药股
    '600276': '恒瑞医药', '000538': '云南白药', '600518': '康美药业',
    '002007': '华兰生物', '002044': '捷新药业',
    # 新能源
    '600011': '华能国际', '601012': '隆基绿能', '002129': '中环股份',
    '002594': '比亚迪', '002709': '天赐材料',
    # 券商
    '600030': '中信证券', '601688': '中国中车', '000776': '甘李药业',
    # 更多精选
    '600900': '长江电力', '600900': '国投电力', '600104': '上汽集团',
    '600309': 'ST万鸿', '600352': '浙江龙盛',
}


def quick_scan(symbols: List[str] = None, 
                use_ml: bool = False,
                min_score: int = 50) -> List[StockPick]:
    """
    快速选股扫描
    
    Args:
        symbols: 股票代码列表 (默认使用预定义池)
        use_ml: 是否使用ML
        min_score: 最低评分
        
    Returns:
        精选股票列表
    """
    picker = SmartStockPicker()
    
    if use_ml:
        picker.enable_ml('random_forest')
    
    if symbols is None:
        symbols = list(A_SHARE_POOL.keys())
    
    # 训练ML模型 (如果启用)
    if use_ml and picker.use_ml:
        picker.train_ml_model(symbols[:20])  # 用前20只训练
    
    # 选股
    picks = picker.get_top_picks(symbols, top_n=30, min_score=min_score)
    
    return picks


def main():
    """主函数 - 演示智能选股"""
    print("=" * 80)
    print("🎯 智能选股系统 - 付费版演示")
    print("=" * 80)
    
    # 演示：使用模拟数据
    print("\n📊 使用模拟数据进行演示...")
    
    # 创建模拟股票数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
    
    stock_data = {}
    for i, (symbol, name) in enumerate(list(A_SHARE_POOL.items())[:10]):
        # 趋势逐渐上涨
        trend = i * 0.02
        close = 10 + i + np.cumsum(np.random.randn(150) * 0.3 + trend)
        volume = np.random.randint(5000000, 20000000, 150)
        
        df = pd.DataFrame({
            'open': close - np.random.randn(150) * 0.1,
            'high': close + np.abs(np.random.randn(150) * 0.2),
            'low': close - np.abs(np.random.randn(150) * 0.2),
            'close': close,
            'volume': volume
        }, index=dates)
        
        stock_data[symbol] = df
    
    # 创建评分系统
    picker = SmartStockPicker()
    
    # 批量评分
    results = []
    for symbol, df in stock_data.items():
        try:
            result = picker.scoring_system.calculate(df)
            results.append({
                'symbol': symbol,
                'name': A_SHARE_POOL.get(symbol, symbol),
                'score': result.total_score,
                'signal': result.signal.value,
                'ma20_angle': result.details.get('ma20_angle', 0),
                'momentum_5': result.details.get('momentum_5', 0),
                'rsi': result.details.get('rsi', 50)
            })
        except Exception as e:
            print(f"分析失败 {symbol}: {e}")
    
    # 排序并打印
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*80}")
    print("📈 评分排名")
    print(f"{'='*80}")
    print(f"{'代码':<10} {'名称':<8} {'评分':>6} {'信号':>10} {'MA20角度':>10} {'5日涨幅':>10} {'RSI':>6}")
    print("-" * 80)
    
    for r in results:
        # 解析百分比字符串
        def parse_pct(val):
            if isinstance(val, str):
                return float(val.replace('%', '')) / 100 if '%' in val else float(val)
            return val if pd.notna(val) else 0

        print(f"{r['symbol']:<10} {r['name']:<8} {r['score']:>6.1f} {r['signal']:>10} "
              f"{parse_pct(r['ma20_angle']):>8.2f}° {parse_pct(r['momentum_5']):>9.2%} {r['rsi']:>6.1f}")
    
    # 精选推荐
    print(f"\n🏆 TOP 5 推荐:")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['symbol']} ({r['name']}) - {r['score']:.1f}分 - {r['signal']}")
    
    # ML演示
    print(f"\n{'='*80}")
    print("🤖 ML模型演示")
    print(f"{'='*80}")
    
    if ML_AVAILABLE:
        # 训练ML模型
        ml_selector = MLSelector(model_type='random_forest')
        all_data = pd.concat(stock_data.values(), ignore_index=True)
        ml_result = ml_selector.train(all_data, verbose=True)
        
        # ML预测
        print(f"\nML预测示例:")
        for symbol, df in list(stock_data.items())[:3]:
            pred = ml_selector.predict(df)
            print(f"  {symbol}: {pred['signal']} (置信度: {pred['confidence']:.2%})")
        
        # 特征重要性
        print(f"\n📊 特征重要性:")
        importance = ml_selector.get_feature_importance()
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.3f}")
    else:
        print("ML模块不可用，跳过ML演示")
    
    print(f"\n{'='*80}")
    print("✅ 智能选股演示完成")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
