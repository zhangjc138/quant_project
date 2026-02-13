#!/usr/bin/env python3
"""
MA20 角度选股策略模块
基于 MA20 均线斜率识别趋势强度
支持 RSI、MACD 等技术指标
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import akshare as ak
from datetime import datetime, timedelta
import os

# 尝试导入本地数据模块
try:
    from data_manager import load_stock_daily, get_stock_ma20_angle
    LOCAL_DATA_AVAILABLE = True
except ImportError:
    LOCAL_DATA_AVAILABLE = False
    print("警告: 本地数据模块不可用，将使用 akshare 在线数据")


@dataclass
class StockSignal:
    """股票信号数据结构"""
    symbol: str           # 股票代码
    name: str            # 股票名称
    price: float          # 当前价格
    change_pct: float     # 涨跌幅
    ma20: float           # MA20 值
    ma20_angle: float     # MA20 角度（度）
    rsi: float           # RSI 指标值
    rsi_signal: str      # RSI 信号 (OVERBOUGHT/OVERSOLD/NEUTRAL)
    macd: float          # MACD 值 (DIF)
    macd_signal: str     # MACD 信号 (GOLD_CROSS/DEAD_CROSS/NEUTRAL)
    signal: str           # 综合信号 BUY/SELL/HOLD
    signal_desc: str      # 信号描述
    update_time: str      # 更新时间


class TechnicalIndicator:
    """
    技术指标计算器
    
    支持:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - MA (Moving Average)
    """
    
    # RSI 参数
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70    # 超买阈值
    RSI_OVERSOLD = 30      # 超卖阈值
    
    # MACD 参数
    MACD_FAST = 12         # 快速 EMA 周期
    MACD_SLOW = 26         # 慢速 EMA 周期
    MACD_SIGNAL = 9        # Signal 线周期
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算 RSI 指标
        
        Args:
            prices: 价格序列
            period: RSI 周期
            
        Returns:
            RSI 值序列
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # 计算价格变化
        delta = prices.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 计算平均涨幅和跌幅
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # 使用 EMA 计算平均（更常用）
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 处理平均损失为 0 的情况
        rsi = rsi.fillna(100)
        
        return rsi
    
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算 MACD 指标
        
        Args:
            prices: 价格序列
            fast_period: 快速 EMA 周期
            slow_period: 慢速 EMA 周期
            signal_period: Signal 线周期
            
        Returns:
            Tuple[DIF, DEA(Signal), MACD(Histogram)]
        """
        if len(prices) < slow_period + signal_period:
            nans = pd.Series([np.nan] * len(prices), index=prices.index)
            return nans, nans, nans
        
        # 计算快速和慢速 EMA
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # DIF (MACD Line) = EMA_fast - EMA_slow
        dif = ema_fast - ema_slow
        
        # DEA (Signal Line) = EMA(DIF, signal_period)
        dea = dif.ewm(span=signal_period, adjust=False).mean()
        
        # MACD Histogram = (DIF - DEA) * 2
        macd = (dif - dea) * 2
        
        return dif, dea, macd
    
    @staticmethod
    def calculate_ma(prices: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def detect_rsi_signal(rsi: float) -> str:
        """
        检测 RSI 信号
        
        Args:
            rsi: RSI 值
            
        Returns:
            信号类型: OVERBOUGHT/OVERSOLD/NEUTRAL
        """
        if rsi >= TechnicalIndicator.RSI_OVERBOUGHT:
            return "OVERBOUGHT"
        elif rsi <= TechnicalIndicator.RSI_OVERSOLD:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def detect_macd_signal(
        dif: float, dif_prev: float,
        dea: float, dea_prev: float
    ) -> str:
        """
        检测 MACD 信号（金叉/死叉）
        
        Args:
            dif: 当前 DIF 值
            dif_prev: 前一日 DIF 值
            dea: 当前 DEA 值
            dea_prev: 前一日 DEA 值
            
        Returns:
            信号类型: GOLD_CROSS/DEAD_CROSS/NEUTRAL
        """
        # 金叉: DIF 从下方穿过 DEA
        if dif_prev <= dea_prev and dif > dea:
            return "GOLD_CROSS"
        # 死叉: DIF 从上方穿过 DEA
        elif dif_prev >= dea_prev and dif < dea:
            return "DEAD_CROSS"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def detect_ma_signal(
        price: float, ma_short: float, ma_long: float,
        price_prev: float, ma_short_prev: float
    ) -> str:
        """
        检测 MA 信号（金叉/死叉/多头/空头）
        
        Args:
            price: 当前价格
            ma_short: 短期均线
            ma_long: 长期均线
            price_prev: 昨日价格
            ma_short_prev: 昨日短期均线
            
        Returns:
            信号类型
        """
        if pd.isna(ma_short) or pd.isna(ma_long):
            return "NEUTRAL"
        
        # 金叉: 短期均线从下方穿过长期均线
        if ma_short_prev <= ma_long and ma_short > ma_long:
            return "GOLD_CROSS"
        # 死叉: 短期均线从上方穿过长期均线
        elif ma_short_prev >= ma_long and ma_short < ma_long:
            return "DEAD_CROSS"
        # 多头: 短期均线在长期均线上方
        elif ma_short > ma_long:
            return "BULLISH"
        # 空头: 短期均线在长期均线下方
        else:
            return "BEARISH"


class StockSelector:
    """
    MA20 角度选股器
    
    核心功能:
    - 计算 MA20 均线角度
    - 生成 BUY/SELL/HOLD 信号
    - 支持自定义股票池扫描
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "angle_threshold_buy": 3.0,    # 买入角度阈值（度）
        "angle_threshold_sell": 0.0,    # 卖出角度阈值（度）
        "rsi_oversold": 30,            # RSI 超卖阈值
        "rsi_overbought": 70,          # RSI 超买阈值
        "require_rsi_oversold": True,  # 买入是否要求 RSI 超卖
        "require_macd_golden": True,   # 买入是否要求 MACD 金叉
        "price_min": 5.0,              # 最低股价
        "price_max": 100.0,            # 最高股价
        "volume_ratio_min": 0.5,       # 最低量比
        "exclude_st": True,             # 排除 ST 股票
        "exclude_new": True,            # 排除新股（上市不满60日）
        "new_stock_days": 60,           # 新股判定天数
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化选股器
        
        Args:
            config: 配置字典，覆盖默认配置
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.watchlist = self._get_default_watchlist()
        self.indicator = TechnicalIndicator()
    
    def _get_default_watchlist(self) -> Dict[str, Dict]:
        """获取默认监控股票池"""
        return {
            # 银行股
            "600000": {"name": "浦发银行", "category": "银行"},
            "600036": {"name": "招商银行", "category": "银行"},
            "600016": {"name": "民生银行", "category": "银行"},
            "600015": {"name": "华夏银行", "category": "银行"},
            # 证券股
            "600030": {"name": "中信证券", "category": "证券"},
            # 高速公路
            "600012": {"name": "皖通高速", "category": "高速"},
            "600033": {"name": "福建高速", "category": "高速"},
            "600035": {"name": "宁沪高速", "category": "高速"},
            # 机场航空
            "600009": {"name": "上海机场", "category": "机场"},
            # 医药消费
            "600085": {"name": "同仁堂", "category": "医药"},
            # 有色金属
            "600352": {"name": "山东黄金", "category": "黄金"},
        }
    
    def set_watchlist(self, watchlist: Dict[str, Dict]):
        """设置自定义股票池"""
        self.watchlist = watchlist
    
    def calculate_ma20_angle(self, df: pd.DataFrame) -> float:
        """
        计算 MA20 角度（度）
        
        使用线性回归计算 MA20 斜率，转换为角度
        
        Args:
            df: 包含 'close' 和 'MA20' 列的 DataFrame
            
        Returns:
            float: MA20 角度（度）
        """
        if df is None or len(df) < 25:
            return 0.0
        
        # 取最近 20 个 MA20 值
        ma20_series = df['MA20'].dropna().tail(20)
        if len(ma20_series) < 20:
            return 0.0
        
        # 计算 MA20 的斜率（度/日）
        x = np.arange(len(ma20_series))
        y = ma20_series.values
        
        # 线性回归
        if np.std(x) == 0:
            return 0.0
        
        slope = np.cov(x, y)[0, 1] / np.var(x)
        
        # 计算角度（度）
        # arctan 返回弧度，转换为度
        angle = np.degrees(np.arctan(slope / ma20_series.mean() * 100))
        
        return angle
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: 价格数据
            
        Returns:
            包含所有指标的 DataFrame
        """
        if df is None or len(df) < 30:
            return df
        
        # 计算 MA
        df['MA5'] = self.indicator.calculate_ma(df['close'], 5)
        df['MA10'] = self.indicator.calculate_ma(df['close'], 10)
        df['MA20'] = self.indicator.calculate_ma(df['close'], 20)
        df['MA60'] = self.indicator.calculate_ma(df['close'], 60)
        
        # 计算 RSI
        df['RSI'] = self.indicator.calculate_rsi(df['close'], self.indicator.RSI_PERIOD)
        
        # 计算 MACD
        df['DIF'], df['DEA'], df['MACD'] = self.indicator.calculate_macd(df['close'])
        
        # 计算 MA20 角度
        df['MA20_angle'] = df['MA20'].rolling(window=20).apply(
            self._calculate_angle_internal, raw=False
        )
        
        return df
    
    def _calculate_angle_internal(self, series: pd.Series) -> float:
        """内部使用的角度计算函数"""
        if len(series) < 20:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        if np.std(x) == 0:
            return 0.0
        
        slope = np.cov(x, y)[0, 1] / np.var(x)
        angle = np.degrees(np.arctan(slope / np.mean(y) * 100))
        
        return angle
    
    def load_stock_data(self, symbol: str, days: int = 250) -> Optional[pd.DataFrame]:
        """
        加载股票历史数据
        
        Args:
            symbol: 股票代码（如 600000）
            days: 获取多少天的数据
            
        Returns:
            DataFrame 或 None
        """
        # 优先使用本地数据
        if LOCAL_DATA_AVAILABLE:
            try:
                df = load_stock_daily(symbol)
                if df is not None and not df.empty:
                    return df.tail(days)
            except Exception as e:
                pass
        
        # 使用 akshare 获取
        try:
            # 转换代码格式
            if symbol.startswith("6"):
                symbol_ak = "sh" + symbol
            else:
                symbol_ak = "sz" + symbol
            
            df = ak.stock_zh_a_hist(
                symbol=symbol_ak,
                period="daily",
                start_date=(datetime.now() - timedelta(days=days)).strftime("%Y%m%d"),
                adjust="qfq"
            )
            
            if df is None or df.empty:
                return None
            
            # 统一格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # 计算 MA20
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            print(f"获取数据失败 {symbol}: {e}")
            return None
    
    def get_signal(self, symbol: str) -> Optional[StockSignal]:
        """
        获取单个股票的 MA20 角度信号
        
        Args:
            symbol: 股票代码
            
        Returns:
            StockSignal 或 None
        """
        config = self.config
        
        # 加载数据
        df = self.load_stock_data(symbol)
        if df is None or len(df) < 25:
            return None
        
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        if len(df) < 30:
            return None
        
        # 获取最新值
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) >= 2 else current
        
        # MA20 角度
        ma20_angle = current.get('MA20_angle', 0.0)
        if pd.isna(ma20_angle):
            ma20_angle = 0.0
        
        # RSI
        rsi = current.get('RSI', 50.0)
        if pd.isna(rsi):
            rsi = 50.0
        rsi_signal = self.indicator.detect_rsi_signal(rsi)
        
        # MACD
        dif = current.get('DIF', 0.0)
        dea = current.get('DEA', 0.0)
        dif_prev = previous.get('DIF', 0.0)
        dea_prev = previous.get('DEA', 0.0)
        
        if pd.isna(dif):
            dif = 0.0
        if pd.isna(dea):
            dea = 0.0
        if pd.isna(dif_prev):
            dif_prev = 0.0
        if pd.isna(dea_prev):
            dea_prev = 0.0
        
        macd_signal = self.indicator.detect_macd_signal(dif, dif_prev, dea, dea_prev)
        
        # 获取最新价格和涨跌幅
        current_price = df['close'].iloc[-1]
        change_pct = df['change_pct'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        
        # 生成综合信号
        signal, signal_desc = self._generate_signal(
            ma20_angle=ma20_angle,
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            price=current_price,
            ma20=ma20
        )
        
        # 股票名称
        name = self.watchlist.get(symbol, {}).get("name", symbol)
        
        return StockSignal(
            symbol=symbol,
            name=name,
            price=current_price,
            change_pct=change_pct,
            ma20=ma20,
            ma20_angle=ma20_angle,
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd=dif,
            macd_signal=macd_signal,
            signal=signal,
            signal_desc=signal_desc,
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_signal(
        self,
        ma20_angle: float,
        rsi: float,
        rsi_signal: str,
        macd_signal: str,
        price: float,
        ma20: float
    ) -> Tuple[str, str]:
        """
        生成综合交易信号
        
        Args:
            ma20_angle: MA20 角度
            rsi: RSI 值
            rsi_signal: RSI 信号
            macd_signal: MACD 信号
            price: 当前价格
            ma20: MA20 值
            
        Returns:
            Tuple[信号, 描述]
        """
        config = self.config
        
        # 买入条件
        buy_conditions = []
        
        # MA20 角度大于阈值
        if ma20_angle > config["angle_threshold_buy"]:
            buy_conditions.append("MA20上升")
        
        # RSI 条件（可选）
        if config.get("require_rsi_oversold", True):
            if rsi_signal == "OVERSOLD":
                buy_conditions.append("RSI超卖")
        else:
            if rsi_signal == "NEUTRAL":
                buy_conditions.append("RSI中性")
        
        # MACD 条件（可选）
        if config.get("require_macd_golden", True):
            if macd_signal == "GOLD_CROSS":
                buy_conditions.append("MACD金叉")
        else:
            if macd_signal in ["GOLD_CROSS", "NEUTRAL"]:
                buy_conditions.append("MACD配合")
        
        # 判断买入信号
        if (ma20_angle > config["angle_threshold_buy"] and
            (not config.get("require_rsi_oversold", True) or rsi_signal == "OVERSOLD") and
            (not config.get("require_macd_golden", True) or macd_signal == "GOLD_CROSS")):
            return "BUY", f"看涨信号: {', '.join(buy_conditions)}"
        
        # 卖出条件
        if ma20_angle < config["angle_threshold_sell"]:
            return "SELL", f"看跌信号: MA20角度{ma20_angle:.2f}° < {config['angle_threshold_sell']}°"
        
        # 震荡/观望
        if ma20_angle >= config["angle_threshold_sell"]:
            return "HOLD", f"震荡整理: 等待明确信号"
        
        return "HOLD", "观望等待"
    
    def scan_watchlist(self) -> List[StockSignal]:
        """
        扫描股票池，获取所有信号
        
        Returns:
            List[StockSignal]: 信号列表
        """
        results = []
        
        for symbol, config in self.watchlist.items():
            if not config.get("enabled", True):
                continue
            
            signal = self.get_signal(symbol)
            if signal:
                results.append(signal)
        
        # 按 MA20 角度降序排列
        results.sort(key=lambda x: x.ma20_angle, reverse=True)
        
        return results
    
    def scan_all_a_shares(self, limit: int = 100) -> List[StockSignal]:
        """
        扫描全部 A 股（使用 akshare 获取股票列表）
        
        Args:
            limit: 限制扫描数量
            
        Returns:
            List[StockSignal]: 信号列表
        """
        try:
            # 获取 A 股列表
            stock_list = ak.stock_info_a_code_name()
            if stock_list is None or stock_list.empty:
                return []
            
            # 过滤条件
            stock_list = stock_list.head(limit)
            
            results = []
            for _, row in stock_list.iterrows():
                symbol = row['code']
                name = row['name']
                
                # 跳过 ST
                if self.config["exclude_st"] and ('ST' in name or '*ST' in name):
                    continue
                
                signal = self.get_signal(symbol)
                if signal:
                    signal.name = name
                    results.append(signal)
            
            # 按 MA20 角度降序排列
            results.sort(key=lambda x: x.ma20_angle, reverse=True)
            
            return results
            
        except Exception as e:
            print(f"扫描全部 A 股失败: {e}")
            return []
    
    def format_report(self, signals: List[StockSignal]) -> str:
        """
        生成信号报告
        
        Args:
            signals: 信号列表
            
        Returns:
            str: Markdown 格式报告
        """
        if not signals:
            return "未扫描到任何信号"
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 分类
        buy_signals = [s for s in signals if s.signal == "BUY"]
        sell_signals = [s for s in signals if s.signal == "SELL"]
        hold_signals = [s for s in signals if s.signal == "HOLD"]
        
        report = f"""## MA20 角度选股报告

**扫描时间**: {now}
**监控股票**: {len(signals)} 只
**买入信号**: {len(buy_signals)} 只
**卖出信号**: {len(sell_signals)} 只
**观望**: {len(hold_signals)} 只

---

### 🟢 买入信号（MA20角度 ≥ {self.config['angle_threshold_buy']}° + RSI/MACD 配合）

| 股票 | 代码 | 价格 | 涨跌幅 | MA20 | MA20角度 | RSI | RSI信号 | MACD信号 | 描述 |
|------|------|------|--------|------|----------|-----|---------|----------|------|
"""
        
        for s in buy_signals:
            rsi_emoji = "🔴" if s.rsi_signal == "OVERBOUGHT" else "🟢" if s.rsi_signal == "OVERSOLD" else "🟡"
            macd_emoji = "🟢" if s.macd_signal == "GOLD_CROSS" else "🔴" if s.macd_signal == "DEAD_CROSS" else "🟡"
            report += f"| {s.name} | {s.symbol} | {s.price:.2f} | {s.change_pct:+.2f}% | {s.ma20:.2f} | **{s.ma20_angle:.2f}°** | {s.rsi:.1f} {rsi_emoji} | {macd_emoji} | {s.signal_desc} |\n"
        
        if not buy_signals:
            report += "| - | - | - | - | - | - | - | - | - |\n"
        
        report += f"""
### 🔴 卖出信号（MA20角度 < {self.config['angle_threshold_sell']}°）

| 股票 | 代码 | 价格 | 涨跌幅 | MA20 | MA20角度 | RSI | RSI信号 | MACD信号 | 描述 |
|------|------|------|--------|------|----------|-----|---------|----------|------|
"""
        
        for s in sell_signals:
            rsi_emoji = "🔴" if s.rsi_signal == "OVERBOUGHT" else "🟢" if s.rsi_signal == "OVERSOLD" else "🟡"
            macd_emoji = "🟢" if s.macd_signal == "GOLD_CROSS" else "🔴" if s.macd_signal == "DEAD_CROSS" else "🟡"
            report += f"| {s.name} | {s.symbol} | {s.price:.2f} | {s.change_pct:+.2f}% | {s.ma20:.2f} | **{s.ma20_angle:.2f}°** | {s.rsi:.1f} {rsi_emoji} | {macd_emoji} | {s.signal_desc} |\n"
        
        if not sell_signals:
            report += "| - | - | - | - | - | - | - | - | - |\n"
        
        report += f"""
### 🟡 观望信号

| 股票 | 代码 | 价格 | 涨跌幅 | MA20 | MA20角度 | RSI | RSI信号 | MACD信号 | 描述 |
|------|------|------|--------|------|----------|-----|---------|----------|------|
"""
        
        for s in hold_signals[:15]:  # 最多显示 15 只
            rsi_emoji = "🔴" if s.rsi_signal == "OVERBOUGHT" else "🟢" if s.rsi_signal == "OVERSOLD" else "🟡"
            macd_emoji = "🟢" if s.macd_signal == "GOLD_CROSS" else "🔴" if s.macd_signal == "DEAD_CROSS" else "🟡"
            report += f"| {s.name} | {s.symbol} | {s.price:.2f} | {s.change_pct:+.2f}% | {s.ma20:.2f} | {s.ma20_angle:.2f}° | {s.rsi:.1f} {rsi_emoji} | {macd_emoji} | {s.signal_desc} |\n"
        
        if len(hold_signals) > 15:
            report += f"| ... | 还有 {len(hold_signals) - 15} 只 | - | - | - | - | - | - | - |\n"
        
        report += f"""
---

### 📊 技术指标说明

**RSI (相对强弱指数)**:
- 超买区域: ≥ 70 (🔴 建议卖出)
- 超卖区域: ≤ 30 (🟢 建议买入)
- 中性区域: 30-70 (🟡 观望)

**MACD (移动平均收敛 divergence)**:
- 金叉: DIF 上穿 DEA (🟢 买入信号)
- 死叉: DIF 下穿 DEA (🔴 卖出信号)
- 中性: 无交叉 (🟡 观望)

**MA20 角度**:
- ≥ 3°: 强势上涨趋势
- 0° ~ 3°: 温和上涨/震荡
- < 0°: 下跌趋势

---

**参数配置**:
- 买入角度阈值: {self.config['angle_threshold_buy']}°
- 卖出角度阈值: {self.config['angle_threshold_sell']}°
- RSI 超卖阈值: {self.config['rsi_oversold']}
- RSI 超买阈值: {self.config['rsi_overbought']}
- 最低股价: {self.config['price_min']}元
- 排除ST股票: {'是' if self.config['exclude_st'] else '否'}

---
*生成时间: {now}*
"""
        
        return report


# ==================== 便捷函数 ====================
def get_stock_ma20_angle(symbol: str) -> Tuple[float, float, str]:
    """
    获取股票 MA20 角度的便捷函数
    
    Args:
        symbol: 股票代码
        
    Returns:
        Tuple[ma20_angle, price, signal]
    """
    selector = StockSelector()
    result = selector.get_signal(symbol)
    
    if result:
        return result.ma20_angle, result.price, result.signal
    else:
        return 0.0, 0.0, "N/A"


def calculate_rsi(symbol: str, period: int = 14) -> Optional[float]:
    """
    计算股票的 RSI 值
    
    Args:
        symbol: 股票代码
        period: RSI 周期
        
    Returns:
        RSI 值或 None
    """
    selector = StockSelector()
    df = selector.load_stock_data(symbol)
    
    if df is None:
        return None
    
    rsi = TechnicalIndicator.calculate_rsi(df['close'], period)
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_macd(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    计算股票的 MACD 值
    
    Args:
        symbol: 股票代码
        
    Returns:
        Tuple[DIF, DEA, MACD]
    """
    selector = StockSelector()
    df = selector.load_stock_data(symbol)
    
    if df is None:
        return None, None, None
    
    dif, dea, macd = TechnicalIndicator.calculate_macd(df['close'])
    return dif.iloc[-1], dea.iloc[-1], macd.iloc[-1]


if __name__ == "__main__":
    # 测试
    selector = StockSelector()
    
    # 扫描股票池
    print("=== 扫描监控股票池 ===")
    signals = selector.scan_watchlist()
    
    # 打印报告
    report = selector.format_report(signals)
    print(report)
