#!/usr/bin/env python3
"""
MA20 角度选股策略模块
基于 MA20 均线斜率识别趋势强度
支持 RSI、MACD、BOLL、KDJ 等技术指标
支持邮件/飞书推送功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
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

# 尝试导入推送模块
try:
    from notifier import NotificationManager, create_notifier
    PUSH_NOTIFICATION_AVAILABLE = True
except ImportError:
    PUSH_NOTIFICATION_AVAILABLE = False
    print("警告: 推送模块不可用")

# 尝试导入技术指标模块
try:
    from indicators import TechnicalIndicators as NewTechnicalIndicators
    NEW_INDICATORS_AVAILABLE = True
except ImportError:
    NEW_INDICATORS_AVAILABLE = False
    print("警告: 高级技术指标模块不可用")

# 尝试导入财务因子模块
try:
    from financial import get_stock_financials, filter_financials
    FINANCIAL_AVAILABLE = True
except ImportError:
    FINANCIAL_AVAILABLE = False
    print("警告: 财务因子模块不可用")


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
    # BOLL 布林带
    boll_upper: float     # BOLL 上轨
    boll_lower: float     # BOLL 下轨
    boll_position: float  # BOLL 位置
    boll_signal: str      # BOLL 信号
    # KDJ 随机指标
    kdj_k: float          # K 值
    kdj_d: float          # D 值
    kdj_j: float          # J 值
    kdj_signal: str      # KDJ 信号
    signal: str           # 综合信号 BUY/SELL/HOLD
    signal_desc: str      # 信号描述
    update_time: str      # 更新时间
    # 行业板块
    industry: str = "未知"   # 所属行业
    # 财务因子
    pe: float = 0         # 市盈率
    pb: float = 0         # 市净率
    roe: float = 0        # 净资产收益率 (%)
    revenue_growth: float = 0  # 营收增速 (%)
    profit_growth: float = 0   # 净利润增速 (%)


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
    - 支持邮件/飞书推送
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
    
    # 推送配置
    DEFAULT_PUSH_CONFIG = {
        "push_enabled": False,        # 是否启用推送
        "push_on_buy": True,           # 买入信号是否推送
        "push_on_sell": True,          # 卖出信号是否推送
        "push_on_hold": False,         # 持有信号是否推送
        "push_buy_only": True,         # 只推送新出现的买入信号（避免重复）
        "notify_email": False,         # 是否发送邮件
        "notify_feishu": False,         # 是否发送飞书
        "min_angle_for_push": 3.0,     # 最小角度阈值触发推送
        "min_rsi_for_buy_push": 50,    # 买入推送的RSI上限（避免高位接盘）
    }
    
    def __init__(self, config: Optional[Dict] = None, push_config: Optional[Dict] = None):
        """
        初始化选股器
        
        Args:
            config: 选股配置字典，覆盖默认配置
            push_config: 推送配置字典
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.push_config = {**self.DEFAULT_PUSH_CONFIG, **(push_config or {})}
        self.watchlist = self._get_default_watchlist()
        self.indicator = TechnicalIndicator()
        self._notifier: Optional[NotificationManager] = None
        self._last_signals: Dict[str, str] = {}  # 记录上次的信号状态
        
    def set_notifier(self, notifier: NotificationManager):
        """
        设置推送通知器
        
        Args:
            notifier: NotificationManager 实例
        """
        self._notifier = notifier
        self.push_config["push_enabled"] = True
    
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
        
        # 计算 BOLL 布林带（使用高级指标模块或内置计算）
        if NEW_INDICATORS_AVAILABLE:
            try:
                from indicators import TechnicalIndicators as Ind
                df['BOLL_upper'], df['BOLL_middle'], df['BOLL_lower'] = \
                    Ind.calculate_boll(df['close'], period=20, std_dev=2)
                _, _, df['BOLL_position'] = Ind.calculate_boll_with_position(
                    df['close'], period=20, std_dev=2
                )[2:]
                # BOLL信号
                df['BOLL_signal'] = df.apply(
                    lambda x: Ind.detect_boll_signal(
                        x['close'], x['BOLL_upper'], x['BOLL_lower'], x['BOLL_position']
                    ), axis=1
                )
            except Exception as e:
                print(f"BOLL计算失败: {e}")
        else:
            # 简化版BOLL计算
            df['BOLL_middle'] = df['close'].rolling(window=20).mean()
            df['BOLL_std'] = df['close'].rolling(window=20).std()
            df['BOLL_upper'] = df['BOLL_middle'] + 2 * df['BOLL_std']
            df['BOLL_lower'] = df['BOLL_middle'] - 2 * df['BOLL_std']
            df['BOLL_position'] = (df['close'] - df['BOLL_lower']) / \
                (df['BOLL_upper'] - df['BOLL_lower']).replace(0, np.nan)
            df['BOLL_signal'] = df.apply(
                lambda x: 'OVERBOUGHT' if x['close'] >= x['BOLL_upper'] 
                else ('OVERSOLD' if x['close'] <= x['BOLL_lower'] else 'NEUTRAL'), axis=1
            )
        
        # 计算 KDJ 随机指标（使用高级指标模块或内置计算）
        if NEW_INDICATORS_AVAILABLE:
            try:
                from indicators import TechnicalIndicators as Ind
                df['KDJ_K'], df['KDJ_D'], df['KDJ_J'] = Ind.calculate_kdj(
                    df['high'], df['low'], df['close'],
                    n=9, m1=3, m2=3
                )
                df['KDJ_signal'] = Ind.detect_kdj_cross(df['KDJ_K'], df['KDJ_D'])
            except Exception as e:
                print(f"KDJ计算失败: {e}")
        else:
            # 简化版KDJ计算
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            rsv = ((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
            df['KDJ_K'] = rsv.rolling(window=3).mean()
            df['KDJ_D'] = df['KDJ_K'].rolling(window=3).mean()
            df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
            df['KDJ_signal'] = df.apply(
                lambda x: 'OVERBOUGHT' if x['KDJ_K'] >= 80 and x['KDJ_D'] >= 80
                else ('OVERSOLD' if x['KDJ_K'] <= 20 and x['KDJ_D'] <= 20 else 'NEUTRAL'), axis=1
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
    
    def get_signal(self, symbol: str, enable_push: bool = True) -> Optional[StockSignal]:
        """
        获取单个股票的 MA20 角度信号
        
        Args:
            symbol: 股票代码
            enable_push: 是否触发推送（与push_config配合使用）
            
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
        
        # BOLL 布林带
        boll_upper = current.get('BOLL_upper', 0.0)
        boll_lower = current.get('BOLL_lower', 0.0)
        boll_position = current.get('BOLL_position', 0.5)
        boll_signal = current.get('BOLL_signal', 'NEUTRAL')
        
        if pd.isna(boll_upper):
            boll_upper = 0.0
        if pd.isna(boll_lower):
            boll_lower = 0.0
        if pd.isna(boll_position):
            boll_position = 0.5
        if pd.isna(boll_signal):
            boll_signal = 'NEUTRAL'
        
        # KDJ 随机指标
        kdj_k = current.get('KDJ_K', 50.0)
        kdj_d = current.get('KDJ_D', 50.0)
        kdj_j = current.get('KDJ_J', 50.0)
        kdj_signal = current.get('KDJ_signal', 'NEUTRAL')
        
        if pd.isna(kdj_k):
            kdj_k = 50.0
        if pd.isna(kdj_d):
            kdj_d = 50.0
        if pd.isna(kdj_j):
            kdj_j = 50.0
        if pd.isna(kdj_signal):
            kdj_signal = 'NEUTRAL'
        
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
            ma20=ma20,
            boll_signal=boll_signal,
            kdj_signal=kdj_signal
        )
        
        # 股票名称
        name = self.watchlist.get(symbol, {}).get("name", symbol)
        
        # 获取财务数据
        pe = 0
        pb = 0
        roe = 0
        revenue_growth = 0
        profit_growth = 0
        
        if FINANCIAL_AVAILABLE:
            try:
                financial = get_stock_financials(symbol)
                if financial:
                    pe = financial.pe
                    pb = financial.pb
                    roe = financial.roe
                    revenue_growth = financial.revenue_growth
                    profit_growth = financial.profit_growth
            except Exception as e:
                print(f"获取财务数据失败: {symbol}, 错误: {e}")
        
        result_signal = StockSignal(
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
            boll_upper=boll_upper,
            boll_lower=boll_lower,
            boll_position=boll_position,
            boll_signal=boll_signal,
            kdj_k=kdj_k,
            kdj_d=kdj_d,
            kdj_j=kdj_j,
            kdj_signal=kdj_signal,
            signal=signal,
            signal_desc=signal_desc,
            industry="未知",  # TODO: 接入行业数据
            pe=pe,
            pb=pb,
            roe=roe,
            revenue_growth=revenue_growth,
            profit_growth=profit_growth,
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 触发推送检查
        if enable_push and self._notifier and self.push_config.get("push_enabled"):
            self._check_and_push_signal(symbol, name, result_signal)
        
        return result_signal
    
    def _check_and_push_signal(self, symbol: str, name: str, signal: StockSignal):
        """
        检查并触发推送
        
        Args:
            symbol: 股票代码
            name: 股票名称
            signal: StockSignal 对象
        """
        pc = self.push_config
        last_signal = self._last_signals.get(symbol)
        
        # 买入信号推送
        if signal.signal == "BUY" and pc.get("push_on_buy"):
            # 检查是否是新出现的买入信号
            should_push = False
            
            if pc.get("push_buy_only"):
                # 只推送新出现的买入信号
                if last_signal != "BUY":
                    should_push = True
            else:
                # 每次都推送
                should_push = True
            
            # 检查最小角度阈值
            if should_push and signal.ma20_angle < pc.get("min_angle_for_push", 0):
                should_push = False
            
            # 检查RSI上限（避免高位接盘）
            if should_push and signal.rsi > pc.get("min_rsi_for_buy_push", 50):
                should_push = False
            
            if should_push:
                self._send_push(signal)
                self._last_signals[symbol] = "BUY"
        
        # 卖出信号推送
        elif signal.signal == "SELL" and pc.get("push_on_sell"):
            if last_signal != "SELL":
                self._send_push(signal)
                self._last_signals[symbol] = "SELL"
        
        # 持有信号推送（可选）
        elif signal.signal == "HOLD" and pc.get("push_on_hold"):
            # 只推送信号变化
            if last_signal and last_signal != "HOLD":
                self._send_push(signal)
            self._last_signals[symbol] = "HOLD"
    
    def _send_push(self, signal: StockSignal):
        """
        发送推送
        
        Args:
            signal: StockSignal 对象
        """
        try:
            if self._notifier:
                self._notifier.send_stock_signal(
                    symbol=signal.symbol,
                    name=signal.name,
                    signal=signal.signal,
                    price=signal.price,
                    change_pct=signal.change_pct,
                    ma20_angle=signal.ma20_angle,
                    rsi=signal.rsi,
                    macd_signal=signal.macd_signal
                )
        except Exception as e:
            print(f"⚠️ 推送失败 {signal.symbol}: {e}")
    
    def _generate_signal(
        self,
        ma20_angle: float,
        rsi: float,
        rsi_signal: str,
        macd_signal: str,
        price: float,
        ma20: float,
        boll_signal: str = 'NEUTRAL',
        kdj_signal: str = 'NEUTRAL'
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
            boll_signal: BOLL 信号
            kdj_signal: KDJ 信号
            
        Returns:
            Tuple[信号, 描述]
        """
        config = self.config
        
        # 买入条件
        buy_conditions = []
        buy_score = 0  # 买入信号评分
        
        # MA20 角度大于阈值
        if ma20_angle > config["angle_threshold_buy"]:
            buy_conditions.append("MA20上升")
            buy_score += 2
        
        # RSI 条件（可选）
        if config.get("require_rsi_oversold", True):
            if rsi_signal == "OVERSOLD":
                buy_conditions.append("RSI超卖")
                buy_score += 1
        else:
            if rsi_signal == "NEUTRAL":
                buy_conditions.append("RSI中性")
                buy_score += 0.5
        
        # MACD 条件（可选）
        if config.get("require_macd_golden", True):
            if macd_signal == "GOLD_CROSS":
                buy_conditions.append("MACD金叉")
                buy_score += 2
        else:
            if macd_signal in ["GOLD_CROSS", "NEUTRAL"]:
                buy_conditions.append("MACD配合")
                buy_score += 1
        
        # BOLL 条件
        if boll_signal == "OVERSOLD":
            buy_conditions.append("BOLL下轨反弹")
            buy_score += 1.5
        elif boll_signal == "NEUTRAL":
            buy_score += 0.5
        
        # KDJ 条件
        if kdj_signal == "GOLD_CROSS":
            buy_conditions.append("KDJ金叉")
            buy_score += 2
        elif kdj_signal == "OVERSOLD":
            buy_conditions.append("KDJ超卖")
            buy_score += 1.5
        elif kdj_signal == "BULLISH":
            buy_conditions.append("KDJ多头")
            buy_score += 1
        
        # 综合买入信号判断（支持复合策略）
        use_boll_kdj = config.get("use_boll_kdj", False)
        
        if use_boll_kdj:
            # 复合策略：MA20 + RSI + BOLL + KDJ
            if (ma20_angle > config["angle_threshold_buy"] and
                (macd_signal == "GOLD_CROSS" or macd_signal == "NEUTRAL") and
                (boll_signal in ["OVERSOLD", "NEUTRAL"]) and
                (kdj_signal in ["GOLD_CROSS", "OVERSOLD", "BULLISH"])):
                return "BUY", f"复合买入: {', '.join(buy_conditions)}"
        else:
            # 原策略：MA20 + RSI + MACD
            if (ma20_angle > config["angle_threshold_buy"] and
                (not config.get("require_rsi_oversold", True) or rsi_signal == "OVERSOLD") and
                (not config.get("require_macd_golden", True) or macd_signal == "GOLD_CROSS")):
                return "BUY", f"看涨信号: {', '.join(buy_conditions)}"
        
        # 卖出条件
        sell_conditions = []
        
        # MA20 角度小于阈值
        if ma20_angle < config["angle_threshold_sell"]:
            sell_conditions.append("MA20下行")
        
        # RSI 超买
        if rsi_signal == "OVERBOUGHT":
            sell_conditions.append("RSI超买")
        
        # MACD 死叉
        if macd_signal == "DEAD_CROSS":
            sell_conditions.append("MACD死叉")
        
        # BOLL 超买
        if boll_signal == "OVERBOUGHT":
            sell_conditions.append("BOLL上轨压力")
        
        # KDJ 死叉或超买
        if kdj_signal == "DEAD_CROSS":
            sell_conditions.append("KDJ死叉")
        elif kdj_signal == "OVERBOUGHT":
            sell_conditions.append("KDJ超买")
        
        # 综合卖出信号
        if (ma20_angle < config["angle_threshold_sell"] or
            rsi_signal == "OVERBOUGHT" or
            macd_signal == "DEAD_CROSS" or
            kdj_signal == "DEAD_CROSS"):
            return "SELL", f"看跌信号: {', '.join(sell_conditions)}"
        
        # 震荡/观望
        if ma20_angle >= config["angle_threshold_sell"]:
            return "HOLD", f"震荡整理: 等待明确信号"
        
        return "HOLD", "观望等待"
    
    def scan_watchlist(self, industry: str = None) -> List[StockSignal]:
        """
        扫描股票池，获取所有信号
        
        Args:
            industry: 行业筛选（可选）
        
        Returns:
            List[StockSignal]: 信号列表
        """
        results = []
        
        for symbol, config in self.watchlist.items():
            if not config.get("enabled", True):
                continue
            
            signal = self.get_signal(symbol)
            if signal:
                # 行业过滤
                if industry and signal.industry != industry and industry not in signal.industry:
                    continue
                results.append(signal)
        
        # 按 MA20 角度降序排列
        results.sort(key=lambda x: x.ma20_angle, reverse=True)
        
        return results
    
    def scan_all_a_shares(self, limit: int = 100, industry: str = None) -> List[StockSignal]:
        """
        扫描全部 A 股（使用 akshare 获取股票列表）
        
        Args:
            limit: 限制扫描数量
            industry: 行业筛选（可选）
            
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
