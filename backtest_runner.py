#!/usr/bin/env python3
"""期货策略回测"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

def gen_mock_data(symbol):
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")
    if symbol == "CU2006":
        base, vol = 50000, 0.02
    elif symbol == "RB2005":
        base, vol = 3500, 0.025
    else:
        base, vol = 4000, 0.015
    returns = np.random.normal(0, vol, len(dates))
    closes = base * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "date": dates,
        "open": closes * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": closes * (1 + np.random.uniform(0, 0.015, len(dates))),
        "low": closes * (1 - np.random.uniform(0, 0.015, len(dates))),
        "close": closes,
        "volume": np.random.randint(100000, 1000000, len(dates)),
        "open_interest": np.random.randint(100000, 500000, len(dates)),
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df

class VolAdaptive(bt.Strategy):
    params = (("fast", 10), ("slow", 30), ("atr_p", 14), ("atr_mul", 2.0),)
    
    def __init__(self):
        self.order = None
        self.entry_price = 0
        self.sma_f = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.sma_s = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_p)
        self.cr = bt.indicators.CrossOver(self.sma_f, self.sma_s)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            self.order = None
            if order.isbuy:
                self.entry_price = order.executed.price
                
    def next(self):
        if self.order:
            return
        if self.cr > 0 and not self.position:
            risk = self.params.atr_mul * self.atr[0] + 0.0001
            sz = int(0.02 * self.broker.getvalue() / risk)
            self.order = self.buy(size=max(1, sz))
        elif self.cr < 0 and self.position:
            self.order = self.sell(size=self.position.size)
        elif self.position and self.entry_price > 0:
            if self.data.close[0] < self.entry_price - self.params.atr_mul * self.atr[0]:
                self.order = self.sell(size=self.position.size)

class Intraday(bt.Strategy):
    params = (("breakout", 0.002), ("atr_mul", 2.0),)
    
    def __init__(self):
        self.order = None
        self.entry_price = 0
        self.atr = bt.indicators.ATR(self.data, period=14)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            self.order = None
            if order.isbuy:
                self.entry_price = order.executed.price
                
    def next(self):
        if self.order:
            return
        op = self.data.open[0]
        hi = self.data.high[0]
        lo = self.data.low[0]
        if hi > op * (1 + self.params.breakout) and not self.position:
            self.order = self.buy()
        elif lo < op * (1 - self.params.breakout) and not self.position:
            self.order = self.sell()
        elif self.position and self.entry_price > 0:
            if self.data.close[0] < self.entry_price - self.params.atr_mul * self.atr[0]:
                self.order = self.close()

class ChanTheory(bt.Strategy):
    params = (("big", 60), ("mid", 30), ("sma", 10), ("atr", 14),)
    
    def __init__(self):
        self.order = None
        self.entry_price = 0
        self.sma_b = bt.indicators.SMA(self.data.close, period=self.params.big)
        self.sma_m = bt.indicators.SMA(self.data.close, period=self.params.mid)
        self.sma_s = bt.indicators.SMA(self.data.close, period=self.params.sma)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            self.order = None
            if order.isbuy:
                self.entry_price = order.executed.price
                
    def next(self):
        if self.order:
            return
        up = self.data.close[0] > self.sma_b[0]
        golden = self.sma_s[0] > self.sma_m[0] and self.sma_s[-1] <= self.sma_m[-1]
        if golden and up and 30 < self.rsi[0] < 70 and not self.position:
            risk = 2.0 * self.atr[0] + 0.0001
            sz = int(0.02 * self.broker.getvalue() / risk)
            self.order = self.buy(size=max(1, sz))
        elif not up and self.position:
            self.order = self.sell(size=self.position.size)

def run_test(df, strat_cls, name):
    print("\n" + "=" * 50)
    print("Strategy: " + name)
    print("=" * 50)
    
    cer = bt.Cerebro()
    cer.addstrategy(strat_cls)
    data = bt.feeds.PandasData(
        dataname=df, datetime="date", 
        fromdate=datetime(2000, 1, 1), 
        todate=datetime(2099, 12, 31)
    )
    cer.adddata(data)
    cer.broker.setcash(100000)
    cer.broker.setcommission(commission=0.0002)
    cer.addanalyzer(bt.analyzers.SharpeRatio, _name="sr")
    cer.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    
    cer.run()
    final = cer.broker.getvalue()
    ret = (final - 100000) / 100000 * 100
    
    sr = cer.runstrats[0][0].analyzers.sr.get_analysis()
    dd = cer.runstrats[0][0].analyzers.dd.get_analysis()
    
    sharpe_val = sr.get("sharperatio", "N/A")
    maxdd_val = dd.get("max", {}).get("drawdown", "N/A")
    
    print("Final Value: {:,.2f}".format(final))
    print("Return: {:.2f}%".format(ret))
    print("Sharpe: {}".format(sharpe_val))
    print("MaxDD: {}%".format(maxdd_val))
    
    return {
        "name": name, 
        "return": ret, 
        "sharpe": sharpe_val, 
        "maxdd": maxdd_val
    }

def main():
    print("=" * 60)
    print("期货策略回测分析")
    print("=" * 60)
    
    symbols = ["RB2005", "CU2006", "IF2006"]
    results = []
    
    for sym in symbols:
        print("\n\n" + "#" * 60)
        print("# Symbol: " + sym)
        print("#" * 60)
        df = gen_mock_data(sym)
        
        r1 = run_test(df, VolAdaptive, "VolAdaptive Trend")
        results.append(r1)
        
        r2 = run_test(df, Intraday, "Intraday")
        results.append(r2)
        
        r3 = run_test(df, ChanTheory, "Multi-Timeframe Chan")
        results.append(r3)
    
    print("\n" + "=" * 60)
    print("综合对比")
    print("=" * 60)
    print("{:25} {:>10} {:>10} {:>10}".format("策略", "收益率", "夏普", "最大回撤"))
    print("-" * 60)
    for r in results:
        print("{:25} {:>9.2f}% {:>10} {:>9}%".format(
            r["name"], r["return"], r["sharpe"], r["maxdd"]
        ))
    
    best = max(results, key=lambda x: x["return"])
    print("\n最优策略: {} ({:.2f}%)".format(best["name"], best["return"]))

if __name__ == "__main__":
    main()
