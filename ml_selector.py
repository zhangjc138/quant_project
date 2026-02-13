# -*- coding: utf-8 -*-
"""
ML选股模块 - 付费版专属功能
基于sklearn的机器学习模型预测股票涨跌概率

功能:
- 简单线性回归/逻辑回归预测明日涨跌概率
- 随机森林分类器
- 特征工程：MA角度、RSI、MACD、成交量变化率
- 输出预测信号和置信度
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MLSelector:
    """
    机器学习选股器
    
    使用历史技术指标特征训练模型，预测明日涨跌概率
    付费版专属功能
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        初始化ML选股器
        
        Args:
            model_type: 模型类型 ('logistic' 或 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_weights = {}
        
        # 特征名称
        self.feature_names = [
            'ma20_angle',      # MA20角度
            'rsi',             # RSI指标
            'macd_diff',       # MACD差值
            'volume_change',   # 成交量变化率
            'price_momentum',  # 价格动量
            'volatility',      # 波动率
            'rsi_position',    # RSI位置(0-1)
            'macd_histogram'   # MACD柱状图
        ]
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        result = df.copy()
        
        # MA20角度
        if 'ma20' in result.columns:
            result['ma20_angle'] = np.arctan(
                (result['ma20'] - result['ma20'].shift(1)) / 1
            ) * 180 / np.pi
        else:
            # 简单MA20角度计算
            ma20 = result['close'].rolling(window=20).mean()
            result['ma20_angle'] = np.arctan(
                (ma20 - ma20.shift(1)) / ma20.shift(1).replace(0, np.nan)
            ) * 180 / np.pi
        
        # RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['close'].ewm(span=12, adjust=False).mean()
        ema26 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd_diff'] = ema12 - ema26
        result['macd_histogram'] = result['macd_diff'] - result['macd_diff'].ewm(span=9, adjust=False).mean()
        
        # 成交量变化率
        result['volume_change'] = result['volume'].pct_change()
        
        # 价格动量 (5日涨幅)
        result['price_momentum'] = result['close'].pct_change(5)
        
        # 波动率 (5日标准差)
        result['volatility'] = result['close'].pct_change().rolling(window=5).std()
        
        # RSI位置 (0-1 归一化)
        result['rsi_position'] = (result['rsi'] - 30) / 40
        result['rsi_position'] = result['rsi_position'].clip(0, 1)
        
        return result
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            df: 历史数据
            
        Returns:
            X: 特征矩阵
            y: 标签 (1=上涨, 0=下跌/持平)
        """
        data = self._calculate_features(df)
        
        # 创建目标变量：明日涨跌
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # 移除包含NaN的行
        data = data.dropna(subset=self.feature_names + ['target'])
        
        X = data[self.feature_names].values
        y = data['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            df: 历史数据 (至少100个交易日)
            verbose: 是否打印训练信息
            
        Returns:
            训练结果字典
        """
        if len(df) < 100:
            raise ValueError("需要至少100个交易日的数据进行训练")
        
        X, y = self._prepare_training_data(df)
        
        if len(X) < 50:
            raise ValueError("有效数据点不足，请检查数据完整性")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 选择模型
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_weights = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            self.feature_weights = dict(zip(
                self.feature_names,
                np.abs(self.model.coef_[0])
            ))
            # 归一化
            total = sum(self.feature_weights.values())
            if total > 0:
                self.feature_weights = {k: v/total for k, v in self.feature_weights.items()}
        
        self.is_trained = True
        
        if verbose:
            print(f"ML模型训练完成")
            print(f"模型类型: {self.model_type}")
            print(f"训练样本数: {len(X_train)}")
            print(f"测试准确率: {accuracy:.2%}")
            print("\n特征权重:")
            for feat, weight in sorted(self.feature_weights.items(), 
                                        key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {weight:.3f}")
        
        return {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_weights': self.feature_weights
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        预测单个股票
        
        Args:
            df: 最新数据 (至少1行)
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用 train() 方法")
        
        if len(df) < 1:
            raise ValueError("需要提供至少1行数据")
        
        # 计算特征
        data = self._calculate_features(df)
        latest = data.iloc[-1]
        
        # 提取特征
        features = latest[self.feature_names].values.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # 预测
        proba = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        # 解析结果
        confidence = float(proba[prediction])
        up_prob = float(proba[1])
        down_prob = float(proba[0])
        
        # 生成信号
        if prediction == 1 and confidence > 0.6:
            signal = 'BUY'
        elif prediction == 0 and confidence > 0.6:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'up_probability': up_prob,
            'down_probability': down_prob,
            'confidence': confidence,
            'prediction': bool(prediction),
            'features': {k: float(latest[k]) for k in self.feature_names},
            'feature_weights': self.feature_weights
        }
    
    def batch_predict(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        批量预测多只股票
        
        Args:
            stock_data: 股票代码到数据的映射
            
        Returns:
            预测结果列表
        """
        results = []
        for symbol, df in stock_data.items():
            try:
                pred = self.predict(df)
                pred['symbol'] = symbol
                results.append(pred)
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'signal': 'ERROR',
                    'error': str(e)
                })
        
        # 按置信度排序
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征名称到重要性的映射
        """
        if not self.is_trained:
            return {}
        return self.feature_weights
    
    def save_model(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'feature_weights': self.feature_weights,
                'is_trained': self.is_trained
            }, f)
        print(f"模型已保存至: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'MLSelector':
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        selector = cls(model_type=data['model_type'])
        selector.model = data['model']
        selector.scaler = data['scaler']
        selector.feature_names = data['feature_names']
        selector.feature_weights = data['feature_weights']
        selector.is_trained = data['is_trained']
        
        print(f"模型已从 {path} 加载")
        return selector


def ml_stock_selector(stock_data: Dict[str, pd.DataFrame], 
                      model_type: str = 'random_forest',
                      min_confidence: float = 0.6) -> List[Dict]:
    """
    便捷函数：机器学习选股
    
    Args:
        stock_data: 股票代码到数据的映射
        model_type: 模型类型 ('logistic' 或 'random_forest')
        min_confidence: 最小置信度阈值
        
    Returns:
        精选股票列表 (BUY信号)
    """
    selector = MLSelector(model_type)
    
    # 合并所有数据训练模型
    all_data = pd.concat(stock_data.values(), ignore_index=True)
    selector.train(all_data, verbose=False)
    
    # 批量预测
    results = selector.batch_predict(stock_data)
    
    # 过滤高置信度BUY信号
    buy_signals = [
        r for r in results 
        if r['signal'] == 'BUY' and r['confidence'] >= min_confidence
    ]
    
    return buy_signals


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ML选股模块测试")
    print("=" * 60)
    
    # 模拟数据测试
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # 创建模拟价格数据
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    open_ = close - np.random.randn(200) * 0.2
    high = close + np.abs(np.random.randn(200) * 0.3)
    low = close - np.abs(np.random.randn(200) * 0.3)
    volume = np.random.randint(1000000, 10000000, 200)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    print(f"\n模拟数据已生成: {len(df)} 个交易日")
    
    # 训练模型
    selector = MLSelector(model_type='random_forest')
    result = selector.train(df, verbose=True)
    
    # 单股预测
    pred = selector.predict(df)
    print(f"\n预测结果:")
    print(f"  信号: {pred['signal']}")
    print(f"  上涨概率: {pred['up_probability']:.2%}")
    print(f"  置信度: {pred['confidence']:.2%}")
    
    # 特征重要性
    print(f"\n特征重要性排名:")
    for feat, weight in sorted(pred['feature_weights'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {weight:.3f}")
