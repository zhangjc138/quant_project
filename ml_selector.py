#!/usr/bin/env python3
"""
机器学习选股模块（开源版）

使用机器学习进行股票预测
基于 scikit-learn，无需付费依赖
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn 不可用")


@dataclass
class MLPredictionResult:
    """ML预测结果"""
    symbol: str            # 股票代码
    signal: str           # 预测信号 (UP/DOWN)
    up_probability: float  # 上涨概率
    down_probability: float  # 下跌概率
    confidence: float     # 置信度
    accuracy: float       # 模型准确率
    feature_importance: Dict  # 特征重要性


class MLSelector:
    """机器学习选股器（开源版）"""
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        n_estimators: int = 100,
        max_depth: int = 5,
        test_size: float = 0.2
    ):
        """
        初始化ML选股器
        
        Args:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'logistic')
            n_estimators: 树的数量
            max_depth: 最大深度
            test_size: 测试集比例
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.accuracy = 0
        
    def _create_model(self):
        """创建模型"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        else:  # logistic
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签
        
        Args:
            df: 股票数据
        
        Returns:
            X: 特征矩阵
            y: 标签 (1=上涨, 0=下跌)
        """
        # 计算特征
        df = df.copy()
        
        # 基础特征
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        
        # 成交量变化
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        
        # 波动率
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # 动量
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # 创建标签：未来5天涨跌
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)
        
        # 特征列表
        feature_cols = [
            'ma5', 'ma10', 'ma20',  # 移动平均
            'price_change', 'price_change_5',  # 价格变化
            'volume_change', 'volume_ma5',  # 成交量
            'rsi',  # RSI
            'macd',  # MACD
            'volatility',  # 波动率
            'momentum_5', 'momentum_10',  # 动量
        ]
        
        self.feature_names = feature_cols
        
        # 删除NaN
        df = df.dropna(subset=feature_cols + ['label'])
        
        if len(df) < 50:
            return None, None
        
        X = df[feature_cols].values
        y = df['label'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            df: 股票数据
            verbose: 是否打印详细信息
        
        Returns:
            Dict: 训练结果
        """
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'scikit-learn 不可用'}
        
        try:
            # 准备数据
            X, y = self._prepare_features(df)
            
            if X is None:
                return {'success': False, 'error': '数据不足'}
            
            # 标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=self.test_size, 
                random_state=42
            )
            
            # 创建并训练模型
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
            
            # 评估
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            if verbose:
                print(f"\n✅ 模型训练完成!")
                print(f"   模型类型: {self.model_type}")
                print(f"   准确率: {self.accuracy:.2%}")
                print(f"   训练样本: {len(X_train)}")
                print(f"   测试样本: {len(X_test)}")
            
            # 交叉验证
            if len(X_scaled) >= 50:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
                cv_accuracy = cv_scores.mean()
            else:
                cv_accuracy = self.accuracy
            
            # 获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(self.feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
            else:
                importance = {}
            
            return {
                'success': True,
                'accuracy': self.accuracy,
                'cv_accuracy': cv_accuracy,
                'feature_importance': importance,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ 训练失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> MLPredictionResult:
        """
        预测
        
        Args:
            df: 股票数据
        
        Returns:
            MLPredictionResult: 预测结果
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            # 返回默认结果
            return MLPredictionResult(
                symbol="",
                signal="HOLD",
                up_probability=0.5,
                down_probability=0.5,
                confidence=0.5,
                accuracy=0.5,
                feature_importance={}
            )
        
        try:
            # 准备最新数据
            df = df.copy()
            
            # 计算特征
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 0.0001)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            feature_cols = [
                'ma5', 'ma10', 'ma20',
                'price_change', 'price_change_5',
                'volume_change', 'volume_ma5',
                'rsi', 'macd', 'volatility',
                'momentum_5', 'momentum_10',
            ]
            
            X = latest[feature_cols].values.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # 预测
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # 确定信号
            up_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            down_prob = probabilities[0] if len(probabilities) > 0 else 0.5
            
            if up_prob > 0.55:
                signal = "UP"
                confidence = up_prob
            elif down_prob > 0.55:
                signal = "DOWN"
                confidence = down_prob
            else:
                signal = "HOLD"
                confidence = 0.5 + abs(up_prob - down_prob)
            
            # 获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                importance = {}
            
            return MLPredictionResult(
                symbol="",
                signal=signal,
                up_probability=up_prob,
                down_probability=down_prob,
                confidence=confidence,
                accuracy=self.accuracy,
                feature_importance=importance
            )
            
        except Exception as e:
            print(f"预测失败: {e}")
            return MLPredictionResult(
                symbol="",
                signal="HOLD",
                up_probability=0.5,
                down_probability=0.5,
                confidence=0.5,
                accuracy=self.accuracy,
                feature_importance={}
            )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        return importance_df


# 便捷函数
def quick_ml_predict(df: pd.DataFrame) -> MLPredictionResult:
    """快速ML预测"""
    selector = MLSelector(model_type='random_forest')
    result = selector.train(df, verbose=False)
    
    if result.get('success'):
        return selector.predict(df)
    else:
        return MLPredictionResult(
            symbol="",
            signal="HOLD",
            up_probability=0.5,
            down_probability=0.5,
            confidence=0.5,
            accuracy=0.5,
            feature_importance={}
        )


if __name__ == "__main__":
    print("ML选股模块测试")
    
    # 创建模拟数据
    dates = pd.date_range(start="2024-01-01", periods=500, freq="D")
    np.random.seed(42)
    
    # 生成带趋势的价格数据
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5 + 0.05)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * (1 + np.random.randn(500) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(500) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(500) * 0.02)),
        'volume': np.random.randint(1000000, 10000000, 500),
    })
    
    # 创建选股器
    selector = MLSelector(model_type='random_forest')
    
    # 训练
    print("\n训练模型...")
    result = selector.train(df, verbose=True)
    
    if result['success']:
        # 预测
        print("\n预测...")
        pred = selector.predict(df)
        
        print(f"\n📊 预测结果:")
        print(f"   信号: {pred.signal}")
        print(f"   上涨概率: {pred.up_probability:.1%}")
        print(f"   下跌概率: {pred.down_probability:.1%}")
        print(f"   置信度: {pred.confidence:.1%}")
        print(f"   模型准确率: {pred.accuracy:.1%}")
        
        # 特征重要性
        print(f"\n📈 特征重要性 (Top 5):")
        importance = pred.feature_importance
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, imp in sorted_importance:
                print(f"   {feat}: {imp:.3f}")
    else:
        print(f"训练失败: {result.get('error')}")
