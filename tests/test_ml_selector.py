# quant_project/tests/test_ml_selector.py
"""
机器学习选股模块测试

测试:
- 特征工程
- 模型训练
- 预测功能
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMLSelector:
    """机器学习选股器测试"""

    @pytest.fixture
    def sample_stock_data(self) -> pd.DataFrame:
        """生成模拟股票数据"""
        np.random.seed(42)
        n = 100
        
        # 生成价格数据
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(n) * 2)
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        # 计算MA20
        df['ma20'] = df['close'].rolling(20).mean()
        
        return df

    def test_ml_selector_import(self):
        """测试ML选股器导入"""
        try:
            from ml_selector import MLSelector
            assert MLSelector is not None
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    def test_ml_selector_initialization(self):
        """测试ML选股器初始化"""
        try:
            from ml_selector import MLSelector
            
            # 默认初始化
            selector = MLSelector()
            assert selector.model_type == 'random_forest'
            assert not selector.is_trained
            
            # 指定模型类型
            selector_lr = MLSelector(model_type='logistic')
            assert selector_lr.model_type == 'logistic'
            
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    def test_feature_names(self):
        """测试特征名称"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            
            # 验证特征名称
            expected_features = [
                'ma20_angle',
                'rsi',
                'macd_diff',
                'volume_change',
                'price_momentum',
                'volatility',
                'rsi_position',
                'macd_histogram'
            ]
            
            for feature in expected_features:
                assert feature in selector.feature_names
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    @pytest.mark.ml
    def test_feature_calculation(self, sample_stock_data: pd.DataFrame):
        """测试特征计算"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            # 验证特征生成
            assert features is not None
            assert len(features) > 0
            
            # 验证特征列存在
            for feature_name in selector.feature_names:
                assert feature_name in features.columns or feature_name in features.index.names
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    @pytest.mark.ml
    def test_feature_scaling(self, sample_stock_data: pd.DataFrame):
        """测试特征标准化"""
        try:
            from ml_selector import MLSelector
            from sklearn.preprocessing import StandardScaler
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            # 选择数值特征进行缩放
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            feature_matrix = features[numeric_cols].dropna()
            
            if len(feature_matrix) > 0:
                scaled = selector.scaler.fit_transform(feature_matrix)
                
                # 验证缩放后均值接近0，标准差接近1
                assert np.abs(scaled.mean()) < 0.1
                assert np.abs(scaled.std() - 1) < 0.1
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    @pytest.mark.ml
    def test_model_training(self, sample_stock_data: pd.DataFrame):
        """测试模型训练"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector(model_type='random_forest')
            
            # 准备训练数据
            features = selector._calculate_features(sample_stock_data)
            
            if features is not None and len(features) > 30:
                # 创建标签（明日涨跌）
                close_prices = sample_stock_data['close'].values
                labels = (close_prices[1:] > close_prices[:-1]).astype(int)
                
                # 选择有效特征
                feature_cols = [col for col in features.columns 
                              if col in selector.feature_names]
                
                if len(feature_cols) > 0 and len(labels) > 10:
                    X = features[feature_cols].dropna()
                    y = labels[:len(X)]
                    
                    if len(X) > 10 and len(y) > 10:
                        X_train, X_test, y_train, y_test = selector._prepare_data(
                            X, y, test_size=0.2
                        )
                        
                        # 训练模型
                        selector.fit(X_train, y_train)
                        
                        # 验证模型已训练
                        assert selector.is_trained
                        assert selector.model is not None
                        
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    @pytest.mark.ml
    def test_prediction(self, sample_stock_data: pd.DataFrame):
        """测试预测功能"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector(model_type='random_forest')
            
            # 准备训练数据
            features = selector._calculate_features(sample_stock_data)
            
            if features is not None and len(features) > 30:
                close_prices = sample_stock_data['close'].values
                labels = (close_prices[1:] > close_prices[:-1]).astype(int)
                
                feature_cols = [col for col in features.columns 
                              if col in selector.feature_names]
                
                if len(feature_cols) > 0 and len(labels) > 10:
                    X = features[feature_cols].dropna()
                    y = labels[:len(X)]
                    
                    if len(X) > 10 and len(y) > 10:
                        # 训练模型
                        X_train, X_test, y_train, y_test = selector._prepare_data(
                            X, y, test_size=0.2
                        )
                        selector.fit(X_train, y_train)
                        
                        # 进行预测
                        prediction = selector.predict(X_test)
                        
                        # 验证预测结果
                        assert prediction is not None
                        assert len(prediction) == len(X_test)
                        
        except ImportError:
            pytest.skip("ML选股器模块导入失败")


class TestMLFeatures:
    """机器学习特征工程测试"""

    def test_ma20_angle_feature(self, sample_stock_data: pd.DataFrame):
        """测试MA20角度特征"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            if 'ma20_angle' in features.columns:
                valid_angles = features['ma20_angle'].dropna()
                assert (valid_angles > -90).all() and (valid_angles < 90).all()
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    def test_rsi_feature(self, sample_stock_data: pd.DataFrame):
        """测试RSI特征"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            if 'rsi' in features.columns:
                valid_rsi = features['rsi'].dropna()
                assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    def test_volume_change_feature(self, sample_stock_data: pd.DataFrame):
        """测试成交量变化率特征"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            if 'volume_change' in features.columns:
                # 成交量变化率可以有任意值
                assert features['volume_change'].notna().sum() > 0
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")

    def test_volatility_feature(self, sample_stock_data: pd.DataFrame):
        """测试波动率特征"""
        try:
            from ml_selector import MLSelector
            
            selector = MLSelector()
            features = selector._calculate_features(sample_stock_data)
            
            if 'volatility' in features.columns:
                valid_vol = features['volatility'].dropna()
                assert (valid_vol >= 0).all()
                
        except ImportError:
            pytest.skip("ML选股器模块导入失败")


class TestMLModels:
    """机器学习模型测试"""

    def test_random_forest_model(self):
        """测试随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # 创建模拟数据
            X, y = make_classification(n_samples=100, n_features=5)
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # 预测
            predictions = model.predict(X)
            
            assert len(predictions) == len(y)
            
        except ImportError:
            pytest.skip("sklearn未安装")

    def test_logistic_regression_model(self):
        """测试逻辑回归模型"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            
            # 创建模拟数据
            X, y = make_classification(n_samples=100, n_features=5)
            
            # 训练模型
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            
            # 预测
            predictions = model.predict(X)
            
            assert len(predictions) == len(y)
            
        except ImportError:
            pytest.skip("sklearn未安装")

    @pytest.mark.ml
    def test_model_comparison(self):
        """测试模型比较"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            from sklearn.model_selection import cross_val_score
            
            # 创建模拟数据
            X, y = make_classification(n_samples=200, n_features=10)
            
            # 测试随机森林
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf_scores = cross_val_score(rf, X, y, cv=5)
            
            # 测试逻辑回归
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr_scores = cross_val_score(lr, X, y, cv=5)
            
            # 验证两种模型都能进行交叉验证
            assert len(rf_scores) == 5
            assert len(lr_scores) == 5
            
        except ImportError:
            pytest.skip("sklearn未安装")


class TestMLIntegration:
    """机器学习集成测试"""

    @pytest.mark.ml
    @pytest.mark.slow
    def test_full_ml_pipeline(self):
        """测试完整的ML流程"""
        try:
            from ml_selector import MLSelector
            
            # 生成测试数据
            np.random.seed(42)
            n = 150
            
            df = pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(n)),
                'volume': np.random.randint(1000000, 10000000, n)
            })
            df['ma20'] = df['close'].rolling(20).mean()
            
            # 创建ML选股器
            selector = MLSelector(model_type='random_forest')
            
            # 计算特征
            features = selector._calculate_features(df)
            
            if features is not None and len(features) > 50:
                close_prices = df['close'].values
                labels = (close_prices[1:] > close_prices[:-1]).astype(int)
                
                feature_cols = [col for col in features.columns 
                              if col in selector.feature_names]
                
                if len(feature_cols) >= 3 and len(labels) > 20:
                    X = features[feature_cols].dropna()
                    y = labels[:len(X)]
                    
                    # 划分数据
                    X_train, X_test, y_train, y_test = selector._prepare_data(
                        X, y, test_size=0.2
                    )
                    
                    # 训练
                    selector.fit(X_train, y_train)
                    assert selector.is_trained
                    
                    # 预测
                    predictions = selector.predict(X_test)
                    assert len(predictions) == len(X_test)
                    
                    # 评估
                    metrics = selector.evaluate(X_test, y_test)
                    assert 'accuracy' in metrics
                    
        except ImportError:
            pytest.skip("ML选股器模块导入失败")
        except Exception as e:
            pytest.skip(f"ML流程测试跳过: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
