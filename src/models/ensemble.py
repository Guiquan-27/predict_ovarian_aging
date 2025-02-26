# -*- coding: utf-8 -*-
"""
集成生存分析模型模块
提供多种集成方法，如堆叠、平均等
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from lifelines.utils import concordance_index
import joblib
import os

from ..models.base_models import BaseSurvivalModel

logger = logging.getLogger(__name__)

class EnsembleSurvivalModel(BaseSurvivalModel):
    """集成生存分析模型基类"""
    
    def __init__(self, name: str = "ensemble", models: List[BaseSurvivalModel] = None):
        """
        初始化集成生存分析模型
        
        参数:
        -----
        name: str, 默认 "ensemble"
            模型名称
        models: List[BaseSurvivalModel], 可选
            基础模型列表
        """
        super().__init__(name=name)
        self.models = models or []
        self.weights = None
    
    def add_model(self, model: BaseSurvivalModel):
        """
        添加基础模型
        
        参数:
        -----
        model: BaseSurvivalModel
            要添加的基础模型
        """
        self.models.append(model)
    
    def set_weights(self, weights: np.ndarray):
        """
        设置模型权重
        
        参数:
        -----
        weights: np.ndarray
            模型权重
        """
        if len(weights) != len(self.models):
            raise ValueError(f"权重数量({len(weights)})与模型数量({len(self.models)})不匹配")
        
        # 归一化权重
        self.weights = weights / np.sum(weights)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'EnsembleSurvivalModel':
        """
        训练集成模型
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        **kwargs:
            额外的训练参数
            
        返回:
        -----
        self: EnsembleSurvivalModel
            训练后的模型实例
        """
        raise NotImplementedError("子类必须实现fit方法")
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        if len(self.models) == 0:
            raise ValueError("没有基础模型")
        
        # 获取每个模型的预测
        predictions = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X, times)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"模型 {i} 预测失败: {str(e)}")
        
        # 如果没有成功的预测，则返回空数组
        if len(predictions) == 0:
            return np.array([])
        
        # 使用权重组合预测
        if self.weights is None:
            # 如果未设置权重，则使用平均值
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            # 使用加权平均
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测风险得分
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
            
        返回:
        -----
        np.ndarray
            预测的风险得分
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        if len(self.models) == 0:
            raise ValueError("没有基础模型")
        
        # 获取每个模型的风险预测
        risk_predictions = []
        for i, model in enumerate(self.models):
            try:
                risk_pred = model.predict_risk(X)
                risk_predictions.append(risk_pred)
            except Exception as e:
                logger.error(f"模型 {i} 风险预测失败: {str(e)}")
        
        # 如果没有成功的预测，则返回空数组
        if len(risk_predictions) == 0:
            return np.array([])
        
        # 使用权重组合预测
        if self.weights is None:
            # 如果未设置权重，则使用平均值
            ensemble_risk = np.mean(risk_predictions, axis=0)
        else:
            # 使用加权平均
            ensemble_risk = np.zeros_like(risk_predictions[0])
            for i, risk_pred in enumerate(risk_predictions):
                ensemble_risk += self.weights[i] * risk_pred
        
        return ensemble_risk
    
    def save(self, path: str):
        """
        保存集成模型
        
        参数:
        -----
        path: str
            保存路径
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存每个基础模型
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = f"{path}_model_{i}"
            model.save(model_path)
            model_paths.append(model_path)
        
        # 保存集成模型信息
        ensemble_info = {
            'name': self.name,
            'model_paths': model_paths,
            'weights': self.weights,
            'fitted': self.fitted,
            'ensemble_type': type(self).__name__
        }
        
        joblib.dump(ensemble_info, f"{path}_ensemble_info.pkl")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleSurvivalModel':
        """
        加载集成模型
        
        参数:
        -----
        path: str
            模型路径
            
        返回:
        -----
        EnsembleSurvivalModel
            加载的模型实例
        """
        # 加载集成模型信息
        ensemble_info = joblib.load(f"{path}_ensemble_info.pkl")
        
        # 创建集成模型实例
        ensemble_model = cls(name=ensemble_info['name'])
        
        # 加载每个基础模型
        for model_path in ensemble_info['model_paths']:
            model = BaseSurvivalModel.load(model_path)
            ensemble_model.add_model(model)
        
        # 设置权重和其他属性
        ensemble_model.weights = ensemble_info['weights']
        ensemble_model.fitted = ensemble_info['fitted']
        
        return ensemble_model


class AveragingEnsemble(EnsembleSurvivalModel):
    """Averaging ensemble for survival models"""
    
    def __init__(self, models: List[BaseSurvivalModel] = None, weights: List[float] = None, name: str = "averaging_ensemble"):
        """
        Initialize averaging ensemble
        
        Parameters:
        -----
        models: List[BaseSurvivalModel], optional
            List of base models
        weights: List[float], optional
            Weights for each model
        name: str, default "averaging_ensemble"
            Model name
        """
        super().__init__(name=name)
        self.models = models or []
        
        # Initialize weights as equal if not provided
        if weights is None and models is not None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights or []
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'AveragingEnsemble':
        """
        Train ensemble model
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target dataframe with time and event columns
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
        **kwargs:
            Additional parameters to pass to base models
        
        Returns:
        -----
        AveragingEnsemble
            Fitted model
        """
        logger.info(f"Training averaging ensemble with {len(self.models)} base models")
        
        self.feature_names = list(X.columns)
        
        # Store time and event columns
        self.time_col = time_col
        self.event_col = event_col
        
        # Fit each base model
        for i, model in enumerate(self.models):
            logger.info(f"Training base model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y, time_col=time_col, event_col=event_col, **kwargs)
        
        self.fitted = True
        return self
    
    def _optimize_weights(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str, event_col: str):
        """
        优化模型权重
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str
            时间列名
        event_col: str
            事件列名
        """
        # 获取每个模型的C-index
        c_indices = []
        for model in self.models:
            c_index = model.score(X, y, time_col, event_col)
            c_indices.append(c_index)
        
        # 将C-index转换为权重
        # 这里使用一个简单的方法：权重与C-index成正比
        weights = np.array(c_indices)
        weights = weights / np.sum(weights)
        
        self.weights = weights
        logger.info(f"优化后的权重: {weights}")


class StackingEnsemble(EnsembleSurvivalModel):
    """堆叠集成模型"""
    
    def __init__(self, name: str = "stacking_ensemble", models: List[BaseSurvivalModel] = None, 
                 meta_learner: Any = None, cv: int = 5):
        """
        初始化堆叠集成模型
        
        参数:
        -----
        name: str, 默认 "stacking_ensemble"
            模型名称
        models: List[BaseSurvivalModel], 可选
            基础模型列表
        meta_learner: Any, 可选
            元学习器
        cv: int, 默认 5
            交叉验证折数
        """
        super().__init__(name=name, models=models)
        self.meta_learner = meta_learner
        self.cv = cv
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'StackingEnsemble':
        """
        训练堆叠集成模型
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        **kwargs:
            额外的训练参数
            
        返回:
        -----
        self: StackingEnsemble
            训练后的模型实例
        """
        logger.info(f"开始训练堆叠集成模型，基础模型数量: {len(self.models)}")
        
        if len(self.models) == 0:
            raise ValueError("没有基础模型")
        
        # 检查元学习器
        if self.meta_learner is None:
            from ..models.base_models import CoxPHModel
            logger.info("未指定元学习器，使用默认的Cox比例风险模型")
            self.meta_learner = CoxPHModel(name="meta_learner")
        
        # 生成元特征
        meta_features = self._generate_meta_features(X, y, time_col, event_col)
        
        # 训练元学习器
        self.meta_learner.fit(meta_features, y, time_col, event_col, **kwargs)
        
        # 重新训练所有基础模型
        for model in self.models:
            model.fit(X, y, time_col, event_col, **kwargs)
        
        self.fitted = True
        logger.info("堆叠集成模型训练完成")
        return self
    
    def _generate_meta_features(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
        """
        生成元特征
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str
            时间列名
        event_col: str
            事件列名
            
        返回:
        -----
        pd.DataFrame
            元特征
        """
        from sklearn.model_selection import KFold
        
        # 创建交叉验证对象
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # 初始化元特征
        meta_features = np.zeros((len(X), len(self.models)))
        
        # 对每个基础模型进行交叉验证预测
        for i, model in enumerate(self.models):
            logger.info(f"为模型 {i} 生成元特征")
            
            # 对每个折进行训练和预测
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                model_clone = model.__class__(**model.__dict__)
                model_clone.fit(X_train, y_train, time_col, event_col)
                
                # 预测风险得分
                meta_features[val_idx, i] = model_clone.predict_risk(X_val)
        
        # 转换为DataFrame
        meta_df = pd.DataFrame(
            meta_features,
            columns=[f"model_{i}_risk" for i in range(len(self.models))]
        )
        
        return meta_df
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 生成元特征
        meta_features = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            meta_features[:, i] = model.predict_risk(X)
        
        # 转换为DataFrame
        meta_df = pd.DataFrame(
            meta_features,
            columns=[f"model_{i}_risk" for i in range(len(self.models))]
        )
        
        # 使用元学习器预测
        return self.meta_learner.predict(meta_df, times)
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测风险得分
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
            
        返回:
        -----
        np.ndarray
            预测的风险得分
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 生成元特征
        meta_features = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            meta_features[:, i] = model.predict_risk(X)
        
        # 转换为DataFrame
        meta_df = pd.DataFrame(
            meta_features,
            columns=[f"model_{i}_risk" for i in range(len(self.models))]
        )
        
        # 使用元学习器预测风险
        return self.meta_learner.predict_risk(meta_df)


def create_ensemble_model(ensemble_type: str, models: List[BaseSurvivalModel] = None, **kwargs) -> EnsembleSurvivalModel:
    """
    创建指定类型的集成模型
    
    参数:
    -----
    ensemble_type: str
        集成类型，可选 'averaging', 'stacking'
    models: List[BaseSurvivalModel], 可选
        基础模型列表
    **kwargs:
        传递给集成模型构造函数的参数
        
    返回:
    -----
    EnsembleSurvivalModel
        创建的集成模型实例
    """
    ensemble_type = ensemble_type.lower()
    
    if ensemble_type == 'averaging':
        return AveragingEnsemble(models=models, **kwargs)
    elif ensemble_type == 'stacking':
        return StackingEnsemble(models=models, **kwargs)
    else:
        raise ValueError(f"不支持的集成类型: {ensemble_type}") 