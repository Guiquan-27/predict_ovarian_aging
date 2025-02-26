# -*- coding: utf-8 -*-
"""
基础生存分析模型模块
提供Cox比例风险模型、随机生存森林等基础生存分析模型
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import joblib
import os

logger = logging.getLogger(__name__)

class BaseSurvivalModel:
    """基础生存分析模型类，提供通用接口"""
    
    def __init__(self, name: str = "base_model"):
        """
        初始化基础生存分析模型
        
        参数:
        -----
        name: str, 默认 "base_model"
            模型名称
        """
        self.name = name
        self.model = None
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'BaseSurvivalModel':
        """
        训练模型
        
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
            额外的模型参数
            
        返回:
        -----
        self: BaseSurvivalModel
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
            预测时间点，默认为None(使用训练数据中的时间点)
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        raise NotImplementedError("子类必须实现predict方法")
    
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
        raise NotImplementedError("子类必须实现predict_risk方法")
    
    def score(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event') -> float:
        """
        计算模型的C-index
        
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
            
        返回:
        -----
        float
            C-index评分
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        risk_scores = self.predict_risk(X)
        c_index = concordance_index(y[time_col], -risk_scores, y[event_col])
        return c_index
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
        -----
        path: str
            保存路径
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"模型已保存至: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseSurvivalModel':
        """
        加载模型
        
        参数:
        -----
        path: str
            模型文件路径
            
        返回:
        -----
        BaseSurvivalModel
            加载的模型实例
        """
        model = joblib.load(path)
        logger.info(f"从 {path} 加载模型")
        return model


class CoxPHModel(BaseSurvivalModel):
    """Cox比例风险模型"""
    
    def __init__(self, name: str = "cox_ph", **kwargs):
        """
        初始化Cox比例风险模型
        
        参数:
        -----
        name: str, 默认 "cox_ph"
            模型名称
        **kwargs:
            传递给CoxPHFitter的参数
        """
        super().__init__(name)
        self.model = CoxPHFitter(**kwargs)
        self.params = kwargs
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'CoxPHModel':
        """
        训练Cox比例风险模型
        
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
            传递给CoxPHFitter.fit的参数
            
        返回:
        -----
        self: CoxPHModel
            训练后的模型实例
        """
        logger.info(f"训练Cox比例风险模型: {self.name}")
        
        # 合并特征和目标变量
        df = pd.concat([X, y], axis=1)
        
        # 训练模型
        self.model.fit(df, duration_col=time_col, event_col=event_col, **kwargs)
        self.fitted = True
        
        # 记录训练结果
        logger.info(f"模型训练完成，concordance_index: {self.model.concordance_index_}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None(使用训练数据中的时间点)
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 预测生存函数
        survival_func = self.model.predict_survival_function(X)
        
        # 如果指定了时间点，则在这些时间点上评估生存函数
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, t in enumerate(times):
                survival_probs[:, i] = survival_func.loc[t].values
            return survival_probs
        else:
            # 否则返回完整的生存函数
            return survival_func
    
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
            预测的风险得分(部分风险)
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 预测部分风险
        return self.model.predict_partial_hazard(X).values
    
    def plot_coefficients(self, top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制模型系数森林图
        
        参数:
        -----
        top_n: int, 默认 10
            显示的特征数量
        figsize: Tuple[int, int], 默认 (10, 8)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 获取系数摘要
        summary = self.model.summary
        
        # 选择前N个特征
        if len(summary) > top_n:
            # 按p值排序
            plot_df = summary.sort_values('p').head(top_n).copy()
        else:
            plot_df = summary.copy()
        
        # 按风险比排序
        plot_df = plot_df.sort_values('exp(coef)')
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制森林图
        y_pos = np.arange(len(plot_df))
        
        # 绘制风险比点和置信区间
        ax.scatter(plot_df['exp(coef)'], y_pos, marker='o', s=50, color='blue')
        
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.plot([row['lower 0.95'], row['upper 0.95']], [i, i], 'b-', alpha=0.6)
        
        # 添加垂直线表示HR=1
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        # 设置Y轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df.index)
        
        # 设置X轴为对数刻度
        ax.set_xscale('log')
        
        # 添加标题和标签
        ax.set_title('Cox模型系数 (95% 置信区间)', fontsize=14)
        ax.set_xlabel('风险比 (HR)', fontsize=12)
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        return fig


class RandomSurvivalForestModel(BaseSurvivalModel):
    """随机生存森林模型"""
    
    def __init__(self, name: str = "rsf", **kwargs):
        """
        初始化随机生存森林模型
        
        参数:
        -----
        name: str, 默认 "rsf"
            模型名称
        **kwargs:
            传递给RandomSurvivalForest的参数
        """
        super().__init__(name)
        self.model = RandomSurvivalForest(**kwargs)
        self.params = kwargs
        self.event_times_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'RandomSurvivalForestModel':
        """
        训练随机生存森林模型
        
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
            传递给RandomSurvivalForest.fit的参数
            
        返回:
        -----
        self: RandomSurvivalForestModel
            训练后的模型实例
        """
        logger.info(f"训练随机生存森林模型: {self.name}")
        
        # 转换为scikit-survival所需的格式
        structured_y = Surv.from_dataframe(event_col, time_col, y)
        
        # 训练模型
        self.model.fit(X, structured_y, **kwargs)
        self.fitted = True
        self.event_times_ = self.model.event_times_
        
        # 记录训练结果
        logger.info(f"模型训练完成，特征数量: {X.shape[1]}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None(使用训练数据中的时间点)
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 预测生存函数
        survival_funcs = self.model.predict_survival_function(X)
        
        # 如果指定了时间点，则在这些时间点上评估生存函数
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, surv_func in enumerate(survival_funcs):
                for j, t in enumerate(times):
                    # 找到最接近的时间点
                    idx = np.searchsorted(self.event_times_, t)
                    if idx == len(self.event_times_):
                        idx = len(self.event_times_) - 1
                    survival_probs[i, j] = surv_func[idx]
            return survival_probs
        else:
            # 否则返回完整的生存函数
            return np.array([sf.y for sf in survival_funcs])
    
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
        
        # 预测风险得分
        return self.model.predict(X)
    
    def plot_feature_importance(self, top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制特征重要性条形图
        
        参数:
        -----
        top_n: int, 默认 10
            显示的特征数量
        figsize: Tuple[int, int], 默认 (10, 8)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 选择前N个特征
        if len(importance_df) > top_n:
            plot_df = importance_df.head(top_n).copy()
        else:
            plot_df = importance_df.copy()
        
        # 反转顺序，使最重要的特征在顶部
        plot_df = plot_df.iloc[::-1]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        ax.barh(plot_df['feature'], plot_df['importance'], color='skyblue', edgecolor='black')
        
        # 添加标题和标签
        ax.set_title('随机生存森林特征重要性', fontsize=14)
        ax.set_xlabel('重要性', fontsize=12)
        
        # 添加网格线
        ax.grid(True, axis='x', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        return fig


class CoxnetModel(BaseSurvivalModel):
    """带弹性网正则化的Cox模型"""
    
    def __init__(self, name: str = "coxnet", **kwargs):
        """
        初始化带弹性网正则化的Cox模型
        
        参数:
        -----
        name: str, 默认 "coxnet"
            模型名称
        **kwargs:
            传递给CoxnetSurvivalAnalysis的参数
        """
        super().__init__(name)
        self.model = CoxnetSurvivalAnalysis(**kwargs)
        self.params = kwargs
        self.event_times_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'CoxnetModel':
        """
        训练带弹性网正则化的Cox模型
        
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
            传递给CoxnetSurvivalAnalysis.fit的参数
            
        返回:
        -----
        self: CoxnetModel
            训练后的模型实例
        """
        logger.info(f"训练带弹性网正则化的Cox模型: {self.name}")
        
        # 转换为scikit-survival所需的格式
        structured_y = Surv.from_dataframe(event_col, time_col, y)
        
        # 训练模型
        self.model.fit(X, structured_y, **kwargs)
        self.fitted = True
        
        # 记录训练结果
        logger.info(f"模型训练完成，非零系数数量: {np.sum(self.model.coef_ != 0)}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None(使用训练数据中的时间点)
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 预测生存函数
        survival_funcs = self.model.predict_survival_function(X)
        
        # 如果指定了时间点，则在这些时间点上评估生存函数
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, surv_func in enumerate(survival_funcs):
                for j, t in enumerate(times):
                    # 找到最接近的时间点
                    idx = np.searchsorted(surv_func.x, t)
                    if idx == len(surv_func.x):
                        idx = len(surv_func.x) - 1
                    survival_probs[i, j] = surv_func.y[idx]
            return survival_probs
        else:
            # 否则返回完整的生存函数
            return np.array([sf.y for sf in survival_funcs])
    
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
        
        # 预测风险得分
        return self.model.predict(X)
    
    def plot_coefficients(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制模型系数条形图
        
        参数:
        -----
        figsize: Tuple[int, int], 默认 (10, 8)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 获取非零系数
        coef = self.model.coef_
        feature_names = self.model.feature_names_in_
        
        # 创建系数DataFrame
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef.flatten()
        })
        
        # 筛选非零系数
        nonzero_coef = coef_df[coef_df['coefficient'] != 0].copy()
        
        # 按系数绝对值排序
        nonzero_coef['abs_coef'] = nonzero_coef['coefficient'].abs()
        nonzero_coef = nonzero_coef.sort_values('abs_coef', ascending=False)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        bars = ax.barh(nonzero_coef['feature'], nonzero_coef['coefficient'], 
                      color=np.where(nonzero_coef['coefficient'] > 0, 'skyblue', 'salmon'))
        
        # 添加标题和标签
        ax.set_title('Coxnet模型非零系数', fontsize=14)
        ax.set_xlabel('系数值', fontsize=12)
        
        # 添加垂直线表示零点
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # 添加网格线
        ax.grid(True, axis='x', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        return fig


def create_model(model_type: str, **kwargs) -> BaseSurvivalModel:
    """
    创建指定类型的生存分析模型
    
    参数:
    -----
    model_type: str
        模型类型，可选 'cox', 'rsf', 'coxnet'
    **kwargs:
        传递给模型构造函数的参数
        
    返回:
    -----
    BaseSurvivalModel
        创建的模型实例
    """
    model_type = model_type.lower()
    
    if model_type == 'cox':
        return CoxPHModel(**kwargs)
    elif model_type == 'rsf':
        return RandomSurvivalForestModel(**kwargs)
    elif model_type == 'coxnet':
        return CoxnetModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 