# -*- coding: utf-8 -*-
"""
交叉验证模块
提供用于生存分析模型的交叉验证功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from lifelines.utils import concordance_index
import joblib
import os
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ..models.base_models import BaseSurvivalModel

logger = logging.getLogger(__name__)

def stratify_by_time_and_event(time: np.ndarray, event: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    根据时间和事件状态进行分层
    
    参数:
    -----
    time: np.ndarray
        生存时间
    event: np.ndarray
        事件状态
    n_bins: int, 默认 5
        时间分箱数量
        
    返回:
    -----
    np.ndarray
        分层标签
    """
    # Create time bins
    time_bins = pd.qcut(time, q=n_bins, labels=False, duplicates='drop')
    
    # Combine time bins and event status
    strata = time_bins * 2 + event
    
    return strata

class SurvivalCV:
    """Survival analysis cross-validation class"""
    
    def __init__(self, model_factory: Callable[[], BaseSurvivalModel], 
                 cv_method: str = 'stratified', n_splits: int = 5, n_repeats: int = 1, 
                 random_state: int = None, metrics: List[str] = None):
        """
        Initialize survival cross-validation
        
        Parameters:
        -----
        model_factory: Callable[[], BaseSurvivalModel]
            Function to create model instances
        cv_method: str, default 'stratified'
            Cross-validation method, can be 'stratified', 'random'
        n_splits: int, default 5
            Number of folds
        n_repeats: int, default 1
            Number of repetitions
        random_state: int, optional
            Random seed
        metrics: List[str], optional
            Evaluation metrics to calculate
        """
        self.model_factory = model_factory
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.metrics = metrics or ['c_index']
        
        self.models = []
        self.cv_results = {}
        self.feature_importances = []
    
    def _get_cv_splitter(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str, event_col: str):
        """获取交叉验证分割器"""
        if self.cv_method == 'random':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_method == 'stratified':
            # 创建分层标签
            strata = stratify_by_time_and_event(y[time_col].values, y[event_col].values)
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_method == 'repeated_random':
            return RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        elif self.cv_method == 'repeated_stratified':
            # 创建分层标签
            strata = stratify_by_time_and_event(y[time_col].values, y[event_col].values)
            return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        else:
            raise ValueError(f"不支持的交叉验证方法: {self.cv_method}")
    
    def run(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', 
           metrics: List[Callable] = None, times: List[float] = None) -> Dict[str, Any]:
        """
        执行交叉验证
        
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
        metrics: List[Callable], 可选
            评估指标函数列表，每个函数应接受y_true, y_pred, time_col, event_col参数
        times: List[float], 可选
            评估时间点，用于时间依赖的评估指标
            
        返回:
        -----
        Dict[str, Any]
            交叉验证结果
        """
        logger.info(f"开始{self.cv_method}交叉验证，折数: {self.n_splits}，重复次数: {self.n_repeats}")
        
        # 如果未指定评估指标，使用C-index
        if metrics is None:
            metrics = [lambda y_true, y_pred, time_col, event_col: 
                      concordance_index(y_true[time_col], -y_pred, y_true[event_col])]
            metric_names = ['c_index']
        else:
            # 假设metrics是一个字典，键为指标名称，值为指标函数
            metric_names = list(metrics.keys())
            metrics = list(metrics.values())
        
        # 获取交叉验证分割器
        cv_splitter = self._get_cv_splitter(X, y, time_col, event_col)
        
        # 初始化结果
        cv_results = {
            'fold': [],
            'repeat': [],
            'train_indices': [],
            'test_indices': []
        }
        
        # 为每个指标初始化结果列表
        for metric_name in metric_names:
            cv_results[f'test_{metric_name}'] = []
        
        # 初始化模型列表
        self.models = []
        
        # 执行交叉验证
        fold_idx = 0
        for repeat_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, stratify_by_time_and_event(y[time_col].values, y[event_col].values) if 'stratified' in self.cv_method else None)):
            repeat = repeat_idx // self.n_splits if 'repeated' in self.cv_method else 0
            fold = repeat_idx % self.n_splits if 'repeated' in self.cv_method else repeat_idx
            
            logger.info(f"训练折 {fold+1}/{self.n_splits} {'重复 ' + str(repeat+1) + '/' + str(self.n_repeats) if 'repeated' in self.cv_method else ''}")
            
            # 分割数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 创建并训练模型
            model = self.model_factory()
            model.fit(X_train, y_train, time_col=time_col, event_col=event_col)
            
            # 保存模型
            self.models.append(model)
            
            # 预测
            y_pred = model.predict_risk(X_test)
            
            # 计算评估指标
            for i, metric_func in enumerate(metrics):
                metric_name = metric_names[i]
                score = metric_func(y_test, y_pred, time_col, event_col)
                cv_results[f'test_{metric_name}'].append(score)
            
            # 保存折信息
            cv_results['fold'].append(fold)
            cv_results['repeat'].append(repeat)
            cv_results['train_indices'].append(train_idx)
            cv_results['test_indices'].append(test_idx)
            
            fold_idx += 1
        
        # 转换为DataFrame
        self.results = pd.DataFrame(cv_results)
        
        # 计算平均分数
        summary = {}
        for metric_name in metric_names:
            scores = self.results[f'test_{metric_name}']
            summary[f'mean_test_{metric_name}'] = scores.mean()
            summary[f'std_test_{metric_name}'] = scores.std()
        
        logger.info(f"交叉验证完成，平均分数: {summary}")
        
        return {
            'results': self.results,
            'summary': summary,
            'models': self.models
        }
    
    def plot_scores(self, metric_name: str = 'c_index', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制交叉验证分数
        
        参数:
        -----
        metric_name: str, 默认 'c_index'
            要绘制的评估指标名称
        figsize: Tuple[int, int], 默认 (10, 6)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if self.results is None:
            raise ValueError("请先运行交叉验证")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取分数
        scores = self.results[f'test_{metric_name}']
        
        # 绘制箱线图
        sns.boxplot(y=scores, ax=ax)
        
        # 添加散点图
        sns.stripplot(y=scores, color='black', size=4, alpha=0.7, ax=ax)
        
        # 添加平均线
        ax.axhline(scores.mean(), color='red', linestyle='--', alpha=0.8, label=f'平均值: {scores.mean():.4f}')
        
        # 添加标题和标签
        ax.set_title(f'交叉验证 {metric_name} 分数分布', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel('', fontsize=12)
        
        # 添加图例
        ax.legend()
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        return fig
    
    def get_feature_importance(self, method: str = 'permutation', n_repeats: int = 10, 
                              random_state: int = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        参数:
        -----
        method: str, 默认 'permutation'
            特征重要性计算方法，可选 'permutation', 'model_specific'
        n_repeats: int, 默认 10
            置换重要性的重复次数
        random_state: int, 可选
            随机种子
            
        返回:
        -----
        pd.DataFrame
            特征重要性结果
        """
        if not self.models:
            raise ValueError("请先运行交叉验证")
        
        # 使用随机种子
        if random_state is None:
            random_state = self.random_state
        
        # 初始化特征重要性列表
        importance_list = []
        
        # 对每个模型计算特征重要性
        for i, model in enumerate(self.models):
            try:
                # 获取特征重要性
                importance = model.get_feature_importance(method=method, n_repeats=n_repeats, random_state=random_state)
                importance['fold'] = i
                importance_list.append(importance)
            except (NotImplementedError, AttributeError):
                logger.warning(f"模型 {i} 不支持特征重要性计算")
        
        # 如果没有特征重要性结果，返回空DataFrame
        if not importance_list:
            return pd.DataFrame()
        
        # 合并特征重要性结果
        importance_df = pd.concat(importance_list, ignore_index=True)
        
        # 计算平均特征重要性
        avg_importance = importance_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        avg_importance = avg_importance.sort_values('mean', ascending=False)
        
        return avg_importance 