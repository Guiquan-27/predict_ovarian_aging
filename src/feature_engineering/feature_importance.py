# -*- coding: utf-8 -*-
"""
Feature importance evaluation module
Provides methods for evaluating feature importance for various survival analysis models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

logger = logging.getLogger(__name__)

def calculate_feature_importance(model: Any, 
                                X: pd.DataFrame, 
                                y: pd.DataFrame,
                                time_col: str = 'time',
                                event_col: str = 'event',
                                method: str = 'permutation',
                                n_repeats: int = 10,
                                random_state: int = None) -> pd.DataFrame:
    """
    Calculate feature importance for survival models
    
    Parameters:
    -----
    model: Any
        Trained survival model
    X: pd.DataFrame
        Feature matrix
    y: pd.DataFrame
        Target dataframe with time and event columns
    time_col: str, default 'time'
        Time column name
    event_col: str, default 'event'
        Event column name
    method: str, default 'permutation'
        Method to calculate feature importance: 'permutation', 'model'
    n_repeats: int, default 10
        Number of times to permute each feature (for permutation importance)
    random_state: int, optional
        Random seed for reproducibility
        
    Returns:
    -----
    pd.DataFrame
        DataFrame with feature importance values
    """
    logger.info(f"Calculating feature importance using {method} method")
    
    if method == 'permutation':
        # Calculate base score (C-index) without permutation
        risk_scores = model.predict_risk(X)
        baseline_score = concordance_index(y[time_col], -risk_scores, y[event_col])
        
        # Initialize importance dataframe
        importance_values = []
        
        # For each feature, permute and calculate performance drop
        for feature in X.columns:
            # Create copied data for permutation
            X_permuted = X.copy()
            
            # Accumulate importance across repeats
            feature_importance = 0
            
            for i in range(n_repeats):
                # Permute the feature
                X_permuted[feature] = X_permuted[feature].sample(frac=1, random_state=random_state + i if random_state else None).values
                
                # Calculate new score
                permuted_risk_scores = model.predict_risk(X_permuted)
                permuted_score = concordance_index(y[time_col], -permuted_risk_scores, y[event_col])
                
                # Calculate importance (drop in performance)
                importance = baseline_score - permuted_score
                feature_importance += importance
            
            # Average importance over repeats
            feature_importance /= n_repeats
            
            importance_values.append({
                'feature': feature,
                'importance': feature_importance
            })
        
        # Create and sort importance dataframe
        importance_df = pd.DataFrame(importance_values)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    elif method == 'model':
        # Use model's built-in feature importance if available
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            # For models like Random Survival Forest
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        elif hasattr(model, 'model') and hasattr(model.model, 'coef_'):
            # For models like Cox regression
            coef = model.model.coef_
            if isinstance(coef, np.ndarray):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(coef)
                })
                return importance_df.sort_values('importance', ascending=False)
        
        # If built-in method not available, fall back to permutation importance
        logger.warning("Model does not provide built-in feature importance. Using permutation importance instead.")
        return calculate_feature_importance(model, X, y, time_col, event_col, 
                                           method='permutation', n_repeats=n_repeats, 
                                           random_state=random_state)
    
    else:
        raise ValueError(f"Unsupported feature importance method: {method}")

def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 20, 
                           figsize: Tuple[int, int] = (10, 8),
                           show_std: bool = True) -> plt.Figure:
    """
    绘制特征重要性条形图

    参数:
    -----
    importance_df: pd.DataFrame
        特征重要性结果
    top_n: int, 默认 20
        显示的特征数量
    figsize: Tuple[int, int], 默认 (10, 8)
        图形大小
    show_std: bool, 默认 True
        是否显示标准差误差条

    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
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
    bars = ax.barh(plot_df['feature'], plot_df['importance'], 
                  color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加误差条
    if show_std and 'importance_std' in plot_df.columns:
        ax.errorbar(plot_df['importance'], plot_df['feature'], 
                   xerr=plot_df['importance_std'], fmt='none', color='black', 
                   capsize=5, elinewidth=1.5)
    
    # 添加标题和标签
    ax.set_title('Feature Importance', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    
    # 添加网格线
    ax.grid(True, axis='x', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def compare_feature_importance(importance_dfs: Dict[str, pd.DataFrame],
                              top_n: int = 10,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    比较不同模型的特征重要性

    参数:
    -----
    importance_dfs: Dict[str, pd.DataFrame]
        不同模型的特征重要性结果字典
    top_n: int, 默认 10
        每个模型显示的特征数量
    figsize: Tuple[int, int], 默认 (12, 8)
        图形大小

    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, axes = plt.subplots(1, len(importance_dfs), figsize=figsize, sharey=False)
    
    # 如果只有一个模型，将axes转换为列表
    if len(importance_dfs) == 1:
        axes = [axes]
    
    # 为每个模型绘制特征重要性
    for i, (model_name, importance_df) in enumerate(importance_dfs.items()):
        # 选择前N个特征
        if len(importance_df) > top_n:
            plot_df = importance_df.head(top_n).copy()
        else:
            plot_df = importance_df.copy()
        
        # 反转顺序，使最重要的特征在顶部
        plot_df = plot_df.iloc[::-1]
        
        # 绘制条形图
        axes[i].barh(plot_df['feature'], plot_df['importance'], 
                    color=f'C{i}', edgecolor='black', alpha=0.7)
        
        # 添加标题和标签
        axes[i].set_title(f'{model_name}', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Feature', fontsize=10)
        axes[i].set_xlabel('Importance', fontsize=10)
        
        # 添加网格线
        axes[i].grid(True, axis='x', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def rank_aggregation(importance_dfs: Dict[str, pd.DataFrame],
                    top_n: int = None) -> pd.DataFrame:
    """
    聚合多个模型的特征重要性排名

    参数:
    -----
    importance_dfs: Dict[str, pd.DataFrame]
        不同模型的特征重要性结果字典
    top_n: int, 可选
        每个模型考虑的特征数量，默认为None（全部特征）

    返回:
    -----
    pd.DataFrame
        聚合后的特征排名
    """
    logger.info(f"聚合{len(importance_dfs)}个模型的特征重要性排名")
    
    # 提取每个模型的特征排名
    feature_ranks = {}
    all_features = set()
    
    for model_name, importance_df in importance_dfs.items():
        # 选择前N个特征
        if top_n is not None and len(importance_df) > top_n:
            model_features = importance_df.head(top_n)['feature'].tolist()
        else:
            model_features = importance_df['feature'].tolist()
        
        # 记录排名
        feature_ranks[model_name] = {feature: rank+1 for rank, feature in enumerate(model_features)}
        
        # 更新所有特征集合
        all_features.update(model_features)
    
    # 创建排名矩阵
    rank_matrix = []
    
    for feature in all_features:
        feature_row = {'feature': feature}
        
        # 添加每个模型的排名
        for model_name in importance_dfs.keys():
            feature_row[f'{model_name}_rank'] = feature_ranks[model_name].get(feature, np.nan)
        
        # 计算平均排名
        valid_ranks = [rank for rank in feature_row.values() if isinstance(rank, (int, float)) and not np.isnan(rank)]
        feature_row['avg_rank'] = np.mean(valid_ranks) if valid_ranks else np.nan
        
        rank_matrix.append(feature_row)
    
    # 创建结果DataFrame并按平均排名排序
    result_df = pd.DataFrame(rank_matrix).sort_values('avg_rank')
    
    logger.info(f"排名聚合完成，共{len(result_df)}个特征")
    return result_df 