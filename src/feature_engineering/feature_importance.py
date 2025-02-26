# -*- coding: utf-8 -*-
"""
特征重要性评估模块
提供用于评估特征重要性的方法，支持多种生存分析模型
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
                                random_state: int = 42) -> pd.DataFrame:
    """
    计算特征重要性

    参数:
    -----
    model: Any
        训练好的模型对象
    X: pd.DataFrame
        特征矩阵
    y: pd.DataFrame
        目标变量(时间和事件)
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    method: str, 默认 'permutation'
        重要性计算方法，可选 'permutation', 'model_specific'
    n_repeats: int, 默认 10
        置换重要性的重复次数
    random_state: int, 默认 42
        随机种子

    返回:
    -----
    pd.DataFrame
        特征重要性结果
    """
    logger.info(f"使用{method}方法计算特征重要性")
    
    # 检查模型类型
    model_type = type(model).__name__
    
    if method == 'permutation':
        # 对于置换重要性，需要准备适当的目标变量格式
        if model_type in ['CoxPHFitter', 'CoxnetSurvivalAnalysis']:
            # 对于Cox模型，使用DataFrame格式的y
            y_for_perm = y
        else:
            # 对于其他模型，可能需要转换为结构化数组
            try:
                from sksurv.util import Surv
                structured_y = Surv.from_dataframe(event_col, time_col, y)
                y_for_perm = structured_y
            except Exception as e:
                logger.error(f"转换目标变量格式失败: {str(e)}")
                y_for_perm = y
        
        # 定义评分函数
        def score_func(model, X, y):
            if model_type == 'CoxPHFitter':
                # 对于lifelines的Cox模型，使用concordance_index
                pred = model.predict_partial_hazard(X)
                return model.score(y)
            else:
                # 对于scikit-survival模型，使用内置评分
                return model.score(X, y)
        
        # 计算置换重要性
        try:
            perm_importance = permutation_importance(
                model, X, y_for_perm,
                scoring=score_func,
                n_repeats=n_repeats,
                random_state=random_state
            )
            
            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance_mean', ascending=False)
            
        except Exception as e:
            logger.error(f"计算置换重要性失败: {str(e)}")
            logger.warning("切换到模型特定的重要性计算方法")
            method = 'model_specific'
    
    if method == 'model_specific':
        # 根据模型类型提取特征重要性
        if model_type == 'CoxPHFitter':
            # 对于lifelines的Cox模型，使用系数绝对值
            importance = np.abs(model.params_)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance,
                'importance_std': np.zeros_like(importance)
            })
            
        elif model_type == 'RandomSurvivalForest':
            # 对于随机生存森林，使用内置特征重要性
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance,
                'importance_std': np.zeros_like(importance)
            })
            
        elif hasattr(model, 'feature_importances_'):
            # 对于其他具有feature_importances_属性的模型
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance,
                'importance_std': np.zeros_like(importance)
            })
            
        elif hasattr(model, 'coef_'):
            # 对于线性模型
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance[0]
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': importance,
                'importance_std': np.zeros_like(importance)
            })
            
        else:
            logger.error(f"无法提取模型 {model_type} 的特征重要性")
            return pd.DataFrame(columns=['feature', 'importance_mean', 'importance_std'])
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    logger.info(f"特征重要性计算完成，共{len(importance_df)}个特征")
    return importance_df

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
    bars = ax.barh(plot_df['feature'], plot_df['importance_mean'], 
                  color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加误差条
    if show_std and 'importance_std' in plot_df.columns:
        ax.errorbar(plot_df['importance_mean'], plot_df['feature'], 
                   xerr=plot_df['importance_std'], fmt='none', color='black', 
                   capsize=5, elinewidth=1.5)
    
    # 添加标题和标签
    ax.set_title('特征重要性', fontsize=14)
    ax.set_xlabel('重要性', fontsize=12)
    
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
        axes[i].barh(plot_df['feature'], plot_df['importance_mean'], 
                    color=f'C{i}', edgecolor='black', alpha=0.7)
        
        # 添加标题和标签
        axes[i].set_title(f'{model_name}', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('特征', fontsize=10)
        axes[i].set_xlabel('重要性', fontsize=10)
        
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