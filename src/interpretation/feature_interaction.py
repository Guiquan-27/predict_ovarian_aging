# -*- coding: utf-8 -*-
"""
特征交互分析模块
提供用于分析生存分析模型中特征交互关系的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
import networkx as nx
import warnings

# 尝试导入shap库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("未找到shap库，SHAP交互分析功能将不可用。请使用pip install shap安装。")

from ..models.base_models import BaseSurvivalModel

logger = logging.getLogger(__name__)

def check_shap_available():
    """检查SHAP库是否可用"""
    if not SHAP_AVAILABLE:
        raise ImportError("未找到shap库，请使用pip install shap安装。")

def calculate_partial_dependence(model: BaseSurvivalModel, X: pd.DataFrame, 
                               features: List[str], 
                               time_point: Optional[float] = None,
                               grid_resolution: int = 20,
                               percentiles: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
    """
    计算部分依赖
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    features: List[str]
        要分析的特征列表
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    grid_resolution: int, 默认 20
        网格分辨率
    percentiles: Tuple[float, float], 默认 (0.05, 0.95)
        特征值范围的百分位数
        
    返回:
    -----
    Dict[str, Any]
        部分依赖结果
    """
    # 检查特征是否在数据集中
    for feature in features:
        if feature not in X.columns:
            raise ValueError(f"特征 '{feature}' 不在数据集中")
    
    # 获取特征索引
    feature_indices = [list(X.columns).index(feature) for feature in features]
    
    # 定义预测函数
    if time_point is not None:
        # 使用特定时间点的风险概率
        def predict_fn(X_array):
            X_df = pd.DataFrame(X_array, columns=X.columns)
            return 1 - model.predict(X_df, times=[time_point])[0]
    else:
        # 使用风险得分
        def predict_fn(X_array):
            X_df = pd.DataFrame(X_array, columns=X.columns)
            return model.predict_risk(X_df)
    
    # 计算部分依赖
    pdp_results = {}
    
    for feature in features:
        feature_idx = list(X.columns).index(feature)
        
        # 计算特征的百分位数范围
        feature_values = X[feature].values
        lower = np.percentile(feature_values, percentiles[0] * 100)
        upper = np.percentile(feature_values, percentiles[1] * 100)
        
        # 创建网格
        if np.issubdtype(X[feature].dtype, np.number):
            # 连续特征
            grid = np.linspace(lower, upper, grid_resolution)
        else:
            # 分类特征
            grid = np.unique(feature_values)
        
        # 计算部分依赖
        pdp_values = []
        for value in grid:
            X_modified = X.copy()
            X_modified[feature] = value
            pdp_values.append(np.mean(predict_fn(X_modified.values)))
        
        pdp_results[feature] = {
            'grid': grid,
            'pdp': np.array(pdp_values)
        }
    
    return pdp_results

def plot_partial_dependence(pdp_results: Dict[str, Any], 
                          features: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          n_cols: int = 2) -> plt.Figure:
    """
    绘制部分依赖图
    
    参数:
    -----
    pdp_results: Dict[str, Any]
        部分依赖结果
    features: List[str], 可选
        要绘制的特征列表，默认为None(绘制所有特征)
    figsize: Tuple[int, int], 默认 (12, 8)
        图形大小
    n_cols: int, 默认 2
        列数
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 如果未指定特征，使用所有特征
    if features is None:
        features = list(pdp_results.keys())
    
    # 计算行数
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 确保axes是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # 绘制每个特征的部分依赖图
    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # 获取部分依赖结果
        grid = pdp_results[feature]['grid']
        pdp = pdp_results[feature]['pdp']
        
        # 绘制部分依赖曲线
        ax.plot(grid, pdp, 'b-', linewidth=2)
        
        # 添加标题和标签
        ax.set_title(f'特征 "{feature}" 的部分依赖', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('预测影响', fontsize=10)
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def calculate_ice_curves(model: BaseSurvivalModel, X: pd.DataFrame, 
                       feature: str, 
                       time_point: Optional[float] = None,
                       n_samples: int = 50,
                       grid_resolution: int = 20,
                       percentiles: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
    """
    计算个体条件期望(ICE)曲线
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    feature: str
        要分析的特征
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    n_samples: int, 默认 50
        用于计算ICE曲线的样本数量
    grid_resolution: int, 默认 20
        网格分辨率
    percentiles: Tuple[float, float], 默认 (0.05, 0.95)
        特征值范围的百分位数
        
    返回:
    -----
    Dict[str, Any]
        ICE曲线结果
    """
    # 检查特征是否在数据集中
    if feature not in X.columns:
        raise ValueError(f"特征 '{feature}' 不在数据集中")
    
    # 随机选择样本
    if n_samples < len(X):
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 计算特征的百分位数范围
    feature_values = X[feature].values
    lower = np.percentile(feature_values, percentiles[0] * 100)
    upper = np.percentile(feature_values, percentiles[1] * 100)
    
    # 创建网格
    if np.issubdtype(X[feature].dtype, np.number):
        # 连续特征
        grid = np.linspace(lower, upper, grid_resolution)
    else:
        # 分类特征
        grid = np.unique(feature_values)
    
    # 定义预测函数
    if time_point is not None:
        # 使用特定时间点的风险概率
        def predict_fn(X_df):
            return 1 - model.predict(X_df, times=[time_point])[0]
    else:
        # 使用风险得分
        def predict_fn(X_df):
            return model.predict_risk(X_df)
    
    # 计算ICE曲线
    ice_curves = []
    
    for _, row in X_sample.iterrows():
        # 为每个网格点创建一个样本副本
        ice_values = []
        for value in grid:
            sample = row.copy()
            sample[feature] = value
            sample_df = pd.DataFrame([sample])
            ice_values.append(predict_fn(sample_df)[0])
        
        ice_curves.append(ice_values)
    
    # 计算部分依赖(平均ICE曲线)
    pdp = np.mean(ice_curves, axis=0)
    
    return {
        'grid': grid,
        'ice_curves': np.array(ice_curves),
        'pdp': pdp
    }

def plot_ice_curves(ice_results: Dict[str, Any], 
                   feature: str,
                   figsize: Tuple[int, int] = (10, 6),
                   alpha: float = 0.1,
                   center: bool = True) -> plt.Figure:
    """
    绘制个体条件期望(ICE)曲线
    
    参数:
    -----
    ice_results: Dict[str, Any]
        ICE曲线结果
    feature: str
        特征名称
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
    alpha: float, 默认 0.1
        ICE曲线的透明度
    center: bool, 默认 True
        是否居中ICE曲线
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 获取ICE曲线结果
    grid = ice_results['grid']
    ice_curves = ice_results['ice_curves']
    pdp = ice_results['pdp']
    
    # 居中ICE曲线
    if center:
        ice_curves = ice_curves - ice_curves[:, 0].reshape(-1, 1)
        pdp = pdp - pdp[0]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制ICE曲线
    for curve in ice_curves:
        ax.plot(grid, curve, 'b-', alpha=alpha)
    
    # 绘制部分依赖曲线
    ax.plot(grid, pdp, 'r-', linewidth=2, label='平均(PDP)')
    
    # 添加标题和标签
    ax.set_title(f'特征 "{feature}" 的ICE曲线', fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('预测影响', fontsize=12)
    
    # 添加图例
    ax.legend()
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def calculate_feature_interactions(model: BaseSurvivalModel, X: pd.DataFrame, 
                                 features: Optional[List[str]] = None,
                                 n_samples: Optional[int] = None,
                                 time_point: Optional[float] = None,
                                 interaction_threshold: float = 0.1) -> pd.DataFrame:
    """
    计算特征交互强度
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    features: List[str], 可选
        要分析的特征列表，默认为None(使用所有特征)
    n_samples: int, 可选
        用于计算的样本数量，默认为None(使用全部样本)
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    interaction_threshold: float, 默认 0.1
        交互强度阈值，低于此值的交互将被忽略
        
    返回:
    -----
    pd.DataFrame
        特征交互矩阵
    """
    check_shap_available()
    
    # 如果未指定特征，使用所有特征
    if features is None:
        features = list(X.columns)
    
    # 如果指定了样本数量，随机选择样本
    if n_samples is not None and n_samples < len(X):
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 计算SHAP值
    from .shap_analysis import calculate_shap_values
    shap_values, explainer = calculate_shap_values(model, X_sample, time_point=time_point)
    
    # 初始化交互矩阵
    n_features = len(features)
    interaction_matrix = np.zeros((n_features, n_features))
    
    # 计算特征交互
    for i, feature1 in enumerate(features):
        if feature1 not in X.columns:
            continue
            
        for j, feature2 in enumerate(features):
            if j <= i or feature2 not in X.columns:
                continue
                
            # 计算SHAP交互值
            try:
                interaction_values = explainer.shap_interaction_values(X_sample)
                
                # 获取特征索引
                idx1 = list(X_sample.columns).index(feature1)
                idx2 = list(X_sample.columns).index(feature2)
                
                # 计算交互强度
                interaction = np.abs(interaction_values[:, idx1, idx2]).mean()
                
                # 存储交互强度
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
            except Exception as e:
                logger.warning(f"计算特征 '{feature1}' 和 '{feature2}' 的交互时出错: {str(e)}")
    
    # 创建交互DataFrame
    interaction_df = pd.DataFrame(interaction_matrix, index=features, columns=features)
    
    # 应用阈值
    interaction_df[interaction_df < interaction_threshold] = 0
    
    return interaction_df

def plot_interaction_heatmap(interaction_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 10),
                           cmap: str = 'viridis',
                           annot: bool = True) -> plt.Figure:
    """
    绘制特征交互热图
    
    参数:
    -----
    interaction_df: pd.DataFrame
        特征交互矩阵
    figsize: Tuple[int, int], 默认 (12, 10)
        图形大小
    cmap: str, 默认 'viridis'
        颜色映射
    annot: bool, 默认 True
        是否显示数值标注
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    mask = np.triu(np.ones_like(interaction_df, dtype=bool), k=0)
    sns.heatmap(interaction_df, mask=mask, cmap=cmap, annot=annot, 
               fmt='.2f', linewidths=0.5, ax=ax, square=True)
    
    # 添加标题
    ax.set_title('特征交互热图', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_interaction_network(interaction_df: pd.DataFrame,
                           threshold: float = 0.1,
                           figsize: Tuple[int, int] = (12, 10),
                           node_size_factor: float = 1000,
                           edge_width_factor: float = 5) -> plt.Figure:
    """
    绘制特征交互网络图
    
    参数:
    -----
    interaction_df: pd.DataFrame
        特征交互矩阵
    threshold: float, 默认 0.1
        交互强度阈值，低于此值的交互将被忽略
    figsize: Tuple[int, int], 默认 (12, 10)
        图形大小
    node_size_factor: float, 默认 1000
        节点大小因子
    edge_width_factor: float, 默认 5
        边宽度因子
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for feature in interaction_df.columns:
        # 计算节点大小(基于自交互强度)
        node_size = interaction_df.loc[feature, feature] * node_size_factor if feature in interaction_df.index else 1
        G.add_node(feature, size=node_size)
    
    # 添加边
    for i, feature1 in enumerate(interaction_df.columns):
        for j, feature2 in enumerate(interaction_df.columns):
            if j <= i:
                continue
                
            # 获取交互强度
            interaction = interaction_df.loc[feature1, feature2]
            
            # 如果交互强度超过阈值，添加边
            if interaction > threshold:
                G.add_edge(feature1, feature2, weight=interaction)
    
    # 计算节点位置
    pos = nx.spring_layout(G, seed=42)
    
    # 获取节点大小
    node_sizes = [G.nodes[node].get('size', 300) for node in G.nodes]
    
    # 获取边宽度
    edge_widths = [G.edges[edge]['weight'] * edge_width_factor for edge in G.edges]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    
    # 添加标题
    ax.set_title('特征交互网络图', fontsize=14)
    
    # 关闭坐标轴
    ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def analyze_two_way_interactions(model: BaseSurvivalModel, X: pd.DataFrame,
                               feature1: str, feature2: str,
                               time_point: Optional[float] = None,
                               grid_resolution: int = 20,
                               percentiles: Tuple[float, float] = (0.05, 0.95),
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    分析两个特征之间的交互
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    feature1: str
        第一个特征
    feature2: str
        第二个特征
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    grid_resolution: int, 默认 20
        网格分辨率
    percentiles: Tuple[float, float], 默认 (0.05, 0.95)
        特征值范围的百分位数
    figsize: Tuple[int, int], 默认 (10, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 检查特征是否在数据集中
    if feature1 not in X.columns:
        raise ValueError(f"特征 '{feature1}' 不在数据集中")
    if feature2 not in X.columns:
        raise ValueError(f"特征 '{feature2}' 不在数据集中")
    
    # 计算特征的百分位数范围
    feature1_values = X[feature1].values
    feature2_values = X[feature2].values
    
    # 创建网格
    if np.issubdtype(X[feature1].dtype, np.number):
        # 连续特征
        lower1 = np.percentile(feature1_values, percentiles[0] * 100)
        upper1 = np.percentile(feature1_values, percentiles[1] * 100)
        grid1 = np.linspace(lower1, upper1, grid_resolution)
    else:
        # 分类特征
        grid1 = np.unique(feature1_values)
    
    if np.issubdtype(X[feature2].dtype, np.number):
        # 连续特征
        lower2 = np.percentile(feature2_values, percentiles[0] * 100)
        upper2 = np.percentile(feature2_values, percentiles[1] * 100)
        grid2 = np.linspace(lower2, upper2, grid_resolution)
    else:
        # 分类特征
        grid2 = np.unique(feature2_values)
    
    # 定义预测函数
    if time_point is not None:
        # 使用特定时间点的风险概率
        def predict_fn(X_df):
            return 1 - model.predict(X_df, times=[time_point])[0]
    else:
        # 使用风险得分
        def predict_fn(X_df):
            return model.predict_risk(X_df)
    
    # 计算交互效应
    interaction_matrix = np.zeros((len(grid1), len(grid2)))
    
    for i, val1 in enumerate(grid1):
        for j, val2 in enumerate(grid2):
            X_modified = X.copy()
            X_modified[feature1] = val1
            X_modified[feature2] = val2
            interaction_matrix[i, j] = np.mean(predict_fn(X_modified))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    im = ax.imshow(interaction_matrix, cmap='viridis', aspect='auto', 
                  extent=[min(grid2), max(grid2), max(grid1), min(grid1)])
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('预测值', fontsize=12)
    
    # 添加标题和标签
    ax.set_title(f'特征 "{feature1}" 和 "{feature2}" 的交互效应', fontsize=14)
    ax.set_xlabel(feature2, fontsize=12)
    ax.set_ylabel(feature1, fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def generate_interaction_report(model: BaseSurvivalModel, X: pd.DataFrame,
                              top_features: Optional[List[str]] = None,
                              n_top_features: int = 10,
                              time_point: Optional[float] = None,
                              n_samples: int = 100,
                              figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    生成特征交互综合报告
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    top_features: List[str], 可选
        要分析的顶级特征列表，默认为None(自动选择)
    n_top_features: int, 默认 10
        如果top_features为None，选择的顶级特征数量
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    n_samples: int, 默认 100
        用于计算的样本数量
    figsize: Tuple[int, int], 默认 (15, 12)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 如果未指定顶级特征，自动选择
    if top_features is None:
        # 计算SHAP值
        from .shap_analysis import calculate_shap_values, get_shap_feature_importance
        shap_values, _ = calculate_shap_values(model, X, n_samples=n_samples, time_point=time_point)
        
        # 获取特征重要性
        importance = get_shap_feature_importance(shap_values, X.columns)
        
        # 选择顶级特征
        top_features = importance.head(n_top_features)['feature'].tolist()
    
    # 计算特征交互
    interaction_df = calculate_feature_interactions(
        model, X, features=top_features, n_samples=n_samples, time_point=time_point
    )
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 添加标题
    fig.suptitle('特征交互分析报告', fontsize=16)
    
    # 创建网格
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 绘制交互热图
    ax1 = fig.add_subplot(gs[0, 0])
    mask = np.triu(np.ones_like(interaction_df, dtype=bool), k=0)
    sns.heatmap(interaction_df, mask=mask, cmap='viridis', annot=True, 
               fmt='.2f', linewidths=0.5, ax=ax1, square=True)
    ax1.set_title('特征交互热图', fontsize=12)
    
    # 绘制交互网络图
    ax2 = fig.add_subplot(gs[0, 1])
    G = nx.Graph()
    
    # 添加节点
    for feature in interaction_df.columns:
        G.add_node(feature)
    
    # 添加边
    for i, feature1 in enumerate(interaction_df.columns):
        for j, feature2 in enumerate(interaction_df.columns):
            if j <= i:
                continue
                
            # 获取交互强度
            interaction = interaction_df.loc[feature1, feature2]
            
            # 如果交互强度超过阈值，添加边
            if interaction > 0.05:
                G.add_edge(feature1, feature2, weight=interaction)
    
    # 计算节点位置
    pos = nx.spring_layout(G, seed=42)
    
    # 获取边宽度
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    
    # 绘制网络图
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax2)
    
    # 添加标题
    ax2.set_title('特征交互网络图', fontsize=12)
    ax2.axis('off')
    
    # 绘制部分依赖图
    ax3 = fig.add_subplot(gs[1, :])
    
    # 选择前3个最重要的特征
    top3_features = top_features[:3]
    
    # 计算部分依赖
    pdp_results = calculate_partial_dependence(
        model, X, features=top3_features, time_point=time_point
    )
    
    # 绘制部分依赖图
    for i, feature in enumerate(top3_features):
        # 获取部分依赖结果
        grid = pdp_results[feature]['grid']
        pdp = pdp_results[feature]['pdp']
        
        # 绘制部分依赖曲线
        ax3.plot(grid, pdp, label=feature, linewidth=2)
    
    # 添加标题和标签
    ax3.set_title('主要特征的部分依赖图', fontsize=12)
    ax3.set_xlabel('特征值', fontsize=10)
    ax3.set_ylabel('预测值', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def calculate_shap_interaction_values(model: BaseSurvivalModel, X: pd.DataFrame, 
                                    n_samples: int = 100, 
                                    time_point: Optional[float] = None) -> np.ndarray:
    """
    计算SHAP交互值
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    n_samples: int, 默认 100
        用于计算的样本数量
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
        
    返回:
    -----
    np.ndarray
        SHAP交互值
    """
    check_shap_available()
    
    # 随机选择样本
    if n_samples < len(X):
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 根据模型类型选择合适的SHAP解释器
    model_type = model.__class__.__name__
    
    try:
        if model_type in ['RandomSurvivalForestModel', 'GradientBoostingSurvivalModel', 'XGBoostSurvivalModel']:
            # 树模型使用TreeExplainer
            explainer = shap.TreeExplainer(model.model)
            
            # 计算SHAP交互值
            shap_interaction_values = explainer.shap_interaction_values(X_sample)
            
            # 如果指定了时间点，选择特定时间点的SHAP交互值
            if time_point is not None and isinstance(shap_interaction_values, list):
                if hasattr(model, 'event_times_'):
                    idx = np.searchsorted(model.event_times_, time_point)
                    if idx == len(model.event_times_):
                        idx = len(model.event_times_) - 1
                    
                    shap_interaction_values = shap_interaction_values[idx]
        else:
            # 其他模型类型不支持SHAP交互值
            raise NotImplementedError(f"模型类型 '{model_type}' 不支持SHAP交互值计算")
        
        return shap_interaction_values
    
    except Exception as e:
        logger.error(f"计算SHAP交互值时出错: {e}")
        raise

def plot_shap_interaction(shap_interaction_values: np.ndarray, X: pd.DataFrame, 
                         feature1: str, feature2: str, 
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制SHAP交互图
    
    参数:
    -----
    shap_interaction_values: np.ndarray
        SHAP交互值
    X: pd.DataFrame
        特征矩阵
    feature1: str
        第一个特征名称
    feature2: str
        第二个特征名称
    figsize: Tuple[int, int], 默认 (10, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 检查特征是否在数据集中
    if feature1 not in X.columns:
        raise ValueError(f"特征 '{feature1}' 不在数据集中")
    if feature2 not in X.columns:
        raise ValueError(f"特征 '{feature2}' 不在数据集中")
    
    # 获取特征索引
    feature1_idx = list(X.columns).index(feature1)
    feature2_idx = list(X.columns).index(feature2)
    
    # 提取交互值
    interaction_values = shap_interaction_values[:, feature1_idx, feature2_idx]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制散点图
    scatter = ax.scatter(X[feature1], X[feature2], c=interaction_values, 
                        cmap='coolwarm', s=50, alpha=0.8, edgecolors='k', linewidths=0.5)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('SHAP交互值', fontsize=12)
    
    # 添加标题和标签
    ax.set_title(f'特征 "{feature1}" 和 "{feature2}" 的SHAP交互图', fontsize=14)
    ax.set_xlabel(feature1, fontsize=12)
    ax.set_ylabel(feature2, fontsize=12)
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def calculate_feature_interactions(model: BaseSurvivalModel, X: pd.DataFrame, 
                                 features: Optional[List[str]] = None, 
                                 n_samples: int = 100, 
                                 time_point: Optional[float] = None) -> pd.DataFrame:
    """
    计算特征交互强度
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    features: List[str], 可选
        要分析的特征列表，默认为None(使用所有特征)
    n_samples: int, 默认 100
        用于计算的样本数量
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
        
    返回:
    -----
    pd.DataFrame
        特征交互强度矩阵
    """
    # 如果未指定特征，使用所有特征
    if features is None:
        features = X.columns.tolist()
    
    # 随机选择样本
    if n_samples < len(X):
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 初始化交互矩阵
    n_features = len(features)
    interaction_matrix = np.zeros((n_features, n_features))
    
    # 尝试使用SHAP交互值
    try:
        if SHAP_AVAILABLE:
            # 计算SHAP交互值
            shap_interaction_values = calculate_shap_interaction_values(
                model, X_sample, time_point=time_point
            )
            
            # 获取特征索引
            feature_indices = [list(X.columns).index(feature) for feature in features]
            
            # 计算交互强度
            for i, feature1_idx in enumerate(feature_indices):
                for j, feature2_idx in enumerate(feature_indices):
                    if i == j:
                        continue
                    
                    # 提取交互值
                    interaction_values = shap_interaction_values[:, feature1_idx, feature2_idx]
                    
                    # 计算交互强度
                    interaction_matrix[i, j] = np.mean(np.abs(interaction_values))
    except:
        # 如果SHAP交互值计算失败，使用替代方法
        logger.warning("SHAP交互值计算失败，使用替代方法")
        
        # 定义预测函数
        if time_point is not None:
            # 使用特定时间点的风险概率
            def predict_fn(X_df):
                return 1 - model.predict(X_df, times=[time_point])[0]
        else:
            # 使用风险得分
            def predict_fn(X_df):
                return model.predict_risk(X_df)
        
        # 计算交互强度
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i == j:
                    continue
                
                # 计算特征1的部分依赖
                pdp1 = np.zeros(len(X_sample))
                for k, row in X_sample.iterrows():
                    X_modified = X_sample.copy()
                    X_modified[feature1] = row[feature1]
                    pdp1[k] = np.mean(predict_fn(X_modified))
                
                # 计算特征2的部分依赖
                pdp2 = np.zeros(len(X_sample))
                for k, row in X_sample.iterrows():
                    X_modified = X_sample.copy()
                    X_modified[feature2] = row[feature2]
                    pdp2[k] = np.mean(predict_fn(X_modified))
                
                # 计算联合部分依赖
                pdp_joint = np.zeros(len(X_sample))
                for k, row in X_sample.iterrows():
                    X_modified = X_sample.copy()
                    X_modified[feature1] = row[feature1]
                    X_modified[feature2] = row[feature2]
                    pdp_joint[k] = np.mean(predict_fn(X_modified))
                
                # 计算交互强度
                interaction = np.mean(np.abs(pdp_joint - pdp1 - pdp2 + np.mean(predict_fn(X_sample))))
                interaction_matrix[i, j] = interaction
    
    # 创建交互DataFrame
    interaction_df = pd.DataFrame(interaction_matrix, index=features, columns=features)
    
    return interaction_df 