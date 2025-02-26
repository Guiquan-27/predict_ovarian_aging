# -*- coding: utf-8 -*-
"""
SHAP值分析模块
提供用于解释生存分析模型预测的SHAP值分析功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 尝试导入shap库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("未找到shap库，SHAP值分析功能将不可用。请使用pip install shap安装。")

from ..models.base_models import BaseSurvivalModel

logger = logging.getLogger(__name__)

def check_shap_available():
    """检查SHAP库是否可用"""
    if not SHAP_AVAILABLE:
        raise ImportError("未找到shap库，请使用pip install shap安装。")

def calculate_shap_values(model: BaseSurvivalModel, X: pd.DataFrame, 
                         n_samples: Optional[int] = None, 
                         time_point: Optional[float] = None) -> Tuple[np.ndarray, Any]:
    """
    计算SHAP值
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    n_samples: int, 可选
        用于计算SHAP值的样本数量，默认为None(使用全部样本)
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
        
    返回:
    -----
    Tuple[np.ndarray, Any]
        (SHAP值, SHAP解释器)
    """
    check_shap_available()
    
    # 如果指定了样本数量，随机选择样本
    if n_samples is not None and n_samples < len(X):
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 根据模型类型选择合适的SHAP解释器
    model_type = model.__class__.__name__
    
    try:
        if model_type in ['RandomSurvivalForestModel', 'GradientBoostingSurvivalModel', 'XGBoostSurvivalModel']:
            # 树模型使用TreeExplainer
            explainer = shap.TreeExplainer(model.model)
            
            # 如果指定了时间点，计算特定时间点的SHAP值
            if time_point is not None:
                # 获取最接近的时间点索引
                if hasattr(model, 'event_times_'):
                    idx = np.searchsorted(model.event_times_, time_point)
                    if idx == len(model.event_times_):
                        idx = len(model.event_times_) - 1
                    
                    # 计算特定时间点的SHAP值
                    shap_values = explainer.shap_values(X_sample, approximate=True)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[idx]
                else:
                    # 如果模型没有event_times_属性，使用风险得分的SHAP值
                    shap_values = explainer.shap_values(X_sample, approximate=True)
            else:
                # 计算风险得分的SHAP值
                shap_values = explainer.shap_values(X_sample, approximate=True)
        
        elif model_type in ['CoxPHModel', 'CoxnetModel']:
            # 线性模型使用LinearExplainer
            explainer = shap.LinearExplainer(model.model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        elif model_type in ['DeepSurvModel', 'MTLCoxModel']:
            # 深度学习模型使用DeepExplainer或GradientExplainer
            try:
                # 尝试使用DeepExplainer
                explainer = shap.DeepExplainer(model.model, X_sample.values)
                shap_values = explainer.shap_values(X_sample.values)
            except:
                # 如果失败，使用GradientExplainer
                explainer = shap.GradientExplainer(model.model, X_sample.values)
                shap_values = explainer.shap_values(X_sample.values)
        
        else:
            # 其他模型使用KernelExplainer
            # 定义预测函数
            if time_point is not None:
                def predict_func(X_):
                    X_df = pd.DataFrame(X_, columns=X_sample.columns)
                    return model.predict(X_df, times=[time_point])[:, 0]
            else:
                def predict_func(X_):
                    X_df = pd.DataFrame(X_, columns=X_sample.columns)
                    return model.predict_risk(X_df)
            
            explainer = shap.KernelExplainer(predict_func, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        return shap_values, explainer
    
    except Exception as e:
        logger.error(f"计算SHAP值时出错: {e}")
        raise

def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, 
                     max_display: int = 20, plot_type: str = "bar", 
                     title: str = "SHAP值全局特征重要性", 
                     figsize: Tuple[int, int] = (10, 12)) -> plt.Figure:
    """
    绘制SHAP值摘要图
    
    参数:
    -----
    shap_values: np.ndarray
        SHAP值
    X: pd.DataFrame
        特征矩阵
    max_display: int, 默认 20
        显示的最大特征数量
    plot_type: str, 默认 "bar"
        图表类型，可选 "bar", "dot", "violin", "compact_dot"
    title: str, 默认 "SHAP值全局特征重要性"
        图表标题
    figsize: Tuple[int, int], 默认 (10, 12)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 根据图表类型绘制SHAP值摘要图
    if plot_type == "bar":
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    elif plot_type == "dot":
        shap.summary_plot(shap_values, X, plot_type="dot", max_display=max_display, show=False)
    elif plot_type == "violin":
        shap.summary_plot(shap_values, X, plot_type="violin", max_display=max_display, show=False)
    elif plot_type == "compact_dot":
        shap.summary_plot(shap_values, X, plot_type="compact_dot", max_display=max_display, show=False)
    else:
        raise ValueError(f"不支持的图表类型: {plot_type}")
    
    # 添加标题
    plt.title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame, 
                        feature: str, interaction_feature: Optional[str] = None,
                        title: Optional[str] = None, 
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制SHAP依赖图
    
    参数:
    -----
    shap_values: np.ndarray
        SHAP值
    X: pd.DataFrame
        特征矩阵
    feature: str
        要分析的特征名称
    interaction_feature: str, 可选
        交互特征名称，默认为None
    title: str, 可选
        图表标题，默认为None
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 绘制SHAP依赖图
    if interaction_feature is not None:
        shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_feature, show=False)
        if title is None:
            title = f"特征 '{feature}' 与 '{interaction_feature}' 的SHAP依赖关系"
    else:
        shap.dependence_plot(feature, shap_values, X, show=False)
        if title is None:
            title = f"特征 '{feature}' 的SHAP依赖关系"
    
    # 添加标题
    plt.title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_shap_waterfall(shap_values: np.ndarray, X: pd.DataFrame, 
                       sample_idx: int = 0, max_display: int = 20,
                       title: Optional[str] = None, 
                       figsize: Tuple[int, int] = (10, 12)) -> plt.Figure:
    """
    绘制SHAP瀑布图(单个样本的特征贡献)
    
    参数:
    -----
    shap_values: np.ndarray
        SHAP值
    X: pd.DataFrame
        特征矩阵
    sample_idx: int, 默认 0
        样本索引
    max_display: int, 默认 20
        显示的最大特征数量
    title: str, 可选
        图表标题，默认为None
    figsize: Tuple[int, int], 默认 (10, 12)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 获取单个样本的SHAP值
    if len(shap_values.shape) > 1:
        sample_shap_values = shap_values[sample_idx]
        sample_features = X.iloc[sample_idx]
    else:
        sample_shap_values = shap_values
        sample_features = X
    
    # 绘制SHAP瀑布图
    shap.plots.waterfall(sample_shap_values, max_display=max_display, show=False)
    
    # 添加标题
    if title is None:
        title = f"样本 {sample_idx} 的SHAP值贡献"
    plt.title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_shap_force(shap_values: np.ndarray, X: pd.DataFrame, 
                   sample_idx: int = 0, 
                   title: Optional[str] = None, 
                   figsize: Tuple[int, int] = (20, 3)) -> plt.Figure:
    """
    绘制SHAP力图(单个样本的特征贡献)
    
    参数:
    -----
    shap_values: np.ndarray
        SHAP值
    X: pd.DataFrame
        特征矩阵
    sample_idx: int, 默认 0
        样本索引
    title: str, 可选
        图表标题，默认为None
    figsize: Tuple[int, int], 默认 (20, 3)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 获取单个样本的SHAP值
    if len(shap_values.shape) > 1:
        sample_shap_values = shap_values[sample_idx]
        sample_features = X.iloc[sample_idx]
    else:
        sample_shap_values = shap_values
        sample_features = X
    
    # 绘制SHAP力图
    shap.force_plot(np.sum(sample_shap_values), sample_shap_values, sample_features, 
                   matplotlib=True, show=False)
    
    # 添加标题
    if title is None:
        title = f"样本 {sample_idx} 的SHAP力图"
    plt.title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_shap_decision(shap_values: np.ndarray, X: pd.DataFrame, 
                      feature_names: Optional[List[str]] = None,
                      title: str = "SHAP决策图", 
                      figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    绘制SHAP决策图
    
    参数:
    -----
    shap_values: np.ndarray
        SHAP值
    X: pd.DataFrame
        特征矩阵
    feature_names: List[str], 可选
        特征名称列表，默认为None(使用X的列名)
    title: str, 默认 "SHAP决策图"
        图表标题
    figsize: Tuple[int, int], 默认 (10, 10)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 如果未指定特征名称，使用X的列名
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # 绘制SHAP决策图
    shap.decision_plot(np.sum(shap_values), shap_values, feature_names=feature_names, show=False)
    
    # 添加标题
    plt.title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def generate_shap_report(model: BaseSurvivalModel, X: pd.DataFrame, 
                        n_samples: int = 100, 
                        time_point: Optional[float] = None,
                        max_display: int = 15,
                        top_features: int = 5,
                        sample_indices: Optional[List[int]] = None,
                        figsize: Tuple[int, int] = (20, 25)) -> plt.Figure:
    """
    生成综合SHAP分析报告
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    n_samples: int, 默认 100
        用于计算SHAP值的样本数量
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    max_display: int, 默认 15
        显示的最大特征数量
    top_features: int, 默认 5
        显示依赖图的顶部特征数量
    sample_indices: List[int], 可选
        要分析的样本索引列表，默认为None(随机选择3个样本)
    figsize: Tuple[int, int], 默认 (20, 25)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    check_shap_available()
    
    # 计算SHAP值
    shap_values, explainer = calculate_shap_values(model, X, n_samples=n_samples, time_point=time_point)
    
    # 如果样本数量大于n_samples，随机选择n_samples个样本
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # 如果未指定样本索引，随机选择3个样本
    if sample_indices is None:
        sample_indices = np.random.choice(len(X_sample), min(3, len(X_sample)), replace=False)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 添加标题
    if time_point is not None:
        fig.suptitle(f"SHAP值分析报告 (时间点: {time_point})", fontsize=16)
    else:
        fig.suptitle("SHAP值分析报告 (风险得分)", fontsize=16)
    
    # 创建网格
    gs = fig.add_gridspec(4, 2)
    
    # 1. 绘制SHAP值摘要图 (条形图)
    ax1 = fig.add_subplot(gs[0, 0])
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False, ax=ax1)
    ax1.set_title("SHAP值全局特征重要性 (条形图)", fontsize=12)
    
    # 2. 绘制SHAP值摘要图 (点图)
    ax2 = fig.add_subplot(gs[0, 1])
    shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=max_display, show=False, ax=ax2)
    ax2.set_title("SHAP值全局特征重要性 (点图)", fontsize=12)
    
    # 3. 绘制顶部特征的SHAP依赖图
    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns.tolist()
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    top_features_df = feature_importance_df.sort_values('importance', ascending=False).head(top_features)
    
    # 为每个顶部特征绘制依赖图
    for i, (_, row) in enumerate(top_features_df.iterrows()):
        feature = row['feature']
        ax = fig.add_subplot(gs[1, i % 2])
        shap.dependence_plot(feature, shap_values, X_sample, show=False, ax=ax)
        ax.set_title(f"特征 '{feature}' 的SHAP依赖关系", fontsize=12)
        
        # 如果已经绘制了所有顶部特征，跳出循环
        if i >= 1:
            break
    
    # 4. 为每个样本绘制SHAP瀑布图
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[2, i % 2])
        if i < 2:  # 最多绘制2个样本的瀑布图
            shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
            ax.set_title(f"样本 {idx} 的SHAP值贡献", fontsize=12)
    
    # 5. 为每个样本绘制SHAP力图
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[3, i % 2])
        if i < 2:  # 最多绘制2个样本的力图
            shap.force_plot(np.sum(shap_values[idx]), shap_values[idx], X_sample.iloc[idx], 
                           matplotlib=True, show=False, figsize=(10, 3))
            ax.set_title(f"样本 {idx} 的SHAP力图", fontsize=12)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def bootstrap_shap_importance(model: BaseSurvivalModel, X: pd.DataFrame, 
                             n_bootstrap: int = 50, 
                             sample_size: Optional[int] = None,
                             time_point: Optional[float] = None,
                             random_state: Optional[int] = None) -> pd.DataFrame:
    """
    使用Bootstrap方法评估SHAP特征重要性的稳定性
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    X: pd.DataFrame
        特征矩阵
    n_bootstrap: int, 默认 50
        Bootstrap重复次数
    sample_size: int, 可选
        每次Bootstrap的样本大小，默认为None(使用原始样本大小)
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    random_state: int, 可选
        随机种子，默认为None
        
    返回:
    -----
    pd.DataFrame
        Bootstrap特征重要性结果
    """
    check_shap_available()
    
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 如果未指定样本大小，使用原始样本大小的80%
    if sample_size is None:
        sample_size = int(len(X) * 0.8)
    
    # 初始化特征重要性列表
    importance_list = []
    
    # 执行Bootstrap
    for i in range(n_bootstrap):
        try:
            # 随机选择样本
            bootstrap_indices = np.random.choice(len(X), sample_size, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            
            # 计算SHAP值
            shap_values, _ = calculate_shap_values(model, X_bootstrap, time_point=time_point)
            
            # 计算特征重要性
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance': feature_importance,
                'bootstrap': i
            })
            
            # 添加到列表
            importance_list.append(importance_df)
            
        except Exception as e:
            logger.warning(f"Bootstrap {i} 失败: {e}")
    
    # 合并所有Bootstrap结果
    if importance_list:
        all_importance = pd.concat(importance_list, ignore_index=True)
        
        # 计算每个特征的平均重要性和标准差
        summary = all_importance.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
        summary = summary.sort_values('mean', ascending=False)
        
        # 计算95%置信区间
        summary['ci_lower'] = summary['mean'] - 1.96 * summary['std'] / np.sqrt(summary['count'])
        summary['ci_upper'] = summary['mean'] + 1.96 * summary['std'] / np.sqrt(summary['count'])
        
        return summary
    else:
        return pd.DataFrame(columns=['feature', 'mean', 'std', 'count', 'ci_lower', 'ci_upper'])

def plot_bootstrap_importance(bootstrap_results: pd.DataFrame, 
                             max_display: int = 20,
                             title: str = "Bootstrap SHAP特征重要性",
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    绘制Bootstrap SHAP特征重要性图
    
    参数:
    -----
    bootstrap_results: pd.DataFrame
        Bootstrap特征重要性结果
    max_display: int, 默认 20
        显示的最大特征数量
    title: str, 默认 "Bootstrap SHAP特征重要性"
        图表标题
    figsize: Tuple[int, int], 默认 (12, 10)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 选择前N个特征
    top_features = bootstrap_results.head(max_display).copy()
    
    # 反转顺序，使最重要的特征在顶部
    top_features = top_features.iloc[::-1]
    
    # 绘制水平条形图
    bars = ax.barh(top_features['feature'], top_features['mean'], 
                  xerr=1.96 * top_features['std'] / np.sqrt(top_features['count']),
                  capsize=5, alpha=0.7, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', va='center')
    
    # 添加标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('平均SHAP值重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    
    # 添加网格线
    ax.grid(True, axis='x', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig 