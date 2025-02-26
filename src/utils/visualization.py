# -*- coding: utf-8 -*-
"""
可视化工具模块
提供统一的可视化接口，用于生成各种图表
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import os
import warnings
from datetime import datetime

from .logger import get_logger

logger = get_logger(name="visualization")

# 设置默认风格
def set_visualization_style(style: str = 'whitegrid', 
                          context: str = 'paper', 
                          font_scale: float = 1.2, 
                          palette: str = 'deep',
                          use_mpl_style: bool = True) -> None:
    """
    Set visualization style for matplotlib and seaborn
    
    Parameters:
    -----
    style: str, default 'whitegrid'
        Seaborn style, options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
    context: str, default 'paper'
        Seaborn context, options: 'paper', 'notebook', 'talk', 'poster'
    font_scale: float, default 1.2
        Font scale factor
    palette: str, default 'deep'
        Color palette name
    use_mpl_style: bool, default True
        Whether to set matplotlib style as well
    """
    # Set seaborn style
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)
    
    # Set matplotlib style if requested
    if use_mpl_style:
        plt.style.use('seaborn')
    
    # Additional customization
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

# 保存图表
def save_figure(fig: plt.Figure, 
              filename: str, 
              directory: str = "figures", 
              formats: List[str] = ["png", "pdf", "svg"],
              dpi: int = 300,
              transparent: bool = False,
              close_figure: bool = True) -> List[str]:
    """
    保存图表到文件
    
    参数:
    -----
    fig: plt.Figure
        matplotlib图形对象
    filename: str
        文件名（不包含扩展名）
    directory: str, 默认 "figures"
        保存目录
    formats: List[str], 默认 ["png", "pdf", "svg"]
        保存格式
    dpi: int, 默认 300
        分辨率
    transparent: bool, 默认 False
        是否使用透明背景
    close_figure: bool, 默认 True
        保存后是否关闭图形
        
    返回:
    -----
    List[str]
        保存的文件路径列表
    """
    # 创建保存目录
    os.makedirs(directory, exist_ok=True)
    
    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}"
    
    # 保存图表
    saved_files = []
    for fmt in formats:
        filepath = os.path.join(directory, f"{filename_with_timestamp}.{fmt}")
        fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches="tight")
        saved_files.append(filepath)
        logger.info(f"图表已保存: {filepath}")
    
    # 关闭图形
    if close_figure:
        plt.close(fig)
    
    return saved_files

# 创建子图网格
def create_subplots(n_plots: int, 
                   n_cols: int = 2, 
                   figsize: Optional[Tuple[float, float]] = None,
                   height_ratios: Optional[List[float]] = None,
                   width_ratios: Optional[List[float]] = None,
                   title: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    创建子图网格
    
    参数:
    -----
    n_plots: int
        子图数量
    n_cols: int, 默认 2
        列数
    figsize: Tuple[float, float], 可选
        图形大小
    height_ratios: List[float], 可选
        行高比例
    width_ratios: List[float], 可选
        列宽比例
    title: str, 可选
        图形标题
        
    返回:
    -----
    Tuple[plt.Figure, List[plt.Axes]]
        (图形对象, 子图列表)
    """
    # 计算行数
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # 设置默认图形大小
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # 创建网格规格
    if height_ratios is not None or width_ratios is not None:
        gridspec_kw = {}
        if height_ratios is not None:
            gridspec_kw["height_ratios"] = height_ratios
        if width_ratios is not None:
            gridspec_kw["width_ratios"] = width_ratios
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, gridspec_kw=gridspec_kw)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 添加标题
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # 将axes转换为列表
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes

# 绘制生存曲线
def plot_survival_curves(survival_curves: Dict[str, np.ndarray], 
                        times: np.ndarray,
                        labels: List[str] = None, 
                        ci_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
                        title: str = 'Survival Curves',
                        xlabel: str = 'Time',
                        ylabel: str = 'Survival Probability',
                        cmap: str = 'viridis',
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot survival curves for multiple groups or models
    
    Parameters:
    -----
    survival_curves: Dict[str, np.ndarray]
        Dictionary mapping curve names to survival probabilities
    times: np.ndarray
        Time points for survival curves
    labels: List[str], optional
        Curve labels for legend
    ci_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], optional
        Dictionary mapping curve names to confidence intervals (lower, upper)
    title: str, default 'Survival Curves'
        Plot title
    xlabel: str, default 'Time'
        X-axis label
    ylabel: str, default 'Survival Probability'
        Y-axis label
    cmap: str, default 'viridis'
        Colormap name
    figsize: Tuple[int, int], default (10, 6)
        Figure size
        
    Returns:
    -----
    plt.Figure
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    n_curves = len(survival_curves)
    if n_curves <= 10:
        colors = sns.color_palette(cmap, n_curves)
    else:
        colors = plt.cm.get_cmap(cmap, n_curves)
    
    # Use provided labels or keys
    if labels is None:
        labels = list(survival_curves.keys())
    
    # Plot each curve
    for i, (key, surv) in enumerate(survival_curves.items()):
        # Plot survival curve
        ax.plot(times, surv, '-', label=labels[i], color=colors[i])
        
        # Plot confidence intervals if provided
        if ci_curves is not None and key in ci_curves:
            lower, upper = ci_curves[key]
            ax.fill_between(times, lower, upper, alpha=0.2, color=colors[i])
    
    # Add reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set axis limits
    ax.set_xlim([0, max(times)])
    ax.set_ylim([0, 1.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# 绘制ROC曲线
def plot_roc_curves(fpr_list: List[np.ndarray], 
                   tpr_list: List[np.ndarray], 
                   auc_list: List[float],
                   labels: List[str],
                   title: str = "ROC曲线",
                   figsize: Tuple[float, float] = (8, 8),
                   colors: Optional[List[str]] = None,
                   linestyles: Optional[List[str]] = None,
                   add_chance_line: bool = True) -> plt.Figure:
    """
    绘制ROC曲线
    
    参数:
    -----
    fpr_list: List[np.ndarray]
        假阳性率列表
    tpr_list: List[np.ndarray]
        真阳性率列表
    auc_list: List[float]
        AUC值列表
    labels: List[str]
        曲线标签
    title: str, 默认 "ROC曲线"
        图表标题
    figsize: Tuple[float, float], 默认 (8, 8)
        图形大小
    colors: List[str], 可选
        曲线颜色列表
    linestyles: List[str], 可选
        曲线样式列表
    add_chance_line: bool, 默认 True
        是否添加随机猜测线
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 设置默认颜色和线型
    if colors is None:
        colors = plt.cm.tab10.colors
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':'] * 3
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制ROC曲线
    for i, (fpr, tpr, auc, label) in enumerate(zip(fpr_list, tpr_list, auc_list, labels)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2, 
               label=f"{label} (AUC = {auc:.3f})")
    
    # 添加随机猜测线
    if add_chance_line:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1, label='随机猜测')
    
    # 设置坐标轴
    ax.set_xlabel('假阳性率', fontsize=12)
    ax.set_ylabel('真阳性率', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 设置坐标轴范围
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='lower right')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 绘制特征重要性
def plot_feature_importance(features: List[str], 
                          importance: np.ndarray, 
                          errors: Optional[np.ndarray] = None,
                          title: str = "特征重要性",
                          figsize: Tuple[float, float] = (10, 8),
                          color: str = "skyblue",
                          max_features: int = 20,
                          horizontal: bool = True) -> plt.Figure:
    """
    绘制特征重要性条形图
    
    参数:
    -----
    features: List[str]
        特征名称列表
    importance: np.ndarray
        特征重要性数组
    errors: np.ndarray, 可选
        特征重要性误差数组
    title: str, 默认 "特征重要性"
        图表标题
    figsize: Tuple[float, float], 默认 (10, 8)
        图形大小
    color: str, 默认 "skyblue"
        条形颜色
    max_features: int, 默认 20
        显示的最大特征数量
    horizontal: bool, 默认 True
        是否使用水平条形图
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    })
    
    # 添加误差
    if errors is not None:
        importance_df['error'] = errors
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 选择前N个特征
    if len(importance_df) > max_features:
        importance_df = importance_df.head(max_features)
    
    # 反转顺序，使最重要的特征在顶部（对于水平条形图）
    if horizontal:
        importance_df = importance_df.iloc[::-1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制条形图
    if horizontal:
        if errors is not None:
            bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                          xerr=importance_df['error'], color=color, capsize=5, alpha=0.7)
        else:
            bars = ax.barh(importance_df['feature'], importance_df['importance'], color=color)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', va='center')
        
        # 设置坐标轴标签
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        
        # 添加网格线
        ax.grid(True, axis='x', alpha=0.3)
    else:
        if errors is not None:
            bars = ax.bar(importance_df['feature'], importance_df['importance'], 
                         yerr=importance_df['error'], color=color, capsize=5, alpha=0.7)
        else:
            bars = ax.bar(importance_df['feature'], importance_df['importance'], color=color)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                   f'{height:.3f}', ha='center')
        
        # 设置坐标轴标签
        ax.set_xlabel('特征', fontsize=12)
        ax.set_ylabel('重要性', fontsize=12)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        # 添加网格线
        ax.grid(True, axis='y', alpha=0.3)
    
    # 设置标题
    ax.set_title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 绘制模型比较
def plot_model_comparison(models: List[str], 
                         metrics: Dict[str, List[float]], 
                         title: str = "模型比较",
                         figsize: Tuple[float, float] = (12, 8),
                         colors: Optional[List[str]] = None,
                         sort_by: Optional[str] = None,
                         ascending: bool = False) -> plt.Figure:
    """
    绘制模型比较图
    
    参数:
    -----
    models: List[str]
        模型名称列表
    metrics: Dict[str, List[float]]
        指标字典，键为指标名称，值为各模型的指标值列表
    title: str, 默认 "模型比较"
        图表标题
    figsize: Tuple[float, float], 默认 (12, 8)
        图形大小
    colors: List[str], 可选
        条形颜色列表
    sort_by: str, 可选
        排序依据的指标名称
    ascending: bool, 默认 False
        是否升序排序
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建数据DataFrame
    data = {'model': models}
    for metric_name, metric_values in metrics.items():
        data[metric_name] = metric_values
    df = pd.DataFrame(data)
    
    # 排序
    if sort_by is not None and sort_by in metrics:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # 设置默认颜色
    if colors is None:
        colors = plt.cm.tab10.colors
    
    # 计算子图数量
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 将axes转换为列表
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 绘制每个指标的条形图
    for i, (metric_name, ax) in enumerate(zip(metrics.keys(), axes)):
        # 绘制条形图
        bars = ax.bar(df['model'], df[metric_name], color=colors[:len(models)])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                   f'{height:.3f}', ha='center')
        
        # 设置标题和标签
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel('值', fontsize=10)
        
        # 旋转x轴标签
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        
        # 添加网格线
        ax.grid(True, axis='y', alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # 设置标题
    fig.suptitle(title, fontsize=16)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

# 绘制相关性热图
def plot_correlation_heatmap(data: pd.DataFrame, 
                           method: str = 'pearson', 
                           title: str = "特征相关性热图",
                           figsize: Tuple[float, float] = (12, 10),
                           cmap: str = "coolwarm",
                           mask_upper: bool = True,
                           annot: bool = True,
                           fmt: str = ".2f") -> plt.Figure:
    """
    绘制相关性热图
    
    参数:
    -----
    data: pd.DataFrame
        数据DataFrame
    method: str, 默认 'pearson'
        相关系数计算方法，可选 'pearson', 'kendall', 'spearman'
    title: str, 默认 "特征相关性热图"
        图表标题
    figsize: Tuple[float, float], 默认 (12, 10)
        图形大小
    cmap: str, 默认 "coolwarm"
        颜色映射
    mask_upper: bool, 默认 True
        是否遮盖上三角部分
    annot: bool, 默认 True
        是否显示数值标注
    fmt: str, 默认 ".2f"
        数值格式
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 计算相关系数
    corr = data.corr(method=method)
    
    # 创建遮罩
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt, 
               linewidths=0.5, ax=ax, square=True, center=0)
    
    # 设置标题
    ax.set_title(title, fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 绘制分布图
def plot_distribution(data: pd.DataFrame, 
                     columns: Optional[List[str]] = None,
                     hue: Optional[str] = None,
                     kind: str = 'hist',
                     title: str = "特征分布",
                     figsize: Optional[Tuple[float, float]] = None,
                     n_cols: int = 3,
                     bins: int = 30,
                     kde: bool = True) -> plt.Figure:
    """
    绘制特征分布图
    
    参数:
    -----
    data: pd.DataFrame
        数据DataFrame
    columns: List[str], 可选
        要绘制的列名列表，默认为None(使用所有数值列)
    hue: str, 可选
        分组变量
    kind: str, 默认 'hist'
        图表类型，可选 'hist', 'kde', 'box', 'violin', 'strip'
    title: str, 默认 "特征分布"
        图表标题
    figsize: Tuple[float, float], 可选
        图形大小，默认根据列数自动计算
    n_cols: int, 默认 3
        列数
    bins: int, 默认 30
        直方图的箱数
    kde: bool, 默认 True
        是否显示核密度估计
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 如果未指定列，使用所有数值列
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # 过滤不存在的列
    columns = [col for col in columns if col in data.columns]
    
    if not columns:
        raise ValueError("没有有效的列可绘制")
    
    # 计算行数
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # 设置默认图形大小
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 将axes转换为列表
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 绘制每个特征的分布图
    for i, (col, ax) in enumerate(zip(columns, axes)):
        if kind == 'hist':
            if hue is not None and hue in data.columns:
                for name, group in data.groupby(hue):
                    sns.histplot(group[col], kde=kde, bins=bins, alpha=0.5, label=name, ax=ax)
                ax.legend()
            else:
                sns.histplot(data[col], kde=kde, bins=bins, ax=ax)
        elif kind == 'kde':
            if hue is not None and hue in data.columns:
                for name, group in data.groupby(hue):
                    sns.kdeplot(group[col], label=name, ax=ax)
                ax.legend()
            else:
                sns.kdeplot(data[col], ax=ax)
        elif kind == 'box':
            if hue is not None and hue in data.columns:
                sns.boxplot(x=hue, y=col, data=data, ax=ax)
            else:
                sns.boxplot(y=col, data=data, ax=ax)
        elif kind == 'violin':
            if hue is not None and hue in data.columns:
                sns.violinplot(x=hue, y=col, data=data, ax=ax)
            else:
                sns.violinplot(y=col, data=data, ax=ax)
        elif kind == 'strip':
            if hue is not None and hue in data.columns:
                sns.stripplot(x=hue, y=col, data=data, ax=ax)
            else:
                sns.stripplot(y=col, data=data, ax=ax)
        else:
            raise ValueError(f"不支持的图表类型: {kind}")
        
        # 设置标题
        ax.set_title(col, fontsize=12)
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    # 设置标题
    fig.suptitle(title, fontsize=16)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

# 绘制校准曲线
def plot_calibration_curves(prob_true_list: List[np.ndarray], 
                          prob_pred_list: List[np.ndarray], 
                          labels: List[str],
                          title: str = "校准曲线",
                          figsize: Tuple[float, float] = (8, 8),
                          colors: Optional[List[str]] = None,
                          hist: bool = True) -> plt.Figure:
    """
    绘制校准曲线
    
    参数:
    -----
    prob_true_list: List[np.ndarray]
        真实概率列表
    prob_pred_list: List[np.ndarray]
        预测概率列表
    labels: List[str]
        曲线标签列表
    title: str, 默认 "校准曲线"
        图表标题
    figsize: Tuple[float, float], 默认 (8, 8)
        图形大小
    colors: List[str], 可选
        曲线颜色列表
    hist: bool, 默认 True
        是否显示直方图
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 设置默认颜色
    if colors is None:
        colors = plt.cm.tab10.colors
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='完美校准')
    
    # 绘制每条校准曲线
    for i, (prob_true, prob_pred, label) in enumerate(zip(prob_true_list, prob_pred_list, labels)):
        color = colors[i % len(colors)]
        ax.plot(prob_pred, prob_true, 's-', color=color, linewidth=2, label=label)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('预测概率', fontsize=12)
    ax.set_ylabel('真实概率', fontsize=12)
    
    # 设置轴范围
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='lower right')
    
    # 添加直方图
    if hist:
        # 创建双轴
        ax2 = ax.twinx()
        
        # 绘制每个模型的预测概率分布
        for i, (prob_pred, label) in enumerate(zip(prob_pred_list, labels)):
            color = colors[i % len(colors)]
            ax2.hist(prob_pred, bins=10, alpha=0.1, color=color)
        
        # 设置y轴标签
        ax2.set_ylabel('样本数', fontsize=12)
        
        # 设置y轴范围
        ax2.set_ylim([0, None])
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 绘制决策曲线
def plot_decision_curves(thresholds: np.ndarray, 
                        net_benefit_list: List[np.ndarray], 
                        labels: List[str],
                        title: str = "决策曲线",
                        figsize: Tuple[float, float] = (10, 6),
                        colors: Optional[List[str]] = None,
                        all_patients_line: bool = True,
                        no_patients_line: bool = True) -> plt.Figure:
    """
    绘制决策曲线
    
    参数:
    -----
    thresholds: np.ndarray
        阈值数组
    net_benefit_list: List[np.ndarray]
        净获益列表
    labels: List[str]
        曲线标签列表
    title: str, 默认 "决策曲线"
        图表标题
    figsize: Tuple[float, float], 默认 (10, 6)
        图形大小
    colors: List[str], 可选
        曲线颜色列表
    all_patients_line: bool, 默认 True
        是否显示"全部患者"线
    no_patients_line: bool, 默认 True
        是否显示"无患者"线
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 设置默认颜色
    if colors is None:
        colors = plt.cm.tab10.colors
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制每条决策曲线
    for i, (net_benefit, label) in enumerate(zip(net_benefit_list, labels)):
        color = colors[i % len(colors)]
        ax.plot(thresholds, net_benefit, color=color, linewidth=2, label=label)
    
    # 绘制"全部患者"线
    if all_patients_line:
        # 假设第一个阈值接近0
        all_patients_benefit = np.zeros_like(thresholds)
        all_patients_benefit[0] = net_benefit_list[0][0]  # 使用第一个模型在阈值0处的净获益
        all_patients_benefit[1:] = all_patients_benefit[0] - thresholds[1:] / (1 - thresholds[1:])
        ax.plot(thresholds, all_patients_benefit, 'k--', linewidth=1, label='全部患者')
    
    # 绘制"无患者"线
    if no_patients_line:
        ax.plot(thresholds, np.zeros_like(thresholds), 'k-', linewidth=1, label='无患者')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('阈值概率', fontsize=12)
    ax.set_ylabel('净获益', fontsize=12)
    
    # 设置x轴范围
    ax.set_xlim([0.0, 1.0])
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 绘制特征重要性
def plot_feature_importance(feature_names: List[str], 
                          importance_values: np.ndarray, 
                          title: str = "特征重要性",
                          figsize: Tuple[float, float] = (10, 8),
                          color: str = "skyblue",
                          error_bars: Optional[np.ndarray] = None,
                          max_features: int = 20,
                          sort: bool = True) -> plt.Figure:
    """
    绘制特征重要性图
    
    参数:
    -----
    feature_names: List[str]
        特征名称列表
    importance_values: np.ndarray
        重要性值数组
    title: str, 默认 "特征重要性"
        图表标题
    figsize: Tuple[float, float], 默认 (10, 8)
        图形大小
    color: str, 默认 "skyblue"
        条形颜色
    error_bars: np.ndarray, 可选
        误差条数组
    max_features: int, 默认 20
        显示的最大特征数量
    sort: bool, 默认 True
        是否按重要性排序
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建数据DataFrame
    data = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # 排序
    if sort:
        data = data.sort_values('importance', ascending=False)
    
    # 限制特征数量
    if len(data) > max_features:
        data = data.head(max_features)
    
    # 反转顺序，使最重要的特征在顶部
    data = data.iloc[::-1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制条形图
    if error_bars is not None:
        # 如果提供了误差条，使用它们
        error_data = error_bars[data.index]
        bars = ax.barh(data['feature'], data['importance'], xerr=error_data, 
                      color=color, edgecolor='black', alpha=0.7, capsize=5)
    else:
        bars = ax.barh(data['feature'], data['importance'], 
                      color=color, edgecolor='black', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', va='center')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('重要性', fontsize=12)
    
    # 添加网格线
    ax.grid(True, axis='x', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

# 主函数：测试可视化功能
if __name__ == "__main__":
    # 设置可视化风格
    set_visualization_style()
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'age': np.random.normal(60, 10, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
        'cholesterol': np.random.normal(200, 30, n_samples),
        'group': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # 测试分布图
    fig1 = plot_distribution(data, kind='hist')
    save_figure(fig1, "distribution_test")
    
    # 测试相关性热图
    fig2 = plot_correlation_heatmap(data.select_dtypes(include=['number']))
    save_figure(fig2, "correlation_test")
    
    # 测试特征重要性图
    feature_names = ['age', 'bmi', 'glucose', 'cholesterol']
    importance_values = np.array([0.3, 0.2, 0.4, 0.1])
    fig3 = plot_feature_importance(feature_names, importance_values)
    save_figure(fig3, "importance_test")
    
    # 测试生存曲线
    times = np.linspace(0, 10, 100)
    survival_probs = [
        np.exp(-0.1 * times),
        np.exp(-0.2 * times),
        np.exp(-0.3 * times)
    ]
    labels = ['Model A', 'Model B', 'Model C']
    fig4 = plot_survival_curves(times, survival_probs, labels)
    save_figure(fig4, "survival_test")
    
    print("可视化测试完成，图表已保存到figures目录")