# -*- coding: utf-8 -*-
"""
校准评估模块
提供用于评估生存分析模型校准性能的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from scipy import stats
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
import warnings

logger = logging.getLogger(__name__)

def calculate_calibration_curve(y_true: pd.DataFrame, survival_probs: np.ndarray, 
                              time_point: float, time_col: str = 'time', event_col: str = 'event', 
                              n_bins: int = 10, strategy: str = 'quantile') -> Tuple[np.ndarray, np.ndarray]:
    """
    计算校准曲线
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    survival_probs: np.ndarray
        预测的生存概率
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    n_bins: int, 默认 10
        分箱数量
    strategy: str, 默认 'quantile'
        分箱策略，可选 'uniform', 'quantile'
        
    返回:
    -----
    Tuple[np.ndarray, np.ndarray]
        (预测概率, 观察概率)
    """
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 计算校准曲线
    # 注意：我们使用1-survival_probs，因为校准曲线期望的是事件发生的概率，而不是生存概率
    prob_pred, prob_obs = calibration_curve(binary_labels, 1 - survival_probs, n_bins=n_bins, strategy=strategy)
    
    return prob_pred, prob_obs

def calculate_calibration_metrics(prob_pred: np.ndarray, prob_obs: np.ndarray) -> Dict[str, float]:
    """
    计算校准指标
    
    参数:
    -----
    prob_pred: np.ndarray
        预测概率
    prob_obs: np.ndarray
        观察概率
        
    返回:
    -----
    Dict[str, float]
        校准指标
    """
    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(prob_pred.reshape(-1, 1), prob_obs)
    
    # 计算校准斜率和截距
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # 计算R²
    r_squared = model.score(prob_pred.reshape(-1, 1), prob_obs)
    
    # 计算均方误差
    mse = np.mean((prob_obs - prob_pred) ** 2)
    
    # 计算平均校准误差
    calibration_error = np.mean(np.abs(prob_obs - prob_pred))
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'mse': mse,
        'calibration_error': calibration_error
    }

def hosmer_lemeshow_test(y_true: pd.DataFrame, survival_probs: np.ndarray, 
                        time_point: float, time_col: str = 'time', event_col: str = 'event', 
                        n_bins: int = 10) -> Dict[str, float]:
    """
    霍斯默-莱梅肖检验
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    survival_probs: np.ndarray
        预测的生存概率
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    n_bins: int, 默认 10
        分箱数量
        
    返回:
    -----
    Dict[str, float]
        检验结果
    """
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 计算事件发生概率
    event_probs = 1 - survival_probs
    
    # 按预测概率排序并分组
    sorted_indices = np.argsort(event_probs)
    sorted_probs = event_probs[sorted_indices]
    sorted_labels = binary_labels[sorted_indices]
    
    # 分箱
    bin_size = len(sorted_probs) // n_bins
    if bin_size == 0:
        raise ValueError(f"样本数量({len(sorted_probs)})不足以分成{n_bins}个箱")
    
    # 计算每个箱的观察事件数和预期事件数
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_probs)
        
        bin_labels = sorted_labels[start_idx:end_idx]
        bin_probs = sorted_probs[start_idx:end_idx]
        
        observed[i] = np.sum(bin_labels)
        expected[i] = np.sum(bin_probs)
    
    # 计算霍斯默-莱梅肖统计量
    hl_statistic = np.sum((observed - expected) ** 2 / (expected * (1 - expected / (end_idx - start_idx))))
    
    # 计算p值
    p_value = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
    
    return {
        'hl_statistic': hl_statistic,
        'p_value': p_value,
        'observed': observed,
        'expected': expected
    }

def plot_calibration_curve(y_true: pd.DataFrame, survival_probs: np.ndarray, 
                          time_point: float, time_col: str = 'time', event_col: str = 'event', 
                          n_bins: int = 10, strategy: str = 'quantile', 
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制校准曲线
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    survival_probs: np.ndarray
        预测的生存概率
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    n_bins: int, 默认 10
        分箱数量
    strategy: str, 默认 'quantile'
        分箱策略，可选 'uniform', 'quantile'
    figsize: Tuple[int, int], 默认 (10, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 计算校准曲线
    prob_pred, prob_obs = calculate_calibration_curve(
        y_true, survival_probs, time_point, time_col, event_col, n_bins, strategy
    )
    
    # 计算校准指标
    metrics = calculate_calibration_metrics(prob_pred, prob_obs)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制校准曲线
    ax.plot(prob_pred, prob_obs, marker='o', linestyle='-', label='校准曲线')
    
    # 绘制拟合线
    x_range = np.linspace(0, 1, 100)
    y_fit = metrics['slope'] * x_range + metrics['intercept']
    ax.plot(x_range, y_fit, linestyle='--', color='red', 
           label=f'拟合线 (斜率={metrics["slope"]:.2f}, 截距={metrics["intercept"]:.2f})')
    
    # 绘制理想线
    ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='理想校准')
    
    # 添加标题和标签
    ax.set_title(f'时间点 {time_point} 的校准曲线', fontsize=14)
    ax.set_xlabel('预测概率', fontsize=12)
    ax.set_ylabel('观察概率', fontsize=12)
    
    # 添加校准指标文本
    textstr = '\n'.join((
        f'斜率: {metrics["slope"]:.2f}',
        f'截距: {metrics["intercept"]:.2f}',
        f'R²: {metrics["r_squared"]:.2f}',
        f'MSE: {metrics["mse"]:.4f}',
        f'校准误差: {metrics["calibration_error"]:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # 设置轴范围
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # 添加图例
    ax.legend(loc='lower right')
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_calibration_histogram(y_true: pd.DataFrame, survival_probs: np.ndarray, 
                              time_point: float, time_col: str = 'time', event_col: str = 'event', 
                              n_bins: int = 10, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    绘制校准直方图
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    survival_probs: np.ndarray
        预测的生存概率
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    n_bins: int, 默认 10
        分箱数量
    figsize: Tuple[int, int], 默认 (12, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 执行霍斯默-莱梅肖检验
    hl_result = hosmer_lemeshow_test(
        y_true, survival_probs, time_point, time_col, event_col, n_bins
    )
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图位置
    x = np.arange(n_bins)
    width = 0.35
    
    # 绘制观察值和预期值
    rects1 = ax.bar(x - width/2, hl_result['observed'], width, label='观察事件数')
    rects2 = ax.bar(x + width/2, hl_result['expected'], width, label='预期事件数')
    
    # 添加标题和标签
    ax.set_title(f'时间点 {time_point} 的校准直方图', fontsize=14)
    ax.set_xlabel('风险分组', fontsize=12)
    ax.set_ylabel('事件数', fontsize=12)
    
    # 添加霍斯默-莱梅肖检验结果
    textstr = '\n'.join((
        f'霍斯默-莱梅肖统计量: {hl_result["hl_statistic"]:.2f}',
        f'p值: {hl_result["p_value"]:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n_bins)])
    
    # 添加图例
    ax.legend()
    
    # 添加网格线
    ax.grid(True, axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def evaluate_calibration(y_train: pd.DataFrame, y_test: pd.DataFrame, model, 
                        times: List[float], time_col: str = 'time', event_col: str = 'event', 
                        n_bins: int = 10) -> Dict[str, Any]:
    """
    评估模型校准性能
    
    参数:
    -----
    y_train: pd.DataFrame
        训练集真实值，包含时间和事件列
    y_test: pd.DataFrame
        测试集真实值，包含时间和事件列
    model: BaseSurvivalModel
        训练好的模型
    times: List[float]
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    n_bins: int, 默认 10
        分箱数量
        
    返回:
    -----
    Dict[str, Any]
        校准评估结果
    """
    # 获取特征矩阵
    X_test = y_test.drop(columns=[time_col, event_col])
    
    # 预测生存概率
    survival_probs = model.predict(X_test, times)
    
    # 初始化结果字典
    results = {
        'calibration_metrics': {},
        'hosmer_lemeshow': {}
    }
    
    # 对每个时间点评估校准性能
    for i, t in enumerate(times):
        # 计算校准曲线
        prob_pred, prob_obs = calculate_calibration_curve(
            y_test, survival_probs[:, i], t, time_col, event_col, n_bins
        )
        
        # 计算校准指标
        metrics = calculate_calibration_metrics(prob_pred, prob_obs)
        results['calibration_metrics'][t] = metrics
        
        # 执行霍斯默-莱梅肖检验
        try:
            hl_result = hosmer_lemeshow_test(
                y_test, survival_probs[:, i], t, time_col, event_col, n_bins
            )
            results['hosmer_lemeshow'][t] = hl_result
        except Exception as e:
            logger.warning(f"执行霍斯默-莱梅肖检验时出错: {e}")
            results['hosmer_lemeshow'][t] = None
    
    return results

def create_calibration_report(calibration_results: Dict[str, Any], times: List[float], 
                             n_bins: int = 10, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    创建校准评估报告
    
    参数:
    -----
    calibration_results: Dict[str, Any]
        校准评估结果
    times: List[float]
        评估时间点
    n_bins: int, 默认 10
        分箱数量
    figsize: Tuple[int, int], 默认 (15, 10)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 添加标题
    fig.suptitle('校准评估报告', fontsize=16)
    
    # 创建网格
    n_times = len(times)
    n_cols = min(3, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols
    
    # 绘制校准指标
    for i, t in enumerate(times):
        # 获取校准指标
        metrics = calibration_results['calibration_metrics'][t]
        
        # 创建子图
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # 绘制校准指标条形图
        metric_names = ['slope', 'intercept', 'r_squared', 'calibration_error']
        metric_values = [metrics[name] for name in metric_names]
        
        bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon', 'lightcoral'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3点垂直偏移
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # 添加标题和标签
        ax.set_title(f'时间点 {t} 的校准指标', fontsize=12)
        ax.set_ylabel('值', fontsize=10)
        
        # 添加网格线
        ax.grid(True, axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig 