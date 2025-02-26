# -*- coding: utf-8 -*-
"""
决策曲线分析模块
提供用于评估生存分析模型临床决策价值的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
import warnings

logger = logging.getLogger(__name__)

def calculate_net_benefit(y_true: pd.DataFrame, risk_probs: np.ndarray, 
                         threshold: float, time_point: float, 
                         time_col: str = 'time', event_col: str = 'event') -> float:
    """
    计算净获益
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    risk_probs: np.ndarray
        预测的风险概率（事件发生概率，不是生存概率）
    threshold: float
        风险阈值
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    float
        净获益值
    """
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 根据阈值进行分类
    predictions = (risk_probs >= threshold).astype(int)
    
    # 计算真阳性和假阳性
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    
    # 计算净获益
    n = len(y_true)
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    return net_benefit

def calculate_interventions_avoided(y_true: pd.DataFrame, risk_probs: np.ndarray, 
                                  threshold: float, time_point: float, 
                                  time_col: str = 'time', event_col: str = 'event') -> float:
    """
    计算干预避免比例
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    risk_probs: np.ndarray
        预测的风险概率（事件发生概率，不是生存概率）
    threshold: float
        风险阈值
    time_point: float
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    float
        干预避免比例
    """
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 根据阈值进行分类
    predictions = (risk_probs >= threshold).astype(int)
    
    # 计算干预避免比例
    n = len(y_true)
    event_rate = np.mean(binary_labels)
    
    # 如果全部干预，干预数为n
    # 如果根据模型干预，干预数为sum(predictions)
    interventions_avoided = (n - np.sum(predictions)) / n
    
    return interventions_avoided

def calculate_decision_curve(y_true: pd.DataFrame, risk_probs: np.ndarray, 
                           time_point: float, thresholds: np.ndarray, 
                           time_col: str = 'time', event_col: str = 'event') -> Dict[str, np.ndarray]:
    """
    计算决策曲线
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    risk_probs: np.ndarray
        预测的风险概率（事件发生概率，不是生存概率）
    time_point: float
        评估时间点
    thresholds: np.ndarray
        风险阈值数组
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    Dict[str, np.ndarray]
        决策曲线结果
    """
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 计算事件发生率
    event_rate = np.mean(binary_labels)
    
    # 初始化结果数组
    net_benefit_model = np.zeros_like(thresholds)
    net_benefit_all = np.zeros_like(thresholds)
    net_benefit_none = np.zeros_like(thresholds)
    interventions_avoided = np.zeros_like(thresholds)
    
    # 计算每个阈值的净获益
    for i, threshold in enumerate(thresholds):
        # 模型的净获益
        net_benefit_model[i] = calculate_net_benefit(
            y_true, risk_probs, threshold, time_point, time_col, event_col
        )
        
        # 全部干预的净获益
        net_benefit_all[i] = event_rate - (1 - event_rate) * threshold / (1 - threshold)
        
        # 不干预的净获益
        net_benefit_none[i] = 0.0
        
        # 干预避免比例
        interventions_avoided[i] = calculate_interventions_avoided(
            y_true, risk_probs, threshold, time_point, time_col, event_col
        )
    
    return {
        'thresholds': thresholds,
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': net_benefit_none,
        'interventions_avoided': interventions_avoided
    }

def plot_decision_curve(decision_curve_results: Dict[str, np.ndarray], 
                       model_name: str = 'Model', 
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制决策曲线
    
    参数:
    -----
    decision_curve_results: Dict[str, np.ndarray]
        决策曲线结果
    model_name: str, 默认 'Model'
        模型名称
    figsize: Tuple[int, int], 默认 (12, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 提取数据
    thresholds = decision_curve_results['thresholds']
    net_benefit_model = decision_curve_results['net_benefit_model']
    net_benefit_all = decision_curve_results['net_benefit_all']
    net_benefit_none = decision_curve_results['net_benefit_none']
    interventions_avoided = decision_curve_results['interventions_avoided']
    
    # 绘制净获益曲线
    ax1.plot(thresholds, net_benefit_model, 'b-', linewidth=2, label=model_name)
    ax1.plot(thresholds, net_benefit_all, 'k--', linewidth=1, label='全部干预')
    ax1.plot(thresholds, net_benefit_none, 'k-', linewidth=1, label='不干预')
    
    # 添加标题和标签
    ax1.set_title('决策曲线', fontsize=14)
    ax1.set_xlabel('风险阈值', fontsize=12)
    ax1.set_ylabel('净获益', fontsize=12)
    
    # 设置x轴范围
    ax1.set_xlim([0.0, 1.0])
    
    # 添加图例
    ax1.legend(loc='best')
    
    # 添加网格线
    ax1.grid(True, alpha=0.3)
    
    # 绘制干预避免比例曲线
    ax2.plot(thresholds, interventions_avoided, 'r-', linewidth=2)
    
    # 添加标题和标签
    ax2.set_title('干预避免比例', fontsize=14)
    ax2.set_xlabel('风险阈值', fontsize=12)
    ax2.set_ylabel('干预避免比例', fontsize=12)
    
    # 设置x轴和y轴范围
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    
    # 添加网格线
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def compare_decision_curves(y_true: pd.DataFrame, models_risk_probs: Dict[str, np.ndarray], 
                           time_point: float, thresholds: np.ndarray = None, 
                           time_col: str = 'time', event_col: str = 'event',
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    比较多个模型的决策曲线
    
    参数:
    -----
    y_true: pd.DataFrame
        真实值，包含时间和事件列
    models_risk_probs: Dict[str, np.ndarray]
        模型名称到风险概率的映射
    time_point: float
        评估时间点
    thresholds: np.ndarray, 可选
        风险阈值数组，默认为None（使用0.01到0.99的100个点）
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
    figsize: Tuple[int, int], 默认 (12, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 如果未指定阈值，使用默认值
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 创建二分类标签：在time_point之前发生事件为1，否则为0
    binary_labels = np.zeros(len(y_true))
    for i, (t, e) in enumerate(zip(y_true[time_col], y_true[event_col])):
        if e == 1 and t <= time_point:
            binary_labels[i] = 1
    
    # 计算事件发生率
    event_rate = np.mean(binary_labels)
    
    # 计算全部干预和不干预的净获益
    net_benefit_all = event_rate - (1 - event_rate) * thresholds / (1 - thresholds)
    net_benefit_none = np.zeros_like(thresholds)
    
    # 绘制全部干预和不干预的净获益曲线
    ax1.plot(thresholds, net_benefit_all, 'k--', linewidth=1, label='全部干预')
    ax1.plot(thresholds, net_benefit_none, 'k-', linewidth=1, label='不干预')
    
    # 计算并绘制每个模型的净获益曲线
    for model_name, risk_probs in models_risk_probs.items():
        # 计算决策曲线
        decision_curve_results = calculate_decision_curve(
            y_true, risk_probs, time_point, thresholds, time_col, event_col
        )
        
        # 绘制净获益曲线
        ax1.plot(thresholds, decision_curve_results['net_benefit_model'], linewidth=2, label=model_name)
        
        # 绘制干预避免比例曲线
        ax2.plot(thresholds, decision_curve_results['interventions_avoided'], linewidth=2, label=model_name)
    
    # 添加标题和标签
    ax1.set_title(f'时间点 {time_point} 的决策曲线比较', fontsize=14)
    ax1.set_xlabel('风险阈值', fontsize=12)
    ax1.set_ylabel('净获益', fontsize=12)
    
    # 设置x轴范围
    ax1.set_xlim([0.0, 1.0])
    
    # 添加图例
    ax1.legend(loc='best')
    
    # 添加网格线
    ax1.grid(True, alpha=0.3)
    
    # 添加标题和标签
    ax2.set_title(f'时间点 {time_point} 的干预避免比例比较', fontsize=14)
    ax2.set_xlabel('风险阈值', fontsize=12)
    ax2.set_ylabel('干预避免比例', fontsize=12)
    
    # 设置x轴和y轴范围
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    
    # 添加图例
    ax2.legend(loc='best')
    
    # 添加网格线
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def evaluate_decision_curve(y_train: pd.DataFrame, y_test: pd.DataFrame, model, 
                           times: List[float], thresholds: np.ndarray = None, 
                           time_col: str = 'time', event_col: str = 'event') -> Dict[float, Dict[str, np.ndarray]]:
    """
    评估模型的决策曲线
    
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
    thresholds: np.ndarray, 可选
        风险阈值数组，默认为None（使用0.01到0.99的100个点）
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    Dict[float, Dict[str, np.ndarray]]
        每个时间点的决策曲线结果
    """
    # 如果未指定阈值，使用默认值
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    # 获取特征矩阵
    X_test = y_test.drop(columns=[time_col, event_col])
    
    # 预测生存概率
    survival_probs = model.predict(X_test, times)
    
    # 初始化结果字典
    results = {}
    
    # 对每个时间点评估决策曲线
    for i, t in enumerate(times):
        # 计算风险概率（1 - 生存概率）
        risk_probs = 1 - survival_probs[:, i]
        
        # 计算决策曲线
        decision_curve_results = calculate_decision_curve(
            y_test, risk_probs, t, thresholds, time_col, event_col
        )
        
        # 保存结果
        results[t] = decision_curve_results
    
    return results

def create_decision_curve_report(decision_curve_results: Dict[float, Dict[str, np.ndarray]], 
                               model_name: str = 'Model', 
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    创建决策曲线评估报告
    
    参数:
    -----
    decision_curve_results: Dict[float, Dict[str, np.ndarray]]
        每个时间点的决策曲线结果
    model_name: str, 默认 'Model'
        模型名称
    figsize: Tuple[int, int], 默认 (15, 10)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 获取时间点
    times = list(decision_curve_results.keys())
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 添加标题
    fig.suptitle('决策曲线评估报告', fontsize=16)
    
    # 创建网格
    n_times = len(times)
    n_cols = min(3, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols * 2  # 每个时间点两行
    
    # 绘制每个时间点的决策曲线
    for i, t in enumerate(times):
        # 获取决策曲线结果
        results = decision_curve_results[t]
        
        # 提取数据
        thresholds = results['thresholds']
        net_benefit_model = results['net_benefit_model']
        net_benefit_all = results['net_benefit_all']
        net_benefit_none = results['net_benefit_none']
        interventions_avoided = results['interventions_avoided']
        
        # 创建净获益子图
        ax1 = fig.add_subplot(n_rows, n_cols, i * 2 + 1)
        
        # 绘制净获益曲线
        ax1.plot(thresholds, net_benefit_model, 'b-', linewidth=2, label=model_name)
        ax1.plot(thresholds, net_benefit_all, 'k--', linewidth=1, label='全部干预')
        ax1.plot(thresholds, net_benefit_none, 'k-', linewidth=1, label='不干预')
        
        # 添加标题和标签
        ax1.set_title(f'时间点 {t} 的决策曲线', fontsize=12)
        ax1.set_xlabel('风险阈值', fontsize=10)
        ax1.set_ylabel('净获益', fontsize=10)
        
        # 设置x轴范围
        ax1.set_xlim([0.0, 1.0])
        
        # 添加图例
        ax1.legend(loc='best', fontsize=8)
        
        # 添加网格线
        ax1.grid(True, alpha=0.3)
        
        # 创建干预避免比例子图
        ax2 = fig.add_subplot(n_rows, n_cols, i * 2 + 2)
        
        # 绘制干预避免比例曲线
        ax2.plot(thresholds, interventions_avoided, 'r-', linewidth=2)
        
        # 添加标题和标签
        ax2.set_title(f'时间点 {t} 的干预避免比例', fontsize=12)
        ax2.set_xlabel('风险阈值', fontsize=10)
        ax2.set_ylabel('干预避免比例', fontsize=10)
        
        # 设置x轴和y轴范围
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        
        # 添加网格线
        ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig 