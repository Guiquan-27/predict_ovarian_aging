# -*- coding: utf-8 -*-
"""
评估指标模块
提供用于评估生存分析模型性能的各种指标
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
    brier_score
)
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings

logger = logging.getLogger(__name__)

def c_index(y_true: pd.DataFrame, risk_scores: np.ndarray, time_col: str = 'time', event_col: str = 'event') -> float:
    """
    Calculate concordance index (C-index)
    
    Parameters:
    -----
    y_true: pd.DataFrame
        True values containing time and event columns
    risk_scores: np.ndarray
        Predicted risk scores
    time_col: str, default 'time'
        Time column name
    event_col: str, default 'event'
        Event column name
        
    Returns:
    -----
    float
        C-index value
    """
    return concordance_index(y_true[time_col], -risk_scores, y_true[event_col])

def c_index_ipcw(y_train: pd.DataFrame, y_test: pd.DataFrame, risk_scores: np.ndarray, 
                time_col: str = 'time', event_col: str = 'event', tau: Optional[float] = None) -> float:
    """
    Calculate inverse probability of censoring weighted concordance index (IPCW C-index)
    
    Parameters:
    -----
    y_train: pd.DataFrame
        Training set true values containing time and event columns
    y_test: pd.DataFrame
        Test set true values containing time and event columns
    risk_scores: np.ndarray
        Predicted risk scores
    time_col: str, default 'time'
        Time column name
    event_col: str, default 'event'
        Event column name
    tau: float, optional
        Truncation time, default None (use maximum observation time)
        
    Returns:
    -----
    float
        IPCW C-index value
    """
    # Convert to sksurv format
    y_train_sksurv = np.array([(e, t) for e, t in zip(y_train[event_col], y_train[time_col])], 
                             dtype=[('event', bool), ('time', float)])
    y_test_sksurv = np.array([(e, t) for e, t in zip(y_test[event_col], y_test[time_col])], 
                            dtype=[('event', bool), ('time', float)])
    
    # If truncation time is not specified, use maximum observation time
    if tau is None:
        tau = y_test[time_col].max()
    
    # Calculate IPCW C-index
    result = concordance_index_ipcw(y_train_sksurv, y_test_sksurv, risk_scores, tau=tau)
    
    return result[0]

def time_dependent_auc(y_train: pd.DataFrame, y_test: pd.DataFrame, risk_scores: np.ndarray, 
                      times: List[float], time_col: str = 'time', event_col: str = 'event') -> Dict[float, float]:
    """
    计算时间依赖的AUC
    
    参数:
    -----
    y_train: pd.DataFrame
        训练集真实值，包含时间和事件列
    y_test: pd.DataFrame
        测试集真实值，包含时间和事件列
    risk_scores: np.ndarray
        预测的风险得分
    times: List[float]
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    Dict[float, float]
        时间点到AUC值的映射
    """
    from sksurv.metrics import cumulative_dynamic_auc
    
    # 转换为sksurv格式
    y_train_sksurv = np.array([(e, t) for e, t in zip(y_train[event_col], y_train[time_col])], 
                             dtype=[('event', bool), ('time', float)])
    y_test_sksurv = np.array([(e, t) for e, t in zip(y_test[event_col], y_test[time_col])], 
                            dtype=[('event', bool), ('time', float)])
    
    # 计算时间依赖的AUC
    auc_scores, _ = cumulative_dynamic_auc(y_train_sksurv, y_test_sksurv, risk_scores, times)
    
    return {t: auc for t, auc in zip(times, auc_scores)}

def brier_score_at_times(y_train: pd.DataFrame, y_test: pd.DataFrame, survival_probs: np.ndarray, 
                        times: List[float], time_col: str = 'time', event_col: str = 'event') -> Dict[float, float]:
    """
    计算指定时间点的Brier分数
    
    参数:
    -----
    y_train: pd.DataFrame
        训练集真实值，包含时间和事件列
    y_test: pd.DataFrame
        测试集真实值，包含时间和事件列
    survival_probs: np.ndarray
        预测的生存概率，形状为(n_samples, n_times)
    times: List[float]
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    Dict[float, float]
        时间点到Brier分数的映射
    """
    from sksurv.metrics import brier_score
    
    # 转换为sksurv格式
    y_train_sksurv = np.array([(e, t) for e, t in zip(y_train[event_col], y_train[time_col])], 
                             dtype=[('event', bool), ('time', float)])
    y_test_sksurv = np.array([(e, t) for e, t in zip(y_test[event_col], y_test[time_col])], 
                            dtype=[('event', bool), ('time', float)])
    
    # 计算Brier分数
    brier_scores = {}
    
    for i, t in enumerate(times):
        try:
            score = brier_score(y_train_sksurv, y_test_sksurv, survival_probs[:, i], t)
            brier_scores[t] = score
        except Exception as e:
            logger.warning(f"计算时间点 {t} 的Brier分数时出错: {e}")
            brier_scores[t] = np.nan
    
    return brier_scores

def integrated_brier_score_at_times(y_train: pd.DataFrame, y_test: pd.DataFrame, 
                                  survival_func: Callable[[float], np.ndarray], 
                                  times: List[float], time_col: str = 'time', 
                                  event_col: str = 'event') -> float:
    """
    计算积分Brier分数
    
    参数:
    -----
    y_train: pd.DataFrame
        训练集真实值，包含时间和事件列
    y_test: pd.DataFrame
        测试集真实值，包含时间和事件列
    survival_func: Callable[[float], np.ndarray]
        生存函数，接受时间点，返回生存概率
    times: List[float]
        评估时间点
    time_col: str, 默认 'time'
        时间列名
    event_col: str, 默认 'event'
        事件列名
        
    返回:
    -----
    float
        积分Brier分数
    """
    from sksurv.metrics import integrated_brier_score
    
    # 转换为sksurv格式
    y_train_sksurv = np.array([(e, t) for e, t in zip(y_train[event_col], y_train[time_col])], 
                             dtype=[('event', bool), ('time', float)])
    y_test_sksurv = np.array([(e, t) for e, t in zip(y_test[event_col], y_test[time_col])], 
                            dtype=[('event', bool), ('time', float)])
    
    # 计算积分Brier分数
    try:
        score = integrated_brier_score(y_train_sksurv, y_test_sksurv, survival_func, times)
        return score
    except Exception as e:
        logger.warning(f"计算积分Brier分数时出错: {e}")
        return np.nan

def plot_time_dependent_auc(td_auc: Dict[float, float], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制时间依赖的AUC曲线
    
    参数:
    -----
    td_auc: Dict[float, float]
        时间点到AUC值的映射
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 提取时间点和AUC值
    times = list(td_auc.keys())
    aucs = list(td_auc.values())
    
    # 绘制AUC曲线
    ax.plot(times, aucs, marker='o', linestyle='-', linewidth=2)
    
    # 添加参考线
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='随机预测')
    
    # 添加标题和标签
    ax.set_title('时间依赖的AUC曲线', fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    
    # 设置y轴范围
    ax.set_ylim([0.4, 1.0])
    
    # 添加图例
    ax.legend()
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_brier_scores(brier_scores: Dict[float, float], figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制Brier分数曲线
    
    参数:
    -----
    brier_scores: Dict[float, float]
        时间点到Brier分数的映射
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 提取时间点和Brier分数
    times = list(brier_scores.keys())
    scores = list(brier_scores.values())
    
    # 绘制Brier分数曲线
    ax.plot(times, scores, marker='o', linestyle='-', linewidth=2)
    
    # 添加参考线
    ax.axhline(0.25, color='red', linestyle='--', alpha=0.7, label='随机预测')
    
    # 添加标题和标签
    ax.set_title('Brier分数曲线', fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('Brier分数', fontsize=12)
    
    # 设置y轴范围
    ax.set_ylim([0.0, 0.3])
    
    # 添加图例
    ax.legend()
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def get_all_metrics(y_train: pd.DataFrame, y_test: pd.DataFrame, model, 
                   times: List[float], time_col: str = 'time', event_col: str = 'event') -> Dict[str, Any]:
    """
    计算所有评估指标
    
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
        
    返回:
    -----
    Dict[str, Any]
        所有评估指标
    """
    # 获取风险得分
    X_test = y_test.drop(columns=[time_col, event_col])
    risk_scores = model.predict_risk(X_test)
    
    # 计算C-index
    cindex = c_index(y_test, risk_scores, time_col, event_col)
    
    # 计算IPCW C-index
    try:
        cindex_ipcw_val = c_index_ipcw(y_train, y_test, risk_scores, time_col, event_col)
    except Exception as e:
        logger.warning(f"计算IPCW C-index时出错: {e}")
        cindex_ipcw_val = np.nan
    
    # 计算时间依赖的AUC
    try:
        td_auc = time_dependent_auc(y_train, y_test, risk_scores, times, time_col, event_col)
    except Exception as e:
        logger.warning(f"计算时间依赖的AUC时出错: {e}")
        td_auc = {t: np.nan for t in times}
    
    # 计算Brier分数
    try:
        # 获取生存概率
        survival_probs = model.predict(X_test, times)
        bs = brier_score_at_times(y_train, y_test, survival_probs, times, time_col, event_col)
        
        # 计算积分Brier分数
        ibs = integrated_brier_score_at_times(y_train, y_test, 
                                            lambda t: model.predict(X_test, [t])[:, 0], 
                                            times, time_col, event_col)
    except Exception as e:
        logger.warning(f"计算Brier分数时出错: {e}")
        bs = {t: np.nan for t in times}
        ibs = np.nan
    
    # 返回所有指标
    return {
        'c_index': cindex,
        'c_index_ipcw': cindex_ipcw_val,
        'time_dependent_auc': td_auc,
        'brier_scores': bs,
        'integrated_brier_score': ibs
    }

def create_metric_report(metrics: Dict[str, Any], times: List[float] = None, 
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    创建评估指标报告
    
    参数:
    -----
    metrics: Dict[str, Any]
        评估指标
    times: List[float], 可选
        评估时间点
    figsize: Tuple[int, int], 默认 (12, 10)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 添加标题
    fig.suptitle('生存分析模型评估报告', fontsize=16)
    
    # 创建子图
    gs = fig.add_gridspec(2, 2)
    
    # 添加C-index
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['C-index', 'IPCW C-index'], [metrics['c_index'], metrics['c_index_ipcw']], color=['skyblue', 'lightgreen'])
    ax1.set_ylim([0.5, 1.0])
    ax1.set_title('一致性指数', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加时间依赖的AUC
    if times is not None and 'time_dependent_auc' in metrics:
        ax2 = fig.add_subplot(gs[0, 1])
        td_auc = metrics['time_dependent_auc']
        times_list = list(td_auc.keys())
        aucs = list(td_auc.values())
        ax2.plot(times_list, aucs, marker='o', linestyle='-', linewidth=2)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='随机预测')
        ax2.set_title('时间依赖的AUC', fontsize=12)
        ax2.set_xlabel('时间', fontsize=10)
        ax2.set_ylabel('AUC', fontsize=10)
        ax2.set_ylim([0.4, 1.0])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 添加Brier分数
    if times is not None and 'brier_scores' in metrics:
        ax3 = fig.add_subplot(gs[1, 0])
        bs = metrics['brier_scores']
        times_list = list(bs.keys())
        scores = list(bs.values())
        ax3.plot(times_list, scores, marker='o', linestyle='-', linewidth=2)
        ax3.axhline(0.25, color='red', linestyle='--', alpha=0.7, label='随机预测')
        ax3.set_title('Brier分数', fontsize=12)
        ax3.set_xlabel('时间', fontsize=10)
        ax3.set_ylabel('Brier分数', fontsize=10)
        ax3.set_ylim([0.0, 0.3])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 添加积分Brier分数
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(['积分Brier分数'], [metrics['integrated_brier_score']], color='salmon')
    ax4.axhline(0.25, color='red', linestyle='--', alpha=0.7, label='随机预测')
    ax4.set_title('积分Brier分数', fontsize=12)
    ax4.set_ylim([0.0, 0.3])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig 