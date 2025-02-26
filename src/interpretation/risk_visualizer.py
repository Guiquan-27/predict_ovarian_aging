# -*- coding: utf-8 -*-
"""
个体风险可视化模块
提供用于可视化个体患者风险预测结果的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

# 尝试导入shap库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("未找到shap库，SHAP值分析功能将不可用。请使用pip install shap安装。")

from ..models.base_models import BaseSurvivalModel
from ..interpretation.shap_analysis import calculate_shap_values, check_shap_available

logger = logging.getLogger(__name__)

def plot_survival_curve(model: BaseSurvivalModel, 
                       patient_data: pd.DataFrame, 
                       times: Optional[List[float]] = None,
                       confidence_intervals: bool = True,
                       n_bootstrap: int = 100,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制个体患者的生存曲线
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    patient_data: pd.DataFrame
        患者数据，单行DataFrame
    times: List[float], 可选
        预测时间点，默认为None(使用模型的默认时间点)
    confidence_intervals: bool, 默认 True
        是否显示置信区间
    n_bootstrap: int, 默认 100
        Bootstrap样本数量，用于计算置信区间
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    if len(patient_data) != 1:
        raise ValueError("patient_data必须是单行DataFrame")
    
    # 预测生存概率
    survival_probs = model.predict(patient_data, times=times)
    
    # 如果times为None，使用模型的默认时间点
    if times is None:
        if hasattr(model, 'event_times_'):
            times = model.event_times_
        else:
            raise ValueError("未指定times，且模型没有默认时间点")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制生存曲线
    ax.step(times, survival_probs[0], where='post', color='blue', linewidth=2, label='预测生存概率')
    
    # 计算置信区间
    if confidence_intervals:
        # 使用Bootstrap方法计算置信区间
        bootstrap_probs = np.zeros((n_bootstrap, len(times)))
        
        for i in range(n_bootstrap):
            # 创建Bootstrap样本
            bootstrap_indices = np.random.choice(len(model.X_train_), len(model.X_train_), replace=True)
            X_bootstrap = model.X_train_.iloc[bootstrap_indices]
            y_bootstrap = model.y_train_.iloc[bootstrap_indices]
            
            # 训练Bootstrap模型
            bootstrap_model = model.__class__(name=f"bootstrap_{i}")
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # 预测生存概率
            bootstrap_probs[i] = bootstrap_model.predict(patient_data, times=times)[0]
        
        # 计算95%置信区间
        lower_ci = np.percentile(bootstrap_probs, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_probs, 97.5, axis=0)
        
        # 绘制置信区间
        ax.fill_between(times, lower_ci, upper_ci, color='blue', alpha=0.2, label='95%置信区间')
    
    # 添加标题和标签
    ax.set_title('个体患者生存曲线预测', fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('生存概率', fontsize=12)
    
    # 设置y轴范围
    ax.set_ylim([0.0, 1.05])
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_risk_timeline(model: BaseSurvivalModel, 
                      patient_data: pd.DataFrame,
                      time_points: List[float],
                      reference_population: Optional[pd.DataFrame] = None,
                      figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    绘制个体患者的风险时间线
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    patient_data: pd.DataFrame
        患者数据，单行DataFrame
    time_points: List[float]
        预测时间点
    reference_population: pd.DataFrame, 可选
        参考人群数据，默认为None(使用模型的训练数据)
    figsize: Tuple[int, int], 默认 (12, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    if len(patient_data) != 1:
        raise ValueError("patient_data必须是单行DataFrame")
    
    # 预测生存概率
    survival_probs = model.predict(patient_data, times=time_points)[0]
    risk_probs = 1 - survival_probs
    
    # 如果提供了参考人群，计算风险百分位
    risk_percentiles = None
    if reference_population is not None:
        # 预测参考人群的风险
        ref_survival_probs = model.predict(reference_population, times=time_points)
        ref_risk_probs = 1 - ref_survival_probs
        
        # 计算风险百分位
        risk_percentiles = np.zeros(len(time_points))
        for i, t in enumerate(time_points):
            risk_percentiles[i] = np.mean(ref_risk_probs[:, i] <= risk_probs[i]) * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制风险时间线
    ax.plot(time_points, risk_probs, marker='o', linestyle='-', linewidth=2, color='red', label='风险概率')
    
    # 添加风险百分位
    if risk_percentiles is not None:
        ax_twin = ax.twinx()
        ax_twin.plot(time_points, risk_percentiles, marker='s', linestyle='--', linewidth=1.5, 
                    color='blue', label='风险百分位')
        ax_twin.set_ylabel('风险百分位 (%)', fontsize=12, color='blue')
        ax_twin.tick_params(axis='y', colors='blue')
        ax_twin.set_ylim([0, 100])
    
    # 添加标题和标签
    ax.set_title('个体患者风险时间线', fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('风险概率', fontsize=12, color='red')
    ax.tick_params(axis='y', colors='red')
    
    # 设置y轴范围
    ax.set_ylim([0.0, 1.0])
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax.get_legend_handles_labels()
    if risk_percentiles is not None:
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_feature_contribution(model: BaseSurvivalModel, 
                             patient_data: pd.DataFrame,
                             time_point: Optional[float] = None,
                             top_n: int = 10,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制特征贡献瀑布图
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    patient_data: pd.DataFrame
        患者数据，单行DataFrame
    time_point: float, 可选
        评估时间点，默认为None(使用风险得分)
    top_n: int, 默认 10
        显示的特征数量
    figsize: Tuple[int, int], 默认 (12, 8)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    if len(patient_data) != 1:
        raise ValueError("patient_data必须是单行DataFrame")
    
    # 检查SHAP库是否可用
    check_shap_available()
    
    # 计算SHAP值
    shap_values, explainer = calculate_shap_values(model, patient_data, time_point=time_point)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 绘制SHAP瀑布图
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, 
                                         shap_values[0], 
                                         patient_data.iloc[0],
                                         max_display=top_n,
                                         show=False)
    
    # 添加标题
    if time_point is not None:
        plt.title(f'特征贡献瀑布图 (时间点: {time_point})', fontsize=14)
    else:
        plt.title('特征贡献瀑布图 (风险得分)', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def plot_risk_stratification(model: BaseSurvivalModel, 
                            patient_data: pd.DataFrame,
                            reference_population: pd.DataFrame,
                            time_point: float,
                            risk_groups: int = 3,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制风险分层图
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    patient_data: pd.DataFrame
        患者数据，单行DataFrame
    reference_population: pd.DataFrame
        参考人群数据
    time_point: float
        评估时间点
    risk_groups: int, 默认 3
        风险组数量
    figsize: Tuple[int, int], 默认 (10, 6)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    if len(patient_data) != 1:
        raise ValueError("patient_data必须是单行DataFrame")
    
    # 预测患者风险
    patient_survival = model.predict(patient_data, times=[time_point])[0][0]
    patient_risk = 1 - patient_survival
    
    # 预测参考人群风险
    ref_survival = model.predict(reference_population, times=[time_point])[:, 0]
    ref_risk = 1 - ref_survival
    
    # 计算风险分位数
    if risk_groups == 3:
        # 低、中、高风险
        thresholds = [np.percentile(ref_risk, 33.33), np.percentile(ref_risk, 66.67)]
        group_names = ['低风险', '中风险', '高风险']
    elif risk_groups == 4:
        # 低、中低、中高、高风险
        thresholds = [np.percentile(ref_risk, 25), np.percentile(ref_risk, 50), np.percentile(ref_risk, 75)]
        group_names = ['低风险', '中低风险', '中高风险', '高风险']
    else:
        # 默认为3组
        thresholds = [np.percentile(ref_risk, 33.33), np.percentile(ref_risk, 66.67)]
        group_names = ['低风险', '中风险', '高风险']
    
    # 确定患者风险组
    patient_group = 0
    for i, threshold in enumerate(thresholds):
        if patient_risk > threshold:
            patient_group = i + 1
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制参考人群风险分布
    sns.histplot(ref_risk, bins=30, kde=True, ax=ax, color='lightgray', alpha=0.7)
    
    # 添加风险阈值线
    for i, threshold in enumerate(thresholds):
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'阈值 {i+1}: {threshold:.3f}')
    
    # 添加患者风险线
    ax.axvline(x=patient_risk, color='blue', linestyle='-', linewidth=2, 
              label=f'患者风险: {patient_risk:.3f} ({group_names[patient_group]})')
    
    # 添加风险组标签
    y_max = ax.get_ylim()[1]
    x_min = ax.get_xlim()[0]
    x_max = ax.get_xlim()[1]
    
    # 添加风险组区域
    if risk_groups == 3:
        ax.fill_between([x_min, thresholds[0]], 0, y_max, color='green', alpha=0.1)
        ax.fill_between([thresholds[0], thresholds[1]], 0, y_max, color='yellow', alpha=0.1)
        ax.fill_between([thresholds[1], x_max], 0, y_max, color='red', alpha=0.1)
        
        # 添加风险组文本
        ax.text((x_min + thresholds[0])/2, y_max*0.9, '低风险', ha='center', fontsize=12)
        ax.text((thresholds[0] + thresholds[1])/2, y_max*0.9, '中风险', ha='center', fontsize=12)
        ax.text((thresholds[1] + x_max)/2, y_max*0.9, '高风险', ha='center', fontsize=12)
    elif risk_groups == 4:
        ax.fill_between([x_min, thresholds[0]], 0, y_max, color='green', alpha=0.1)
        ax.fill_between([thresholds[0], thresholds[1]], 0, y_max, color='lightgreen', alpha=0.1)
        ax.fill_between([thresholds[1], thresholds[2]], 0, y_max, color='yellow', alpha=0.1)
        ax.fill_between([thresholds[2], x_max], 0, y_max, color='red', alpha=0.1)
        
        # 添加风险组文本
        ax.text((x_min + thresholds[0])/2, y_max*0.9, '低风险', ha='center', fontsize=12)
        ax.text((thresholds[0] + thresholds[1])/2, y_max*0.9, '中低风险', ha='center', fontsize=12)
        ax.text((thresholds[1] + thresholds[2])/2, y_max*0.9, '中高风险', ha='center', fontsize=12)
        ax.text((thresholds[2] + x_max)/2, y_max*0.9, '高风险', ha='center', fontsize=12)
    
    # 添加标题和标签
    ax.set_title(f'风险分层 (时间点: {time_point})', fontsize=14)
    ax.set_xlabel('风险概率', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    
    # 添加图例
    ax.legend(loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def generate_patient_risk_report(model: BaseSurvivalModel, 
                                patient_data: pd.DataFrame,
                                reference_population: Optional[pd.DataFrame] = None,
                                time_points: List[float] = [1, 3, 5],
                                figsize: Tuple[int, int] = (15, 20)) -> plt.Figure:
    """
    生成患者风险综合报告
    
    参数:
    -----
    model: BaseSurvivalModel
        训练好的生存分析模型
    patient_data: pd.DataFrame
        患者数据，单行DataFrame
    reference_population: pd.DataFrame, 可选
        参考人群数据，默认为None(使用模型的训练数据)
    time_points: List[float], 默认 [1, 3, 5]
        预测时间点
    figsize: Tuple[int, int], 默认 (15, 20)
        图形大小
        
    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    if len(patient_data) != 1:
        raise ValueError("patient_data必须是单行DataFrame")
    
    # 如果未提供参考人群，使用模型的训练数据
    if reference_population is None:
        if hasattr(model, 'X_train_'):
            reference_population = model.X_train_
        else:
            logger.warning("未提供参考人群，且模型没有训练数据，部分图表将不可用")
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 设置网格
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1.5, 1])
    
    # 添加标题
    fig.suptitle('个体患者风险预测报告', fontsize=16, y=0.98)
    
    # 1. 生存曲线
    ax1 = fig.add_subplot(gs[0, 0])
    survival_fig = plot_survival_curve(model, patient_data, confidence_intervals=True)
    for ax in survival_fig.get_axes():
        for item in ax.get_children():
            if isinstance(item, plt.Line2D) or isinstance(item, plt.PolyCollection):
                item_copy = item.copy()
                ax1.add_artist(item_copy)
    ax1.set_title('生存曲线预测', fontsize=12)
    ax1.set_xlabel('时间', fontsize=10)
    ax1.set_ylabel('生存概率', fontsize=10)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, alpha=0.3)
    plt.close(survival_fig)
    
    # 2. 风险时间线
    ax2 = fig.add_subplot(gs[0, 1])
    if reference_population is not None:
        risk_fig = plot_risk_timeline(model, patient_data, time_points, reference_population)
        for ax in risk_fig.get_axes():
            for item in ax.get_children():
                if isinstance(item, plt.Line2D):
                    item_copy = item.copy()
                    ax2.add_artist(item_copy)
        ax2.set_title('风险时间线', fontsize=12)
        ax2.set_xlabel('时间', fontsize=10)
        ax2.set_ylabel('风险概率', fontsize=10, color='red')
        ax2.tick_params(axis='y', colors='red')
        ax2.set_ylim([0.0, 1.0])
        ax2.grid(True, alpha=0.3)
        plt.close(risk_fig)
    
    # 3. 风险分层
    ax3 = fig.add_subplot(gs[1, :])
    if reference_population is not None:
        strat_fig = plot_risk_stratification(model, patient_data, reference_population, time_points[1])
        for ax in strat_fig.get_axes():
            for item in ax.get_children():
                ax3.add_artist(item.copy())
        ax3.set_title(f'风险分层 (时间点: {time_points[1]})', fontsize=12)
        ax3.set_xlabel('风险概率', fontsize=10)
        ax3.set_ylabel('频率', fontsize=10)
        plt.close(strat_fig)
    
    # 4. 特征贡献
    ax4 = fig.add_subplot(gs[2, :])
    try:
        contrib_fig = plot_feature_contribution(model, patient_data, time_point=time_points[1])
        for ax in contrib_fig.get_axes():
            for item in ax.get_children():
                ax4.add_artist(item.copy())
        ax4.set_title(f'特征贡献 (时间点: {time_points[1]})', fontsize=12)
        plt.close(contrib_fig)
    except:
        ax4.text(0.5, 0.5, "特征贡献分析不可用", ha='center', va='center', fontsize=12)
        ax4.axis('off')
    
    # 5. 患者信息表格
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # 提取患者关键信息
    patient_info = patient_data.iloc[0].to_dict()
    
    # 选择最多10个关键特征
    if len(patient_info) > 10:
        # 尝试使用SHAP值选择最重要的特征
        try:
            shap_values, _ = calculate_shap_values(model, patient_data)
            feature_importance = np.abs(shap_values[0])
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [patient_data.columns[i] for i in top_indices]
            patient_info = {k: patient_info[k] for k in top_features}
        except:
            # 如果SHAP值计算失败，简单选择前10个特征
            patient_info = dict(list(patient_info.items())[:10])
    
    # 创建表格数据
    table_data = []
    for feature, value in patient_info.items():
        if isinstance(value, (int, float)):
            table_data.append([feature, f"{value:.2f}"])
        else:
            table_data.append([feature, str(value)])
    
    # 添加风险预测结果
    survival_probs = model.predict(patient_data, times=time_points)[0]
    risk_probs = 1 - survival_probs
    
    for i, t in enumerate(time_points):
        table_data.append([f"{t}年风险", f"{risk_probs[i]:.2%}"])
    
    # 绘制表格
    table = ax5.table(cellText=table_data, colLabels=['特征', '值'], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig 