# -*- coding: utf-8 -*-
"""
特征选择模块
提供用于生存分析的特征选择方法，包括单变量Cox回归筛选、效应量筛选等
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def univariate_cox_selection(df: pd.DataFrame, 
                            time_col: str, 
                            event_col: str,
                            features: Optional[List[str]] = None,
                            alpha: float = 0.05,
                            fdr_correction: bool = True) -> pd.DataFrame:
    """
    使用单变量Cox回归进行特征筛选

    参数:
    -----
    df: pd.DataFrame
        数据框
    time_col: str
        时间列名
    event_col: str
        事件列名
    features: List[str], 可选
        要评估的特征列表，默认为None（使用所有非目标列）
    alpha: float, 默认 0.05
        显著性水平
    fdr_correction: bool, 默认 True
        是否进行FDR校正

    返回:
    -----
    pd.DataFrame
        包含Cox回归结果的数据框
    """
    logger.info("开始单变量Cox回归特征筛选")
    
    # 如果未指定特征，使用所有非目标列
    if features is None:
        features = [col for col in df.columns if col not in [time_col, event_col]]
    
    results = []
    
    # 对每个特征进行单变量Cox回归
    for feature in features:
        try:
            # 创建包含当前特征和目标变量的数据框
            temp_df = df[[feature, time_col, event_col]].copy()
            
            # 跳过包含缺失值的特征
            if temp_df[feature].isnull().any():
                logger.warning(f"特征 '{feature}' 包含缺失值，已跳过")
                continue
            
            # 拟合Cox模型
            cph = CoxPHFitter()
            cph.fit(temp_df, duration_col=time_col, event_col=event_col)
            
            # 提取结果
            summary = cph.summary
            
            # 保存结果
            results.append({
                'feature': feature,
                'coef': summary.loc[feature, 'coef'],
                'exp(coef)': summary.loc[feature, 'exp(coef)'],
                'se(coef)': summary.loc[feature, 'se(coef)'],
                'z': summary.loc[feature, 'z'],
                'p': summary.loc[feature, 'p'],
                'lower 0.95': summary.loc[feature, 'lower 0.95'],
                'upper 0.95': summary.loc[feature, 'upper 0.95']
            })
            
        except Exception as e:
            logger.error(f"处理特征 '{feature}' 时出错: {str(e)}")
    
    # 创建结果数据框
    result_df = pd.DataFrame(results)
    
    # 应用FDR校正
    if fdr_correction and not result_df.empty:
        _, corrected_pvals, _, _ = multipletests(
            result_df['p'].values, 
            alpha=alpha, 
            method='fdr_bh'  # Benjamini-Hochberg方法
        )
        result_df['p_adjusted'] = corrected_pvals
    
    # 按p值排序
    if not result_df.empty:
        p_col = 'p_adjusted' if fdr_correction else 'p'
        result_df = result_df.sort_values(by=p_col)
    
    logger.info(f"单变量Cox回归完成，共评估{len(results)}个特征")
    return result_df

def filter_by_effect_size(cox_results: pd.DataFrame,
                         hr_threshold_upper: float = 1.2,
                         hr_threshold_lower: float = 0.8,
                         p_threshold: float = 0.05,
                         p_col: str = 'p_adjusted') -> pd.DataFrame:
    """
    基于效应量(风险比)筛选特征

    参数:
    -----
    cox_results: pd.DataFrame
        单变量Cox回归结果
    hr_threshold_upper: float, 默认 1.2
        风险比上限阈值
    hr_threshold_lower: float, 默认 0.8
        风险比下限阈值
    p_threshold: float, 默认 0.05
        p值阈值
    p_col: str, 默认 'p_adjusted'
        使用的p值列名

    返回:
    -----
    pd.DataFrame
        筛选后的特征结果
    """
    logger.info(f"基于效应量筛选特征，HR阈值: [{hr_threshold_lower}, {hr_threshold_upper}]，p值阈值: {p_threshold}")
    
    # 确保p值列存在
    if p_col not in cox_results.columns:
        p_col = 'p'  # 回退到未校正的p值
        logger.warning(f"找不到列 '{p_col}'，使用未校正的p值")
    
    # 筛选显著的特征
    significant = cox_results[cox_results[p_col] < p_threshold].copy()
    
    # 基于风险比筛选
    filtered = significant[
        (significant['exp(coef)'] > hr_threshold_upper) | 
        (significant['exp(coef)'] < hr_threshold_lower)
    ]
    
    logger.info(f"效应量筛选完成，从{len(significant)}个显著特征中筛选出{len(filtered)}个特征")
    return filtered

def logrank_feature_selection(df: pd.DataFrame,
                             time_col: str,
                             event_col: str,
                             categorical_features: List[str],
                             alpha: float = 0.05,
                             fdr_correction: bool = True) -> pd.DataFrame:
    """
    使用Log-rank检验对分类特征进行筛选

    参数:
    -----
    df: pd.DataFrame
        数据框
    time_col: str
        时间列名
    event_col: str
        事件列名
    categorical_features: List[str]
        分类特征列表
    alpha: float, 默认 0.05
        显著性水平
    fdr_correction: bool, 默认 True
        是否进行FDR校正

    返回:
    -----
    pd.DataFrame
        包含Log-rank检验结果的数据框
    """
    logger.info("开始Log-rank特征筛选")
    
    results = []
    
    # 对每个分类特征进行Log-rank检验
    for feature in categorical_features:
        try:
            # 获取特征的唯一值
            unique_values = df[feature].unique()
            
            # 如果唯一值太多，跳过
            if len(unique_values) > 10:
                logger.warning(f"特征 '{feature}' 的唯一值过多 ({len(unique_values)})，已跳过")
                continue
            
            # 对每对类别进行比较
            for i, value1 in enumerate(unique_values):
                for value2 in unique_values[i+1:]:
                    # 获取两组数据
                    group1 = df[df[feature] == value1]
                    group2 = df[df[feature] == value2]
                    
                    # 执行Log-rank检验
                    result = logrank_test(
                        group1[time_col], group2[time_col],
                        group1[event_col], group2[event_col]
                    )
                    
                    # 保存结果
                    results.append({
                        'feature': feature,
                        'value1': value1,
                        'value2': value2,
                        'test_statistic': result.test_statistic,
                        'p': result.p_value
                    })
            
        except Exception as e:
            logger.error(f"处理特征 '{feature}' 时出错: {str(e)}")
    
    # 创建结果数据框
    result_df = pd.DataFrame(results)
    
    # 应用FDR校正
    if fdr_correction and not result_df.empty:
        _, corrected_pvals, _, _ = multipletests(
            result_df['p'].values, 
            alpha=alpha, 
            method='fdr_bh'  # Benjamini-Hochberg方法
        )
        result_df['p_adjusted'] = corrected_pvals
    
    # 按p值排序
    if not result_df.empty:
        p_col = 'p_adjusted' if fdr_correction else 'p'
        result_df = result_df.sort_values(by=p_col)
    
    logger.info(f"Log-rank检验完成，共评估{len(categorical_features)}个特征")
    return result_df

def plot_hazard_ratios(cox_results: pd.DataFrame, 
                      top_n: int = 20, 
                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    绘制风险比森林图

    参数:
    -----
    cox_results: pd.DataFrame
        Cox回归结果
    top_n: int, 默认 20
        显示的特征数量
    figsize: Tuple[int, int], 默认 (12, 10)
        图形大小

    返回:
    -----
    plt.Figure
        matplotlib图形对象
    """
    # 选择前N个特征
    if len(cox_results) > top_n:
        if 'p_adjusted' in cox_results.columns:
            plot_df = cox_results.sort_values('p_adjusted').head(top_n).copy()
        else:
            plot_df = cox_results.sort_values('p').head(top_n).copy()
    else:
        plot_df = cox_results.copy()
    
    # 按风险比排序
    plot_df = plot_df.sort_values('exp(coef)')
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制森林图
    y_pos = np.arange(len(plot_df))
    
    # 绘制风险比点和置信区间
    ax.scatter(plot_df['exp(coef)'], y_pos, marker='o', s=50, color='blue')
    
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot([row['lower 0.95'], row['upper 0.95']], [i, i], 'b-', alpha=0.6)
    
    # 添加垂直线表示HR=1
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # 设置Y轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    
    # 设置X轴为对数刻度
    ax.set_xscale('log')
    
    # 添加标题和标签
    ax.set_title('风险比森林图 (95% 置信区间)', fontsize=14)
    ax.set_xlabel('风险比 (HR)', fontsize=12)
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def group_features(features: List[str], 
                  clinical_prefix: List[str] = None, 
                  protein_prefix: List[str] = None) -> Dict[str, List[str]]:
    """
    将特征分组为临床特征和蛋白质特征

    参数:
    -----
    features: List[str]
        特征列表
    clinical_prefix: List[str], 可选
        临床特征前缀列表
    protein_prefix: List[str], 可选
        蛋白质特征前缀列表

    返回:
    -----
    Dict[str, List[str]]
        分组后的特征字典
    """
    if clinical_prefix is None:
        clinical_prefix = ['clinical_', 'demo_', 'lab_']
    
    if protein_prefix is None:
        protein_prefix = ['protein_', 'prot_', 'p_']
    
    clinical_features = []
    protein_features = []
    other_features = []
    
    for feature in features:
        if any(feature.startswith(prefix) for prefix in clinical_prefix):
            clinical_features.append(feature)
        elif any(feature.startswith(prefix) for prefix in protein_prefix):
            protein_features.append(feature)
        else:
            other_features.append(feature)
    
    return {
        'clinical': clinical_features,
        'protein': protein_features,
        'other': other_features
    } 