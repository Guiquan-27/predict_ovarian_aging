# -*- coding: utf-8 -*-
"""
数据加载与初步检查模块
提供用于读取生存分析数据并进行基本质量检查的功能
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载生存分析数据，支持多种格式（CSV、TXT、Excel）

    参数:
    -----
    file_path: str
        数据文件路径

    返回:
    -----
    pd.DataFrame
        加载的数据集
    """
    logger.info(f"Loading data: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.csv', '.txt']:
            # Auto-detect separator
            df = pd.read_csv(file_path, sep=None, engine='python')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Data loaded successfully, shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查数据质量，包括缺失值、数据类型、异常值等

    参数:
    -----
    df: pd.DataFrame
        待检查的数据框

    返回:
    -----
    Dict[str, Any]
        数据质量报告
    """
    logger.info("正在检查数据质量...")
    
    # 基本信息
    n_rows, n_cols = df.shape
    
    # 缺失值检查
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / n_rows) * 100
    missing_cols = missing_count[missing_count > 0].index.tolist()
    
    # 数据类型检查
    dtypes = df.dtypes
    
    # 列类型分类
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # 检查可能的常量列（唯一值只有一个的列）
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    # 检查高基数分类变量
    high_cardinality_cols = [col for col in categorical_cols 
                             if df[col].nunique() > min(50, n_rows*0.5)]
    
    # 生成质量报告
    quality_report = {
        '基本信息': {
            '行数': n_rows, 
            '列数': n_cols,
            '内存占用(MB)': df.memory_usage(deep=True).sum() / (1024 * 1024)
        },
        '缺失值': {
            '缺失值总计': missing_count.sum(),
            '缺失百分比': missing_count.sum() / (n_rows * n_cols) * 100,
            '含缺失值的列': dict(zip(missing_cols, 
                            [(missing_count[col], missing_percent[col]) for col in missing_cols]))
        },
        '数据类型': {
            '数值型列': numeric_cols,
            '分类型列': categorical_cols,
            '时间型列': datetime_cols,
            '列数据类型': dict(zip(dtypes.index, dtypes.astype(str)))
        },
        '数据问题': {
            '常量列': constant_cols,
            '高基数分类列': high_cardinality_cols
        }
    }
    
    logger.info("数据质量检查完成")
    return quality_report

def check_survival_data(df: pd.DataFrame, 
                       time_col: str, 
                       event_col: str) -> Dict[str, Any]:
    """
    检查生存分析数据是否满足基本要求

    参数:
    -----
    df: pd.DataFrame
        数据框
    time_col: str
        时间列名
    event_col: str
        事件列名

    返回:
    -----
    Dict[str, Any]
        生存数据检查报告
    """
    logger.info("正在验证生存数据...")
    
    survival_check = {'有效': True, '警告': [], '错误': []}
    
    # 检查列是否存在
    if time_col not in df.columns:
        survival_check['有效'] = False
        survival_check['错误'].append(f"时间列 '{time_col}' 不存在")
    
    if event_col not in df.columns:
        survival_check['有效'] = False
        survival_check['错误'].append(f"事件列 '{event_col}' 不存在")
    
    # 如果必要的列不存在，直接返回
    if not survival_check['有效']:
        return survival_check
    
    # 检查时间列
    time_values = df[time_col]
    if not pd.api.types.is_numeric_dtype(time_values):
        survival_check['有效'] = False
        survival_check['错误'].append(f"时间列 '{time_col}' 不是数值类型")
    elif time_values.min() < 0:
        survival_check['有效'] = False
        survival_check['错误'].append(f"时间列 '{time_col}' 包含负值")
    
    # 检查事件列
    event_values = df[event_col]
    unique_events = event_values.unique()
    
    # 检查事件列是否只包含0和1
    if not set(unique_events).issubset({0, 1}):
        survival_check['有效'] = False
        survival_check['错误'].append(f"事件列 '{event_col}' 包含非二进制值: {unique_events}")
    
    # 检查事件发生率
    event_rate = event_values.mean()
    if event_rate < 0.05:
        survival_check['警告'].append(f"事件发生率很低 ({event_rate:.2%})，可能影响模型训练")
    elif event_rate > 0.95:
        survival_check['警告'].append(f"事件发生率很高 ({event_rate:.2%})，考虑反转事件定义")
    
    # 生存相关统计
    if survival_check['有效']:
        n_subjects = len(df)
        n_events = event_values.sum()
        event_rate = n_events / n_subjects
        median_followup = time_values.median()
        
        survival_check['统计'] = {
            '样本数': n_subjects,
            '事件数': n_events,
            '事件率': event_rate,
            '中位随访时间': median_followup,
            '时间范围': (time_values.min(), time_values.max())
        }
    
    return survival_check

def summarize_data(df: pd.DataFrame, 
                  time_col: Optional[str] = None, 
                  event_col: Optional[str] = None) -> Dict[str, Any]:
    """
    生成数据摘要统计

    参数:
    -----
    df: pd.DataFrame
        待分析的数据框
    time_col: str, 可选
        时间列名
    event_col: str, 可选
        事件列名

    返回:
    -----
    Dict[str, Any]
        数据摘要统计
    """
    logger.info("正在生成数据摘要统计...")
    
    # 数值型变量摘要
    numeric_df = df.select_dtypes(include=['number'])
    numeric_summary = numeric_df.describe().T
    
    # 添加更多统计量
    if not numeric_df.empty:
        numeric_summary['skewness'] = numeric_df.skew()
        numeric_summary['kurtosis'] = numeric_df.kurtosis()
        numeric_summary['missing'] = numeric_df.isnull().sum()
        numeric_summary['missing_pct'] = numeric_df.isnull().mean() * 100
    
    # 分类变量摘要
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_summary = {}
    
    for col in cat_cols:
        value_counts = df[col].value_counts()
        top_n = min(10, len(value_counts))
        
        categorical_summary[col] = {
            'unique_count': df[col].nunique(),
            'missing': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().mean() * 100,
            'top_values': value_counts.head(top_n).to_dict(),
            'top_pct': (value_counts.head(top_n) / len(df) * 100).to_dict()
        }
    
    # 生存相关摘要
    survival_summary = None
    if time_col is not None and event_col is not None:
        survival_summary = check_survival_data(df, time_col, event_col)
    
    summary = {
        '数值变量': numeric_summary.to_dict() if not numeric_df.empty else {},
        '分类变量': categorical_summary,
        '生存信息': survival_summary
    }
    
    logger.info("数据摘要统计生成完成")
    return summary

def split_features_target(df: pd.DataFrame, 
                         time_col: str, 
                         event_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将数据分为特征矩阵和目标变量(生存时间和事件状态)

    参数:
    -----
    df: pd.DataFrame
        数据框
    time_col: str
        时间列名
    event_col: str
        事件列名

    返回:
    -----
    Tuple[pd.DataFrame, pd.DataFrame]
        特征矩阵X和目标变量y(时间和事件)
    """
    # 提取特征矩阵，排除目标变量
    X = df.drop(columns=[time_col, event_col])
    
    # 创建目标变量DataFrame
    y = df[[time_col, event_col]].copy()
    
    return X, y 