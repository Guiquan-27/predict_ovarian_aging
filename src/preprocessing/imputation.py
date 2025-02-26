# -*- coding: utf-8 -*-
"""
缺失值处理模块
提供多种缺失值处理方法，包括基本插补和多重插补
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # 必须导入这个才能使用IterativeImputer
from sklearn.impute import IterativeImputer
import os
import tempfile
from joblib import Parallel, delayed

# 设置日志
logger = logging.getLogger(__name__)

def simple_imputation(df: pd.DataFrame, 
                     strategy: str = 'mean',
                     categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
    """
    使用sklearn的SimpleImputer进行简单插补

    参数:
    -----
    df: pd.DataFrame
        含缺失值的数据框
    strategy: str, 默认 'mean'
        数值列插补策略，可选 'mean', 'median', 'most_frequent', 'constant'
    categorical_strategy: str, 默认 'most_frequent'
        分类列插补策略，可选 'most_frequent', 'constant'

    返回:
    -----
    pd.DataFrame
        插补后的数据框
    """
    logger.info(f"使用SimpleImputer进行插补，数值列策略: {strategy}, 分类列策略: {categorical_strategy}")
    
    # 创建结果DataFrame的副本
    imputed_df = df.copy()
    
    # 分离数值列和分类列
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 数值列插补
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy=strategy)
        imputed_df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
    # 分类列插补
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        imputed_df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    logger.info("简单插补完成")
    return imputed_df

def knn_imputation(df: pd.DataFrame, 
                  n_neighbors: int = 5, 
                  weights: str = 'uniform') -> pd.DataFrame:
    """
    使用KNN方法进行缺失值插补

    参数:
    -----
    df: pd.DataFrame
        含缺失值的数据框
    n_neighbors: int, 默认 5
        KNN算法中的邻居数
    weights: str, 默认 'uniform'
        权重类型, 'uniform' 或 'distance'

    返回:
    -----
    pd.DataFrame
        插补后的数据框
    """
    logger.info(f"使用KNN算法进行插补, n_neighbors={n_neighbors}, weights={weights}")
    
    # 保存原始列名和索引
    columns = df.columns
    index = df.index
    
    # 分离数值列和分类列
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object', 'category'])
    
    # 对分类变量进行独热编码
    if not categorical_df.empty:
        categorical_df = pd.get_dummies(categorical_df)
    
    # 合并数值和编码后的分类变量
    combined_df = pd.concat([numeric_df, categorical_df], axis=1)
    
    # 使用KNN插补
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_array = imputer.fit_transform(combined_df)
    
    # 将结果转换回原始DataFrame结构
    # 注意：这不会恢复独热编码的分类变量，仅保留数值列
    imputed_df = pd.DataFrame(imputed_array[:, :len(numeric_df.columns)], 
                             index=index, 
                             columns=numeric_df.columns)
    
    # 使用最常见值填充分类变量
    if not categorical_df.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        imputed_cats = pd.DataFrame(
            cat_imputer.fit_transform(df[categorical_df.columns.tolist()]), 
            index=index,
            columns=categorical_df.columns.tolist()
        )
        imputed_df = pd.concat([imputed_df, imputed_cats], axis=1)
    
    # 确保列顺序与原始数据框一致
    imputed_df = imputed_df[columns]
    
    logger.info("KNN插补完成")
    return imputed_df

def iterative_imputation(df: pd.DataFrame, 
                        max_iter: int = 10, 
                        random_state: int = 42) -> pd.DataFrame:
    """
    使用迭代插补(MICE)进行缺失值处理

    参数:
    -----
    df: pd.DataFrame
        含缺失值的数据框
    max_iter: int, 默认 10
        最大迭代次数
    random_state: int, 默认 42
        随机种子

    返回:
    -----
    pd.DataFrame
        插补后的数据框
    """
    logger.info(f"使用迭代插补算法(MICE), max_iter={max_iter}")
    
    # 保存原始列名和索引
    columns = df.columns
    index = df.index
    
    # 分离数值列和分类列
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 创建结果DataFrame的副本
    imputed_df = df.copy()
    
    # 数值列使用迭代插补
    if numeric_cols:
        mice_imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])
    
    # 分类列使用最常见值插补
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        imputed_df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    logger.info("迭代插补完成")
    return imputed_df

def impute_with_mice(df: pd.DataFrame, k: int = 5, seed: int = 42) -> List[pd.DataFrame]:
    """
    使用rpy2调用R的miceRanger进行多重插补

    参数:
    -----
    df: pd.DataFrame
        含缺失值的数据框
    k: int, 默认 5
        生成的插补数据集数量
    seed: int, 默认 42
        随机种子

    返回:
    -----
    List[pd.DataFrame]
        包含k个插补后数据框的列表
    """
    logger.info(f"使用R的miceRanger包进行多重插补, k={k}")
    
    try:
        # 导入必要的R包
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        # 激活pandas和R的转换
        pandas2ri.activate()
        
        # 导入R包
        base = importr('base')
        mice_ranger = importr('miceRanger')
        
        # 设置随机种子
        base.set_seed(seed)
        
        # 转换Python DataFrame到R
        r_df = pandas2ri.py2rpy(df)
        
        # 执行mice插补
        mice_result = mice_ranger.miceRanger(
            r_df,
            m=k,
            returnModels=False,
            verbose=True
        )
        
        # 从R中提取插补后的数据集
        imputed_dfs = []
        for i in range(1, k+1):
            # 从mice结果中获取第i个插补数据集
            imputed_r_df = mice_ranger.completeData(mice_result, dataset=i)
            # 转换回Python DataFrame
            imputed_df = pandas2ri.rpy2py(imputed_r_df)
            imputed_dfs.append(imputed_df)
        
        logger.info(f"成功生成{k}个插补数据集")
        return imputed_dfs
        
    except Exception as e:
        logger.error(f"R的miceRanger插补失败: {str(e)}")
        logger.warning("切换到Python的迭代插补方法")
        
        # 如果R调用失败，使用Python的方法生成多个插补数据集
        return [iterative_imputation(df, random_state=seed+i) for i in range(k)]

def combine_imputed_datasets(imputed_dfs: List[pd.DataFrame], 
                            method: str = 'robins_rule') -> pd.DataFrame:
    """
    合并多个插补后的数据集

    参数:
    -----
    imputed_dfs: List[pd.DataFrame]
        插补后的数据框列表
    method: str, 默认 'robins_rule'
        合并方法，可选 'mean', 'median', 'robins_rule'

    返回:
    -----
    pd.DataFrame
        合并后的数据框
    """
    logger.info(f"使用{method}方法合并{len(imputed_dfs)}个插补数据集")
    
    if not imputed_dfs:
        raise ValueError("输入的插补数据集列表为空")
    
    # 获取第一个数据框作为模板
    df_template = imputed_dfs[0].copy()
    
    # 分离数值列和分类列
    numeric_cols = df_template.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_template.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 对数值列应用合并策略
    for col in numeric_cols:
        # 从所有插补数据集中收集该列的值
        col_values = [df[col].values for df in imputed_dfs]
        
        if method == 'mean':
            # 计算每个位置的平均值
            combined_values = np.mean(col_values, axis=0)
        elif method == 'median':
            # 计算每个位置的中位数
            combined_values = np.median(col_values, axis=0)
        elif method == 'robins_rule':
            # 实现Rubin's规则，对于数值变量使用平均值
            combined_values = np.mean(col_values, axis=0)
            
            # 计算标准误差
            variances = [np.var(df[col]) for df in imputed_dfs]
            between_var = np.var([np.mean(df[col]) for df in imputed_dfs])
            within_var = np.mean(variances)
            total_var = within_var + between_var * (1 + 1/len(imputed_dfs))
            
            logger.debug(f"列 {col} 的总方差: {total_var}")
        else:
            raise ValueError(f"不支持的合并方法: {method}")
        
        # 更新模板数据框中的值
        df_template[col] = combined_values
    
    # 对分类列使用众数
    for col in categorical_cols:
        # 对每个位置找出众数
        mode_values = []
        for i in range(len(df_template)):
            values = [df[col].iloc[i] for df in imputed_dfs]
            # 找出最常见的值
            value_counts = pd.Series(values).value_counts()
            mode_value = value_counts.index[0]
            mode_values.append(mode_value)
        
        # 更新模板数据框中的值
        df_template[col] = mode_values
    
    logger.info("合并插补数据集完成")
    return df_template

def parallel_imputation(df: pd.DataFrame, 
                       method: str = 'simple', 
                       k: int = 5, 
                       n_jobs: int = -1, 
                       **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    并行执行插补，根据不同方法选择合适的策略

    参数:
    -----
    df: pd.DataFrame
        含缺失值的数据框
    method: str, 默认 'simple'
        插补方法，可选 'simple', 'knn', 'iterative', 'mice'
    k: int, 默认 5
        生成的插补数据集数量 (用于'mice'方法)
    n_jobs: int, 默认 -1
        并行作业数量，-1表示使用所有可用核心
    **kwargs: 
        传递给各插补方法的额外参数

    返回:
    -----
    Union[pd.DataFrame, List[pd.DataFrame]]
        单个合并后的数据框或多个插补数据框的列表
    """
    logger.info(f"使用{method}方法进行并行插补")
    
    if method == 'simple':
        return simple_imputation(df, **kwargs)
    
    elif method == 'knn':
        return knn_imputation(df, **kwargs)
    
    elif method == 'iterative':
        return iterative_imputation(df, **kwargs)
    
    elif method == 'mice':
        # 并行生成多个插补数据集
        seed = kwargs.get('seed', 42)
        
        if method == 'mice' and 'rpy2' in globals():
            # 使用R的miceRanger
            imputed_dfs = impute_with_mice(df, k=k, seed=seed)
        else:
            # 并行生成多个Python迭代插补数据集
            imputed_dfs = Parallel(n_jobs=n_jobs)(
                delayed(iterative_imputation)(df, random_state=seed+i, **kwargs) 
                for i in range(k)
            )
        
        # 是否合并数据集
        combine_method = kwargs.get('combine_method', 'robins_rule')
        if kwargs.get('combine', True):
            return combine_imputed_datasets(imputed_dfs, method=combine_method)
        else:
            return imputed_dfs
    
    else:
        raise ValueError(f"不支持的插补方法: {method}") 