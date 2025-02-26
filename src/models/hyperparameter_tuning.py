# -*- coding: utf-8 -*-
"""
超参数优化模块
提供用于优化生存分析模型超参数的功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from lifelines.utils import concordance_index
import joblib
import os
import time
from functools import partial

# 导入优化库
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from ..models.base_models import BaseSurvivalModel, create_model

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """超参数优化器基类"""
    
    def __init__(self, model_type: str, param_space: Dict[str, Any], 
                 cv: int = 5, n_trials: int = 100, random_state: int = 42,
                 metric: str = 'c_index', direction: str = 'maximize'):
        """
        初始化超参数优化器
        
        参数:
        -----
        model_type: str
            模型类型
        param_space: Dict[str, Any]
            参数空间
        cv: int, 默认 5
            交叉验证折数
        n_trials: int, 默认 100
            优化迭代次数
        random_state: int, 默认 42
            随机种子
        metric: str, 默认 'c_index'
            评估指标
        direction: str, 默认 'maximize'
            优化方向，'maximize'或'minimize'
        """
        self.model_type = model_type
        self.param_space = param_space
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.metric = metric
        self.direction = direction
        self.best_params = None
        self.best_score = None
        self.results = None
    
    def optimize(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> Dict[str, Any]:
        """
        执行超参数优化
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        **kwargs:
            额外的优化参数
            
        返回:
        -----
        Dict[str, Any]
            最佳参数
        """
        raise NotImplementedError("子类必须实现optimize方法")
    
    def evaluate_model(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.DataFrame, 
                      time_col: str = 'time', event_col: str = 'event') -> float:
        """
        评估给定参数的模型性能
        
        参数:
        -----
        params: Dict[str, Any]
            模型参数
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
            
        返回:
        -----
        float
            评估分数
        """
        # 创建交叉验证对象
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # 初始化分数列表
        scores = []
        
        # 对每个折进行训练和评估
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 创建并训练模型
            model = create_model(self.model_type, **params)
            model.fit(X_train, y_train, time_col=time_col, event_col=event_col)
            
            # 评估模型
            if self.metric == 'c_index':
                risk_scores = model.predict_risk(X_val)
                score = concordance_index(y_val[time_col], -risk_scores, y_val[event_col])
            else:
                # 其他评估指标
                raise ValueError(f"不支持的评估指标: {self.metric}")
            
            scores.append(score)
        
        # 计算平均分数
        mean_score = np.mean(scores)
        
        return mean_score
    
    def create_best_model(self) -> BaseSurvivalModel:
        """
        使用最佳参数创建模型
        
        返回:
        -----
        BaseSurvivalModel
            使用最佳参数创建的模型
        """
        if self.best_params is None:
            raise ValueError("尚未执行优化")
        
        return create_model(self.model_type, **self.best_params)
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制优化历史
        
        参数:
        -----
        figsize: Tuple[int, int], 默认 (10, 6)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if self.results is None:
            raise ValueError("尚未执行优化")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制优化历史
        ax.plot(self.results['iteration'], self.results['value'], 'o-', color='blue', alpha=0.6)
        
        # 标记最佳点
        best_idx = self.results['value'].argmax() if self.direction == 'maximize' else self.results['value'].argmin()
        ax.plot(self.results['iteration'][best_idx], self.results['value'][best_idx], 'o', color='red', markersize=10)
        
        # 添加标题和标签
        ax.set_title('超参数优化历史', fontsize=14)
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel(self.metric, fontsize=12)
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        return fig


class OptunaOptimizer(HyperparameterOptimizer):
    """使用Optuna进行超参数优化"""
    
    def __init__(self, model_type: str, param_space: Dict[str, Any], 
                 cv: int = 5, n_trials: int = 100, random_state: int = 42,
                 metric: str = 'c_index', direction: str = 'maximize'):
        """
        初始化Optuna优化器
        
        参数:
        -----
        model_type: str
            模型类型
        param_space: Dict[str, Any]
            参数空间
        cv: int, 默认 5
            交叉验证折数
        n_trials: int, 默认 100
            优化迭代次数
        random_state: int, 默认 42
            随机种子
        metric: str, 默认 'c_index'
            评估指标
        direction: str, 默认 'maximize'
            优化方向，'maximize'或'minimize'
        """
        super().__init__(model_type, param_space, cv, n_trials, random_state, metric, direction)
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请使用pip install optuna安装")
    
    def optimize(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> Dict[str, Any]:
        """
        使用Optuna执行超参数优化
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        **kwargs:
            额外的优化参数
            
        返回:
        -----
        Dict[str, Any]
            最佳参数
        """
        logger.info(f"开始使用Optuna进行超参数优化，模型类型: {self.model_type}")
        
        # 设置随机种子
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 定义目标函数
        def objective(trial):
            # 从参数空间中采样参数
            params = {}
            for param_name, param_config in self.param_space.items():
                param_type = param_config['type']
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config.get('step', 1))
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
                elif param_type == 'bool':
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])
            
            # 评估模型
            score = self.evaluate_model(params, X, y, time_col, event_col)
            
            return score
        
        # 创建学习器
        study = optuna.create_study(
            direction='maximize' if self.direction == 'maximize' else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 执行优化
        study.optimize(objective, n_trials=self.n_trials)
        
        # 获取最佳参数
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # 保存结果
        self.results = pd.DataFrame({
            'iteration': list(range(len(study.trials))),
            'value': [trial.value for trial in study.trials],
            'params': [trial.params for trial in study.trials]
        })
        
        logger.info(f"超参数优化完成，最佳{self.metric}: {self.best_score}")
        logger.info(f"最佳参数: {self.best_params}")
        
        return self.best_params


class HyperoptOptimizer(HyperparameterOptimizer):
    """使用Hyperopt进行超参数优化"""
    
    def __init__(self, model_type: str, param_space: Dict[str, Any], 
                 cv: int = 5, n_trials: int = 100, random_state: int = 42,
                 metric: str = 'c_index', direction: str = 'maximize'):
        """
        初始化Hyperopt优化器
        
        参数:
        -----
        model_type: str
            模型类型
        param_space: Dict[str, Any]
            参数空间
        cv: int, 默认 5
            交叉验证折数
        n_trials: int, 默认 100
            优化迭代次数
        random_state: int, 默认 42
            随机种子
        metric: str, 默认 'c_index'
            评估指标
        direction: str, 默认 'maximize'
            优化方向，'maximize'或'minimize'
        """
        super().__init__(model_type, param_space, cv, n_trials, random_state, metric, direction)
        
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt未安装，请使用pip install hyperopt安装")
    
    def _convert_param_space(self) -> Dict[str, Any]:
        """
        将参数空间转换为Hyperopt格式
        
        返回:
        -----
        Dict[str, Any]
            Hyperopt格式的参数空间
        """
        hyperopt_space = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                hyperopt_space[param_name] = hp.choice(param_name, param_config['values'])
            elif param_type == 'int':
                if param_config.get('log', False):
                    hyperopt_space[param_name] = hp.qloguniform(
                        param_name, 
                        np.log(param_config['low']), 
                        np.log(param_config['high']), 
                        q=param_config.get('step', 1)
                    )
                else:
                    hyperopt_space[param_name] = hp.quniform(
                        param_name, 
                        param_config['low'], 
                        param_config['high'], 
                        q=param_config.get('step', 1)
                    )
            elif param_type == 'float':
                if param_config.get('log', False):
                    hyperopt_space[param_name] = hp.loguniform(
                        param_name, 
                        np.log(param_config['low']), 
                        np.log(param_config['high'])
                    )
                else:
                    hyperopt_space[param_name] = hp.uniform(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
            elif param_type == 'bool':
                hyperopt_space[param_name] = hp.choice(param_name, [True, False])
        
        return hyperopt_space
    
    def optimize(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> Dict[str, Any]:
        """
        使用Hyperopt执行超参数优化
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        **kwargs:
            额外的优化参数
            
        返回:
        -----
        Dict[str, Any]
            最佳参数
        """
        logger.info(f"开始使用Hyperopt进行超参数优化，模型类型: {self.model_type}")
        
        # 转换参数空间
        hyperopt_space = self._convert_param_space()
        
        # 定义目标函数
        def objective(params):
            # 处理整数参数
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = int(params[param_name])
            
            # 评估模型
            score = self.evaluate_model(params, X, y, time_col, event_col)
            
            # 根据优化方向返回结果
            if self.direction == 'maximize':
                return {'loss': -score, 'status': STATUS_OK}
            else:
                return {'loss': score, 'status': STATUS_OK}
        
        # 创建Trials对象
        trials = Trials()
        
        # 执行优化
        best = fmin(
            fn=objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        # 处理最佳参数
        from hyperopt.pyll.base import scope
        from hyperopt.pyll.stochastic import sample
        
        # 获取最佳参数
        self.best_params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'categorical':
                self.best_params[param_name] = param_config['values'][best[param_name]]
            elif param_config['type'] == 'int':
                self.best_params[param_name] = int(best[param_name])
            else:
                self.best_params[param_name] = best[param_name]
        
        # 获取最佳分数
        if self.direction == 'maximize':
            self.best_score = -min(trials.losses())
        else:
            self.best_score = min(trials.losses())
        
        # 保存结果
        self.results = pd.DataFrame({
            'iteration': list(range(len(trials.trials))),
            'value': [-loss if self.direction == 'maximize' else loss for loss in trials.losses()],
            'params': [trial['misc']['vals'] for trial in trials.trials]
        })
        
        logger.info(f"超参数优化完成，最佳{self.metric}: {self.best_score}")
        logger.info(f"最佳参数: {self.best_params}")
        
        return self.best_params


def create_optimizer(optimizer_type: str, model_type: str, param_space: Dict[str, Any], **kwargs) -> HyperparameterOptimizer:
    """
    创建指定类型的超参数优化器
    
    参数:
    -----
    optimizer_type: str
        优化器类型，可选 'optuna', 'hyperopt'
    model_type: str
        模型类型
    param_space: Dict[str, Any]
        参数空间
    **kwargs:
        传递给优化器构造函数的参数
        
    返回:
    -----
    HyperparameterOptimizer
        创建的优化器实例
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'optuna':
        return OptunaOptimizer(model_type, param_space, **kwargs)
    elif optimizer_type == 'hyperopt':
        return HyperoptOptimizer(model_type, param_space, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def get_default_param_space(model_type: str) -> Dict[str, Any]:
    """
    获取指定模型类型的默认参数空间
    
    参数:
    -----
    model_type: str
        模型类型
        
    返回:
    -----
    Dict[str, Any]
        默认参数空间
    """
    model_type = model_type.lower()
    
    if model_type == 'cox':
        return {
            'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
            'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'penalizer': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True}
        }
    elif model_type == 'rsf':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'float', 'low': 0.1, 'high': 1.0}
        }
    elif model_type == 'coxnet':
        return {
            'alpha_min_ratio': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
            'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'n_alphas': {'type': 'int', 'low': 10, 'high': 100}
        }
    elif model_type == 'deepsurv':
        return {
            'hidden_layers': {'type': 'categorical', 'values': [[32], [64], [128], [64, 32], [128, 64], [128, 64, 32]]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
            'batch_size': {'type': 'categorical', 'values': [32, 64, 128, 256]},
            'batch_norm': {'type': 'bool'}
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def optimize_hyperparameters(model_class: Any, 
                            X_train: pd.DataFrame, 
                            y_train: pd.DataFrame,
                            X_val: pd.DataFrame = None,
                            y_val: pd.DataFrame = None,
                            time_col: str = 'time',
                            event_col: str = 'event',
                            param_space: Dict[str, Any] = None,
                            method: str = 'optuna',
                            n_trials: int = 100,
                            metric: str = 'c_index',
                            direction: str = 'maximize',
                            cv: int = 5,
                            random_state: int = None,
                            n_jobs: int = -1) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for survival models
    
    Parameters:
    -----
    model_class: Any
        Model class to optimize
    X_train: pd.DataFrame
        Training feature matrix
    y_train: pd.DataFrame
        Training target with time and event columns
    X_val: pd.DataFrame, optional
        Validation feature matrix
    y_val: pd.DataFrame, optional
        Validation target with time and event columns
    time_col: str, default 'time'
        Time column name
    event_col: str, default 'event'
        Event column name
    param_space: Dict[str, Any], optional
        Dictionary of parameter space
    method: str, default 'optuna'
        Optimization method: 'optuna', 'hyperopt', 'grid', 'random'
    n_trials: int, default 100
        Number of optimization trials
    metric: str, default 'c_index'
        Evaluation metric: 'c_index', 'ibs', etc.
    direction: str, default 'maximize'
        Optimization direction: 'maximize' or 'minimize'
    cv: int, default 5
        Number of cross-validation folds
    random_state: int, optional
        Random seed
    n_jobs: int, default -1
        Number of parallel jobs
        
    Returns:
    -----
    Tuple[Dict[str, Any], float]
        Best parameters and best score
    """
    logger.info(f"Starting hyperparameter optimization with {method} method")
    
    # Get default parameter space if not provided
    if param_space is None:
        model_type = model_class.__name__.lower()
        if 'cox' in model_type:
            param_space = get_coxph_param_space()
        elif 'rsf' in model_type or 'forest' in model_type:
            param_space = get_rsf_param_space()
        elif 'boost' in model_type:
            param_space = get_boosting_param_space()
        elif 'deepsurv' in model_type:
            param_space = get_deepsurv_param_space()
        else:
            raise ValueError(f"No default parameter space for model type: {model_type}")
    
    # Define objective function
    def objective(params):
        # Create and train model
        model = model_class(**params)
        
        if X_val is not None and y_val is not None:
            # Use separate validation set
            model.fit(X_train, y_train, time_col=time_col, event_col=event_col)
            
            # Evaluate on validation set
            if metric == 'c_index':
                risk_scores = model.predict_risk(X_val)
                score = concordance_index(y_val[time_col], -risk_scores, y_val[event_col])
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:
            # Use cross-validation
            scores = []
            
            # Create cross-validation folds
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            
            for train_idx, test_idx in kf.split(X_train):
                # Split data
                X_cv_train = X_train.iloc[train_idx]
                y_cv_train = y_train.iloc[train_idx]
                X_cv_test = X_train.iloc[test_idx]
                y_cv_test = y_train.iloc[test_idx]
                
                # Train model
                model.fit(X_cv_train, y_cv_train, time_col=time_col, event_col=event_col)
                
                # Evaluate model
                if metric == 'c_index':
                    risk_scores = model.predict_risk(X_cv_test)
                    fold_score = concordance_index(y_cv_test[time_col], -risk_scores, y_cv_test[event_col])
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                scores.append(fold_score)
            
            # Average scores
            score = np.mean(scores)
        
        # Return score (direction will be handled by the optimizer)
        return score if direction == 'maximize' else -score

    # Create optimizer
    optimizer = create_optimizer(method, model_class.__name__.lower(), param_space)
    
    # Optimize
    best_params = optimizer.optimize(X_train, y_train, time_col, event_col)
    
    # Evaluate best model
    best_score = objective(best_params)
    
    return best_params, best_score 