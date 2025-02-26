#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主运行脚本
提供命令行接口，整合完整的生存分析工作流程
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import traceback

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from src.utils.logger import get_logger, setup_logger
from src.utils.visualization import set_visualization_style, save_figure
from src.preprocessing.data_loader import load_data
from src.preprocessing.imputation import impute_missing_values
from src.preprocessing.feature_engineering import create_features
from src.preprocessing.feature_selection import select_features
from src.models.base_models import CoxPHModel, RandomSurvivalForestModel
from src.models.deep_models import DeepSurvModel
from src.models.ensemble import EnsembleSurvivalModel
from src.models.hyperparameter_tuning import tune_hyperparameters
from src.evaluation.cross_validation import cross_validate
from src.evaluation.metrics import calculate_metrics, plot_metrics
from src.evaluation.calibration import calibration_curve, plot_calibration
from src.evaluation.decision_curve import decision_curve_analysis, plot_decision_curve
from src.interpretation.shap_analysis import generate_shap_report
from src.interpretation.feature_interaction import generate_interaction_report
from src.interpretation.risk_visualizer import generate_risk_report

# 创建日志记录器
logger = get_logger(name="main")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生存分析模型训练与评估")
    
    # 配置文件
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="配置文件路径")
    
    # 运行模式
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "interpret"],
                       default="train", help="运行模式")
    
    # 数据相关
    parser.add_argument("--data_path", type=str, help="数据文件路径")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    
    # 模型相关
    parser.add_argument("--model_type", type=str, 
                       choices=["cox", "rsf", "deepsurv", "ensemble"],
                       help="模型类型")
    parser.add_argument("--model_path", type=str, help="模型保存/加载路径")
    
    # 实验相关
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 解析参数
    args = parser.parse_args()
    
    return args

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
    -----
    config_path: str
        配置文件路径
        
    返回:
    -----
    Dict[str, Any]
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise

def setup_experiment(args, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    设置实验环境
    
    参数:
    -----
    args: argparse.Namespace
        命令行参数
    config: Dict[str, Any]
        配置字典
        
    返回:
    -----
    Tuple[str, Dict[str, Any]]
        (实验目录, 更新后的配置)
    """
    # 设置随机种子
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    np.random.seed(seed)
    
    # 设置实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = args.model_type or config.get('model', {}).get('type', 'unknown')
        experiment_name = f"{model_type}_{timestamp}"
    
    # 创建实验目录
    output_dir = args.output_dir or config.get('output_dir', 'results')
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    for subdir in ['models', 'figures', 'logs', 'metrics', 'reports']:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(experiment_dir, 'logs', 'experiment.log')
    logger = setup_logger(
        name="experiment",
        log_level=config.get('logging', {}).get('level', 'info'),
        log_file=log_file,
        use_color=True
    )
    
    # 保存配置
    config_file = os.path.join(experiment_dir, 'config.yaml')
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 更新配置
    updated_config = config.copy()
    updated_config['experiment_name'] = experiment_name
    updated_config['experiment_dir'] = experiment_dir
    updated_config['seed'] = seed
    
    # 命令行参数覆盖配置文件
    if args.data_path:
        updated_config['data'] = updated_config.get('data', {})
        updated_config['data']['path'] = args.data_path
    
    if args.model_type:
        updated_config['model'] = updated_config.get('model', {})
        updated_config['model']['type'] = args.model_type
    
    if args.model_path:
        updated_config['model'] = updated_config.get('model', {})
        updated_config['model']['path'] = args.model_path
    
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"实验目录: {experiment_dir}")
    logger.info(f"随机种子: {seed}")
    
    return experiment_dir, updated_config

def create_model(model_config: Dict[str, Any]) -> Any:
    """
    创建模型
    
    参数:
    -----
    model_config: Dict[str, Any]
        模型配置
        
    返回:
    -----
    Any
        创建的模型
    """
    model_type = model_config.get('type', 'cox')
    model_params = model_config.get('params', {})
    
    logger.info(f"创建模型: {model_type}")
    
    if model_type == 'cox':
        return CoxPHModel(name=model_config.get('name', 'CoxPH'), **model_params)
    elif model_type == 'rsf':
        return RandomSurvivalForestModel(name=model_config.get('name', 'RSF'), **model_params)
    elif model_type == 'deepsurv':
        return DeepSurvModel(name=model_config.get('name', 'DeepSurv'), **model_params)
    elif model_type == 'ensemble':
        # 创建基础模型
        base_models = []
        for base_config in model_config.get('base_models', []):
            base_model = create_model(base_config)
            base_models.append(base_model)
        
        # 创建集成模型
        return EnsembleSurvivalModel(
            base_models=base_models,
            name=model_config.get('name', 'Ensemble'),
            **model_params
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_model(config: Dict[str, Any], experiment_dir: str) -> Any:
    """
    训练模型
    
    参数:
    -----
    config: Dict[str, Any]
        配置字典
    experiment_dir: str
        实验目录
        
    返回:
    -----
    Any
        训练好的模型
    """
    logger.info("开始训练模型")
    start_time = time.time()
    
    # 加载数据
    data_config = config.get('data', {})
    data = load_data(
        data_path=data_config.get('path'),
        time_col=data_config.get('time_col', 'time'),
        event_col=data_config.get('event_col', 'event'),
        id_col=data_config.get('id_col'),
        feature_cols=data_config.get('feature_cols'),
        categorical_cols=data_config.get('categorical_cols'),
        test_size=data_config.get('test_size', 0.2),
        random_state=config.get('seed', 42)
    )
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 数据预处理
    preprocessing_config = config.get('preprocessing', {})
    
    # 缺失值填充
    if preprocessing_config.get('imputation', {}).get('enabled', True):
        imputation_config = preprocessing_config.get('imputation', {})
        X_train, X_test = impute_missing_values(
            X_train, X_test,
            strategy=imputation_config.get('strategy', 'mean'),
            categorical_strategy=imputation_config.get('categorical_strategy', 'most_frequent')
        )
    
    # 特征工程
    if preprocessing_config.get('feature_engineering', {}).get('enabled', True):
        feature_config = preprocessing_config.get('feature_engineering', {})
        X_train, X_test = create_features(
            X_train, X_test,
            interactions=feature_config.get('interactions', False),
            polynomials=feature_config.get('polynomials', False),
            degree=feature_config.get('degree', 2)
        )
    
    # 特征选择
    if preprocessing_config.get('feature_selection', {}).get('enabled', True):
        feature_selection_config = preprocessing_config.get('feature_selection', {})
        X_train, X_test = select_features(
            X_train, X_test, y_train,
            method=feature_selection_config.get('method', 'univariate'),
            k=feature_selection_config.get('k', 20),
            threshold=feature_selection_config.get('threshold', 0.05)
        )
    
    # 创建模型
    model_config = config.get('model', {})
    model = create_model(model_config)
    
    # 超参数调优
    if config.get('hyperparameter_tuning', {}).get('enabled', False):
        tuning_config = config.get('hyperparameter_tuning', {})
        model = tune_hyperparameters(
            model, X_train, y_train,
            param_grid=tuning_config.get('param_grid', {}),
            cv=tuning_config.get('cv', 5),
            scoring=tuning_config.get('scoring', 'concordance_index'),
            n_jobs=tuning_config.get('n_jobs', -1)
        )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 保存模型
    model_path = os.path.join(experiment_dir, 'models', f"{model.name}.pkl")
    model.save(model_path)
    logger.info(f"模型已保存: {model_path}")
    
    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
    
    return model, data

def evaluate_model(model: Any, data: Dict[str, Any], config: Dict[str, Any], experiment_dir: str) -> Dict[str, Any]:
    """
    评估模型
    
    参数:
    -----
    model: Any
        训练好的模型
    data: Dict[str, Any]
        数据字典
    config: Dict[str, Any]
        配置字典
    experiment_dir: str
        实验目录
        
    返回:
    -----
    Dict[str, Any]
        评估结果
    """
    logger.info("开始评估模型")
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # 设置可视化风格
    set_visualization_style()
    
    # 评估配置
    eval_config = config.get('evaluation', {})
    time_points = eval_config.get('time_points', [1, 3, 5])
    
    # 交叉验证
    if eval_config.get('cross_validation', {}).get('enabled', True):
        cv_config = eval_config.get('cross_validation', {})
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv_config.get('cv', 5),
            metrics=cv_config.get('metrics', ['concordance_index']),
            time_points=time_points
        )
        
        # 保存交叉验证结果
        cv_path = os.path.join(experiment_dir, 'metrics', 'cross_validation.json')
        with open(cv_path, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, indent=4)
        
        logger.info(f"交叉验证结果: {cv_results['mean']}")
    
    # 计算测试集指标
    metrics = calculate_metrics(
        model, X_test, y_test,
        time_points=time_points,
        metrics=eval_config.get('metrics', ['concordance_index', 'brier_score', 'auc'])
    )
    
    # 保存指标
    metrics_path = os.path.join(experiment_dir, 'metrics', 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"测试集指标: {metrics}")
    
    # 绘制指标图
    metrics_fig = plot_metrics(metrics, time_points)
    save_figure(metrics_fig, "metrics", directory=os.path.join(experiment_dir, 'figures'))
    
    # 校准评估
    if eval_config.get('calibration', {}).get('enabled', True):
        for t in time_points:
            cal_curve = calibration_curve(model, X_test, y_test, time_point=t)
            cal_fig = plot_calibration(cal_curve, title=f"校准曲线 (t={t})")
            save_figure(cal_fig, f"calibration_t{t}", directory=os.path.join(experiment_dir, 'figures'))
    
    # 决策曲线分析
    if eval_config.get('decision_curve', {}).get('enabled', True):
        for t in time_points:
            dc_results = decision_curve_analysis(model, X_test, y_test, time_point=t)
            dc_fig = plot_decision_curve(dc_results, title=f"决策曲线 (t={t})")
            save_figure(dc_fig, f"decision_curve_t{t}", directory=os.path.join(experiment_dir, 'figures'))
    
    # 特征重要性
    if hasattr(model, 'plot_feature_importance'):
        importance_fig = model.plot_feature_importance()
        save_figure(importance_fig, "feature_importance", directory=os.path.join(experiment_dir, 'figures'))
    
    return metrics

def interpret_model(model: Any, data: Dict[str, Any], config: Dict[str, Any], experiment_dir: str) -> None:
    """
    解释模型
    
    参数:
    -----
    model: Any
        训练好的模型
    data: Dict[str, Any]
        数据字典
    config: Dict[str, Any]
        配置字典
    experiment_dir: str
        实验目录
    """
    logger.info("开始解释模型")
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # 设置可视化风格
    set_visualization_style()
    
    # 解释配置
    interpret_config = config.get('interpretation', {})
    time_points = interpret_config.get('time_points', [1, 3, 5])
    
    # SHAP值分析
    if interpret_config.get('shap', {}).get('enabled', True):
        try:
            shap_config = interpret_config.get('shap', {})
            shap_report = generate_shap_report(
                model, X_test,
                n_samples=shap_config.get('n_samples', 100),
                time_point=time_points[0]
            )
            save_figure(shap_report, "shap_report", directory=os.path.join(experiment_dir, 'figures'))
            logger.info("SHAP值分析完成")
        except Exception as e:
            logger.error(f"SHAP值分析失败: {str(e)}")
    
    # 特征交互分析
    if interpret_config.get('feature_interaction', {}).get('enabled', True):
        try:
            interaction_config = interpret_config.get('feature_interaction', {})
            interaction_report = generate_interaction_report(
                model, X_test,
                n_samples=interaction_config.get('n_samples', 100),
                time_point=time_points[0]
            )
            save_figure(interaction_report, "interaction_report", directory=os.path.join(experiment_dir, 'figures'))
            logger.info("特征交互分析完成")
        except Exception as e:
            logger.error(f"特征交互分析失败: {str(e)}")
    
    # 个体风险可视化
    if interpret_config.get('risk_visualization', {}).get('enabled', True):
        try:
            risk_config = interpret_config.get('risk_visualization', {})
            
            # 选择几个样本进行可视化
            n_samples = risk_config.get('n_samples', 3)
            sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                patient_data = X_test.iloc[[idx]]
                risk_report = generate_risk_report(
                    model, patient_data, X_train,
                    time_points=time_points
                )
                save_figure(risk_report, f"risk_report_sample{i}", directory=os.path.join(experiment_dir, 'figures'))
            
            logger.info("个体风险可视化完成")
        except Exception as e:
            logger.error(f"个体风险可视化失败: {str(e)}")

def predict(model: Any, data_path: str, config: Dict[str, Any], output_path: str) -> None:
    """
    使用模型进行预测
    
    参数:
    -----
    model: Any
        训练好的模型
    data_path: str
        预测数据路径
    config: Dict[str, Any]
        配置字典
    output_path: str
        输出路径
    """
    logger.info(f"开始预测: {data_path}")
    
    # 加载数据
    data_config = config.get('data', {})
    try:
        predict_data = pd.read_csv(data_path)
        logger.info(f"加载预测数据: {data_path}, 形状: {predict_data.shape}")
    except Exception as e:
        logger.error(f"加载预测数据失败: {str(e)}")
        return
    
    # 预处理
    # 这里应该应用与训练时相同的预处理步骤
    # 简化起见，假设数据已经预处理好
    
    # 预测
    time_points = config.get('prediction', {}).get('time_points', [1, 3, 5])
    survival_probs = model.predict(predict_data, times=time_points)
    
    # 创建结果DataFrame
    results = predict_data.copy()
    
    # 添加预测结果
    for i, t in enumerate(time_points):
        results[f'survival_prob_t{t}'] = survival_probs[:, i]
        results[f'risk_prob_t{t}'] = 1 - survival_probs[:, i]
    
    # 保存结果
    results.to_csv(output_path, index=False)
    logger.info(f"预测结果已保存: {output_path}")

def main():
    """Main function for running the survival analysis workflow"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default configuration
        config = {}
    
    # Setup experiment directory
    experiment_name = args.experiment_name or config.get('experiment_name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = args.output_dir or config.get('output_dir', 'outputs')
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logger
    log_dir = os.path.join(experiment_dir, 'logs')
    logger = setup_logger(
        name="survival_model",
        log_level=args.log_level or config.get('log_level', 'info'),
        log_dir=log_dir,
        use_color=True
    )
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Set random seed for reproducibility
    seed = args.seed or config.get('seed', 42)
    set_random_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Set visualization style
    set_visualization_style()
    
    # Determine execution mode
    if args.mode == "train":
        # Load data
        data_config = config.get('data', {})
        data = load_data(
            data_path=args.data_path or data_config.get('path'),
            time_col=data_config.get('time_col', 'time'),
            event_col=data_config.get('event_col', 'event'),
            id_col=data_config.get('id_col'),
            feature_cols=data_config.get('feature_cols'),
            categorical_cols=data_config.get('categorical_cols'),
            test_size=data_config.get('test_size', 0.2),
            random_state=seed
        )
        
        # Train model
        model, data = train_model(config, experiment_dir)
        
        # Evaluate model
        evaluate_model(model, data, config, experiment_dir)
        
        # Interpret model
        if config.get('interpretation', {}).get('enabled', True):
            interpret_model(model, data, config, experiment_dir)
    
    elif args.mode == "evaluate":
        # Load model
        model_path = args.model_path or config.get('model', {}).get('path')
        if not model_path:
            logger.error("未指定模型路径")
            return
        
        model_type = args.model_type or config.get('model', {}).get('type', 'cox')
        if model_type == 'cox':
            model = CoxPHModel.load(model_path)
        elif model_type == 'rsf':
            model = RandomSurvivalForestModel.load(model_path)
        elif model_type == 'deepsurv':
            model = DeepSurvModel.load(model_path)
        elif model_type == 'ensemble':
            model = EnsembleSurvivalModel.load(model_path)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return
        
        # Load data
        data_config = config.get('data', {})
        data = load_data(
            data_path=args.data_path or data_config.get('path'),
            time_col=data_config.get('time_col', 'time'),
            event_col=data_config.get('event_col', 'event'),
            id_col=data_config.get('id_col'),
            feature_cols=data_config.get('feature_cols'),
            categorical_cols=data_config.get('categorical_cols'),
            test_size=data_config.get('test_size', 0.2),
            random_state=seed
        )
        
        # Evaluate model
        evaluate_model(model, data, config, experiment_dir)
    
    elif args.mode == "predict":
        # Load model
        model_path = args.model_path or config.get('model', {}).get('path')
        if not model_path:
            logger.error("未指定模型路径")
            return
        
        model_type = args.model_type or config.get('model', {}).get('type', 'cox')
        if model_type == 'cox':
            model = CoxPHModel.load(model_path)
        elif model_type == 'rsf':
            model = RandomSurvivalForestModel.load(model_path)
        elif model_type == 'deepsurv':
            model = DeepSurvModel.load(model_path)
        elif model_type == 'ensemble':
            model = EnsembleSurvivalModel.load(model_path)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return
        
        # Predict
        data_path = args.data_path or config.get('prediction', {}).get('data_path')
        if not data_path:
            logger.error("未指定预测数据路径")
            return
        
        output_path = os.path.join(experiment_dir, 'predictions.csv')
        predict(model, data_path, config, output_path)
    
    elif args.mode == "interpret":
        # Load model
        model_path = args.model_path or config.get('model', {}).get('path')
        if not model_path:
            logger.error("未指定模型路径")
            return
        
        model_type = args.model_type or config.get('model', {}).get('type', 'cox')
        if model_type == 'cox':
            model = CoxPHModel.load(model_path)
        elif model_type == 'rsf':
            model = RandomSurvivalForestModel.load(model_path)
        elif model_type == 'deepsurv':
            model = DeepSurvModel.load(model_path)
        elif model_type == 'ensemble':
            model = EnsembleSurvivalModel.load(model_path)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return
        
        # Load data
        data_config = config.get('data', {})
        data = load_data(
            data_path=args.data_path or data_config.get('path'),
            time_col=data_config.get('time_col', 'time'),
            event_col=data_config.get('event_col', 'event'),
            id_col=data_config.get('id_col'),
            feature_cols=data_config.get('feature_cols'),
            categorical_cols=data_config.get('categorical_cols'),
            test_size=data_config.get('test_size', 0.2),
            random_state=seed
        )
        
        # Interpret model
        interpret_model(model, data, config, experiment_dir)
    
    logger.info(f"实验完成: {config['experiment_name']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        sys.exit(1) 