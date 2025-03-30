# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from bayes_opt import BayesianOptimization
from functools import partial
from joblib import Parallel, delayed
import multiprocessing
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import sys
import os
sys.stdout.reconfigure(write_through=True)

# 导入功能模块
from func.data_utils import load_data, load_variable_definitions, load_lasso_features, prepare_data
from func.utils import save_best_params, save_model

# 参数解析
parser = argparse.ArgumentParser(description='GBM Survival Model Selection with Bayesian Optimization')
parser.add_argument('--seed', type=int, default=3456, help='Random seed for reproducibility')
parser.add_argument('--predictor', type=str, choices=['pro', 'pro_clin', 'clin'], default='pro', help='Predictor type')
parser.add_argument('--output_dir', type=str, default="/home/louchen/UKB_meno_pre/s2_model/model", help='Output directory')
args = parser.parse_args()
seed, predictor, output_dir = args.seed, args.predictor, args.output_dir
print(f"配置: seed={seed}, predictor={predictor}", flush=True)

# 数据准备
print("数据加载中...", flush=True)
surv_pred_train = load_data(seed)
exp_var, out_surv, _ = load_variable_definitions()
pro_lasso = load_lasso_features(seed)
X_model, Y, groups = prepare_data(surv_pred_train, pro_lasso, out_surv, predictor=predictor, exp_var=exp_var)
print(f"数据准备完成: {predictor}类型, {X_model.shape[1]}个特征, {X_model.shape[0]}个样本", flush=True)


# Function to process a single fold
def process_single_fold(fold_idx, train_idx, test_idx, X, y, groups, n_estimators, learning_rate, max_depth, 
                       min_samples_split, min_samples_leaf, max_features, subsample, dropout_rate):
    """Process a single fold of cross-validation"""
    test_regions = np.unique(groups[test_idx])
    # print(f"Fold {fold_idx+1} - Test set regions: {test_regions}, sample count: {len(test_idx)}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Initialize and fit model
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        subsample=subsample,
        dropout_rate=dropout_rate,
        random_state=42
    )
    
    gbsa.fit(X_train, y_train)
    
    # Predict and evaluate
    risk_scores = gbsa.predict(X_test)
    c_index = concordance_index_censored(
        y_test['e.tdm'],
        y_test['t.tdm'],
        risk_scores
    )[0]
    
    print(f"第{fold_idx+1}折 - c-index: {c_index:.4f}", flush=True)
    return c_index

# Define evaluation function for GradientBoostingSurvivalAnalysis
def evaluate_gbsa_cv(X, y, groups, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, max_features, subsample, dropout_rate=0.0, n_jobs=-1):
    """Evaluate GBSA model using parallel GroupKFold cross-validation"""
    # Convert float parameters to int with rounding
    max_depth = round(max_depth)
    min_samples_split = round(min_samples_split)
    min_samples_leaf = round(min_samples_leaf)
    n_estimators = round(n_estimators)
    
    # Set default n_jobs to use all cores if not specified
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Check region distribution
    unique_regions = np.unique(groups)
    print(f"Total unique regions: {len(unique_regions)}: {unique_regions}", flush=True)
    
    # Initialize cross-validation
    cv = GroupKFold(n_splits=10)
    fold_splits = list(enumerate(cv.split(X, y, groups)))
    
    # Execute folds in parallel
    c_indices = Parallel(n_jobs=n_jobs)(
        delayed(process_single_fold)(
            fold_idx, train_idx, test_idx, X, y, groups, 
            n_estimators, learning_rate, max_depth, min_samples_split, 
            min_samples_leaf, max_features, subsample, dropout_rate
        ) for fold_idx, (train_idx, test_idx) in fold_splits
    )
    
    # Calculate and return average concordance index
    mean_c_index = np.mean(c_indices)
    print(f"平均c-index: {mean_c_index:.4f}", flush=True)
    return mean_c_index


# https://bayesian-optimization.github.io/BayesianOptimization/2.0.3/reference/bayes_opt.html#
# Define Bayesian optimization function
def gbsa_bayesian_optimize(X, y, groups, n_jobs=-1):
    """Perform Bayesian optimization for GradientBoostingSurvivalAnalysis hyperparameters"""
    # Set default n_jobs to use all cores if not specified
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Define the objective function for Bayesian optimization with parallel cross-validation
    def objective_function(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, max_features, subsample, dropout_rate):
        """Objective function to maximize concordance index"""
        # Run parallel cross-validation with current parameters
        c_index = evaluate_gbsa_cv(
            X=X,
            y=y,
            groups=groups,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            subsample=subsample,
            dropout_rate=dropout_rate,
            n_jobs=n_jobs  # Use parallel processing
        )
        
        return c_index  # BayesianOptimization maximizes the objective

    # Define parameter bounds
    param_bounds = {
        'n_estimators': (50, 500),
        'learning_rate': (0.01, 0.2),
        'max_depth': (3, 10),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': (0.1, 1.0),  # Proportion of features
        'subsample': (0.5, 1.0),
        'dropout_rate': (0.0, 0.3)   # 增加dropout_rate正则化参数
    }
    
    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )
    
    # Run optimization
    optimizer.maximize(init_points=5, n_iter=25)
    
    # Get the best parameters
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    
    # Convert parameters to appropriate types with rounding
    best_params['n_estimators'] = round(best_params['n_estimators'])
    best_params['max_depth'] = round(best_params['max_depth'])
    best_params['min_samples_split'] = round(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = round(best_params['min_samples_leaf'])
    
    print(f"Best concordance index: {best_score}", flush=True)
    print(f"Best parameters: {best_params}", flush=True)
    
    return best_params, best_score

# Run Bayesian optimization with parallel processing
print("Starting Bayesian optimization for GradientBoostingSurvivalAnalysis...", flush=True)
# 设置CPU核心数，可以根据需要调整，默认使用所有可用核心
n_cpus = multiprocessing.cpu_count()
print(f"使用{n_cpus}个CPU核心进行并行计算", flush=True)
best_params, best_score = gbsa_bayesian_optimize(X_model, Y, groups, n_jobs=n_cpus)

# 保存最优参数到CSV
from func.utils import save_best_params, save_model
save_best_params(best_params, best_score, 'gbm', seed, output_dir, predictor)

# 使用最优参数训练最终模型
final_model = GradientBoostingSurvivalAnalysis(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    subsample=best_params['subsample'],
    dropout_rate=best_params.get('dropout_rate', 0.0),  # 兼容不存在dropout_rate的情况
    random_state=42
)

# 使用全部数据进行训练
final_model.fit(X_model, Y)

# 保存训练好的模型
save_model(final_model, 'gbm', seed, output_dir, predictor)
