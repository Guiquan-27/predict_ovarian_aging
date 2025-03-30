import warnings
import os
import sys
import random
# ignore pytorch warning
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings('ignore', message="You are using `torch.load`", category=FutureWarning)
# 导入其他必要库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.model_selection import GroupKFold
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
import multiprocessing
sys.stdout.reconfigure(write_through=True)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
from func.data_utils import load_data, load_variable_definitions, load_lasso_features, prepare_data
from func.utils import save_best_params, save_model


parser = argparse.ArgumentParser(description='DeepSurv Model Selection with Bayesian Optimization')
parser.add_argument('--seed', type=int, default=3456, help='Random seed for reproducibility')
parser.add_argument('--predictor', type=str, choices=['pro', 'pro_clin', 'clin'], default='clin', help='Predictor type')
parser.add_argument('--output_dir', type=str, default="F:/repro/UKB_meno_pre/s2_model/model", help='Output directory')

args = parser.parse_args()
seed, predictor, output_dir = args.seed, args.predictor, args.output_dir
print(f"配置: seed={seed}, predictor={predictor}", flush=True)

# 设置随机种子以确保可重现性
def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 即使不使用GPU，也设置CUDA种子以确保代码在GPU环境下也能复现
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 为DataLoader设置worker初始化函数
def seed_worker(worker_id):
    """Function to make dataloader workers deterministic"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 应用种子设置
set_seed(seed)

# 数据准备
print("数据加载中...", flush=True)
surv_pred_train = load_data(seed)
exp_var, out_surv, _ = load_variable_definitions()
pro_lasso = load_lasso_features(seed)
X_model, Y, groups = prepare_data(surv_pred_train, pro_lasso, out_surv, predictor=predictor, exp_var=exp_var)
print(f"数据准备完成: {predictor}类型, {X_model.shape[1]}个特征, {X_model.shape[0]}个样本", flush=True)

# 转换数据为DeepSurv可用格式
def convert_to_deepsurv_format(X, y):
    """Convert data to format suitable for DeepSurv model"""
    X_array = X.values.astype('float32')
    time = y['t.tdm'].astype('float32')
    event = y['e.tdm'].astype('float32')
    return X_array, (time, event)

# Function to create DeepSurv neural network with variable hidden layers
def create_deepsurv_net(in_features, num_nodes, num_hidden_layers, dropout):
    """Create a DeepSurv neural network with specified architecture and variable hidden layers"""

    out_features = 1
    output_bias = False
    # Create a list of nodes for each hidden layer
    hidden_layers = [num_nodes] * num_hidden_layers
    return tt.practical.MLPVanilla(
        in_features=in_features, num_nodes=hidden_layers, out_features=out_features,
        batch_norm=True,  # Fixed batch_norm parameter to True
        dropout=dropout, output_bias=output_bias
    )

# Function to process a single fold
def process_single_fold(fold_idx, train_idx, test_idx, X, y, groups, num_nodes, num_hidden_layers, dropout, learning_rate, batch_size):
    """Process a single fold of cross-validation for DeepSurv"""
    fold_seed =  fold_idx
    set_seed(fold_seed)
    
    test_regions = np.unique(groups[test_idx])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Convert to DeepSurv format
    x_train_arr, y_train_tuple = convert_to_deepsurv_format(X_train, y_train)
    x_test_arr, y_test_tuple = convert_to_deepsurv_format(X_test, y_test)
    
    # Use test data as validation data for early stopping
    val_data = (x_test_arr, y_test_tuple)
    print(f"第{fold_idx+1}折 - 训练集样本数: {x_train_arr.shape[0]}", flush=True)
    print(f"第{fold_idx+1}折 - 测试集样本数: {x_test_arr.shape[0]}", flush=True)
    print(batch_size)
    # Create network with variable hidden layers
    in_features = x_train_arr.shape[1]
    net = create_deepsurv_net(
        in_features=in_features,
        num_nodes=round(num_nodes),
        num_hidden_layers=round(num_hidden_layers),
        dropout=dropout
    )
    
    # Create and train model
    model = CoxPH(net, tt.optim.Adam,device = "cpu")
    model.optimizer.set_lr(learning_rate)
    
    # 设置回调函数：早停和学习率衰减
    callbacks = [
        tt.callbacks.EarlyStopping(patience=5, min_delta=0.0001)
    ]
    epochs = 512  # Max epochs, early stopping will prevent overfitting
    
    # 确保批处理大小不会导致最后一个批次只有1个样本
    actual_batch_size = round(batch_size)
    n_samples = len(x_train_arr)
    if n_samples % actual_batch_size == 1:
        actual_batch_size = actual_batch_size - 1  # 减小批次大小以避免最后一个批次只有1个样本,val_batch_size不会受到这个影响
        print(f"第{fold_idx+1}折 - 调整训练批次大小为{actual_batch_size}以避免BatchNorm错误", flush=True)
    

    # Train the model - 添加关键参数以确保结果可重复
    model.fit(
        x_train_arr, y_train_tuple, 
        batch_size=actual_batch_size, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=False,
        val_data=val_data, val_batch_size=round(batch_size),
        shuffle=False,  # 禁用数据洗牌，确保训练顺序一致
        num_workers=0  # 使用单线程可以避免多进程带来的不确定性
    )
    
    # Compute baseline hazards (required for prediction)
    _ = model.compute_baseline_hazards()
    
    # Predict on test set
    surv_df = model.predict_surv_df(x_test_arr)
    
    # Evaluate c-index using EvalSurv
    ev = EvalSurv(
        surv_df, 
        y_test_tuple[0],  # durations
        y_test_tuple[1],  # events
        censor_surv='km'
    )
    c_index = ev.concordance_td()
    
    print(f"第{fold_idx+1}折 - c-index: {c_index:.4f}", flush=True)
    return c_index

# Define evaluation function for DeepSurv with CV
def evaluate_deepsurv_cv(X, y, groups, num_nodes, num_hidden_layers, dropout, 
                         learning_rate, batch_size, n_jobs=-1):
    """Evaluate DeepSurv model using parallel GroupKFold cross-validation"""
    print("Evaluate DeepSurv model using parallel GroupKFold cross-validation", flush=True)
    # Set default n_jobs to use all cores if not specified
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Check region distribution
    unique_regions = np.unique(groups)
    # print(f"Total unique regions: {len(unique_regions)}: {unique_regions}", flush=True)

    # Initialize cross-validation
    cv = GroupKFold(n_splits=10)
    fold_splits = list(enumerate(cv.split(X, y, groups)))
    
    # Execute folds in parallel
    c_indices = Parallel(n_jobs=n_jobs)(
        delayed(process_single_fold)(
            fold_idx, train_idx, test_idx, X, y, groups, 
            num_nodes, num_hidden_layers, dropout, learning_rate, batch_size
        ) for fold_idx, (train_idx, test_idx) in fold_splits
    )
    
    # Calculate and return average concordance index
    mean_c_index = np.mean(c_indices)
    print(f"平均c-index: {mean_c_index:.4f}", flush=True)
    return mean_c_index

# Define Bayesian optimization function
def deepsurv_bayesian_optimize(X, y, groups, n_jobs=-1):
    """Perform Bayesian optimization for DeepSurv hyperparameters"""
    print("Perform Bayesian optimization for DeepSurv hyperparameters", flush=True)
    # 重新设置主进程的随机种子
    set_seed(seed)
    
    # Set default n_jobs to use all cores if not specified
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Define the objective function for Bayesian optimization
    def objective_function(num_nodes, num_hidden_layers, dropout, learning_rate, batch_size):
        """Objective function to maximize concordance index"""
        # Run parallel cross-validation with current parameters
        c_index = evaluate_deepsurv_cv(
            X=X,
            y=y,
            groups=groups,
            num_nodes=num_nodes,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_jobs=n_jobs
        )
        
        return c_index  # BayesianOptimization maximizes the objective

    # Define parameter bounds - 调整为适应高维特征的参数范围
    param_bounds = {
        'num_nodes': (32, 256),            # 增大节点数范围，适应高维特征
        'num_hidden_layers': (2, 6),       # 添加隐藏层数量作为超参数
        'dropout': (0.1, 0.5),             # Dropout rate
        'learning_rate': (0.001, 0.01),     # Learning rate
        'batch_size': (256,1024)          # 增大批处理大小上限到1024
    }
    
    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=seed,  # 使用传入的seed确保可重现性
        verbose=2
    )
    
    # Run optimization - 增加初始点和迭代次数
    optimizer.maximize(init_points=20, n_iter=50)
    
    # Get the best parameters
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    
    # Convert parameters to appropriate types using round() instead of int()
    best_params['num_nodes'] = round(best_params['num_nodes'])
    best_params['num_hidden_layers'] = round(best_params['num_hidden_layers'])
    best_params['batch_size'] = round(best_params['batch_size'])
    
    print(f"Best concordance index: {best_score}", flush=True)
    print(f"Best parameters: {best_params}", flush=True)
    
    return best_params, best_score

# Run Bayesian optimization with parallel processing
print("Starting Bayesian optimization for DeepSurv...", flush=True)

# 设置CPU核心数，可以根据需要调整，默认使用所有可用核心
n_cpus = multiprocessing.cpu_count()
print(f"使用{n_cpus}个CPU核心进行并行计算", flush=True)

best_params, best_score = deepsurv_bayesian_optimize(X_model, Y, groups, n_jobs=n_cpus)
save_best_params(best_params, best_score, 'deepsurv', seed, output_dir, predictor)

X_array, y_tuple = convert_to_deepsurv_format(X_model, Y)

in_features = X_array.shape[1]
final_net = create_deepsurv_net(
    in_features=in_features,
    num_nodes=best_params['num_nodes'],
    num_hidden_layers=best_params['num_hidden_layers'],
    dropout=best_params['dropout']
)

final_model = CoxPH(final_net, tt.optim.Adam)
final_model.optimizer.set_lr(best_params['learning_rate'])

callbacks = [
    tt.callbacks.EarlyStopping(patience=5, min_delta=0.0001)
]
epochs = 512  
set_seed(seed)

final_batch_size = best_params['batch_size']
n_samples = len(X_array)
if n_samples % final_batch_size == 1:
    final_batch_size = final_batch_size - 1  # 减小批次大小以避免最后一个批次只有1个样本
    print(f"调整最终批次大小为{final_batch_size}以避免BatchNorm错误", flush=True)

print("训练最终模型...", flush=True)
final_model.fit(
    X_array, 
    y_tuple, 
    batch_size=final_batch_size, 
    epochs=epochs, 
    callbacks=callbacks, 
    verbose=True,
    shuffle=False,  # 禁用数据洗牌，确保训练顺序一致
    num_workers=0  # 使用单线程可以避免多进程带来的不确定性
)



nested_dir = os.path.join(output_dir, predictor, 'deepsurv')
os.makedirs(nested_dir, exist_ok=True)
model_file = f"deepsurv_{predictor}_best_model_{seed}.pt"
torch.save(final_model.net.state_dict(), os.path.join(nested_dir, model_file))
print(f"DeepSurv模型权重已保存到 '{os.path.join(nested_dir, model_file)}'", flush=True)
print(f"DeepSurv模型训练完成，最佳c-index: {best_score:.4f}", flush=True)


from func.data_utils_test import load_data_test, prepare_data_test
surv_pred_test = load_data_test(seed)
exp_var, out_surv, pro_var = load_variable_definitions()
pro_lasso = load_lasso_features(seed)
X_test_model, Y_test, groups_test = prepare_data_test(surv_pred_test, pro_lasso, out_surv, predictor=predictor, exp_var=exp_var)
X_test_array, y_test_tuple = convert_to_deepsurv_format(X_test_model, Y_test)


_ = final_model.compute_baseline_hazards()

surv = final_model.predict_surv_df(X_test_array)
ev = EvalSurv(surv,  y_test_tuple[0],  y_test_tuple[1], censor_surv='km')
c_index = ev.concordance_td()
print(f"C-index (测试集): {c_index:.4f}")