# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Import utility modules
from func.data_utils import load_data, load_variable_definitions, load_lasso_features, prepare_data
from func.data_utils_test import load_data_test, prepare_data_test
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

# Command line arguments
parser = argparse.ArgumentParser(description='DeepSurv Survival Model Evaluation')
parser.add_argument('--seed', type=int, default=3456, help='Random seed for reproducibility')
parser.add_argument('--predictor', type=str, choices=['pro', 'pro_clin', 'clin'], default='pro', help='Predictor type')
parser.add_argument('--output_dir', type=str, default="F:/repro/UKB_meno_pre/s2_model", help='Output directory')
args = parser.parse_args()
seed, predictor, output_dir = args.seed, args.predictor, args.output_dir
print(f"配置: seed={seed}, predictor={predictor}", flush=True)

# Set random seed
def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Also set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)

# Data preparation
print("数据加载中...", flush=True)
surv_pred_train = load_data(seed)
surv_pred_test = load_data_test(seed)
exp_var, out_surv, pro_var = load_variable_definitions()
pro_lasso = load_lasso_features(seed)
X_train_model, Y_train, groups_train = prepare_data(surv_pred_train, pro_lasso, out_surv, predictor=predictor, exp_var=exp_var)
X_test_model, Y_test, groups_test = prepare_data_test(surv_pred_test, pro_lasso, out_surv, predictor=predictor, exp_var=exp_var)

print(f"数据准备完成: {predictor}类型, 训练集{X_train_model.shape[1]}个特征, {X_train_model.shape[0]}个样本", flush=True)
print(f"测试集: {X_test_model.shape[1]}个特征, {X_test_model.shape[0]}个样本", flush=True)

# Create output directory for results
os.makedirs(output_dir, exist_ok=True)

# Convert data to DeepSurv format
def convert_to_deepsurv_format(X, y):
    """Convert data to format suitable for DeepSurv model"""
    X_array = X.values.astype('float32')
    time = y['t.tdm'].astype('float32')
    event = y['e.tdm'].astype('float32')
    return X_array, (time, event)

# Load best parameters for model reconstruction
model_dir = os.path.join(output_dir, f"model/{predictor}/deepsurv")
params_path = os.path.join(model_dir, f"deepsurv_{predictor}_best_params_{seed}.csv")
best_params = pd.read_csv(params_path)
print(f"加载参数: {params_path}")

# Extract parameters for model reconstruction
num_nodes = int(best_params['num_nodes'].iloc[0])
num_hidden_layers = int(best_params['num_hidden_layers'].iloc[0])
dropout = float(best_params['dropout'].iloc[0])

# Prepare data in the format required by DeepSurv
X_train_array, y_train_tuple = convert_to_deepsurv_format(X_train_model, Y_train)
X_test_array, y_test_tuple = convert_to_deepsurv_format(X_test_model, Y_test)

# Recreate the network architecture
in_features = X_train_array.shape[1]
net = create_deepsurv_net(
    in_features=in_features,
    num_nodes=num_nodes,
    num_hidden_layers=num_hidden_layers,
    dropout=dropout
)

# Load the trained model weights
model_path = os.path.join(model_dir, f"deepsurv_{predictor}_best_model_{seed}.pt")
net.load_state_dict(torch.load(model_path))
print(f"加载模型: {model_path}")

# Set model to evaluation mode
net.eval()

# Create and initialize the model with proper configuration
model = CoxPH(net, tt.optim.Adam)

# This is the important part - compute baseline hazards
# Fit the model with the baseline data to ensure compatibility
model.fit(X_train_array, y_train_tuple, batch_size=256, epochs=0, verbose=False)
_ = model.compute_baseline_hazards()

# Predict survival function
surv_df = model.predict_surv_df(X_test_array)

# Evaluate with concordance index
ev = EvalSurv(
    surv_df,
    y_test_tuple[0],  # durations
    y_test_tuple[1],  # events
    censor_surv='km'
)
c_index = ev.concordance_td()
print(f"C-index (测试集): {c_index:.4f}")

# Calculate integrated brier score
# Get time points between 10th and 90th percentile
lower, upper = np.percentile(Y_test["t.tdm"], [10, 90])
time_grid = np.arange(lower, upper + 1)

# Calculate the integrated brier score
ibs = ev.integrated_brier_score(time_grid)
print(f"Integrated Brier Score: {ibs:.4f}")

# Calculate random prediction and Kaplan-Meier baseline prediction scores for comparison
# Random prediction (coin flip)
random_surv = pd.DataFrame(
    0.5,  # constant 0.5 survival probability
    index=surv_df.index,
    columns=surv_df.columns
)
ev_random = EvalSurv(random_surv, y_test_tuple[0], y_test_tuple[1], censor_surv='km')
random_ibs = ev_random.integrated_brier_score(time_grid)

# Kaplan-Meier prediction (population average)
x_test_zeros = np.zeros_like(X_test_array)  # All predictors set to 0 (mean after standardization)
km_surv = model.predict_surv_df(x_test_zeros)  # Get baseline survival function
ev_km = EvalSurv(km_surv, y_test_tuple[0], y_test_tuple[1], censor_surv='km')
km_ibs = ev_km.integrated_brier_score(time_grid)

# Store comparison of integrated brier scores
score_brier = pd.Series(
    [ibs, random_ibs, km_ibs],
    index=["DeepSurv", "Random", "Kaplan-Meier"],
    name="IBS"
)
print(score_brier)

# Calculate Brier scores for different time points (years)
years = [3, 5, 10]
brier_scores = {}

# Loop through each year to calculate Brier score
for year in years:
    time_point = 365.25 * year
    if time_point <= surv_df.columns.max():
        # For a specific time point, get the brier score
        # This returns a tuple (times, score)
        times, score = ev.brier_score(time_point)
        brier_scores[year] = score
        print(f"{year}年Brier分数: {score:.4f}")

# Store Brier scores in a pandas Series
score_brier_years = pd.Series(
    brier_scores,
    index=list(brier_scores.keys()),
    name="Brier Score"
)

# Save evaluation metrics to CSV files
pd.DataFrame({'C_index': [c_index]}).to_csv(os.path.join(model_dir, f"deepsurv_{predictor}_cindex_{seed}.csv"), index=False)
score_brier.to_frame().to_csv(os.path.join(model_dir, f"deepsurv_{predictor}_ibs_{seed}.csv"))
score_brier_years.to_frame().to_csv(os.path.join(model_dir, f"deepsurv_{predictor}_brier_years_{seed}.csv"))
