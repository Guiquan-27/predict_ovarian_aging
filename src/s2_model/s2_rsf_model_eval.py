# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.ticker as mtick
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# 导入功能模块
from func.data_utils import load_data, load_variable_definitions, load_lasso_features, prepare_data
from func.data_utils_test import load_data_test, prepare_data_test

# 参数解析
parser = argparse.ArgumentParser(description='RSF Survival Model Evaluation')
parser.add_argument('--seed', type=int, default=3456, help='Random seed for reproducibility')
parser.add_argument('--predictor', type=str, choices=['pro', 'pro_clin', 'clin'], default='pro', help='Predictor type')
parser.add_argument('--output_dir', type=str, default="/home/louchen/UKB_meno_pre/s2_model", help='Output directory')
args = parser.parse_args()
seed, predictor, output_dir = args.seed, args.predictor, args.output_dir
print(f"配置: seed={seed}, predictor={predictor}", flush=True)

# 数据准备
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

# Load the trained model with seed and predictor
model_dir = os.path.join(output_dir, f"model/{predictor}/rsf")
model_path = os.path.join(model_dir, f"rsf_{predictor}_best_model_{seed}.pkl")
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

print(f"加载模型: {model_path}")

# Predict risk scores on test data
risk_scores = loaded_model.predict(X_test_model)
c_index_result = concordance_index_censored(
    Y_test['e.tdm'],
    Y_test['t.tdm'],
    risk_scores
)
c_index = c_index_result[0]
print(f"C-index (测试集): {c_index:.4f}")

# brier score
lower, upper = np.percentile(Y_test["t.tdm"], [10, 90])
brier_times = np.arange(lower, upper + 1)
surv_prob = np.vstack([fn(brier_times) for fn in loaded_model.predict_survival_function(X_test_model)])
random_surv_prob = 0.5 * np.ones((Y_test.shape[0], brier_times.shape[0]))
km_func = StepFunction(*kaplan_meier_estimator(Y_test["e.tdm"], Y_test["t.tdm"]))
km_surv_prob = np.tile(km_func(brier_times), (Y_test.shape[0], 1))
score_brier = pd.Series(
    [
        integrated_brier_score(Y_train, Y_test, prob, brier_times)
        for prob in (surv_prob, random_surv_prob, km_surv_prob)
    ],
    index=["RSF", "Random", "Kaplan-Meier"],
    name="IBS",
)
print(score_brier)

# Calculate Brier scores for different time points
years = [3, 5, 10]
survs = loaded_model.predict_survival_function(X_test_model)
brier_scores = {}

# Loop through each year to calculate Brier score
for year in years:
    time_point = 365.25 * year
    preds = [fn(time_point) for fn in survs]
    times, score = brier_score(Y_train, Y_test, preds, time_point)
    brier_scores[year] = score
    print(f"{year}年Brier分数: {score}")

# Store Brier scores in a pandas Series
score_brier_years = pd.Series(
    brier_scores,
    index=years,
    name="Brier Score"
)

# 保存评估指标为CSV文件
pd.DataFrame({'C_index': [c_index]}).to_csv(os.path.join(model_dir, f"rsf_{predictor}_cindex_{seed}.csv"), index=False)
score_brier.to_frame().to_csv(os.path.join(model_dir, f"rsf_{predictor}_ibs_{seed}.csv"))
score_brier_years.to_frame().to_csv(os.path.join(model_dir, f"rsf_{predictor}_brier_years_{seed}.csv"))
