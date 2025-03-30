# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import os
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import sys
sys.stdout.reconfigure(write_through=True)

# 导入功能模块
from func.data_utils import load_data, load_variable_definitions, load_lasso_features, prepare_data
from func.utils import save_model

# 参数解析
parser = argparse.ArgumentParser(description='CoxPH Survival Model Analysis')
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

# CoxPH model fit
print("Fitting CoxPH model...", flush=True)
cox_ph = CoxPHSurvivalAnalysis()
cox_ph.fit(X_model, Y)

# Evaluate model performance
risk_scores = cox_ph.predict(X_model)
c_index = concordance_index_censored(Y['e.tdm'], Y['t.tdm'], risk_scores)[0]
print(f"Concordance index: {c_index:.4f}", flush=True)

# Print model coefficients
coefficients = pd.Series(cox_ph.coef_, index=X_model.columns)
print("\nModel coefficients (top 10):", flush=True)
print(coefficients.sort_values(ascending=False).head(10), flush=True)

# Save trained model
from func.utils import save_model
save_model(cox_ph, 'coxph', seed, output_dir, predictor)
