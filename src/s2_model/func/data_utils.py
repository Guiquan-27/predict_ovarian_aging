# Data processing utilities for survival models
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(seed):
    """Load training data using seed."""
    surv_pred_train = pd.read_csv(f"F:/repro/UKB_meno_pre/imputation/impute_merge/surv_pred/surv_pred_train_{seed}.csv")
    return surv_pred_train

def load_variable_definitions():
    """Load variable definition files."""
    # Fix base path for Windows environment
    base_path = "F:/repro/UKB_meno_pre/merge_and_extract/data_files/variable_define/"
    # Fix separator from incorrect "/t" to proper tab separator "\t"
    exp_var = pd.read_csv(f"{base_path}exp_var_v2.txt", sep="\t")
    out_surv = pd.read_csv(f"{base_path}out_surv_predi_v1.txt", sep="\t")
    pro_var = pd.read_csv(f"{base_path}pro_var_v0.txt", sep="\t")
    return exp_var, out_surv, pro_var

def load_lasso_features(seed):
    """Load LASSO-selected features."""
    pro_lasso = pd.read_csv(f"F:/repro/UKB_meno_pre/s1_lasso/lasso_result/surv_pred_train/surv_pred_train_{seed}.csv")
    return pro_lasso

def prepare_data(surv_pred_train, pro_lasso, out_surv, predictor='pro', exp_var=None):
    """Prepare data for modeling based on predictor type."""
    # Load exp_var if needed and not provided
    if (predictor in ['pro_clin', 'clin']) and exp_var is None:
        exp_var, _, _ = load_variable_definitions()
    
    # Extract features based on predictor type
    if predictor == 'pro':
        # Proteomics features only
        X = surv_pred_train[[col for col in pro_lasso['lasso_pro'] if col in surv_pred_train.columns] + ['region_code_10']].copy()
    elif predictor == 'pro_clin':
        # Proteomics + clinical features
        X = surv_pred_train[[col for col in pro_lasso['lasso_pro'] if col in surv_pred_train.columns] + 
                           [col for col in exp_var['Variable'] if col in surv_pred_train.columns and col not in ['eid', 'region_code_ori']]].copy()
    elif predictor == 'clin':
        # Clinical features only
        X = surv_pred_train[[col for col in exp_var['Variable'] if col in surv_pred_train.columns and col not in ['eid', 'region_code_ori']]].copy()
    
    # Extract outcomes
    Y_df = surv_pred_train[[col for col in out_surv['Variable'] if col in surv_pred_train.columns]]
    
    # Process categorical variables for clinical predictors
    if predictor in ['pro_clin', 'clin']:
        X = _process_categorical_variables(X)
    
    # Remove eid if present
    if 'eid' in X.columns:
        X = X.drop('eid', axis=1)
    
    # Scale features (excluding region_code_10)
    other_cols = [col for col in X.columns if col != 'region_code_10']
    scaler = MinMaxScaler()
    X[other_cols] = scaler.fit_transform(X[other_cols])
    
    # Prepare survival data
    event = Y_df['meno_status'].astype(bool).values
    time = Y_df['time_follow'].values
    Y = np.array([(e, t) for e, t in zip(event, time)], dtype=[('e.tdm', 'bool'), ('t.tdm', '<f8')])
    
    # Extract group information
    groups = X['region_code_10'].values
    X_model = X.drop('region_code_10', axis=1)
    
    return X_model, Y, groups

def _process_categorical_variables(X):
    """Process categorical variables for clinical data."""
    from sklearn.preprocessing import OrdinalEncoder
    
    # Variables that need one-hot encoding
    onehot_vars = [
        'ethnic_bg', 'length_mens_cyc_category', 'pregnancy_loss', 'ever_hrt',
        'blood_pressure_medication_take_cbde', 'radiotherapy_chemotherapy',
        'postpartum_depression', 'prolonged_pregnancy', 'preterm_labour_and_delivery',
        'endometriosis', 'gh_pe', 'gestational_diabetes', 'female_infertility',
        'ectopic_pregnancy', 'primary_ovarian_failure', 'ovarian_dysfunction',
        'leiomyoma_of_uterus', 'excessive_vomiting_in_pregnancy', 
        'spontaneous_abortion', 'habitual_aborter', 'eclampsia', 'adenomyosis',
        'menorrhagia', 'irregular_menses', 'polycystic_ovarian_syndrome',
        'cvd_prevalent', 'diabetes_history'
    ]
    
    # Perform one-hot encoding
    for var in onehot_vars:
        if var in X.columns:
            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=float)
            X = pd.concat([X.drop(var, axis=1), dummies], axis=1)
    
    # Variables that need ordinal encoding with defined orders
    ordinal_vars = {
        'smoke_status': ["Never", "Previous", "Current"],
        'age_first_birth_category': ["No_live_birth", "Age<=21", "Age22-25", "Age26-29", "Age>=30"],
        'age_last_birth_category': ["No_live_birth", "Age<=26", "Age27-30", "Age31-34", "Age>=35"],
        'oestradiol_category': ["Lower_limit", "Oestradiol<=Q1", "Oestradiol>Q1<=Q2", "Oestradiol>Q2<=Q3", "Oestradiol>Q3"],
        'total_t_category': ["Lower_limit", "Total_T<=Q1", "Total_T>Q1<=Q2", "Total_T>Q2<=Q3", "Total_T>Q3"]
    }
    
    # Perform ordinal encoding
    for var, categories in ordinal_vars.items():
        if var in X.columns:
            encoder = OrdinalEncoder(categories=[categories])
            X[var] = encoder.fit_transform(X[[var]])
    
    return X
