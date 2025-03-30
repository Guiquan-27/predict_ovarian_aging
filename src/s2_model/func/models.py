# Survival models implementation
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd

# Default parameters for Random Survival Forest
RSF_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_samples': 0.8,
    'n_jobs': -1,
    'random_state': 42
}

# Default parameters for Gradient Boosting Survival Analysis
GBM_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'subsample': 0.8,
    'dropout_rate': 0.0,
    'random_state': 42
}

# Default parameters for Cox Proportional Hazards model
COXPH_DEFAULT_PARAMS = {
    'alpha': 0.0,  # L2 regularization strength
    'ties': 'efron'  # handle tied event
}

def create_rsf_model(params=None):
    """Create a Random Survival Forest model with default or custom parameters."""
    model_params = RSF_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    
    return RandomSurvivalForest(
        n_estimators=round(model_params['n_estimators']),
        max_depth=model_params['max_depth'],
        min_samples_split=round(model_params['min_samples_split']),
        min_samples_leaf=round(model_params['min_samples_leaf']),
        max_features=model_params['max_features'],
        max_samples=model_params['max_samples'],
        min_weight_fraction_leaf=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=model_params['n_jobs'],
        random_state=model_params['random_state']
    )

def create_gbm_model(params=None):
    """Create a Gradient Boosting Survival Analysis model with default or custom parameters."""
    model_params = GBM_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    
    return GradientBoostingSurvivalAnalysis(
        n_estimators=round(model_params['n_estimators']),
        learning_rate=model_params['learning_rate'],
        max_depth=round(model_params['max_depth']),
        min_samples_split=round(model_params['min_samples_split']),
        min_samples_leaf=round(model_params['min_samples_leaf']),
        max_features=model_params['max_features'],
        subsample=model_params['subsample'],
        dropout_rate=model_params['dropout_rate'],
        random_state=model_params['random_state']
    )

def create_coxph_model(params=None):
    """Create a Cox Proportional Hazards model with default or custom parameters."""
    model_params = COXPH_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    
    return CoxPHSurvivalAnalysis(
        alpha=model_params['alpha'],
        ties=model_params['ties']
    )

def get_coxph_coefficients(model, feature_names):
    """Get CoxPH model coefficients as pandas Series."""
    return pd.Series(model.coef_, index=feature_names)
