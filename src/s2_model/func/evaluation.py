# Model evaluation utilities
import numpy as np
from sklearn.model_selection import GroupKFold
from sksurv.metrics import concordance_index_censored
from joblib import Parallel, delayed
import multiprocessing

def evaluate_model_cv(model_class, model_params, X, y, groups, n_splits=10, n_jobs=-1):
    """Evaluate model with GroupKFold cross-validation."""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    print("Evaluate model with GroupKFold cross-validation", flush=True)
    # Initialize cross-validation
    cv = GroupKFold(n_splits=n_splits)
    fold_splits = list(enumerate(cv.split(X, y, groups)))
    
    # Execute folds in parallel
    c_indices = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_fold)(
            fold_idx, train_idx, test_idx, X, y, model_class, model_params
        ) for fold_idx, (train_idx, test_idx) in fold_splits
    )
    
    # Calculate and return average concordance index
    mean_c_index = np.mean(c_indices)
    print(f"平均c-index: {mean_c_index:.4f}", flush=True)
    return mean_c_index

def _process_single_fold(fold_idx, train_idx, test_idx, X, y, model_creator, model_params):
    """Process a single fold of cross-validation."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Initialize and fit model
    model = model_creator(model_params)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    risk_scores = model.predict(X_test)
    c_index = concordance_index_censored(
        y_test['e.tdm'],
        y_test['t.tdm'],
        risk_scores
    )[0]
    
    print(f"第{fold_idx+1}折 - c-index: {c_index:.4f}", flush=True)
    return c_index

# Model evaluation objective functions for optimization

def rsf_objective(n_estimators, min_samples_split, min_samples_leaf, 
                  max_features, max_samples, X, y, groups, n_jobs=-1):
    """Objective function for RSF model optimization."""
    from func.models import create_rsf_model
    
    params = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'max_samples': max_samples,
        'n_jobs': n_jobs
    }
    
    return evaluate_model_cv(create_rsf_model, params, X, y, groups, n_jobs=n_jobs)

def gbm_objective(n_estimators, learning_rate, max_depth, min_samples_split, 
                  min_samples_leaf, max_features, subsample, dropout_rate, 
                  X, y, groups, n_jobs=-1):
    """Objective function for GBM model optimization."""
    from func.models import create_gbm_model
    
    params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'subsample': subsample,
        'dropout_rate': dropout_rate
        # 移除 n_jobs 参数，因为 GradientBoostingSurvivalAnalysis 不支持并行训练
    }
    
    return evaluate_model_cv(create_gbm_model, params, X, y, groups, n_jobs=n_jobs)
