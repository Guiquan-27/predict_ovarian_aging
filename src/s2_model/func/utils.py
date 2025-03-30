# Utility functions for survival models
import os
import pickle
import pandas as pd
import argparse
import sys

def setup_logging():
    """Configure standard output for proper logging."""
    sys.stdout.reconfigure(write_through=True)

def parse_arguments(description="Survival Model Training"):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=3456, 
                        help='Random seed for reproducibility and allocate data')
    parser.add_argument('--model', type=str, choices=['rsf', 'gbm', 'coxph'], 
                        default='rsf', help='Model type to train')
    parser.add_argument('--predictor', type=str, choices=['pro', 'pro_clin', 'clin'],
                        default='pro', help='Predictor type to use (proteomics, clinical, or both)')
    parser.add_argument('--output_dir', type=str, 
                        default="/home/louchen/UKB_meno_pre/s2_model/model",
                        help='Directory to save results')
    return parser.parse_args()

def save_best_params(params, score, model_type, seed, output_dir, predictor):
    """Save best parameters to CSV file."""
    # Create nested directory structure
    nested_dir = os.path.join(output_dir, predictor, model_type)
    os.makedirs(nested_dir, exist_ok=True)
    
    # Add best score to params
    params_df = pd.DataFrame([params])
    params_df['best_score'] = score
    
    # Save to CSV
    output_file = f"{model_type}_{predictor}_best_params_{seed}.csv"
    params_df.to_csv(os.path.join(nested_dir, output_file), index=False)
    print(f"最优参数已保存到 '{os.path.join(nested_dir, output_file)}'", flush=True)
    
    return os.path.join(nested_dir, output_file)

def save_model(model, model_type, seed, output_dir, predictor):
    """Save trained model to pickle file."""
    # Create nested directory structure
    nested_dir = os.path.join(output_dir, predictor, model_type)
    os.makedirs(nested_dir, exist_ok=True)
    
    # Save model
    model_file = f"{model_type}_{predictor}_best_model_{seed}.pkl"
    with open(os.path.join(nested_dir, model_file), "wb") as f:
        pickle.dump(model, f)
    print(f"最优模型已保存到 '{os.path.join(nested_dir, model_file)}'", flush=True)
    
    return os.path.join(nested_dir, model_file)

def load_model(model_path):
    """Load saved model from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
