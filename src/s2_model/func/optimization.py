# Hyperparameter optimization utilities
from bayes_opt import BayesianOptimization
import numpy as np

class ModelOptimizer:
    """Bayesian optimization for survival models."""
    
    def __init__(self, model_type, X, y, groups, random_state=42):
        """Initialize optimizer for a specific model type."""
        self.model_type = model_type
        self.X = X
        self.y = y
        self.groups = groups
        self.random_state = random_state
        
        # Define parameter spaces for different models
        self.param_spaces = {
            'rsf': {
                'n_estimators': (50, 1000),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': (0.1, 1.0),
                'max_samples': (0.5, 1.0)
            },
            'gbm': {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.2),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': (0.1, 1.0),
                'subsample': (0.5, 1.0),
                'dropout_rate': (0.0, 0.3)
            }
        }
    
    def optimize(self, eval_function, init_points=5, n_iter=25, n_jobs=-1):
        """Run Bayesian optimization to find best parameters."""
        from functools import partial
        
        # Select parameter space
        if self.model_type not in self.param_spaces:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        param_bounds = self.param_spaces[self.model_type]
        
        # Create partial function with fixed arguments
        fixed_eval = partial(eval_function, 
                            X=self.X, 
                            y=self.y, 
                            groups=self.groups,
                            n_jobs=n_jobs)
        
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=fixed_eval,
            pbounds=param_bounds,
            random_state=self.random_state,
            verbose=2
        )
        
        # Run optimization
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Get best parameters
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        # Process parameters based on model type
        self._process_params(best_params)
        
        return best_params, best_score
    
    def _process_params(self, params):
        """Process parameters to correct types."""
        # Convert int parameters
        int_params = ['n_estimators', 'min_samples_split', 'min_samples_leaf']
        if self.model_type == 'gbm':
            int_params.append('max_depth')
        
        for param in int_params:
            if param in params:
                params[param] = round(params[param])
        
        return params
