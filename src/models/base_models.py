# -*- coding: utf-8 -*-
"""
Base survival analysis model module
Provide Cox proportional hazard model, random survival forest, etc. base survival analysis models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import joblib
import os

logger = logging.getLogger(__name__)

class BaseSurvivalModel:
    """Base survival analysis model class with common interface"""
    
    def __init__(self, name: str = "base_model"):
        """
        Initialize base survival model
        
        Parameters:
        -----
        name: str, default "base_model"
            Model name
        """
        self.name = name
        self.model = None
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'BaseSurvivalModel':
        """
        Train model
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target variable (time and event)
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
        **kwargs:
            Additional model parameters
            
        Returns:
        -----
        self: BaseSurvivalModel
            Trained model instance
        """
        raise NotImplementedError("Subclass must implement fit method")
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict survival probability
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        times: List[float], optional
            Prediction time points, default None (use time points from training data)
            
        Returns:
        -----
        np.ndarray
            Predicted survival probability
        """
        raise NotImplementedError("Subclass must implement predict method")
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk score
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
            
        Returns:
        -----
        np.ndarray
            Predicted risk score
        """
        raise NotImplementedError("Subclass must implement predict_risk method")
    
    def score(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event') -> float:
        """
        Calculate model C-index
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target variable (time and event)
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
            
        Returns:
        -----
        float
            C-index score
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        risk_scores = self.predict_risk(X)
        c_index = concordance_index(y[time_col], -risk_scores, y[event_col])
        return c_index
    
    def save(self, path: str) -> None:
        """
        Save model
        
        Parameters:
        -----
        path: str
            Save path
        """
        if not self.fitted:
            raise ValueError("Model not trained, cannot save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseSurvivalModel':
        """
        Load model
        
        Parameters:
        -----
        path: str
            Model file path
            
        Returns:
        -----
        BaseSurvivalModel
            Loaded model instance
        """
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model


class CoxPHModel(BaseSurvivalModel):
    """Cox proportional hazard model"""
    
    def __init__(self, name: str = "cox_ph", **kwargs):
        """
        Initialize Cox proportional hazard model
        
        Parameters:
        -----
        name: str, default "cox_ph"
            Model name
        **kwargs:
            Parameters to pass to CoxPHFitter
        """
        super().__init__(name)
        self.model = CoxPHFitter(**kwargs)
        self.params = kwargs
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'CoxPHModel':
        """
        Train Cox proportional hazard model
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target variable (time and event)
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
        **kwargs:
            Parameters to pass to CoxPHFitter.fit
            
        Returns:
        -----
        self: CoxPHModel
            Trained model instance
        """
        logger.info(f"Training Cox proportional hazard model: {self.name}")
        
        # Merge feature and target variable
        df = pd.concat([X, y], axis=1)
        
        # Train model
        self.model.fit(df, duration_col=time_col, event_col=event_col, **kwargs)
        self.fitted = True
        
        # Record training results
        logger.info(f"Model training completed, concordance_index: {self.model.concordance_index_}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict survival probability
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        times: List[float], optional
            Prediction time points, default None (use time points from training data)
            
        Returns:
        -----
        np.ndarray
            Predicted survival probability
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict survival function
        survival_func = self.model.predict_survival_function(X)
        
        # If specified time points, evaluate survival function at these points
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, t in enumerate(times):
                survival_probs[:, i] = survival_func.loc[t].values
            return survival_probs
        else:
            # Otherwise return full survival function
            return survival_func
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk score
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
            
        Returns:
        -----
        np.ndarray
            Predicted risk score (partial risk)
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict partial risk
        return self.model.predict_partial_hazard(X).values
    
    def plot_coefficients(self, top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot model coefficient forest plot
        
        Parameters:
        -----
        top_n: int, default 10
            Number of features to display
        figsize: Tuple[int, int], default (10, 8)
            Figure size
            
        Returns:
        -----
        plt.Figure
            matplotlib figure object
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Get coefficient summary
        summary = self.model.summary
        
        # Select top N features
        if len(summary) > top_n:
            # Sort by p-value
            plot_df = summary.sort_values('p').head(top_n).copy()
        else:
            plot_df = summary.copy()
        
        # Sort by risk ratio
        plot_df = plot_df.sort_values('exp(coef)')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot forest plot
        y_pos = np.arange(len(plot_df))
        
        # Plot risk ratio points and confidence intervals
        ax.scatter(plot_df['exp(coef)'], y_pos, marker='o', s=50, color='blue')
        
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.plot([row['lower 0.95'], row['upper 0.95']], [i, i], 'b-', alpha=0.6)
        
        # Add vertical line indicating HR=1
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        # Set Y axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df.index)
        
        # Set X axis to logarithmic scale
        ax.set_xscale('log')
        
        # Add title and labels
        ax.set_title('Cox model coefficients (95% confidence interval)', fontsize=14)
        ax.set_xlabel('Risk ratio (HR)', fontsize=12)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class RandomSurvivalForestModel(BaseSurvivalModel):
    """Random survival forest model"""
    
    def __init__(self, name: str = "rsf", **kwargs):
        """
        Initialize random survival forest model
        
        Parameters:
        -----
        name: str, default "rsf"
            Model name
        **kwargs:
            Parameters to pass to RandomSurvivalForest
        """
        super().__init__(name)
        self.model = RandomSurvivalForest(**kwargs)
        self.params = kwargs
        self.event_times_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'RandomSurvivalForestModel':
        """
        Train random survival forest model
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target variable (time and event)
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
        **kwargs:
            Parameters to pass to RandomSurvivalForest.fit
            
        Returns:
        -----
        self: RandomSurvivalForestModel
            Trained model instance
        """
        logger.info(f"Training random survival forest model: {self.name}")
        
        # Convert to scikit-survival required format
        structured_y = Surv.from_dataframe(event_col, time_col, y)
        
        # Train model
        self.model.fit(X, structured_y, **kwargs)
        self.fitted = True
        self.event_times_ = self.model.event_times_
        
        # Record training results
        logger.info(f"Model training completed, feature count: {X.shape[1]}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict survival probability
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        times: List[float], optional
            Prediction time points, default None (use time points from training data)
            
        Returns:
        -----
        np.ndarray
            Predicted survival probability
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict survival function
        survival_funcs = self.model.predict_survival_function(X)
        
        # If specified time points, evaluate survival function at these points
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, surv_func in enumerate(survival_funcs):
                for j, t in enumerate(times):
                    # Find closest time point
                    idx = np.searchsorted(self.event_times_, t)
                    if idx == len(self.event_times_):
                        idx = len(self.event_times_) - 1
                    survival_probs[i, j] = surv_func[idx]
            return survival_probs
        else:
            # Otherwise return full survival function
            return np.array([sf.y for sf in survival_funcs])
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk score
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
            
        Returns:
        -----
        np.ndarray
            Predicted risk score
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict risk score
        return self.model.predict(X)
    
    def plot_feature_importance(self, top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance bar plot
        
        Parameters:
        -----
        top_n: int, default 10
            Number of features to display
        figsize: Tuple[int, int], default (10, 8)
            Figure size
            
        Returns:
        -----
        plt.Figure
            matplotlib figure object
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Get feature importance
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        if len(importance_df) > top_n:
            plot_df = importance_df.head(top_n).copy()
        else:
            plot_df = importance_df.copy()
        
        # Reverse order so most important features are at the top
        plot_df = plot_df.iloc[::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar plot
        ax.barh(plot_df['feature'], plot_df['importance'], color='skyblue', edgecolor='black')
        
        # Add title and labels
        ax.set_title('Random survival forest feature importance', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        
        # Add grid lines
        ax.grid(True, axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class CoxnetModel(BaseSurvivalModel):
    """Cox model with elastic net regularization"""
    
    def __init__(self, name: str = "coxnet", **kwargs):
        """
        Initialize Cox model with elastic net regularization
        
        Parameters:
        -----
        name: str, default "coxnet"
            Model name
        **kwargs:
            Parameters to pass to CoxnetSurvivalAnalysis
        """
        super().__init__(name)
        self.model = CoxnetSurvivalAnalysis(**kwargs)
        self.params = kwargs
        self.event_times_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', **kwargs) -> 'CoxnetModel':
        """
        Train Cox model with elastic net regularization
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        y: pd.DataFrame
            Target variable (time and event)
        time_col: str, default 'time'
            Time column name
        event_col: str, default 'event'
            Event column name
        **kwargs:
            Parameters to pass to CoxnetSurvivalAnalysis.fit
            
        Returns:
        -----
        self: CoxnetModel
            Trained model instance
        """
        logger.info(f"Training Cox model with elastic net regularization: {self.name}")
        
        # Convert to scikit-survival required format
        structured_y = Surv.from_dataframe(event_col, time_col, y)
        
        # Train model
        self.model.fit(X, structured_y, **kwargs)
        self.fitted = True
        
        # Record training results
        logger.info(f"Model training completed, non-zero coefficient count: {np.sum(self.model.coef_ != 0)}")
        
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        Predict survival probability
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
        times: List[float], optional
            Prediction time points, default None (use time points from training data)
            
        Returns:
        -----
        np.ndarray
            Predicted survival probability
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict survival function
        survival_funcs = self.model.predict_survival_function(X)
        
        # If specified time points, evaluate survival function at these points
        if times is not None:
            survival_probs = np.zeros((len(X), len(times)))
            for i, surv_func in enumerate(survival_funcs):
                for j, t in enumerate(times):
                    # Find closest time point
                    idx = np.searchsorted(surv_func.x, t)
                    if idx == len(surv_func.x):
                        idx = len(surv_func.x) - 1
                    survival_probs[i, j] = surv_func.y[idx]
            return survival_probs
        else:
            # Otherwise return full survival function
            return np.array([sf.y for sf in survival_funcs])
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk score
        
        Parameters:
        -----
        X: pd.DataFrame
            Feature matrix
            
        Returns:
        -----
        np.ndarray
            Predicted risk score
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Predict risk score
        return self.model.predict(X)
    
    def plot_coefficients(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot model coefficient bar plot
        
        Parameters:
        -----
        figsize: Tuple[int, int], default (10, 8)
            Figure size
            
        Returns:
        -----
        plt.Figure
            matplotlib figure object
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Get non-zero coefficients
        coef = self.model.coef_
        feature_names = self.model.feature_names_in_
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef.flatten()
        })
        
        # Filter non-zero coefficients
        nonzero_coef = coef_df[coef_df['coefficient'] != 0].copy()
        
        # Sort by absolute coefficient value
        nonzero_coef['abs_coef'] = nonzero_coef['coefficient'].abs()
        nonzero_coef = nonzero_coef.sort_values('abs_coef', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar plot
        bars = ax.barh(nonzero_coef['feature'], nonzero_coef['coefficient'], 
                      color=np.where(nonzero_coef['coefficient'] > 0, 'skyblue', 'salmon'))
        
        # Add title and labels
        ax.set_title('Coxnet model non-zero coefficients', fontsize=14)
        ax.set_xlabel('Coefficient value', fontsize=12)
        
        # Add vertical line indicating zero point
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add grid lines
        ax.grid(True, axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


def create_model(model_type: str, **kwargs) -> BaseSurvivalModel:
    """
    Create specified type of survival analysis model
    
    Parameters:
    -----
    model_type: str
        Model type, optional 'cox', 'rsf', 'coxnet'
    **kwargs:
        Parameters to pass to model constructor
        
    Returns:
    -----
    BaseSurvivalModel
        Created model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'cox':
        return CoxPHModel(**kwargs)
    elif model_type == 'rsf':
        return RandomSurvivalForestModel(**kwargs)
    elif model_type == 'coxnet':
        return CoxnetModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 