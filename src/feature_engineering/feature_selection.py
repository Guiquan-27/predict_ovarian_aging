# -*- coding: utf-8 -*-
"""
Feature selection module
Provides methods for feature selection in survival analysis, including univariate Cox regression, effect size filtering, and Log-rank test.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def univariate_cox_selection(df: pd.DataFrame, 
                            time_col: str, 
                            event_col: str,
                            features: Optional[List[str]] = None,
                            alpha: float = 0.05,
                            fdr_correction: bool = True) -> pd.DataFrame:
    """
    Use univariate Cox regression for feature selection

    Parameters:
    -----
    df: pd.DataFrame
        Dataframe
    time_col: str
        Name of time column
    event_col: str
        Name of event column
    features: List[str], optional
        List of features to evaluate, default None (use all non-target columns)
    alpha: float, default 0.05
        Significance level
    fdr_correction: bool, default True
        Whether to apply FDR correction

    Returns:
    -----
    pd.DataFrame
        Dataframe containing Cox regression results
    """
    logger.info("Starting univariate Cox regression feature selection")
    
    # If features not specified, use all non-target columns
    if features is None:
        features = [col for col in df.columns if col not in [time_col, event_col]]
    
    results = []
    
    # Perform univariate Cox regression for each feature
    for feature in features:
        try:
            # Create dataframe for Cox model
            cph_df = df[[feature, time_col, event_col]].copy()
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cph_df, duration_col=time_col, event_col=event_col)
            
            # Extract results
            summary = cph.summary
            p_value = summary.loc[feature, 'p']
            hazard_ratio = np.exp(summary.loc[feature, 'coef'])
            hr_lower = np.exp(summary.loc[feature, 'coef lower 95%'])
            hr_upper = np.exp(summary.loc[feature, 'coef upper 95%'])
            
            results.append({
                'feature': feature,
                'hazard_ratio': hazard_ratio,
                'hr_lower_95': hr_lower,
                'hr_upper_95': hr_upper,
                'p_value': p_value,
                'log_rank': -np.log10(p_value) if p_value > 0 else 0
            })
        except Exception as e:
            logger.warning(f"Error in Cox regression for feature {feature}: {str(e)}")
    
    # Create results dataframe
    result_df = pd.DataFrame(results)
    
    # Apply FDR correction if requested
    if fdr_correction and len(result_df) > 0:
        _, corrected_pvals, _, _ = multipletests(
            result_df['p_value'].values, 
            alpha=alpha, 
            method='fdr_bh'
        )
        result_df['p_value_corrected'] = corrected_pvals
        result_df['significant'] = result_df['p_value_corrected'] < alpha
    else:
        result_df['p_value_corrected'] = result_df['p_value']
        result_df['significant'] = result_df['p_value'] < alpha
    
    # Sort by p-value
    result_df = result_df.sort_values('p_value')
    
    logger.info(f"Univariate Cox regression completed. {sum(result_df['significant'])} significant features found.")
    return result_df

def filter_by_effect_size(cox_results: pd.DataFrame,
                         hr_threshold_upper: float = 1.2,
                         hr_threshold_lower: float = 0.8,
                         p_threshold: float = 0.05,
                         p_col: str = 'p_adjusted') -> pd.DataFrame:
    """
    Filter features based on effect size (risk ratio)

    Parameters:
    -----
    cox_results: pd.DataFrame
        Results from univariate Cox regression
    hr_threshold_upper: float, default 1.2
        Upper threshold for risk ratio
    hr_threshold_lower: float, default 0.8
        Lower threshold for risk ratio
    p_threshold: float, default 0.05
        p-value threshold
    p_col: str, default 'p_adjusted'
        Column name for p-values

    Returns:
    -----
    pd.DataFrame
        Filtered feature results
    """
    logger.info(f"Filtering features based on effect size, HR thresholds: [{hr_threshold_lower}, {hr_threshold_upper}], p-value threshold: {p_threshold}")
    
    # Ensure p-value column exists
    if p_col not in cox_results.columns:
        p_col = 'p'  # Fallback to uncorrected p-values
        logger.warning(f"Column '{p_col}' not found, using uncorrected p-values")
    
    # Filter significant features
    significant = cox_results[cox_results[p_col] < p_threshold].copy()
    
    # Filter based on risk ratio
    filtered = significant[
        (significant['exp(coef)'] > hr_threshold_upper) | 
        (significant['exp(coef)'] < hr_threshold_lower)
    ]
    
    logger.info(f"Effect size filtering completed. {len(significant)} significant features filtered down to {len(filtered)} features")
    return filtered

def logrank_feature_selection(df: pd.DataFrame,
                             time_col: str,
                             event_col: str,
                             categorical_features: List[str],
                             alpha: float = 0.05,
                             fdr_correction: bool = True) -> pd.DataFrame:
    """
    Use Log-rank test to filter categorical features

    Parameters:
    -----
    df: pd.DataFrame
        Dataframe
    time_col: str
        Name of time column
    event_col: str
        Name of event column
    categorical_features: List[str]
        List of categorical features
    alpha: float, default 0.05
        Significance level
    fdr_correction: bool, default True
        Whether to apply FDR correction

    Returns:
    -----
    pd.DataFrame
        Dataframe containing Log-rank test results
    """
    logger.info("Starting Log-rank feature selection")
    
    results = []
    
    # Perform Log-rank test for each categorical feature
    for feature in categorical_features:
        try:
            # Get unique values of the feature
            unique_values = df[feature].unique()
            
            # If there are too many unique values, skip
            if len(unique_values) > 10:
                logger.warning(f"Feature '{feature}' has too many unique values ({len(unique_values)}) and will be skipped")
                continue
            
            # Compare each pair of categories
            for i, value1 in enumerate(unique_values):
                for value2 in unique_values[i+1:]:
                    # Get two groups of data
                    group1 = df[df[feature] == value1]
                    group2 = df[df[feature] == value2]
                    
                    # Perform Log-rank test
                    result = logrank_test(
                        group1[time_col], group2[time_col],
                        group1[event_col], group2[event_col]
                    )
                    
                    # Save results
                    results.append({
                        'feature': feature,
                        'value1': value1,
                        'value2': value2,
                        'test_statistic': result.test_statistic,
                        'p': result.p_value
                    })
            
        except Exception as e:
            logger.error(f"Error in Log-rank test for feature {feature}: {str(e)}")
    
    # Create results dataframe
    result_df = pd.DataFrame(results)
    
    # Apply FDR correction if requested
    if fdr_correction and not result_df.empty:
        _, corrected_pvals, _, _ = multipletests(
            result_df['p'].values, 
            alpha=alpha, 
            method='fdr_bh'
        )
        result_df['p_adjusted'] = corrected_pvals
    
    # Sort by p-value
    if not result_df.empty:
        p_col = 'p_adjusted' if fdr_correction else 'p'
        result_df = result_df.sort_values(by=p_col)
    
    logger.info(f"Log-rank test completed. {len(categorical_features)} features evaluated")
    return result_df

def plot_hazard_ratios(cox_results: pd.DataFrame, 
                      top_n: int = 20, 
                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot hazard ratio forest plot

    Parameters:
    -----
    cox_results: pd.DataFrame
        Cox regression results
    top_n: int, default 20
        Number of features to display
    figsize: Tuple[int, int], default (12, 10)
        Figure size

    Returns:
    -----
    plt.Figure
        matplotlib figure object
    """
    # Select top N features
    if len(cox_results) > top_n:
        if 'p_adjusted' in cox_results.columns:
            plot_df = cox_results.sort_values('p_adjusted').head(top_n).copy()
        else:
            plot_df = cox_results.sort_values('p').head(top_n).copy()
    else:
        plot_df = cox_results.copy()
    
    # Sort by hazard ratio
    plot_df = plot_df.sort_values('exp(coef)')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot forest plot
    y_pos = np.arange(len(plot_df))
    
    # Plot hazard ratio points and confidence intervals
    ax.scatter(plot_df['exp(coef)'], y_pos, marker='o', s=50, color='blue')
    
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot([row['lower 0.95'], row['upper 0.95']], [i, i], 'b-', alpha=0.6)
    
    # Add vertical line indicating HR=1
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Set Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    
    # Set X-axis to logarithmic scale
    ax.set_xscale('log')
    
    # Add title and labels
    ax.set_title('Hazard Ratio Forest Plot (95% Confidence Interval)', fontsize=14)
    ax.set_xlabel('Hazard Ratio (HR)', fontsize=12)
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def group_features(features: List[str], 
                  clinical_prefix: List[str] = None, 
                  protein_prefix: List[str] = None) -> Dict[str, List[str]]:
    """
    Group features into clinical and protein features

    Parameters:
    -----
    features: List[str]
        List of features
    clinical_prefix: List[str], optional
        List of clinical feature prefixes
    protein_prefix: List[str], optional
        List of protein feature prefixes

    Returns:
    -----
    Dict[str, List[str]]
        Dictionary of grouped features
    """
    if clinical_prefix is None:
        clinical_prefix = ['clinical_', 'demo_', 'lab_']
    
    if protein_prefix is None:
        protein_prefix = ['protein_', 'prot_', 'p_']
    
    clinical_features = []
    protein_features = []
    other_features = []
    
    for feature in features:
        if any(feature.startswith(prefix) for prefix in clinical_prefix):
            clinical_features.append(feature)
        elif any(feature.startswith(prefix) for prefix in protein_prefix):
            protein_features.append(feature)
        else:
            other_features.append(feature)
    
    return {
        'clinical': clinical_features,
        'protein': protein_features,
        'other': other_features
    } 