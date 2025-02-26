# Survival Analysis Model Development Framework

This is a Python-based survival analysis model development framework for building and evaluating survival prediction models. The framework supports various survival analysis methods, including Cox proportional hazards model, random survival forest, deep learning survival models, and provides comprehensive tools for data preprocessing, feature engineering, model training, evaluation, and interpretation.

## Features

-   **Data Preprocessing**:
    -   Multiple imputation for missing values (MICE method, implementing Rubin's rules, generating k=5 imputed datasets)
    -   Outlier detection (Z-score method) and handling (1%-99% truncation)
    -   Feature standardization and encoding (Z-score for continuous variables, one-hot/target/ordinal encoding for categorical variables)
    -   Data quality checking and reporting

-   **Feature Engineering**:
    -   Univariate Cox regression selection + FDR correction (Benjamini-Hochberg method)
    -   Effect size filtering (keeping features with HR>1.2 or <0.8)
    -   Bootstrap feature stability assessment
    -   Feature grouping (clinical and protein groups)

-   **Model Development**:
    -   **Base Models**:
        -   CoxPH: Baseline reference model
        -   RSF (Random Survival Forest): Capturing non-linear relationships
        -   CoxBoost: Handling high-dimensional data
    -   **Advanced Models**:
        -   DeepSurv: Deep learning survival model
        -   MTL-Cox: Multi-task learning model (requires TensorFlow)
    -   **Ensemble Strategies**:
        -   **Multi-level Ensemble Approach**:
            1.  **First Level**: Baseline models and advanced models, each trained on different feature subsets
            2.  **Second Level**: Using Stacking method with predictions from the first level as features and a simple but robust meta-learner
            3.  **Third Level**: Dynamic weight adjustment based on model performance using Bayesian Model Averaging
        -   **Ensemble Optimization Strategies**:
            -   Cross-validation for evaluating base model stability
            -   SHAP value analysis for model contribution assessment
            -   Dynamic elimination of unstable models
            -   Adaptive weight adjustment based on prediction confidence

-   **Model Training and Validation**:
    -   **Repeated Stratified Cross-Validation**:
        -   10-fold stratified cross-validation ensuring consistent event ratios in each fold
        -   Repeated 5 times to reduce randomness, especially important for small samples
    -   **Bayesian Hyperparameter Optimization**:
        -   Using `hyperopt` library
        -   Maximizing C-index
    -   **Evaluation Metrics**:
        -   **Primary Metrics**: C-index (discrimination), Time-dependent AUC (3/5/10-year time windows)
        -   **Calibration**: Segmented Brier Score and calibration curves (0-3, 3-5, 5-10 years)
        -   **Clinical Decision Utility**: Decision Curve Analysis (DCA) to evaluate clinical value of risk stratification

-   **Model Interpretation**:
    -   **Global Model Interpretation**:
        -   SHAP value calculation and visualization
        -   Bootstrap stability assessment
        -   Feature interaction analysis (PDP, SHAP interaction values, biological pathway correlation analysis)
    -   **Individualized Risk Prediction**:
        -   Generate 5-year/10-year risk prediction for individuals
        -   Display uncertainty range of predictions
        -   Highlight key factors affecting individual risk

## Installation

### Requirements

-   Python 3.8+
-   System dependencies: gcc, g++ (for compiling certain dependency packages)

### Installation with pip

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/survival_model.git
    cd survival_model
    ```

2.  **Create a virtual environment (optional but recommended)**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate    # Windows
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install in development mode**:

    ```bash
    pip install -e .
    ```

### Optional Components

-   **GPU support (for deep learning)**:

    ```bash
    pip install tensorflow-gpu  # Replace tensorflow
    ```
    or
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Choose based on your CUDA version
    ```

-   **R language support (for multiple imputation)**:

    ```bash
    # Install rpy2
    pip install rpy2

    # Install miceRanger package in R
    R -e "install.packages('miceRanger', repos='https://cran.r-project.org')"
    ```