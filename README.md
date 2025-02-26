# 生存分析模型开发框架

这是一个基于Python的生存分析模型开发框架，用于构建和评估生存预测模型。该框架支持多种生存分析方法，包括Cox比例风险模型、随机生存森林、深度学习生存模型等，并提供完整的数据预处理、特征工程、模型训练、评估和解释工具。

## 功能特点

-   **数据预处理**：
    -   缺失值多重插补（MICE方法，实现Rubin's规则，生成k=5套插补后数据）
    -   异常值检测（Z-score方法）与处理（1%-99%截断法）
    -   特征标准化与编码（连续变量Z-score标准化，分类变量独热/目标/有序编码）
    -   数据质量检查与报告

-   **特征工程**：
    -   单变量Cox回归筛选 + FDR校正（Benjamini-Hochberg方法）
    -   效应量过滤（保留HR>1.2或<0.8的特征）
    -   Bootstrap特征稳定性评估
    -   特征分组（临床组和蛋白质组）

-   **模型开发**：
    -   **基准模型**：
        -   CoxPH：基线参考模型
        -   RSF（Random Survival Forest）：捕捉非线性关系
        -   CoxBoost：处理高维数据
    -   **进阶模型**：
        -   DeepSurv：深度学习生存模型
        -   MTL-Cox：多任务学习模型 (需要安装 TensorFlow)
    -   **集成策略**：
        -   **多层级集成方案**：
            1.  **第一层**：基线模型、进阶模型，每个模型使用不同特征子集训练多个版本
            2.  **第二层**：使用Stacking方法，以第一层模型预测作为特征，采用简单但稳健的meta-learner
            3.  **第三层**：基于模型表现动态调整权重，使用Bayesian Model Averaging整合预测结果
        -   **集成优化策略**：
            -   使用交叉验证评估每个基模型的稳定性
            -   采用SHAP值分析模型贡献度
            -   动态剔除表现不稳定的模型
            -   根据预测置信度自适应调整权重

-   **模型训练与验证**：
    -   **重复分层交叉验证**：
        -   采用10折分层交叉验证，确保每折中事件比例一致
        -   重复5次以降低随机性影响，特别重要在小样本情况下
    -   **贝叶斯超参数优化**：
        -   使用`hyperopt`库实现
        -   目标最大化C-index
    -   **评估指标**：
        -   **主要指标**：C-index（判别能力）、Time-dependent AUC（3年/5年/10年时间窗）
        -   **校准性**：分段Brier Score及校准曲线（0-3年、3-5年、5-10年）
        -   **临床决策效用**：决策曲线分析（DCA）评估不同风险分层对干预决策的临床价值

-   **模型解释**：
    -   **全局模型解释**：
        -   SHAP值计算与可视化
        -   Bootstrap稳定性评估
        -   特征交互分析（部分依赖图、SHAP交互值、生物学通路关联分析）
    -   **个体化风险预测**：
        -   生成个体5年/10年风险预测
        -   展示预测的不确定性范围
        -   标注影响个体风险的主要因素

## 安装方法

### 必要环境

-   Python 3.8+
-   系统依赖：gcc, g++（用于编译某些依赖包）

### 使用pip安装

1.  **克隆仓库**：

    ```bash
    git clone https://github.com/yourusername/survival_model.git
    cd survival_model
    ```

2.  **创建虚拟环境 (可选，强烈建议)**：

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate    # Windows
    ```

3.  **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

4.  **安装开发模式**：

    ```bash
    pip install -e .
    ```

### 可选组件

-   **GPU支持 (深度学习)**：

    ```bash
    pip install tensorflow-gpu  # 替代tensorflow
    ```
    或者
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # 根据您的CUDA版本选择
    ```

-   **R语言支持 (多重插补)**：

    ```bash
    # 安装rpy2
    pip install rpy2

    # 在R中安装miceRanger包
    R -e "install.packages('miceRanger', repos='https://cran.r-project.org')"
    ```