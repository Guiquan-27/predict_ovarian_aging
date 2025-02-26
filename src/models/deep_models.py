# -*- coding: utf-8 -*-
"""
深度学习生存分析模型模块
提供DeepSurv、DeepHit等深度学习生存分析模型
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from lifelines.utils import concordance_index

# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchtuples as tt

# 导入TensorFlow相关库
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..models.base_models import BaseSurvivalModel

logger = logging.getLogger(__name__)

class SurvivalDataset(Dataset):
    """PyTorch生存分析数据集"""
    
    def __init__(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        """
        初始化生存分析数据集
        
        参数:
        -----
        X: np.ndarray
            特征矩阵
        time: np.ndarray
            生存时间
        event: np.ndarray
            事件状态
        """
        self.X = torch.FloatTensor(X)
        self.time = torch.FloatTensor(time)
        self.event = torch.FloatTensor(event)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], (self.time[idx], self.event[idx])


class DeepSurvNet(nn.Module):
    """DeepSurv神经网络模型"""
    
    def __init__(self, in_features: int, hidden_layers: List[int], dropout: float = 0.1, batch_norm: bool = True):
        """
        初始化DeepSurv神经网络
        
        参数:
        -----
        in_features: int
            输入特征数量
        hidden_layers: List[int]
            隐藏层节点数列表
        dropout: float, 默认 0.1
            Dropout比例
        batch_norm: bool, 默认 True
            是否使用批归一化
        """
        super(DeepSurvNet, self).__init__()
        
        layers = []
        prev_nodes = in_features
        
        # 构建隐藏层
        for nodes in hidden_layers:
            # 添加线性层
            layers.append(nn.Linear(prev_nodes, nodes))
            
            # 添加批归一化
            if batch_norm:
                layers.append(nn.BatchNorm1d(nodes))
            
            # 添加激活函数
            layers.append(nn.ReLU())
            
            # 添加Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_nodes = nodes
        
        # 输出层
        layers.append(nn.Linear(prev_nodes, 1))
        
        # 构建网络
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.net(x)


class DeepSurvModel(BaseSurvivalModel):
    """Deep learning survival model (DeepSurv)"""
    
    def __init__(self, name: str = "deepsurv", 
                hidden_layers: List[int] = [32, 16], 
                activation: str = 'relu',
                dropout: float = 0.1,
                batch_norm: bool = True,
                learning_rate: float = 0.001,
                batch_size: int = 64,
                epochs: int = 100,
                patience: int = 10):
        """
        Initialize DeepSurv model
        
        Parameters:
        -----
        name: str, default "deepsurv"
            Model name
        hidden_layers: List[int], default [32, 16]
            Hidden layer dimensions
        activation: str, default 'relu'
            Activation function
        dropout: float, default 0.1
            Dropout rate
        batch_norm: bool, default True
            Whether to use batch normalization
        learning_rate: float, default 0.001
            Learning rate
        batch_size: int, default 64
            Batch size
        epochs: int, default 100
            Number of epochs
        patience: int, default 10
            Patience for early stopping
        """
        super().__init__(name=name)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.feature_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 保存其他参数
        self.kwargs = {}
        
        # 初始化网络
        self.net = None
        self.optimizer = None
        self.criterion = None
        
        # 保存训练历史
        self.history = None
        
        # 保存基线生存函数
        self.baseline_hazard = None
        self.baseline_survival = None
        self.event_times = None
    
    def _negative_log_likelihood(self, risk_pred, y):
        """
        计算负对数似然损失
        
        参数:
        -----
        risk_pred: torch.Tensor
            预测的风险得分
        y: Tuple[torch.Tensor, torch.Tensor]
            (时间, 事件)元组
            
        返回:
        -----
        torch.Tensor
            负对数似然损失
        """
        time, event = y
        
        # 对每个样本，找出所有生存时间大于等于该样本的样本
        # 这是Cox模型中的风险集
        mask = time.unsqueeze(0) <= time.unsqueeze(1)
        
        # 计算每个样本的风险集中的风险得分之和的对数
        risk_set_sum = torch.log(torch.sum(torch.exp(risk_pred) * mask, dim=1) + 1e-5)
        
        # 计算负对数似然
        neg_likelihood = risk_set_sum - risk_pred * event
        
        # 返回平均负对数似然
        return torch.mean(neg_likelihood)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', 
            validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None, **kwargs) -> 'DeepSurvModel':
        """
        训练DeepSurv模型
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        validation_data: Tuple[pd.DataFrame, pd.DataFrame], 可选
            验证数据，格式为(X_val, y_val)
        **kwargs:
            额外的训练参数
            
        返回:
        -----
        self: DeepSurvModel
            训练后的模型实例
        """
        logger.info(f"开始训练DeepSurv模型，设备: {self.device}")
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 准备训练数据
        X_train = X.values.astype(np.float32)
        time_train = y[time_col].values.astype(np.float32)
        event_train = y[event_col].values.astype(np.float32)
        
        # 创建数据集和数据加载器
        train_dataset = SurvivalDataset(X_train, time_train, event_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 准备验证数据
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.values.astype(np.float32)
            time_val = y_val[time_col].values.astype(np.float32)
            event_val = y_val[event_col].values.astype(np.float32)
            
            val_dataset = SurvivalDataset(X_val, time_val, event_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 初始化网络
        self.net = DeepSurvNet(
            in_features=X_train.shape[1],
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        ).to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = self._negative_log_likelihood
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_c_index': []
        }
        
        # 早停设置
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练模式
            self.net.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 将数据移动到设备
                batch_X = batch_X.to(self.device)
                batch_time = batch_y[0].to(self.device)
                batch_event = batch_y[1].to(self.device)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                risk_pred = self.net(batch_X)
                
                # 计算损失
                loss = self.criterion(risk_pred, (batch_time, batch_event))
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                # 累加损失
                train_loss += loss.item() * batch_X.size(0)
            
            # 计算平均训练损失
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss, val_c_index = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_c_index'].append(val_c_index)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state_dict = self.net.state_dict().copy()
                
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val C-index: {val_c_index:.4f}")
                
                # 早停
                if epoch - best_epoch >= self.patience:
                    logger.info(f"早停触发，最佳轮次: {best_epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}")
        
        # 加载最佳模型
        if val_loader is not None and best_state_dict is not None:
            self.net.load_state_dict(best_state_dict)
        
        # 保存训练历史
        self.history = history
        
        # 计算基线生存函数
        self._compute_baseline_hazard(X, y, time_col, event_col)
        
        # 设置模型为已训练
        self.fitted = True
        
        return self
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        在验证集上评估模型
        
        参数:
        -----
        val_loader: DataLoader
            验证数据加载器
            
        返回:
        -----
        Tuple[float, float]
            (验证损失, 验证C-index)
        """
        self.net.eval()
        val_loss = 0.0
        all_risk_scores = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # 将数据移动到设备
                batch_X = batch_X.to(self.device)
                batch_time = batch_y[0].to(self.device)
                batch_event = batch_y[1].to(self.device)
                
                # 前向传播
                risk_pred = self.net(batch_X)
                
                # 计算损失
                loss = self.criterion(risk_pred, (batch_time, batch_event))
                
                # 累加损失
                val_loss += loss.item() * batch_X.size(0)
                
                # 收集预测和真实值
                all_risk_scores.append(risk_pred.cpu().numpy())
                all_times.append(batch_time.cpu().numpy())
                all_events.append(batch_event.cpu().numpy())
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        
        # 计算C-index
        all_risk_scores = np.concatenate(all_risk_scores).flatten()
        all_times = np.concatenate(all_times)
        all_events = np.concatenate(all_events)
        
        val_c_index = concordance_index(all_times, -all_risk_scores, all_events)
        
        return val_loss, val_c_index
    
    def _compute_baseline_hazard(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str, event_col: str) -> None:
        """
        计算基线风险和生存函数
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str
            时间列名
        event_col: str
            事件列名
        """
        # 获取风险得分
        risk_scores = self.predict_risk(X)
        
        # 获取时间和事件
        times = y[time_col].values
        events = y[event_col].values
        
        # 获取唯一的事件时间，并排序
        unique_times = np.sort(np.unique(times[events == 1]))
        self.event_times = unique_times
        
        # 计算基线风险函数
        baseline_hazard = {}
        for t in unique_times:
            # 找出风险集（时间大于等于t的样本）
            risk_set = times >= t
            # 找出在时间t发生事件的样本
            events_at_t = (times == t) & (events == 1)
            
            # 计算风险集中的风险得分之和
            risk_sum = np.sum(np.exp(risk_scores) * risk_set)
            
            # 计算在时间t的基线风险
            if risk_sum > 0:
                baseline_hazard[t] = np.sum(events_at_t) / risk_sum
            else:
                baseline_hazard[t] = 0
        
        self.baseline_hazard = baseline_hazard
        
        # 计算基线生存函数
        baseline_survival = {}
        cumulative_hazard = 0
        
        for t in unique_times:
            cumulative_hazard += baseline_hazard[t]
            baseline_survival[t] = np.exp(-cumulative_hazard)
        
        self.baseline_survival = baseline_survival
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None(使用训练数据中的事件时间)
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 获取风险得分
        risk_scores = self.predict_risk(X)
        
        # 如果未指定时间点，使用训练数据中的事件时间
        if times is None:
            times = self.event_times
        
        # 预测每个样本在每个时间点的生存概率
        survival_probs = np.zeros((len(X), len(times)))
        
        for i, t in enumerate(times):
            # 找到最接近的时间点
            idx = np.searchsorted(self.event_times, t)
            if idx == len(self.event_times):
                idx = len(self.event_times) - 1
            
            # 获取基线生存概率
            baseline_surv = self.baseline_survival[self.event_times[idx]]
            
            # 计算每个样本的生存概率
            survival_probs[:, i] = np.power(baseline_surv, np.exp(risk_scores))
        
        return survival_probs
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测风险得分
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
            
        返回:
        -----
        np.ndarray
            预测的风险得分
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 准备数据
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        # 评估模式
        self.net.eval()
        
        # 预测
        with torch.no_grad():
            risk_scores = self.net(X_tensor).cpu().numpy().flatten()
        
        return risk_scores
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        绘制训练历史
        
        参数:
        -----
        figsize: Tuple[int, int], 默认 (12, 5)
            图形大小
            
        返回:
        -----
        plt.Figure
            matplotlib图形对象
        """
        if self.history is None:
            raise ValueError("模型尚未训练或没有训练历史")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制损失曲线
        axes[0].plot(self.history['train_loss'], label='训练损失')
        if 'val_loss' in self.history and self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='验证损失')
        
        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制C-index曲线
        if 'val_c_index' in self.history and self.history['val_c_index']:
            axes[1].plot(self.history['val_c_index'], label='验证C-index')
            axes[1].set_xlabel('轮次')
            axes[1].set_ylabel('C-index')
            axes[1].set_title('验证C-index')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        return fig


class MTLCoxModel(BaseSurvivalModel):
    """多任务学习Cox模型"""
    
    def __init__(self, name: str = "mtl_cox", shared_layers: List[int] = [64, 32], 
                 task_specific_layers: List[int] = [16], num_tasks: int = 2,
                 dropout: float = 0.1, batch_norm: bool = True, learning_rate: float = 0.001, 
                 batch_size: int = 64, epochs: int = 100, patience: int = 10, **kwargs):
        """
        初始化多任务学习Cox模型
        
        参数:
        -----
        name: str, 默认 "mtl_cox"
            模型名称
        shared_layers: List[int], 默认 [64, 32]
            共享层节点数列表
        task_specific_layers: List[int], 默认 [16]
            任务特定层节点数列表
        num_tasks: int, 默认 2
            任务数量
        dropout: float, 默认 0.1
            Dropout比例
        batch_norm: bool, 默认 True
            是否使用批归一化
        learning_rate: float, 默认 0.001
            学习率
        batch_size: int, 默认 64
            批大小
        epochs: int, 默认 100
            训练轮数
        patience: int, 默认 10
            早停耐心值
        **kwargs:
            其他参数
        """
        super(MTLCoxModel, self).__init__(name=name)
        
        self.shared_layers = shared_layers
        self.task_specific_layers = task_specific_layers
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # 检查TensorFlow是否可用
        if not TF_AVAILABLE:
            raise ImportError("MTLCoxModel需要TensorFlow，但未能导入")
        
        # 保存其他参数
        self.kwargs = kwargs
        
        # 初始化模型
        self.model = None
        
        # 保存训练历史
        self.history = None
        
        # 保存特征名称
        self.feature_names = None
        
        # 保存基线生存函数
        self.baseline_hazards = None
        self.baseline_survivals = None
        self.event_times = None
    
    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """
        构建多任务学习模型
        
        参数:
        -----
        input_dim: int
            输入特征维度
            
        返回:
        -----
        tf.keras.Model
            构建的模型
        """
        # 输入层
        inputs = tf.keras.Input(shape=(input_dim,))
        
        # 共享层
        x = inputs
        for units in self.shared_layers:
            x = layers.Dense(units)(x)
            if self.batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        # 任务特定层和输出
        outputs = []
        for task in range(self.num_tasks):
            task_x = x
            for units in self.task_specific_layers:
                task_x = layers.Dense(units)(task_x)
                if self.batch_norm:
                    task_x = layers.BatchNormalization()(task_x)
                task_x = layers.Activation('relu')(task_x)
                if self.dropout > 0:
                    task_x = layers.Dropout(self.dropout)(task_x)
            
            # 输出层
            task_output = layers.Dense(1, name=f'task_{task}_output')(task_x)
            outputs.append(task_output)
        
        # 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _negative_log_likelihood(self, y_true, y_pred):
        """
        计算负对数似然损失
        
        参数:
        -----
        y_true: tf.Tensor
            真实值，包含时间和事件
        y_pred: tf.Tensor
            预测的风险得分
            
        返回:
        -----
        tf.Tensor
            负对数似然损失
        """
        # 分离时间和事件
        time, event = tf.split(y_true, 2, axis=1)
        
        # 计算风险集掩码
        mask = tf.cast(time >= tf.transpose(time), dtype=tf.float32)
        
        # 计算风险集中的风险得分之和的对数
        risk_set_sum = tf.math.log(tf.reduce_sum(tf.exp(y_pred) * mask, axis=1) + 1e-5)
        
        # 计算负对数似然
        neg_likelihood = risk_set_sum - y_pred * event
        
        # 返回平均负对数似然
        return tf.reduce_mean(neg_likelihood)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, time_col: str = 'time', event_col: str = 'event', 
            task_groups: Optional[List[List[str]]] = None, validation_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None, 
            **kwargs) -> 'MTLCoxModel':
        """
        训练多任务学习Cox模型
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        y: pd.DataFrame
            目标变量(时间和事件)
        time_col: str, 默认 'time'
            时间列名
        event_col: str, 默认 'event'
            事件列名
        task_groups: List[List[str]], 可选
            任务特征分组，每个子列表包含一个任务的特征名称
        validation_data: Tuple[pd.DataFrame, pd.DataFrame], 可选
            验证数据，格式为(X_val, y_val)
        **kwargs:
            额外的训练参数
            
        返回:
        -----
        self: MTLCoxModel
            训练后的模型实例
        """
        logger.info(f"开始训练多任务学习Cox模型")
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 如果未指定任务分组，则平均分配特征
        if task_groups is None:
            features = X.columns.tolist()
            features_per_task = len(features) // self.num_tasks
            task_groups = [features[i:i+features_per_task] for i in range(0, len(features), features_per_task)]
            
            # 确保任务数量正确
            if len(task_groups) > self.num_tasks:
                # 将最后一组合并到倒数第二组
                task_groups[-2].extend(task_groups[-1])
                task_groups.pop()
        
        # 更新任务数量
        self.num_tasks = len(task_groups)
        
        # 准备训练数据
        X_train = X.values.astype(np.float32)
        time_train = y[time_col].values.astype(np.float32).reshape(-1, 1)
        event_train = y[event_col].values.astype(np.float32).reshape(-1, 1)
        y_train = np.concatenate([time_train, event_train], axis=1)
        
        # 准备验证数据
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.values.astype(np.float32)
            time_val = y_val[time_col].values.astype(np.float32).reshape(-1, 1)
            # 接上文代码
            event_val = y_val[event_col].values.astype(np.float32).reshape(-1, 1)
            y_val = np.concatenate([time_val, event_val], axis=1)
        else:
            X_val, y_val = None, None
        
        # 构建模型
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim)
        
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=[self._negative_log_likelihood] * self.num_tasks
        )
        
        # 设置早停
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # 训练模型
        history = self.model.fit(
            X_train,
            [y_train] * self.num_tasks,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, [y_val] * self.num_tasks) if validation_data is not None else None,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.fitted = True
        self.history = history.history
        
        logger.info(f"多任务学习Cox模型训练完成")
        return self
    
    def predict(self, X: pd.DataFrame, times: Optional[List[float]] = None) -> np.ndarray:
        """
        预测生存概率
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
        times: List[float], 可选
            预测时间点，默认为None
            
        返回:
        -----
        np.ndarray
            预测的生存概率
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 转换输入数据
        X_test = X.values.astype(np.float32)
        
        # 获取风险得分
        risk_scores = self.predict_risk(X)
        
        # 如果未指定时间点，则无法计算生存概率
        if times is None:
            logger.warning("未指定时间点，无法计算生存概率")
            return risk_scores
        
        # 使用基线生存函数计算生存概率
        # 注意：这里需要实现基线生存函数的估计，这是一个简化版本
        baseline_survival = np.exp(-np.outer(risk_scores, np.array(times)))
        
        return baseline_survival
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测风险得分
        
        参数:
        -----
        X: pd.DataFrame
            特征矩阵
            
        返回:
        -----
        np.ndarray
            预测的风险得分
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 转换输入数据
        X_test = X.values.astype(np.float32)
        
        # 获取所有任务的预测
        task_predictions = self.model.predict(X_test)
        
        # 合并任务预测（简单平均）
        if isinstance(task_predictions, list):
            risk_scores = np.mean([pred.flatten() for pred in task_predictions], axis=0)
        else:
            risk_scores = task_predictions.flatten()
        
        return risk_scores
    
    def save(self, path: str):
        """
        保存模型
        
        参数:
        -----
        path: str
            保存路径
        """
        if not self.fitted:
            raise ValueError("模型尚未训练")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        
        # 保存模型参数
        params = {
            'name': self.name,
            'shared_layers': self.shared_layers,
            'task_specific_layers': self.task_specific_layers,
            'num_tasks': self.num_tasks,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }
        
        joblib.dump(params, f"{path}_params.pkl")
    
    @classmethod
    def load(cls, path: str) -> 'MTLCoxModel':
        """
        加载模型
        
        参数:
        -----
        path: str
            模型路径
            
        返回:
        -----
        MTLCoxModel
            加载的模型实例
        """
        # 加载模型参数
        params = joblib.load(f"{path}_params.pkl")
        
        # 创建模型实例
        model_instance = cls(
            name=params['name'],
            shared_layers=params['shared_layers'],
            task_specific_layers=params['task_specific_layers'],
            num_tasks=params['num_tasks'],
            dropout=params['dropout'],
            batch_norm=params['batch_norm'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            patience=params['patience']
        )
        
        # 加载模型
        model_instance.model = tf.keras.models.load_model(
            path,
            custom_objects={'_negative_log_likelihood': model_instance._negative_log_likelihood}
        )
        
        # 设置其他属性
        model_instance.feature_names = params['feature_names']
        model_instance.fitted = params['fitted']
        
        return model_instance


def create_deep_model(model_type: str, **kwargs) -> BaseSurvivalModel:
    """
    创建指定类型的深度学习生存分析模型
    
    参数:
    -----
    model_type: str
        模型类型，可选 'deepsurv', 'mtlcox'
    **kwargs:
        传递给模型构造函数的参数
        
    返回:
    -----
    BaseSurvivalModel
        创建的模型实例
    """
    model_type = model_type.lower()
    
    if model_type == 'deepsurv':
        return DeepSurvModel(**kwargs)
    elif model_type == 'mtlcox':
        if not TF_AVAILABLE:
            raise ImportError("MTLCoxModel需要TensorFlow，但未找到TensorFlow")
        return MTLCoxModel(**kwargs)
    else:
        raise ValueError(f"不支持的深度学习模型类型: {model_type}")