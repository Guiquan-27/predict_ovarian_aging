# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志记录功能，支持记录到控制台和文件
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, List, Optional, Union, Any
import yaml
import colorlog
import traceback
import inspect
import platform
import socket
import getpass
from datetime import datetime

# 默认日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# 彩色日志格式
COLOR_LOG_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 彩色日志颜色映射
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white"
}

def setup_logger(name: str, 
                log_level: str = "info", 
                log_file: Optional[str] = None,
                log_dir: Optional[str] = None,
                use_color: bool = True,
                console_output: bool = True,
                file_rotation: bool = True,
                max_bytes: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5,
                capture_warnings: bool = True,
                log_system_info: bool = True) -> logging.Logger:
    """
    Setup logger
    
    Parameters:
    -----
    name: str
        Logger name
    log_level: str, default "info"
        Log level, can be "debug", "info", "warning", "error", "critical"
    log_file: str, optional
        Log file path, default None (no file logging)
    log_dir: str, optional
        Log directory path, if provided, log file will be created in this directory
    use_color: bool, default True
        Whether to use colored logs
    console_output: bool, default True
        Whether to output to console
    file_rotation: bool, default True
        Whether to enable log file rotation
    max_bytes: int, default 10MB
        Maximum size of a single log file
    backup_count: int, default 5
        Number of log files to keep
    capture_warnings: bool, default True
        Whether to capture Python warnings
    log_system_info: bool, default True
        Whether to log system information
        
    Returns:
    -----
    logging.Logger
        Configured logger
    """
    # 获取日志级别
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式化器
    if use_color and console_output:
        formatter = colorlog.ColoredFormatter(
            COLOR_LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=LOG_COLORS
        )
    else:
        formatter = logging.Formatter(
            DEFAULT_LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file or log_dir:
        # 如果提供了日志目录，创建日志文件路径
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if not log_file:
                # 使用当前时间创建日志文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            else:
                log_file = os.path.join(log_dir, log_file)
        
        # 创建文件处理器
        if file_rotation:
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 捕获Python警告
    if capture_warnings:
        logging.captureWarnings(True)
    
    # 记录系统信息
    if log_system_info:
        logger.info("=" * 80)
        logger.info("日志系统初始化")
        logger.info(f"日志级别: {log_level.upper()}")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"操作系统: {platform.platform()}")
        logger.info(f"主机名: {socket.gethostname()}")
        logger.info(f"用户: {getpass.getuser()}")
        logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
    
    return logger

def load_logger_config(config_file: str) -> logging.Logger:
    """
    从配置文件加载日志配置
    
    参数:
    -----
    config_file: str
        配置文件路径
        
    返回:
    -----
    logging.Logger
        配置好的日志记录器
    """
    # 加载配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取日志配置
    logger_config = config.get('logger', {})
    
    # 设置日志记录器
    return setup_logger(
        name=logger_config.get('name', 'survival_model'),
        log_level=logger_config.get('level', 'info'),
        log_file=logger_config.get('file'),
        log_dir=logger_config.get('dir'),
        use_color=logger_config.get('use_color', True),
        console_output=logger_config.get('console_output', True),
        file_rotation=logger_config.get('file_rotation', True),
        max_bytes=logger_config.get('max_bytes', 10 * 1024 * 1024),
        backup_count=logger_config.get('backup_count', 5),
        capture_warnings=logger_config.get('capture_warnings', True),
        log_system_info=logger_config.get('log_system_info', True)
    )

class LoggerManager:
    """日志管理器，用于管理多个日志记录器"""
    
    def __init__(self):
        """初始化日志管理器"""
        self.loggers = {}
    
    def get_logger(self, name: str, **kwargs) -> logging.Logger:
        """
        获取日志记录器，如果不存在则创建
        
        参数:
        -----
        name: str
            日志记录器名称
        **kwargs:
            传递给setup_logger的参数
            
        返回:
        -----
        logging.Logger
            日志记录器
        """
        if name not in self.loggers:
            self.loggers[name] = setup_logger(name, **kwargs)
        return self.loggers[name]
    
    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """
        获取所有日志记录器
        
        返回:
        -----
        Dict[str, logging.Logger]
            日志记录器字典
        """
        return self.loggers

# 创建全局日志管理器
logger_manager = LoggerManager()

def get_logger(name: str = "survival_model", **kwargs) -> logging.Logger:
    """
    获取日志记录器
    
    参数:
    -----
    name: str, 默认 "survival_model"
        日志记录器名称
    **kwargs:
        传递给setup_logger的参数
        
    返回:
    -----
    logging.Logger
        日志记录器
    """
    return logger_manager.get_logger(name, **kwargs)

def log_function_call(logger: logging.Logger, level: str = "debug"):
    """
    装饰器：记录函数调用
    
    参数:
    -----
    logger: logging.Logger
        日志记录器
    level: str, 默认 "debug"
        日志级别
        
    返回:
    -----
    Callable
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取函数信息
            func_name = func.__name__
            module_name = func.__module__
            
            # 记录函数调用
            log_method = getattr(logger, level.lower())
            log_method(f"调用函数: {module_name}.{func_name}")
            
            # 记录参数
            arg_str = ", ".join([str(arg) for arg in args])
            kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            params_str = ", ".join(filter(None, [arg_str, kwarg_str]))
            log_method(f"参数: {params_str}")
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 调用函数
                result = func(*args, **kwargs)
                
                # 记录结束时间和执行时间
                end_time = time.time()
                execution_time = end_time - start_time
                log_method(f"函数 {func_name} 执行完成，耗时: {execution_time:.4f}秒")
                
                return result
            except Exception as e:
                # 记录异常
                logger.error(f"函数 {func_name} 执行出错: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        return wrapper
    
    return decorator

def log_exception(logger: logging.Logger, level: str = "error"):
    """
    装饰器：记录异常
    
    参数:
    -----
    logger: logging.Logger
        日志记录器
    level: str, 默认 "error"
        日志级别
        
    返回:
    -----
    Callable
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取函数信息
                func_name = func.__name__
                module_name = func.__module__
                
                # 记录异常
                log_method = getattr(logger, level.lower())
                log_method(f"函数 {module_name}.{func_name} 发生异常: {str(e)}")
                log_method(traceback.format_exc())
                
                # 重新抛出异常
                raise
        
        return wrapper
    
    return decorator

# 主函数：测试日志功能
if __name__ == "__main__":
    # 创建日志记录器
    logger = setup_logger(
        name="test_logger",
        log_level="debug",
        log_dir="logs",
        use_color=True
    )
    
    # 测试不同级别的日志
    logger.debug("这是一条调试日志")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")
    
    # 测试装饰器
    @log_function_call(logger)
    def test_function(a, b, c=None):
        logger.info(f"函数内部: a={a}, b={b}, c={c}")
        return a + b
    
    result = test_function(1, 2, c=3)
    logger.info(f"函数返回值: {result}")
    
    # 测试异常记录
    @log_exception(logger)
    def error_function():
        logger.info("即将抛出异常")
        raise ValueError("测试异常")
    
    try:
        error_function()
    except ValueError:
        logger.info("异常已捕获") 