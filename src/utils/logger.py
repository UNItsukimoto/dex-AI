# -*- coding: utf-8 -*-
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """シンプルなロガーを取得"""
    logger = logging.getLogger(name)
    
    # すでに設定されている場合はそのまま返す
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger