# -*- coding: utf-8 -*-
import time
import functools
from typing import Callable
import asyncio

def timer(func: Callable) -> Callable:
    """関数の実行時間を計測するデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def async_timer(func: Callable) -> Callable:
    """非同期関数の実行時間を計測するデコレータ"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """リトライ機能を追加するデコレータ"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        print(f"Max retries reached for {func.__name__}")
                        raise
                    print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    await asyncio.sleep(delay * (attempt + 1))
            
        return wrapper
    return decorator