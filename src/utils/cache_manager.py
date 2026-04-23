"""
统一缓存管理器

为新闻、股吧、量化情绪等数据添加TTL缓存，减少重复API调用。
"""

import time
import json
import os
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from src.utils.logging_config import setup_logger

logger = setup_logger('cache_manager')


class CacheEntry:
    """缓存条目"""
    def __init__(self, data: Any, ttl_seconds: int):
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class CacheManager:
    """
    缓存管理器
    
    支持：
    1. 内存缓存（快速访问）
    2. 磁盘缓存（持久化）
    3. TTL过期策略
    4. 自动清理
    """
    
    def __init__(self, cache_dir: str = "src/data/cache", default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.memory_cache: dict[str, CacheEntry] = {}
        
        # 不同数据类型的TTL配置（秒）
        self.ttl_config = {
            "stock_news": 3600,          # 新闻：1小时
            "guba_sentiment": 1800,      # 股吧：30分钟
            "quant_sentiment": 900,      # 量化情绪：15分钟
            "north_money": 86400,        # 北向资金：1天
            "financial_metrics": 86400,  # 财务指标：1天
            "macro_analysis": 3600,      # 宏观分析：1小时
        }
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载磁盘缓存
        self._load_disk_cache()
    
    def get_or_fetch(self, key: str, fetcher: Callable, ttl: Optional[int] = None, **kwargs) -> Any:
        """
        获取缓存或执行fetcher
        
        Args:
            key: 缓存键
            fetcher: 数据获取函数
            ttl: 过期时间（秒），None则使用默认值
            **kwargs: 传递给fetcher的参数
            
        Returns:
            缓存数据或新获取的数据
        """
        # 1. 检查内存缓存
        if key in self.memory_cache and not self.memory_cache[key].is_expired():
            logger.debug(f"📦 [CACHE HIT] 内存缓存: {key}")
            return self.memory_cache[key].data
        
        # 2. 检查磁盘缓存
        disk_data = self._load_from_disk(key)
        if disk_data is not None:
            logger.debug(f"💾 [CACHE HIT] 磁盘缓存: {key}")
            # 写入内存缓存
            entry_ttl = ttl or self._get_default_ttl(key)
            self.memory_cache[key] = CacheEntry(disk_data, entry_ttl)
            return disk_data
        
        # 3. 缓存未命中，执行fetcher
        logger.info(f"🔄 [CACHE MISS] 获取新数据: {key}")
        try:
            data = fetcher(**kwargs)
            
            # 存入缓存
            entry_ttl = ttl or self._get_default_ttl(key)
            self.memory_cache[key] = CacheEntry(data, entry_ttl)
            self._save_to_disk(key, data)
            
            logger.info(f"✅ [CACHE STORED] 已缓存: {key}, TTL={entry_ttl}s")
            return data
            
        except Exception as e:
            logger.error(f"❌ [CACHE ERROR] 获取数据失败: {key}, error={e}")
            raise
    
    def invalidate(self, key: str):
        """使缓存失效"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        self._remove_from_disk(key)
        logger.info(f"🗑️  [CACHE INVALIDATED] {key}")
    
    def clear_all(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        self._clear_disk_cache()
        logger.info("🗑️  [CACHE CLEARED] 所有缓存已清空")
    
    def cleanup_expired(self):
        """清理过期缓存"""
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.info(f"🧹 [CACHE CLEANUP] 清理了{len(expired_keys)}个过期缓存")
    
    def _get_default_ttl(self, key: str) -> int:
        """根据key获取默认TTL"""
        for prefix, ttl in self.ttl_config.items():
            if prefix in key:
                return ttl
        return self.default_ttl
    
    def _load_disk_cache(self):
        """从磁盘加载缓存索引"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    cache_index = json.load(f)
                logger.info(f"💾 加载磁盘缓存索引: {len(cache_index)}个条目")
            except Exception as e:
                logger.warning(f"加载磁盘缓存索引失败: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """从磁盘加载单个缓存"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                # 检查是否过期
                if 'expires_at' in cached and time.time() > cached['expires_at']:
                    os.remove(cache_file)
                    return None
                
                return cached.get('data')
            except Exception as e:
                logger.warning(f"从磁盘加载缓存失败 {key}: {e}")
        return None
    
    def _save_to_disk(self, key: str, data: Any):
        """保存缓存到磁盘"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        ttl = self._get_default_ttl(key)
        
        try:
            cached = {
                'data': data,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存到磁盘失败 {key}: {e}")
    
    def _remove_from_disk(self, key: str):
        """从磁盘删除缓存"""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception as e:
                logger.warning(f"删除磁盘缓存失败 {key}: {e}")
    
    def _clear_disk_cache(self):
        """清空磁盘缓存"""
        cache_files = [
            f for f in os.listdir(self.cache_dir)
            if f.endswith('.json') and f != 'cache_index.json'
        ]
        for filename in cache_files:
            try:
                os.remove(os.path.join(self.cache_dir, filename))
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {filename}: {e}")


# 全局缓存实例
_global_cache = None


def get_cache() -> CacheManager:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


# 便捷装饰器
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    缓存装饰器
    
    Usage:
        @cached(ttl=3600, key_prefix="news")
        def get_stock_news(symbol: str, max_news: int = 20):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # 生成缓存键
            key_parts = [key_prefix or func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)
            
            # 获取或执行
            return cache.get_or_fetch(cache_key, func, ttl=ttl, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator
