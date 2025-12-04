"""Cache implementations for fractal alpha system"""

from typing import Any, Optional, Dict
import json
import time
from collections import OrderedDict


class MemoryCache:
    """Simple in-memory LRU cache"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
            
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None
            
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.timestamps[oldest]
            
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()


class RedisCache:
    """Redis cache implementation (stub for now)"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        # In production, would connect to Redis
        # For now, use memory cache as fallback
        self.memory_cache = MemoryCache()
        print("⚠️ Redis not configured, using memory cache")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.memory_cache.get(key)
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        self.memory_cache.set(key, value)