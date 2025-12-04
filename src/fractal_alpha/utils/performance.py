"""Performance monitoring utilities"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class PerformanceMonitor:
    """Track performance metrics for fractal indicators"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.active_timers = {}
        
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.active_timers[name] = time.perf_counter()
        
    def stop_timer(self, name: str) -> float:
        """Stop timer and record duration"""
        if name not in self.active_timers:
            return 0.0
            
        duration = time.perf_counter() - self.active_timers[name]
        del self.active_timers[name]
        
        self.metrics[name].append(duration)
        return duration
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(name, [])
        
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
            
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.active_timers.clear()