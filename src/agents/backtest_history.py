"""
📚 BACKTEST HISTORY MANAGER
Tracks which symbols have been tested to avoid redundant backtesting

Features:
- Record test dates and results
- Check if symbol was recently tested
- Configurable retest intervals
- Automatic cleanup of old records
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

class BacktestHistory:
    """Manages backtest history to prevent redundant testing"""
    
    def __init__(self, history_file='aegs_backtest_history.json'):
        self.history_file = history_file
        self.history = self._load_history()
        
        # Configuration
        self.retest_interval_days = 30  # Retest symbols after 30 days
        self.max_history_days = 90  # Keep history for 90 days
    
    def _load_history(self) -> Dict:
        """Load history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'backtest_history': {},
                'metadata': {
                    'created': datetime.now().strftime('%Y-%m-%d'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d'),
                    'total_symbols_tested': 0
                }
            }
    
    def save_history(self):
        """Save history to file"""
        self.history['metadata']['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def was_recently_tested(self, symbol: str) -> bool:
        """Check if symbol was tested recently"""
        if symbol not in self.history['backtest_history']:
            return False
        
        last_test = self.history['backtest_history'][symbol].get('last_tested')
        if not last_test:
            return False
        
        # Parse date and check interval
        last_test_date = datetime.strptime(last_test, '%Y-%m-%d')
        days_since_test = (datetime.now() - last_test_date).days
        
        return days_since_test < self.retest_interval_days
    
    def record_test(self, symbol: str, results: Dict):
        """Record that a symbol was tested"""
        if symbol not in self.history['backtest_history']:
            self.history['backtest_history'][symbol] = {
                'first_tested': datetime.now().strftime('%Y-%m-%d'),
                'test_count': 0,
                'results': []
            }
        
        record = self.history['backtest_history'][symbol]
        record['last_tested'] = datetime.now().strftime('%Y-%m-%d')
        record['test_count'] += 1
        
        # Store key results
        result_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'excess_return': results.get('excess_return', 0),
            'win_rate': results.get('win_rate', 0),
            'trades': results.get('total_trades', 0)
        }
        
        record['results'].append(result_summary)
        
        # Keep only recent results
        if len(record['results']) > 5:
            record['results'] = record['results'][-5:]
        
        # Update metadata
        self.history['metadata']['total_symbols_tested'] = len(self.history['backtest_history'])
        
        self.save_history()
    
    def get_untested_symbols(self, candidates: list) -> list:
        """Filter out recently tested symbols"""
        untested = []
        skipped = []
        
        for symbol in candidates:
            if self.was_recently_tested(symbol):
                skipped.append(symbol)
            else:
                untested.append(symbol)
        
        if skipped:
            print(f"   ℹ️ Skipping {len(skipped)} recently tested symbols: {', '.join(skipped[:5])}")
        
        return untested
    
    def cleanup_old_records(self):
        """Remove records older than max_history_days"""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        
        symbols_to_remove = []
        for symbol, data in self.history['backtest_history'].items():
            last_tested = datetime.strptime(data['last_tested'], '%Y-%m-%d')
            if last_tested < cutoff_date:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.history['backtest_history'][symbol]
        
        if symbols_to_remove:
            print(f"   🧹 Cleaned up {len(symbols_to_remove)} old records")
            self.save_history()
    
    def get_test_history(self, symbol: str) -> Optional[Dict]:
        """Get test history for a symbol"""
        return self.history['backtest_history'].get(symbol)
    
    def get_summary(self) -> Dict:
        """Get summary of backtest history"""
        total_symbols = len(self.history['backtest_history'])
        
        # Count by test recency
        tested_today = 0
        tested_week = 0
        tested_month = 0
        
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        for symbol, data in self.history['backtest_history'].items():
            last_tested = datetime.strptime(data['last_tested'], '%Y-%m-%d')
            
            if last_tested.date() == today.date():
                tested_today += 1
            if last_tested >= week_ago:
                tested_week += 1
            if last_tested >= month_ago:
                tested_month += 1
        
        return {
            'total_symbols_tested': total_symbols,
            'tested_today': tested_today,
            'tested_this_week': tested_week,
            'tested_this_month': tested_month,
            'retest_interval_days': self.retest_interval_days
        }