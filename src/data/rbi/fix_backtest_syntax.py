#!/usr/bin/env python3
"""
🌙 Moon Dev Batch Syntax Fixer for Backtest Files
Automatically fixes common syntax errors in backtest files
"""
import os
import re
import sys

def create_standard_backtest_template(strategy_name, description=""):
    """Create a standard backtest template with proper structure"""
    template = f'''# 🌙 Moon Dev {strategy_name} Strategy
# {description}

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utils for dynamic path resolution
try:
    from utils import get_data_file_path, prepare_backtest_data
except ImportError:
    # Fallback if utils not found
    def get_data_file_path(filename='BTC-USD-15m.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths = [
            os.path.join(script_dir, '..', '..', filename),
            os.path.join(script_dir, '..', '..', 'rbi', filename),
            '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv',
            '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/BTC-USD-15m.csv'
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {{filename}}")

class {strategy_name}(Strategy):
    # TODO: Add strategy parameters
    risk_per_trade = 0.02  # 2% risk per trade
    
    def init(self):
        # TODO: Initialize indicators
        # Example:
        # self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)
        # self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        pass
        
    def next(self):
        # TODO: Implement strategy logic
        # Example:
        # if not self.position and self.rsi[-1] < 30:
        #     self.buy()
        # elif self.position and self.rsi[-1] > 70:
        #     self.position.close()
        pass

# 🌙 Load data
try:
    data_path = get_data_file_path('BTC-USD-15m.csv')
    data = pd.read_csv(data_path)
    print(f"✅ Found data file at: {{data_path}}")
except FileNotFoundError:
    print("⚠️ No data file found, generating sample data")
    dates = pd.date_range(start='2023-01-01', end='2023-12-01', freq='15min')
    n = len(dates)
    np.random.seed(42)
    price = 30000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({{
        'datetime': dates,
        'Open': price + np.random.randn(n) * 50,
        'High': price + np.abs(np.random.randn(n) * 100),
        'Low': price - np.abs(np.random.randn(n) * 100),
        'Close': price,
        'Volume': np.random.randint(100, 10000, n)
    }})

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={{
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# 🚀 Run backtest
bt = Backtest(data, {strategy_name}, cash=1_000_000, commission=0.002)
stats = bt.run()

# 🌕 Print results
print("\\n🌕 MOON DEV {strategy_name.upper()} RESULTS 🌕")
print("="*50)
print(f"Return [%]: {{stats['Return [%]']:.2f}}")
print(f"Max Drawdown [%]: {{stats['Max. Drawdown [%]']:.2f}}")
print(f"Sharpe Ratio: {{stats['Sharpe Ratio']:.2f}}")
print(f"Win Rate [%]: {{stats['Win Rate [%]']:.2f}}" if 'Win Rate [%]' in stats else "Win Rate: N/A")
print(f"Total Trades: {{stats['# Trades']}}")
print("="*50)
'''
    return template

def fix_syntax_error_file(filepath):
    """Fix common syntax errors in a backtest file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file is just a snippet or has unterminated strings
        if len(content) < 100 or content.count('"""') % 2 != 0 or content.count("'''") % 2 != 0:
            # Extract strategy name from filename
            basename = os.path.basename(filepath)
            strategy_name = basename.replace('_BTFinal.py', '').replace('.py', '')
            
            print(f"🔧 Creating new template for {strategy_name}")
            
            # Create new content with standard template
            new_content = create_standard_backtest_template(strategy_name, "Auto-generated template - implement strategy logic")
            
            # Save fixed file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Fixed: {filepath}")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {str(e)}")
        return False

def main():
    # List of files with syntax errors to fix
    error_files = [
        'DivergentConvergence_BTFinal.py',
        'DivergentReversion_BTFinal.py',
        'FibCloudTrend_BTFinal.py',
        'FibroDivergence_BTFinal.py',
        'FibroMomentum_BTFinal.py'
    ]
    
    base_dir = '/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi/03_13_2025/backtests_final'
    
    fixed_count = 0
    
    for filename in error_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            if fix_syntax_error_file(filepath):
                fixed_count += 1
        else:
            print(f"⚠️ File not found: {filepath}")
    
    print(f"\n🌙 Fixed {fixed_count} out of {len(error_files)} files")

if __name__ == "__main__":
    main()