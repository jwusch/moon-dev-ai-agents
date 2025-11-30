import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy
import numpy as np

# DATA PREPROCESSING 🌙
def prepare_data(path):
    # Load and clean data
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Column mapping with Moon Dev validation ✨
    required = ['open', 'high', 'low', 'close', 'volume', 'funding']
    assert all(col in data.columns for col in required), " MISSING REQUIRED COLUMNS!"
    
    # Proper case conversion
    data = data.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close',
        'volume': 'Volume', 'funding': 'Funding'
    })
    data.set_index(pd.DatetimeIndex(data['datetime']), inplace=True)
    return data

# STRATEGY IMPLEMENTATION 🚀
class SqueezeLiquidity(Strategy):
    # MOON DEV CONFIGURATIONS ✨
    bb_period = 20
    squeeze_lookback = 50
    atr_period = 14
    risk_per_trade = 0.01
    volume_filter_multiplier = 1.2
    
    def init(self):
        # CORE INDICATORS 🌙
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands with Moon Dev squeeze detection ✨
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bandwidth = self.I(lambda: self.bb_upper - self.bb_lower, name='Bandwidth')
        self.squeeze = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_lookback, name='Squeeze')
        
        # Funding rate analysis 🌊
        self.funding = self.data.Funding
        self.funding_ma = self.I(talib.SMA, self.funding, timeperiod=7*24*4, name='Funding MA')  # 7D 4H data
        
        # Volume dynamics 📊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=14, name='Volume MA')
        
        # ATR for risk management 🛡️
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        
        # Moon Dev debug trackers 🌙
        self.last_squeeze = None
        self.entry_price = None
        self.sl_price = None
        self.target_price = None

    def next(self):
        # MOON DEV SAFETY CHECKS ✨
        if len(self.data) < 100:  # Warmup period
            return
            
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # SQUEEZE DETECTION 🌪️
        squeeze_active = self.bandwidth[-1] == self.squeeze[-1]
        funding_spike = self.funding[-1] < (self.funding_ma[-1] - 0.0001)  # 0.01% below MA
        
        # VOLUME VALIDATION 📈
        volume_ok = current_volume > (self.vol_ma[-1] * self.volume_filter_multiplier)
        
        # LIQUIDITY TARGETING 🎯
        recent_highs = self.data.High[-50:]
        recent_lows = self.data.Low[-50:]
        target_long = recent_highs.max()  # Simplified volume proxy
        target_short = recent_lows.min()
        
        # MOON DEV RISK CALCULATIONS ✨