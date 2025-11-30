import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# Cosmic data preparation 🌌
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Lunar data cleansing ritual 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class AnchoredBreakout(Strategy):
    vwap_period = 20  #  VWAP orbital period
    std_dev = 2       #  Standard deviation multiplier
    adx_period = 14   #  ADX measurement window
    volume_lookback = 50  #  Volume analysis period
    risk_pct = 0.01   #  Cosmic risk shield percentage
    
    def init(self):
        # Calculate VWAP-based Bollinger Bands with lunar precision 🌙
        # Calculate VWAP with fallback
        vwap_result = ta.vwap(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume
        )
        
        if vwap_result is None or (hasattr(vwap_result, '__len__') and len(vwap_result) == 0):
            vwap_values = (self.data.High + self.data.Low + self.data.Close) / 3
        else:
            vwap_values = vwap_result.ffill().fillna((self.data.High + self.data.Low + self.data.Close) / 3).values
            
        self.vwap = self.I(lambda: vwap_values, name='VWAP')
        
        # Standard deviation for cosmic bands 🌠
        stddev = self.I(talib.STDDEV, self.data.Close, self.vwap_period)
        self.upper = self.I(lambda: self.vwap + self.std_dev*stddev, name='UPPER_BAND')
        self.lower = self.I(lambda: self.vwap - self.std_dev*stddev, name='LOWER_BAND')
        
        # ADX trend strength filter ✨
        self.adx = self.I(talib.ADX, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.adx_period,
                         name='ADX')
        
        # Volume surge detector 🚀
        self.vol_threshold = self.I(ta.quantile,
                                   self.data.Volume,
                                   length=self.volume_lookback,
                                   q=0.9,
                                   name='VOL_90TH')
        
    def next(self):
        price = self.data.Close[-1]
        upper_band = self.upper[-1]
        lower_band = self.lower[-1]
        
        if not self.position:
            # Long entry constellation 🌌
            if (price > upper_band and
                self.adx[-1] > 25 and
                self.data.Volume[-1] > self.vol_threshold[-1]):
                
                sl = self.data.Low[-1] * 0.995
                risk_amount = self.equity * self.risk_pct
                risk_per_share = price - sl
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.buy(size=size, sl=sl)
                    print(f" MOON SHOT ACTIVATED! Long {size} units at {price:.2f} | Cosmic SL {sl:.2f}")
            
            # Short entry constellation 🌠
            elif (price < lower_band and
                  self.adx[-1] > 25 and
                  self.data.Volume[-1] > self.vol_threshold[-1]):
                
                sl = self.data.High[-1] * 1.005
                risk_amount = self.equity * self.risk_pct
                risk_per_share = sl - price
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.sell(size=size, sl=sl)
                    print(f" BLACK HOLE ENGAGED! Short {size} units at {price:.2f} | Galactic SL {sl:.2f}")
        
        else:
            # Exit with cosmic precision 🌟
            if self.position.is_long and price < self.vwap[-1]:
                self.position.close()
                print(f" LUNAR ECLIPSE! Closing long at {price:.2f}")
            elif self.position.is_short and price > self.vwap[-1]:
                self.position.close