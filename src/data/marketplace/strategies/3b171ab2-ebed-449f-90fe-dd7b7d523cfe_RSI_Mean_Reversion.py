"""
Sample RSI Mean Reversion Strategy for Marketplace Demo
Built by Moon Dev 🌙
"""

from backtesting import Strategy
import pandas_ta as ta


class RSIMeanReversionStrategy(Strategy):
    """
    Classic RSI oversold/overbought mean reversion strategy
    
    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - oversold: RSI level to consider oversold (default: 30)
    - overbought: RSI level to consider overbought (default: 70)
    - stop_loss: Stop loss percentage (default: 2%)
    - take_profit: Take profit percentage (default: 3%)
    """
    
    # Strategy parameters
    rsi_period = 14
    oversold = 30
    overbought = 70
    stop_loss = 0.02
    take_profit = 0.03
    
    def init(self):
        """Initialize indicators"""
        # Calculate RSI
        self.rsi = self.I(ta.rsi, self.data.Close, self.rsi_period)
        
    def next(self):
        """Strategy logic for each bar"""
        # Skip if we don't have enough data
        if len(self.data) < self.rsi_period:
            return
            
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        
        # Entry logic
        if not self.position:
            # Buy when RSI is oversold
            if current_rsi < self.oversold:
                self.buy(
                    size=0.95,  # Use 95% of available capital
                    sl=current_price * (1 - self.stop_loss),
                    tp=current_price * (1 + self.take_profit)
                )
                
        # Exit logic
        elif self.position.is_long:
            # Sell when RSI is overbought
            if current_rsi > self.overbought:
                self.position.close()


# Metadata for marketplace
STRATEGY_METADATA = {
    "name": "RSI Mean Reversion",
    "description": "A classic mean reversion strategy using RSI to identify oversold/overbought conditions. Buys when RSI < 30 and sells when RSI > 70 with configurable stop loss and take profit levels.",
    "author": "moon_dev",
    "category": ["mean_reversion", "technical", "momentum"],
    "timeframes": ["15m", "30m", "1H", "4H"],
    "instruments": ["BTC", "ETH", "SOL"],
    "min_capital": 100.0,
    "risk_level": "low",
    "dependencies": ["pandas_ta"],
    "parameters": {
        "rsi_period": {"default": 14, "min": 5, "max": 30, "description": "RSI calculation period"},
        "oversold": {"default": 30, "min": 10, "max": 40, "description": "RSI oversold threshold"},
        "overbought": {"default": 70, "min": 60, "max": 90, "description": "RSI overbought threshold"},
        "stop_loss": {"default": 0.02, "min": 0.005, "max": 0.05, "description": "Stop loss percentage"},
        "take_profit": {"default": 0.03, "min": 0.01, "max": 0.1, "description": "Take profit percentage"}
    }
}