#!/usr/bin/env python3
"""
🎯 TLRY Exit Signal Tracker
Real-time monitoring of exit conditions for your TLRY position
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class TLRYExitTracker:
    def __init__(self):
        self.symbol = "TLRY"
        self.ticker = yf.Ticker(self.symbol)
        
    def get_current_data(self):
        """Get current and recent price data"""
        # Get intraday data for more precise signals
        df_1h = self.ticker.history(period="5d", interval="1h")
        df_daily = self.ticker.history(period="30d", interval="1d")
        
        if df_1h.empty or df_daily.empty:
            raise Exception("Failed to download data")
            
        return df_1h, df_daily
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for exit signals"""
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df['BB_Upper'] = bbands.iloc[:, 0]  # Upper band
            df['BB_Middle'] = bbands.iloc[:, 1]  # Middle band
            df['BB_Lower'] = bbands.iloc[:, 2]  # Lower band
            df['BB_%'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        else:
            # Fallback calculation
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            rolling_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (2 * rolling_std)
            df['BB_Lower'] = df['BB_Middle'] - (2 * rolling_std)
            df['BB_%'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Moving Averages
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        
        # ATR for volatility
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_%'] = (df['ATR'] / df['Close']) * 100
        
        # Volume analysis
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['ROC'] = ta.roc(df['Close'], length=5)
        
        return df
    
    def check_aegs_exit_signals(self, df_1h, df_daily):
        """Check for AEGS-style exit conditions"""
        
        current_price = df_1h['Close'].iloc[-1]
        current_time = df_1h.index[-1]
        
        print(colored(f"\n🎯 TLRY EXIT SIGNAL ANALYSIS", 'cyan', attrs=['bold']))
        print("="*60)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Last Update: {current_time.strftime('%Y-%m-%d %H:%M')}")
        
        exit_signals = {
            'immediate': [],
            'warning': [],
            'monitoring': []
        }
        
        # Get latest indicators
        rsi_1h = df_1h['RSI'].iloc[-1]
        rsi_daily = df_daily['RSI'].iloc[-1]
        bb_percent_1h = df_1h['BB_%'].iloc[-1]
        bb_percent_daily = df_daily['BB_%'].iloc[-1]
        volume_ratio = df_1h['Volume_Ratio'].iloc[-1]
        atr_percent = df_1h['ATR_%'].iloc[-1]
        roc = df_1h['ROC'].iloc[-1]
        
        # 1. RSI Exit Signals (AEGS uses RSI > 50 or 70 for exits)
        if rsi_1h > 70:
            exit_signals['immediate'].append(f"🔴 RSI Overbought (1H): {rsi_1h:.1f} > 70")
        elif rsi_1h > 60:
            exit_signals['warning'].append(f"🟡 RSI Rising (1H): {rsi_1h:.1f} > 60")
        
        if rsi_daily > 70:
            exit_signals['warning'].append(f"🟡 RSI Overbought (Daily): {rsi_daily:.1f} > 70")
        
        # 2. Bollinger Band Exit Signals
        if bb_percent_1h > 0.95:
            exit_signals['immediate'].append(f"🔴 At Upper BB (1H): {bb_percent_1h:.1%}")
        elif bb_percent_1h > 0.80:
            exit_signals['warning'].append(f"🟡 Near Upper BB (1H): {bb_percent_1h:.1%}")
            
        # 3. Price Distance from Moving Average (AEGS exit condition)
        distance_from_sma20 = ((current_price - df_1h['SMA_20'].iloc[-1]) / df_1h['SMA_20'].iloc[-1]) * 100
        if distance_from_sma20 > 5:
            exit_signals['warning'].append(f"🟡 Price {distance_from_sma20:.1f}% above SMA20")
        elif distance_from_sma20 > 10:
            exit_signals['immediate'].append(f"🔴 Price {distance_from_sma20:.1f}% extended above SMA20")
            
        # 4. Volume Exhaustion
        if volume_ratio < 0.5 and rsi_1h > 60:
            exit_signals['warning'].append(f"🟡 Volume drying up: {volume_ratio:.1f}x average")
            
        # 5. Momentum Loss
        if roc < -2:
            exit_signals['warning'].append(f"🟡 Momentum weakening: ROC {roc:.1f}%")
            
        # 6. Volatility Expansion (AEGS Vol_Expansion exit)
        recent_atr = df_1h['ATR_%'].rolling(5).mean().iloc[-1]
        if atr_percent > recent_atr * 1.5:
            exit_signals['monitoring'].append(f"📊 Volatility expanding: {atr_percent:.1f}%")
            
        # Calculate exit score
        exit_score = len(exit_signals['immediate']) * 3 + len(exit_signals['warning']) * 2 + len(exit_signals['monitoring'])
        
        return exit_signals, exit_score
    
    def get_position_metrics(self, entry_price):
        """Calculate position P&L and metrics"""
        df_1h, df_daily = self.get_current_data()
        current_price = df_1h['Close'].iloc[-1]
        
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_dollar = current_price - entry_price
        
        # Get high/low since entry
        recent_high = df_1h['High'].max()
        recent_low = df_1h['Low'].min()
        
        return {
            'current_price': current_price,
            'pnl_percent': pnl_percent,
            'pnl_dollar': pnl_dollar,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'high_drawdown': ((recent_high - current_price) / recent_high) * 100
        }
    
    def run_exit_analysis(self, entry_price=None):
        """Run complete exit analysis"""
        
        print(colored("🔍 TLRY EXIT TRACKER - AEGS STRATEGY", 'yellow', attrs=['bold']))
        print("="*60)
        
        try:
            # Get data
            df_1h, df_daily = self.get_current_data()
            
            # Calculate indicators
            df_1h = self.calculate_indicators(df_1h)
            df_daily = self.calculate_indicators(df_daily)
            
            # Check exit signals
            exit_signals, exit_score = self.check_aegs_exit_signals(df_1h, df_daily)
            
            # Display signals
            print("\n🚨 EXIT SIGNALS:")
            print("-"*40)
            
            if exit_signals['immediate']:
                print(colored("IMMEDIATE ACTION:", 'red', attrs=['bold']))
                for signal in exit_signals['immediate']:
                    print(f"  {signal}")
                    
            if exit_signals['warning']:
                print(colored("\nWARNING SIGNALS:", 'yellow'))
                for signal in exit_signals['warning']:
                    print(f"  {signal}")
                    
            if exit_signals['monitoring']:
                print(colored("\nMONITORING:", 'blue'))
                for signal in exit_signals['monitoring']:
                    print(f"  {signal}")
            
            # Position metrics
            if entry_price:
                print("\n💰 POSITION METRICS:")
                print("-"*40)
                metrics = self.get_position_metrics(entry_price)
                
                color = 'green' if metrics['pnl_percent'] > 0 else 'red'
                print(f"Entry Price: ${entry_price:.2f}")
                print(f"Current Price: ${metrics['current_price']:.2f}")
                print(colored(f"P&L: {metrics['pnl_percent']:+.1f}% (${metrics['pnl_dollar']:+.2f})", color))
                print(f"Recent High: ${metrics['recent_high']:.2f}")
                print(f"Drawdown from High: {metrics['high_drawdown']:.1f}%")
            
            # Exit recommendation
            print("\n📊 EXIT RECOMMENDATION:")
            print("-"*40)
            
            if exit_score >= 9:
                print(colored("🔴 STRONG SELL - Exit immediately!", 'red', attrs=['bold']))
                print("Multiple overbought signals converging")
            elif exit_score >= 6:
                print(colored("🟡 SELL SIGNAL - Consider exiting on strength", 'yellow'))
                print("Exit indicators warming up")
            elif exit_score >= 3:
                print(colored("🟢 HOLD - Monitor closely", 'green'))
                print("Some caution signals, but not critical")
            else:
                print(colored("🟢 STRONG HOLD - No exit signals", 'green', attrs=['bold']))
                print("Position looks healthy")
                
            # AEGS-specific exit rules
            print("\n📋 AEGS EXIT RULES:")
            print("-"*40)
            print("1. RSI > 70 (1H) → Strong exit signal")
            print("2. RSI > 50 + Price > SMA20 → Consider exit")
            print("3. At Upper Bollinger Band → Take profits")
            print("4. Volume < 50% average + RSI > 60 → Weakness")
            print("5. 10%+ above SMA20 → Overextended")
            
            # Next check time
            print(f"\n⏰ Next update in 1 hour")
            print(f"🔄 Run again: python {__file__}")
            
            if entry_price:
                print(f"\n💡 Quick command: python {__file__} --entry {entry_price}")
                
        except Exception as e:
            print(colored(f"❌ Error: {e}", 'red'))

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TLRY Exit Signal Tracker')
    parser.add_argument('--entry', type=float, help='Your entry price for P&L calculation')
    
    args = parser.parse_args()
    
    tracker = TLRYExitTracker()
    tracker.run_exit_analysis(entry_price=args.entry)

if __name__ == "__main__":
    main()