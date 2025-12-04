#!/usr/bin/env python3
"""
🔥💎 TLRY AEGS EXIT TRIGGER ANALYSIS 💎🔥
Real-time exit analysis for current TLRY position
Entry: $7.19 on 2025-12-03, 100 shares
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class TLRYAEGSExitAnalysis:
    """Real-time AEGS exit analysis for TLRY position"""
    
    def __init__(self):
        self.symbol = 'TLRY'
        self.entry_price = 7.19
        self.entry_date = '2025-12-03 15:22:29'
        self.shares = 100
        self.position_size = 719.00
        
    def get_current_data(self):
        """Get current TLRY data and calculate position status"""
        
        print(colored("🔍 Fetching Current TLRY Data...", 'yellow', attrs=['bold']))
        print("=" * 60)
        
        ticker = yf.Ticker(self.symbol)
        
        # Get recent data for analysis
        df_daily = ticker.history(period='30d', interval='1d')
        df_1h = ticker.history(period='5d', interval='1h')
        df_15m = ticker.history(period='2d', interval='15m')
        
        current_price = df_daily['Close'].iloc[-1]
        current_return = (current_price - self.entry_price) / self.entry_price
        current_pnl = (current_price - self.entry_price) * self.shares
        
        # Calculate time held
        entry_dt = pd.to_datetime(self.entry_date)
        current_dt = datetime.now()
        time_held = current_dt - entry_dt
        hours_held = time_held.total_seconds() / 3600
        days_held = time_held.days + time_held.seconds / 86400
        
        print(f"📊 TLRY Position Status:")
        print(f"   Entry Price: ${self.entry_price:.2f}")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Position Return: {current_return*100:+.2f}%")
        print(f"   P&L: ${current_pnl:+.2f}")
        print(f"   Time Held: {days_held:.1f} days ({hours_held:.1f} hours)")
        
        return df_daily, df_1h, df_15m, current_price, current_return, current_pnl, days_held
    
    def calculate_aegs_indicators(self, df):
        """Calculate AEGS indicators for exit analysis"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Daily_Change'] = df['Close'].pct_change()
        
        # ATR for volatility
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        return df
    
    def analyze_exit_triggers(self, df_daily, current_price, current_return, days_held):
        """Analyze all AEGS exit triggers"""
        
        print(colored(f"\n🎯 AEGS EXIT TRIGGER ANALYSIS", 'cyan', attrs=['bold']))
        print("=" * 60)
        
        # Calculate indicators
        df_daily = self.calculate_aegs_indicators(df_daily)
        latest = df_daily.iloc[-1]
        
        exit_triggers = {
            'active_triggers': [],
            'approaching_triggers': [],
            'current_levels': {},
            'recommendation': 'HOLD'
        }
        
        # Current technical levels
        current_rsi = latest['RSI']
        current_bb_position = latest['BB_Position']
        current_atr_ratio = latest['ATR_Ratio']
        volume_ratio = latest['Volume_Ratio']
        
        exit_triggers['current_levels'] = {
            'rsi': current_rsi,
            'bb_position': current_bb_position,
            'atr_ratio': current_atr_ratio,
            'volume_ratio': volume_ratio
        }
        
        print(f"📈 Current Technical Levels:")
        print(f"   RSI: {current_rsi:.1f}")
        print(f"   BB Position: {current_bb_position:.3f}")
        print(f"   ATR Ratio: {current_atr_ratio:.3f} ({current_atr_ratio*100:.1f}%)")
        print(f"   Volume Ratio: {volume_ratio:.1f}x")
        
        # 1. PROFIT TARGET TRIGGERS
        print(colored(f"\n🎯 PROFIT TARGET ANALYSIS:", 'green', attrs=['bold']))
        
        # Dynamic profit targets based on volatility
        atr_pct = current_atr_ratio * 100
        if atr_pct < 5:
            profit_targets = [15, 25, 40]  # Low vol targets
        elif atr_pct < 10:
            profit_targets = [20, 35, 50]  # Medium vol targets
        else:
            profit_targets = [30, 50, 70]  # High vol targets
        
        current_return_pct = current_return * 100
        
        for i, target in enumerate(profit_targets):
            target_price = self.entry_price * (1 + target/100)
            distance = (target_price - current_price) / current_price * 100
            
            if current_return_pct >= target:
                exit_triggers['active_triggers'].append(f"Profit Target {target}% HIT: ${target_price:.2f}")
                print(colored(f"   🚨 PROFIT TARGET {target}% HIT! Target: ${target_price:.2f}", 'green', attrs=['bold']))
            elif distance <= 2:
                exit_triggers['approaching_triggers'].append(f"Approaching {target}% target: ${target_price:.2f}")
                print(f"   ⚡ Approaching {target}% target: ${target_price:.2f} (need {distance:+.1f}%)")
            else:
                print(f"   📊 {target}% target: ${target_price:.2f} (need {distance:+.1f}%)")
        
        # 2. STOP LOSS TRIGGERS
        print(colored(f"\n🛑 STOP LOSS ANALYSIS:", 'red', attrs=['bold']))
        
        # Dynamic stop loss based on volatility
        if atr_pct < 5:
            stop_loss = -15  # Tight stop for low vol
        elif atr_pct < 10:
            stop_loss = -20  # Medium stop
        else:
            stop_loss = -25  # Wide stop for high vol
        
        stop_price = self.entry_price * (1 + stop_loss/100)
        stop_distance = (current_price - stop_price) / current_price * 100
        
        if current_return_pct <= stop_loss:
            exit_triggers['active_triggers'].append(f"Stop Loss {stop_loss}% HIT: ${stop_price:.2f}")
            print(colored(f"   🚨 STOP LOSS {stop_loss}% HIT! Stop: ${stop_price:.2f}", 'red', attrs=['bold']))
        elif stop_distance <= 3:
            exit_triggers['approaching_triggers'].append(f"Approaching {stop_loss}% stop: ${stop_price:.2f}")
            print(colored(f"   ⚠️ Near stop loss: ${stop_price:.2f} (buffer: {stop_distance:+.1f}%)", 'red'))
        else:
            print(f"   🛡️ Stop loss: ${stop_price:.2f} (buffer: {stop_distance:+.1f}%)")
        
        # 3. TIME-BASED EXIT TRIGGERS
        print(colored(f"\n⏰ TIME-BASED EXIT ANALYSIS:", 'yellow', attrs=['bold']))
        
        # AEGS typical hold periods
        if days_held >= 60:
            exit_triggers['active_triggers'].append(f"Force exit after {days_held:.1f} days")
            print(colored(f"   🚨 FORCE EXIT: Held {days_held:.1f} days (max 60)", 'yellow', attrs=['bold']))
        elif days_held >= 30 and current_return > 0:
            exit_triggers['active_triggers'].append(f"Profitable time exit after {days_held:.1f} days")
            print(colored(f"   🎯 PROFITABLE TIME EXIT: {days_held:.1f} days with {current_return_pct:+.1f}% gain", 'green'))
        elif days_held >= 45:
            exit_triggers['approaching_triggers'].append(f"Approaching 60-day force exit")
            print(f"   ⚡ Approaching force exit: {days_held:.1f}/60 days")
        else:
            print(f"   ✅ Time OK: {days_held:.1f}/60 days held")
        
        # 4. TECHNICAL EXIT TRIGGERS
        print(colored(f"\n📊 TECHNICAL EXIT ANALYSIS:", 'blue', attrs=['bold']))
        
        # RSI overbought exit (if profitable)
        if current_rsi > 70 and current_return > 0.1:
            exit_triggers['active_triggers'].append(f"RSI overbought exit: {current_rsi:.1f}")
            print(colored(f"   🚨 RSI OVERBOUGHT EXIT: {current_rsi:.1f} with {current_return_pct:+.1f}% gain", 'blue', attrs=['bold']))
        elif current_rsi > 65 and current_return > 0:
            exit_triggers['approaching_triggers'].append(f"RSI approaching overbought: {current_rsi:.1f}")
            print(f"   ⚡ RSI getting high: {current_rsi:.1f} (profitable)")
        else:
            print(f"   📊 RSI: {current_rsi:.1f} (exit at 70+ if profitable)")
        
        # Bollinger Band upper exit
        if current_bb_position > 0.9 and current_return > 0.1:
            exit_triggers['active_triggers'].append(f"BB upper band exit: {current_bb_position:.3f}")
            print(colored(f"   🚨 BB UPPER EXIT: Position {current_bb_position:.3f}", 'blue', attrs=['bold']))
        elif current_bb_position > 0.8:
            exit_triggers['approaching_triggers'].append(f"BB approaching upper: {current_bb_position:.3f}")
            print(f"   ⚡ BB getting high: {current_bb_position:.3f}")
        else:
            print(f"   📊 BB Position: {current_bb_position:.3f} (exit at 0.9+ if profitable)")
        
        # Volume exhaustion
        if volume_ratio < 0.5 and current_return > 0.05:
            exit_triggers['approaching_triggers'].append(f"Low volume: {volume_ratio:.1f}x")
            print(f"   ⚠️ Low volume warning: {volume_ratio:.1f}x average")
        
        return exit_triggers
    
    def generate_exit_recommendation(self, exit_triggers, current_return, days_held):
        """Generate specific exit recommendation"""
        
        print(colored(f"\n🎯 AEGS EXIT RECOMMENDATION", 'magenta', attrs=['bold']))
        print("=" * 60)
        
        active_triggers = exit_triggers['active_triggers']
        approaching_triggers = exit_triggers['approaching_triggers']
        
        if active_triggers:
            print(colored("🚨 IMMEDIATE EXIT RECOMMENDED!", 'red', attrs=['bold']))
            print("Active Triggers:")
            for trigger in active_triggers:
                print(f"   ⚠️ {trigger}")
            recommendation = "SELL NOW"
            urgency = "HIGH"
        elif len(approaching_triggers) >= 2:
            print(colored("⚡ PREPARE TO EXIT", 'yellow', attrs=['bold']))
            print("Approaching Triggers:")
            for trigger in approaching_triggers:
                print(f"   ⚠️ {trigger}")
            recommendation = "SELL SOON"
            urgency = "MEDIUM"
        elif current_return < -0.15:  # -15% loss
            print(colored("🛑 CONSIDER EXIT ON WEAKNESS", 'red'))
            recommendation = "REVIEW STOP LOSS"
            urgency = "MEDIUM"
        else:
            print(colored("✅ CONTINUE HOLDING", 'green'))
            recommendation = "HOLD"
            urgency = "LOW"
        
        print(f"\n📋 RECOMMENDATION: {recommendation}")
        print(f"📋 URGENCY: {urgency}")
        
        # Specific action steps
        print(f"\n📝 ACTION STEPS:")
        if recommendation == "SELL NOW":
            print("   1. Place market sell order for 100 shares")
            print("   2. Log exit reason in position tracker")
            print("   3. Review trade for lessons learned")
        elif recommendation == "SELL SOON":
            print("   1. Set limit sell order at next resistance level")
            print("   2. Monitor for any active trigger activation")
            print("   3. Be ready for quick exit decision")
        elif recommendation == "REVIEW STOP LOSS":
            print("   1. Consider tightening stop loss")
            print("   2. Monitor for any bounce signals")
            print("   3. Prepare for potential exit")
        else:
            print("   1. Continue monitoring daily")
            print("   2. Watch for approaching triggers")
            print("   3. Be patient for full profit targets")
        
        return {
            'recommendation': recommendation,
            'urgency': urgency,
            'active_triggers': active_triggers,
            'approaching_triggers': approaching_triggers
        }
    
    def save_exit_analysis(self, exit_analysis, current_data):
        """Save exit analysis to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'tlry_aegs_exit_analysis_{timestamp}.json'
        
        analysis_data = {
            'symbol': self.symbol,
            'position': {
                'entry_price': self.entry_price,
                'entry_date': self.entry_date,
                'shares': self.shares,
                'current_price': float(current_data['current_price']),
                'current_return_pct': float(current_data['current_return'] * 100),
                'current_pnl': float(current_data['current_pnl']),
                'days_held': float(current_data['days_held'])
            },
            'exit_analysis': exit_analysis,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"\n💾 Exit analysis saved to: {filename}")
        return filename

def main():
    """Run TLRY AEGS exit analysis"""
    
    print(colored("🔥💎 TLRY AEGS EXIT TRIGGER ANALYSIS 💎🔥", 'red', attrs=['bold']))
    print("Position: 100 shares @ $7.19 (Cannabis Oversold Bounce)")
    print("=" * 70)
    
    analyzer = TLRYAEGSExitAnalysis()
    
    # Get current market data
    df_daily, df_1h, df_15m, current_price, current_return, current_pnl, days_held = analyzer.get_current_data()
    
    # Analyze exit triggers
    exit_triggers = analyzer.analyze_exit_triggers(df_daily, current_price, current_return, days_held)
    
    # Generate recommendation
    recommendation = analyzer.generate_exit_recommendation(exit_triggers, current_return, days_held)
    
    # Save analysis
    current_data = {
        'current_price': current_price,
        'current_return': current_return,
        'current_pnl': current_pnl,
        'days_held': days_held
    }
    
    analyzer.save_exit_analysis(recommendation, current_data)
    
    print(colored(f"\n🎯 TLRY EXIT ANALYSIS COMPLETE!", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()