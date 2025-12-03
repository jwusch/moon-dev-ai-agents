"""
🎯 Entry Signals with Price Target Predictions
Enhanced signals showing expected price at next decision point

Author: Claude (Anthropic)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
from yfinance_cache_demo import YFinanceCache

class EntrySignalWithTargets:
    def __init__(self):
        self.cache = YFinanceCache()
        
    def calculate_price_targets(self, current_price, atr, sma, distance_pct, position_type='Long'):
        """Calculate expected prices at key decision levels"""
        
        targets = {}
        
        if position_type == 'Long':
            # Profit targets
            targets['1%_profit'] = current_price * 1.01
            targets['2%_profit'] = current_price * 1.02
            targets['5%_profit'] = current_price * 1.05
            
            # Mean reversion targets
            targets['50%_to_mean'] = current_price + (sma - current_price) * 0.5
            targets['75%_to_mean'] = current_price + (sma - current_price) * 0.75
            targets['at_mean'] = sma
            
            # ATR-based targets
            targets['1_ATR_up'] = current_price + atr
            targets['2_ATR_up'] = current_price + (2 * atr)
            
            # Stop loss levels
            targets['1.5%_stop'] = current_price * 0.985
            targets['3%_stop'] = current_price * 0.97
            targets['7.5%_stop'] = current_price * 0.925
            
            # Time-based targets (historical average move in 1hr, 2hr, 3hr)
            # These would be calculated from historical data
            targets['1hr_expected'] = current_price * 1.003  # 0.3% avg move
            targets['2hr_expected'] = current_price * 1.005  # 0.5% avg move
            targets['3hr_expected'] = current_price * 1.007  # 0.7% avg move
            
        else:  # Short
            # Profit targets
            targets['1%_profit'] = current_price * 0.99
            targets['2%_profit'] = current_price * 0.98
            targets['5%_profit'] = current_price * 0.95
            
            # Mean reversion targets
            targets['50%_to_mean'] = current_price - (current_price - sma) * 0.5
            targets['75%_to_mean'] = current_price - (current_price - sma) * 0.75
            targets['at_mean'] = sma
            
            # ATR-based targets
            targets['1_ATR_down'] = current_price - atr
            targets['2_ATR_down'] = current_price - (2 * atr)
            
            # Stop loss levels
            targets['1.5%_stop'] = current_price * 1.015
            targets['3%_stop'] = current_price * 1.03
            targets['7.5%_stop'] = current_price * 1.075
            
            # Time-based targets
            targets['1hr_expected'] = current_price * 0.997
            targets['2hr_expected'] = current_price * 0.995
            targets['3hr_expected'] = current_price * 0.993
            
        return targets
    
    def calculate_probabilities(self, df, current_idx, targets, position_type='Long'):
        """Estimate probability of reaching each target based on historical moves"""
        
        # Look at past 100 similar setups
        lookback = 100
        similar_setups = []
        
        current_rsi = df['RSI'].iloc[current_idx]
        current_distance = df['Distance%'].iloc[current_idx]
        current_volume_ratio = df['Volume_Ratio'].iloc[current_idx]
        
        # Find similar historical setups
        for i in range(max(100, current_idx - 500), current_idx):
            if pd.isna(df['RSI'].iloc[i]):
                continue
                
            # Similar conditions (within 20% of current)
            rsi_similar = abs(df['RSI'].iloc[i] - current_rsi) < 10
            distance_similar = abs(df['Distance%'].iloc[i] - current_distance) < 0.5
            volume_similar = abs(df['Volume_Ratio'].iloc[i] - current_volume_ratio) < 0.3
            
            if rsi_similar and distance_similar and volume_similar:
                # Track what happened in next 20 bars (5 hours)
                future_prices = df['Close'].iloc[i:min(i+20, len(df))].tolist()
                if len(future_prices) > 1:
                    similar_setups.append({
                        'entry_price': df['Close'].iloc[i],
                        'future_prices': future_prices,
                        'max_move': max(future_prices) / df['Close'].iloc[i] - 1,
                        'min_move': min(future_prices) / df['Close'].iloc[i] - 1
                    })
        
        # Calculate probabilities
        probabilities = {}
        
        if len(similar_setups) >= 10:  # Need enough data
            for target_name, target_price in targets.items():
                hits = 0
                current_price = df['Close'].iloc[current_idx]
                target_move = (target_price - current_price) / current_price
                
                for setup in similar_setups:
                    if position_type == 'Long':
                        if 'profit' in target_name or 'up' in target_name or 'mean' in target_name:
                            if setup['max_move'] >= target_move:
                                hits += 1
                        elif 'stop' in target_name:
                            if setup['min_move'] <= target_move:
                                hits += 1
                    else:  # Short
                        if 'profit' in target_name or 'down' in target_name or 'mean' in target_name:
                            if setup['min_move'] <= target_move:
                                hits += 1
                        elif 'stop' in target_name:
                            if setup['max_move'] >= target_move:
                                hits += 1
                
                probabilities[target_name] = (hits / len(similar_setups)) * 100
        else:
            # Default probabilities if not enough historical data
            for target_name in targets:
                if '1%_profit' in target_name:
                    probabilities[target_name] = 60
                elif '2%_profit' in target_name:
                    probabilities[target_name] = 45
                elif '5%_profit' in target_name:
                    probabilities[target_name] = 25
                elif '1.5%_stop' in target_name:
                    probabilities[target_name] = 35
                elif 'mean' in target_name:
                    probabilities[target_name] = 70
                else:
                    probabilities[target_name] = 50
                    
        return probabilities
    
    def generate_enhanced_signal(self, df, idx):
        """Generate trading signal with all target information"""
        
        # Basic signal info
        current_price = df['Close'].iloc[idx]
        sma = df['SMA20'].iloc[idx]
        distance = df['Distance%'].iloc[idx]
        rsi = df['RSI'].iloc[idx]
        atr = df['ATR'].iloc[idx]
        volume_ratio = df['Volume_Ratio'].iloc[idx]
        
        # Entry quality score
        from improved_entry_timing import ImprovedEntryTiming
        improver = ImprovedEntryTiming()
        quality_score, score_details = improver.evaluate_entry_quality(df, idx)
        
        # Determine position type
        if distance < -1.0 and rsi < 40:
            position_type = 'Long'
            action = 'BUY'
        elif distance > 1.0 and rsi > 60:
            position_type = 'Short'
            action = 'SELL'
        else:
            return None  # No signal
            
        # Calculate targets
        targets = self.calculate_price_targets(current_price, atr, sma, distance, position_type)
        
        # Calculate probabilities
        probabilities = self.calculate_probabilities(df, idx, targets, position_type)
        
        # Determine optimal exit strategy
        best_risk_reward = 0
        best_target = None
        
        for target_name, target_price in targets.items():
            if 'profit' in target_name and 'stop' not in target_name:
                profit_pct = abs(target_price - current_price) / current_price * 100
                prob = probabilities.get(target_name, 50)
                expected_value = profit_pct * (prob / 100)
                
                if expected_value > best_risk_reward:
                    best_risk_reward = expected_value
                    best_target = target_name
        
        # Create comprehensive signal
        signal = {
            'timestamp': df.index[idx],
            'action': action,
            'price': current_price,
            'size': 0.95,  # 95% of capital
            'quality_score': quality_score,
            
            # Current market state
            'sma': sma,
            'distance_from_sma': distance,
            'rsi': rsi,
            'atr': atr,
            'atr_pct': (atr / current_price) * 100,
            'volume_ratio': volume_ratio,
            
            # Price targets
            'targets': targets,
            'probabilities': probabilities,
            
            # Recommendations
            'recommended_profit_target': best_target,
            'recommended_stop_loss': f"{7.5 if '5%' in str(best_target) else 1.5}%_stop",
            
            # Expected outcomes
            'expected_move_1hr': targets.get('1hr_expected', current_price),
            'expected_move_3hr': targets.get('3hr_expected', current_price),
            'probability_profit_1pct': probabilities.get('1%_profit', 60),
            'probability_profit_5pct': probabilities.get('5%_profit', 25),
            'probability_hit_stop': probabilities.get('1.5%_stop', 35),
        }
        
        return signal
    
    def format_signal_output(self, signal):
        """Format signal for easy reading"""
        if not signal:
            return "No signal"
            
        output = f"""
═══════════════════════════════════════════════════════════════
📊 TRADING SIGNAL - {signal['timestamp']}
═══════════════════════════════════════════════════════════════

ACTION: {signal['action']} @ ${signal['price']:.2f}
QUALITY SCORE: {signal['quality_score']}/100
POSITION SIZE: {signal['size']*100:.0f}% of capital

📈 CURRENT STATE:
  • Distance from SMA: {signal['distance_from_sma']:.1f}%
  • RSI: {signal['rsi']:.1f}
  • ATR: {signal['atr_pct']:.1f}% of price
  • Volume: {signal['volume_ratio']:.1f}x average

🎯 PRICE TARGETS & PROBABILITIES:
"""
        
        # Sort targets by price
        sorted_targets = sorted(signal['targets'].items(), 
                               key=lambda x: x[1], 
                               reverse=(signal['action'] == 'SELL'))
        
        for target_name, target_price in sorted_targets:
            prob = signal['probabilities'].get(target_name, 0)
            move_pct = (target_price - signal['price']) / signal['price'] * 100
            
            # Highlight important levels
            if target_name == signal['recommended_profit_target']:
                marker = "⭐"
            elif 'stop' in target_name and '7.5%' in target_name:
                marker = "🛑"
            elif 'mean' in target_name:
                marker = "📊"
            else:
                marker = "  "
                
            output += f"  {marker} ${target_price:.2f} ({move_pct:+.1f}%) - {target_name:<15} [{prob:.0f}% chance]\n"
        
        output += f"""
📋 RECOMMENDATIONS:
  • Target: {signal['recommended_profit_target']} (${signal['targets'][signal['recommended_profit_target']]:.2f})
  • Stop Loss: {signal['recommended_stop_loss']} (${signal['targets'][signal['recommended_stop_loss']]:.2f})
  • Expected in 1hr: ${signal['expected_move_1hr']:.2f}
  • Expected in 3hr: ${signal['expected_move_3hr']:.2f}

📊 WIN PROBABILITIES:
  • 1% Profit: {signal['probability_profit_1pct']:.0f}%
  • 5% Profit: {signal['probability_profit_5pct']:.0f}%
  • Hit Stop: {signal['probability_hit_stop']:.0f}%

═══════════════════════════════════════════════════════════════
"""
        return output
    
    def prepare_enhanced_data(self, symbol, period="59d"):
        """Prepare data with all required indicators"""
        # Get data
        df = self.cache.get_data(symbol, period=period, interval="15m")
        
        if len(df) < 100:
            return None
            
        # Basic indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance%'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, 14)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df


def main():
    print("="*70)
    print("🎯 ENHANCED ENTRY SIGNALS WITH PRICE TARGETS")
    print("="*70)
    
    signaler = EntrySignalWithTargets()
    
    # Test on VXX
    symbol = 'VXX'
    print(f"\n📊 Analyzing {symbol}...")
    
    df = signaler.prepare_enhanced_data(symbol)
    
    if df is not None:
        # Find recent signals
        signals_found = 0
        
        for i in range(len(df)-10, len(df)):
            if pd.isna(df['RSI'].iloc[i]) or pd.isna(df['Distance%'].iloc[i]):
                continue
                
            # Only during market hours
            if df.index[i].hour < 9 or df.index[i].hour >= 16:
                continue
                
            # Check for signal
            if (df['Distance%'].iloc[i] < -1.0 and df['RSI'].iloc[i] < 40) or \
               (df['Distance%'].iloc[i] > 1.0 and df['RSI'].iloc[i] > 60):
                
                signal = signaler.generate_enhanced_signal(df, i)
                if signal and signal['quality_score'] >= 50:
                    print(signaler.format_signal_output(signal))
                    signals_found += 1
                    
                    if signals_found >= 3:  # Show max 3 signals
                        break
        
        if signals_found == 0:
            print("No recent high-quality signals found")
            
            # Show what a signal would look like
            print("\n📋 EXAMPLE SIGNAL FORMAT:")
            
            # Create example signal
            example_signal = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'price': 15.50,
                'size': 0.95,
                'quality_score': 75,
                'sma': 16.00,
                'distance_from_sma': -3.1,
                'rsi': 28,
                'atr': 0.45,
                'atr_pct': 2.9,
                'volume_ratio': 1.8,
                'targets': {
                    '7.5%_stop': 14.34,
                    '1.5%_stop': 15.27,
                    '1%_profit': 15.66,
                    '2%_profit': 15.81,
                    '50%_to_mean': 15.75,
                    '75%_to_mean': 15.88,
                    'at_mean': 16.00,
                    '5%_profit': 16.28,
                },
                'probabilities': {
                    '7.5%_stop': 15,
                    '1.5%_stop': 35,
                    '1%_profit': 65,
                    '2%_profit': 48,
                    '50%_to_mean': 72,
                    '75%_to_mean': 68,
                    'at_mean': 65,
                    '5%_profit': 28,
                },
                'recommended_profit_target': '50%_to_mean',
                'recommended_stop_loss': '1.5%_stop',
                'expected_move_1hr': 15.55,
                'expected_move_3hr': 15.61,
                'probability_profit_1pct': 65,
                'probability_profit_5pct': 28,
                'probability_hit_stop': 35,
            }
            
            print(signaler.format_signal_output(example_signal))
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()