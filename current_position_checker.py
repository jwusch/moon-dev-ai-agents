"""
📊 Current Position Checker - Are We In Positions Today?
Check if our goldmine symbols would trigger BUY/SELL signals based on yesterday's close

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored

class CurrentPositionChecker:
    """
    Check current position status for goldmine symbols
    """
    
    def __init__(self):
        # Our goldmine symbols with massive return potential
        self.goldmine_symbols = [
            'WULF',  # +13,041% excess return
            'EQT',   # +1,038% excess return  
            'NOK',   # +3,355% excess return
            'WKHS',  # +1,133% excess return
        ]
        
        # Additional high-potential symbols to check
        self.bonus_symbols = [
            'SH',    # Our proven winner (+31.8% excess)
            'MARA',  # Bitcoin mining (+1,457% from earlier)
            'SAVA',  # Biotech rocket (+170% from earlier)
            'TNA',   # Leveraged small cap (+347% from earlier)
        ]
        
        self.all_symbols = self.goldmine_symbols + self.bonus_symbols
    
    def check_current_signals(self):
        """Check if we'd be in positions based on recent data"""
        
        print(colored("📊 CURRENT POSITION STATUS CHECKER", 'cyan', attrs=['bold']))
        print("=" * 70)
        print(f"Checking signals for {len(self.all_symbols)} goldmine symbols...")
        print(f"Based on most recent trading data through {datetime.now().strftime('%Y-%m-%d')}")
        
        current_positions = []
        potential_entries = []
        
        for symbol in self.all_symbols:
            print(f"\\n🔍 Checking {symbol}...")
            
            try:
                # Get recent data (last 100 days for indicators)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='100d')
                
                if len(df) < 50:
                    print(f"❌ Insufficient recent data for {symbol}")
                    continue
                
                # Calculate key indicators
                df = self._calculate_indicators(df)
                
                # Get latest values
                latest = df.iloc[-1]
                current_price = latest['Close']
                
                # Check for entry/exit signals
                signal_status = self._check_signal_status(df, symbol)
                
                if signal_status['in_position']:
                    current_positions.append({
                        'symbol': symbol,
                        'entry_price': signal_status['entry_price'],
                        'current_price': current_price,
                        'pnl_pct': signal_status['pnl_pct'],
                        'days_held': signal_status['days_held'],
                        'signal_strength': signal_status['signal_strength']
                    })
                    
                    pnl_color = 'green' if signal_status['pnl_pct'] > 0 else 'red'
                    print(colored(f"   🎯 IN POSITION: Entry ${signal_status['entry_price']:.2f} → Current ${current_price:.2f}", 'yellow'))
                    print(colored(f"   💰 P&L: {signal_status['pnl_pct']:+.1f}% | Held: {signal_status['days_held']} days", pnl_color))
                
                elif signal_status['entry_signal']:
                    potential_entries.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'signal_strength': signal_status['signal_strength'],
                        'reasoning': signal_status['reasoning']
                    })
                    
                    print(colored(f"   🚀 BUY SIGNAL: Entry @ ${current_price:.2f}", 'green', attrs=['bold']))
                    print(colored(f"   📊 Strength: {signal_status['signal_strength']}/100 | {signal_status['reasoning']}", 'green'))
                
                else:
                    print(f"   ⏸️  NO SIGNAL: Wait for entry | Price: ${current_price:.2f}")
                    print(f"   📊 RSI: {latest.get('RSI', 'N/A'):.1f} | Distance: {latest.get('Distance_Pct', 'N/A'):.1f}%")
                
            except Exception as e:
                print(f"❌ Error checking {symbol}: {str(e)[:50]}...")
                continue
        
        # Summary report
        self._print_position_summary(current_positions, potential_entries)
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators for signal detection"""
        
        # Simple Moving Average
        df['SMA20'] = df['Close'].rolling(20).mean()
        
        # Distance from SMA
        df['Distance_Pct'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        df['BB_Middle'] = df['Close'].rolling(bb_window).mean()
        bb_std = df['Close'].rolling(bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume ratio
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def _check_signal_status(self, df, symbol):
        """Check if we're in a position or have entry signal"""
        
        # Look back 30 days for recent signals
        recent_data = df.tail(30).copy()
        
        # Entry conditions (ensemble strategy patterns)
        # 1. Extreme oversold (Distance < -3% AND RSI < 35)
        # 2. Bollinger Band breach (BB_Position < 0.1)
        # 3. Volume spike (Volume_Ratio > 1.5)
        
        entry_signals = []
        current_pos = None
        
        for i in range(len(recent_data)):
            row = recent_data.iloc[i]
            
            signal_strength = 0
            reasons = []
            
            # Check entry conditions
            if pd.notna(row['Distance_Pct']) and row['Distance_Pct'] < -2.0:
                signal_strength += 30
                reasons.append(f"Oversold: {row['Distance_Pct']:.1f}%")
            
            if pd.notna(row['RSI']) and row['RSI'] < 40:
                signal_strength += 25
                reasons.append(f"RSI: {row['RSI']:.1f}")
            
            if pd.notna(row['BB_Position']) and row['BB_Position'] < 0.2:
                signal_strength += 25
                reasons.append(f"BB breach: {row['BB_Position']:.2f}")
            
            if pd.notna(row['Volume_Ratio']) and row['Volume_Ratio'] > 1.3:
                signal_strength += 20
                reasons.append(f"Volume: {row['Volume_Ratio']:.1f}x")
            
            # Strong entry signal
            if signal_strength >= 60:
                entry_date = row.name
                entry_price = row['Close']
                
                # Check if we'd still be in this position
                days_since = (recent_data.index[-1] - entry_date).days
                
                if days_since <= 61:  # Typical hold period
                    current_price = recent_data.iloc[-1]['Close']
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    
                    current_pos = {
                        'in_position': True,
                        'entry_price': entry_price,
                        'pnl_pct': pnl_pct,
                        'days_held': days_since,
                        'signal_strength': signal_strength
                    }
                    break
        
        # If not in position, check for current entry signal
        if current_pos is None:
            latest = recent_data.iloc[-1]
            
            signal_strength = 0
            reasons = []
            
            if pd.notna(latest['Distance_Pct']) and latest['Distance_Pct'] < -1.5:
                signal_strength += 30
                reasons.append(f"Oversold: {latest['Distance_Pct']:.1f}%")
            
            if pd.notna(latest['RSI']) and latest['RSI'] < 40:
                signal_strength += 25
                reasons.append(f"Low RSI: {latest['RSI']:.1f}")
            
            if pd.notna(latest['BB_Position']) and latest['BB_Position'] < 0.25:
                signal_strength += 25
                reasons.append(f"BB support: {latest['BB_Position']:.2f}")
            
            if pd.notna(latest['Volume_Ratio']) and latest['Volume_Ratio'] > 1.2:
                signal_strength += 20
                reasons.append(f"Volume: {latest['Volume_Ratio']:.1f}x")
            
            return {
                'in_position': False,
                'entry_signal': signal_strength >= 50,
                'signal_strength': signal_strength,
                'reasoning': ', '.join(reasons) if reasons else 'No strong signals'
            }
        
        return current_pos
    
    def _print_position_summary(self, current_positions, potential_entries):
        """Print summary of current positions and opportunities"""
        
        print(f"\\n{'='*70}")
        print(colored("📊 POSITION & OPPORTUNITY SUMMARY", 'yellow', attrs=['bold']))
        print(f"{'='*70}")
        
        # Current Positions
        if current_positions:
            print(colored(f"\\n🎯 CURRENT POSITIONS ({len(current_positions)}):", 'yellow', attrs=['bold']))
            
            total_pnl = 0
            for pos in current_positions:
                symbol = pos['symbol']
                pnl = pos['pnl_pct']
                total_pnl += pnl
                
                pnl_color = 'green' if pnl > 0 else 'red'
                status = "🟢 PROFIT" if pnl > 0 else "🔴 LOSS"
                
                print(f"   {symbol}: ${pos['entry_price']:.2f} → ${pos['current_price']:.2f}")
                print(colored(f"      {status}: {pnl:+.1f}% | Held: {pos['days_held']} days", pnl_color))
                
                # Suggest action
                if pos['days_held'] > 50:
                    print(colored("      ⚠️  Consider taking profits (near max hold period)", 'yellow'))
                elif pnl > 20:
                    print(colored("      💰 Strong profits - consider partial exit", 'green'))
                elif pnl < -15:
                    print(colored("      🛑 Consider stop loss", 'red'))
            
            avg_pnl = total_pnl / len(current_positions)
            pnl_color = 'green' if avg_pnl > 0 else 'red'
            print(colored(f"\\n   PORTFOLIO P&L: {avg_pnl:+.1f}% average", pnl_color, attrs=['bold']))
        
        else:
            print(colored("\\n💤 NO CURRENT POSITIONS", 'blue'))
            print("   All goldmine symbols are currently out of position")
        
        # Potential Entries
        if potential_entries:
            print(colored(f"\\n🚀 IMMEDIATE BUY OPPORTUNITIES ({len(potential_entries)}):", 'green', attrs=['bold']))
            
            # Sort by signal strength
            potential_entries.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            for entry in potential_entries:
                symbol = entry['symbol']
                strength = entry['signal_strength']
                price = entry['current_price']
                reasoning = entry['reasoning']
                
                if strength >= 80:
                    urgency = "🔥 URGENT"
                    color = 'red'
                elif strength >= 70:
                    urgency = "⚡ STRONG"
                    color = 'yellow'
                else:
                    urgency = "📈 MODERATE"
                    color = 'green'
                
                print(colored(f"   {symbol}: {urgency} BUY @ ${price:.2f}", color, attrs=['bold']))
                print(f"      Signal Strength: {strength}/100")
                print(f"      Reasoning: {reasoning}")
                
                # Position sizing suggestion
                if strength >= 80:
                    size = "Large position (5-10% portfolio)"
                elif strength >= 70:
                    size = "Medium position (2-5% portfolio)"
                else:
                    size = "Small position (1-2% portfolio)"
                
                print(f"      Suggested Size: {size}")
                print()
        
        else:
            print(colored("\\n⏸️  NO IMMEDIATE BUY SIGNALS", 'blue'))
            print("   All goldmine symbols are waiting for better entry points")
        
        # Trading recommendations
        print(colored("\\n💡 IMMEDIATE ACTION PLAN:", 'cyan', attrs=['bold']))
        
        if potential_entries:
            best_entry = potential_entries[0]
            print(f"   🎯 PRIORITY BUY: {best_entry['symbol']} @ ${best_entry['current_price']:.2f}")
            print(f"      Strength: {best_entry['signal_strength']}/100")
        
        if current_positions:
            best_performer = max(current_positions, key=lambda x: x['pnl_pct'])
            worst_performer = min(current_positions, key=lambda x: x['pnl_pct'])
            
            if best_performer['pnl_pct'] > 20:
                print(f"   💰 TAKE PROFITS: {best_performer['symbol']} (+{best_performer['pnl_pct']:.1f}%)")
            
            if worst_performer['pnl_pct'] < -15:
                print(f"   🛑 CONSIDER STOP: {worst_performer['symbol']} ({worst_performer['pnl_pct']:.1f}%)")
        
        if not current_positions and not potential_entries:
            print("   😴 WAIT FOR SIGNALS: Market is not presenting clear opportunities")
            print("   📊 Monitor for extreme oversold conditions in goldmine symbols")


def main():
    """Run current position check"""
    
    print("📊 Checking current position status for goldmine symbols...")
    print("🎯 Analyzing if we'd be in positions based on recent data...")
    
    checker = CurrentPositionChecker()
    checker.check_current_signals()
    
    print("\\n✅ Position check complete!")

if __name__ == "__main__":
    main()