"""
🌿🔮 TLRY AUTO MONITOR WITH HURST EXPONENT INTEGRATION
Enhanced exit signal detection using fractal regime analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Import Hurst Exponent indicator
from src.fractal_alpha.indicators.fractals.hurst_exponent import HurstExponentIndicator


class TLRYHurstMonitor:
    """TLRY position monitor with Hurst regime detection"""
    
    def __init__(self):
        self.symbol = "TLRY"
        self.position = self.load_position()
        self.hurst_indicator = HurstExponentIndicator(
            lookback_periods=100,
            rolling_window=50
        )
        self.alerts_sent = []
        
        # Exit thresholds
        self.profit_target = 0.20  # 20% profit
        self.stop_loss = -0.10    # 10% loss
        self.regime_exit_threshold = 0.45  # Exit if Hurst < 0.45 (mean reverting)
        
        # Technical thresholds
        self.rsi_overbought = 70
        self.volume_spike_multiplier = 3
        
    def load_position(self):
        """Load TLRY position from file"""
        try:
            with open('tlry_position_card.json', 'r') as f:
                position = json.load(f)
                # Convert string date to datetime
                position['entry_date'] = pd.to_datetime(position['entry_date'])
                return position
        except:
            # Default position if no file
            return {
                'symbol': 'TLRY',
                'entry_price': 1.48,
                'shares': 1000,
                'entry_date': pd.to_datetime('2024-11-26'),
                'strategy': 'Oversold bounce from $1.39 support'
            }
    
    def save_position(self):
        """Save position to file"""
        position = self.position.copy()
        position['entry_date'] = position['entry_date'].isoformat()
        
        with open('tlry_position_card.json', 'w') as f:
            json.dump(position, f, indent=2)
    
    def get_current_data(self):
        """Fetch current TLRY data with technical indicators"""
        
        # Get recent data for indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)  # Extra for Hurst calculation
        
        df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
            
        # Fix multi-level column issue if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure lowercase columns for Hurst indicator compatibility
        df.columns = df.columns.str.lower()
            
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def calculate_hurst_regime(self, df):
        """Calculate Hurst exponent and determine regime"""
        
        if len(df) < 100:
            return None, "insufficient_data"
            
        # Calculate Hurst
        result = self.hurst_indicator.calculate(df, self.symbol)
        
        if result and 'hurst_value' in result.metadata:
            hurst_value = result.metadata['hurst_value']
            regime = result.metadata.get('market_regime', 'unknown')
            confidence = result.confidence
            
            return {
                'hurst': hurst_value,
                'regime': regime,
                'confidence': confidence,
                'interpretation': result.metadata.get('interpretation', ''),
                'signal': result.signal.value
            }
        
        return None
    
    def check_exit_signals(self, df, current_price):
        """Check for exit signals including Hurst regime"""
        
        exit_signals = []
        
        # Get position metrics
        days_held = (datetime.now() - self.position['entry_date']).days
        pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
        pnl_dollar = (current_price - self.position['entry_price']) * self.position['shares']
        
        # 1. Price-based exits
        if pnl_pct >= self.profit_target:
            exit_signals.append({
                'type': 'PROFIT_TARGET',
                'urgency': 'HIGH',
                'reason': f'Reached {pnl_pct:.1%} profit target',
                'action': 'SELL_75%'  # Take partial profits
            })
            
        if pnl_pct <= self.stop_loss:
            exit_signals.append({
                'type': 'STOP_LOSS',
                'urgency': 'CRITICAL',
                'reason': f'Hit {pnl_pct:.1%} stop loss',
                'action': 'SELL_ALL'
            })
        
        # 2. Hurst regime analysis
        hurst_analysis = self.calculate_hurst_regime(df)
        if hurst_analysis:
            hurst_value = hurst_analysis['hurst']
            regime = hurst_analysis['regime']
            
            # Regime-based exits
            if hurst_value < self.regime_exit_threshold:
                exit_signals.append({
                    'type': 'REGIME_CHANGE',
                    'urgency': 'HIGH',
                    'reason': f'Hurst {hurst_value:.3f} indicates {regime} regime - unfavorable for holding',
                    'action': 'SELL_50%',
                    'hurst_analysis': hurst_analysis
                })
                
            # Regime transition detection
            if hurst_analysis['confidence'] < 50 and abs(hurst_value - 0.5) < 0.05:
                exit_signals.append({
                    'type': 'REGIME_TRANSITION',
                    'urgency': 'MEDIUM',
                    'reason': f'Market in transition (Hurst={hurst_value:.3f}) - consider reducing',
                    'action': 'SELL_25%',
                    'hurst_analysis': hurst_analysis
                })
        
        # 3. Technical indicator exits
        latest = df.iloc[-1]
        
        # RSI overbought
        if latest['rsi'] > self.rsi_overbought:
            exit_signals.append({
                'type': 'RSI_OVERBOUGHT',
                'urgency': 'MEDIUM',
                'reason': f'RSI {latest["rsi"]:.1f} is overbought',
                'action': 'SELL_50%'
            })
        
        # Volume spike with red candle
        if latest['volume_ratio'] > self.volume_spike_multiplier and latest['close'] < latest['open']:
            exit_signals.append({
                'type': 'DISTRIBUTION',
                'urgency': 'HIGH',
                'reason': f'High volume ({latest["volume_ratio"]:.1f}x avg) selling detected',
                'action': 'SELL_50%'
            })
        
        # Death cross
        if len(df) > 50 and latest['sma20'] < latest['sma50'] and df.iloc[-2]['sma20'] >= df.iloc[-2]['sma50']:
            exit_signals.append({
                'type': 'DEATH_CROSS',
                'urgency': 'HIGH',
                'reason': 'SMA20 crossed below SMA50',
                'action': 'SELL_75%'
            })
        
        # 4. Time-based exit (if held too long)
        if days_held > 30 and pnl_pct < 0.05:
            exit_signals.append({
                'type': 'TIME_EXIT',
                'urgency': 'LOW',
                'reason': f'Position held {days_held} days with minimal gain ({pnl_pct:.1%})',
                'action': 'SELL_50%'
            })
        
        return exit_signals, {
            'days_held': days_held,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'current_price': current_price,
            'hurst_analysis': hurst_analysis
        }
    
    def display_position_status(self, metrics, exit_signals):
        """Display current position status with Hurst analysis"""
        
        print("\n" + "=" * 80)
        print(colored("🌿 TLRY POSITION MONITOR WITH HURST REGIME ANALYSIS 🔮", 'green', attrs=['bold']))
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Position details
        print(colored("\n📊 POSITION DETAILS:", 'cyan'))
        print(f"Entry Price: ${self.position['entry_price']:.2f}")
        print(f"Current Price: ${metrics['current_price']:.2f}")
        print(f"Shares: {self.position['shares']:,}")
        print(f"Days Held: {metrics['days_held']}")
        
        # P&L
        pnl_color = 'green' if metrics['pnl_pct'] > 0 else 'red'
        print(colored(f"\n💰 P&L: {metrics['pnl_pct']:+.1%} (${metrics['pnl_dollar']:+,.2f})", pnl_color, attrs=['bold']))
        
        # Hurst Analysis
        if metrics['hurst_analysis']:
            hurst = metrics['hurst_analysis']
            print(colored("\n🔮 HURST REGIME ANALYSIS:", 'yellow'))
            print(f"Hurst Exponent: {hurst['hurst']:.3f}")
            print(f"Market Regime: {hurst['regime'].upper()}")
            print(f"Confidence: {hurst['confidence']:.0f}%")
            print(f"Interpretation: {hurst['interpretation']}")
            print(f"Regime Signal: {hurst['signal']}")
            
            # Regime-specific advice
            if hurst['hurst'] > 0.55:
                print(colored("✅ Trending regime favorable for position holding", 'green'))
            elif hurst['hurst'] < 0.45:
                print(colored("⚠️ Mean-reverting regime - consider exit or reduction", 'yellow'))
            else:
                print(colored("⏸️ Random walk regime - monitor closely", 'blue'))
        
        # Exit signals
        if exit_signals:
            print(colored(f"\n🚨 EXIT SIGNALS DETECTED ({len(exit_signals)}):", 'red', attrs=['bold']))
            
            # Sort by urgency
            urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            exit_signals.sort(key=lambda x: urgency_order.get(x['urgency'], 4))
            
            for signal in exit_signals:
                urgency_color = {
                    'CRITICAL': 'red',
                    'HIGH': 'yellow',
                    'MEDIUM': 'blue',
                    'LOW': 'white'
                }.get(signal['urgency'], 'white')
                
                print(f"\n{colored(f'[{signal["urgency"]}]', urgency_color, attrs=['bold'])} {signal['type']}:")
                print(f"  Reason: {signal['reason']}")
                print(f"  Action: {colored(signal['action'], 'cyan', attrs=['bold'])}")
                
                # Show Hurst details for regime signals
                if 'hurst_analysis' in signal and signal['hurst_analysis']:
                    h = signal['hurst_analysis']
                    print(f"  Hurst Details: H={h['hurst']:.3f}, Regime={h['regime']}, Confidence={h['confidence']:.0f}%")
        else:
            print(colored("\n✅ NO EXIT SIGNALS - HOLD POSITION", 'green', attrs=['bold']))
        
        # Trading recommendation
        print(colored("\n💡 RECOMMENDATION:", 'cyan', attrs=['bold']))
        
        if any(s['urgency'] == 'CRITICAL' for s in exit_signals):
            print(colored("🚨 IMMEDIATE ACTION REQUIRED - EXIT POSITION", 'red', attrs=['bold', 'blink']))
        elif any(s['urgency'] == 'HIGH' for s in exit_signals):
            print(colored("⚠️ Consider reducing position size (50-75%)", 'yellow', attrs=['bold']))
        elif any(s['urgency'] == 'MEDIUM' for s in exit_signals):
            print(colored("👀 Monitor closely - prepare to exit if conditions worsen", 'blue'))
        else:
            print(colored("✅ Continue holding - regime favorable", 'green'))
        
        print("\n" + "=" * 80)
    
    def run_continuous_monitor(self, check_interval=300):
        """Run continuous monitoring with alerts"""
        
        print(colored("🚀 Starting TLRY Auto Monitor with Hurst Integration", 'cyan', attrs=['bold']))
        print(f"Checking every {check_interval} seconds...")
        
        while True:
            try:
                # Get current data
                df = self.get_current_data()
                
                if df is not None and len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    
                    # Check exit signals
                    exit_signals, metrics = self.check_exit_signals(df, current_price)
                    
                    # Display status
                    self.display_position_status(metrics, exit_signals)
                    
                    # Send alerts for critical signals
                    for signal in exit_signals:
                        if signal['urgency'] in ['CRITICAL', 'HIGH']:
                            alert_key = f"{signal['type']}_{datetime.now().strftime('%Y%m%d')}"
                            if alert_key not in self.alerts_sent:
                                print("\n" + colored(f"🔔 ALERT: {signal['type']} - {signal['reason']}", 'red', attrs=['bold', 'blink']))
                                self.alerts_sent.append(alert_key)
                else:
                    print(colored("\n❌ Failed to fetch data", 'red'))
                
                # Wait for next check
                print(f"\nNext check in {check_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print(colored("\n\n👋 Monitoring stopped by user", 'yellow'))
                break
            except Exception as e:
                print(colored(f"\n❌ Error: {str(e)}", 'red'))
                time.sleep(check_interval)
    
    def run_single_check(self):
        """Run a single check"""
        df = self.get_current_data()
        
        if df is not None and len(df) > 0:
            current_price = df['close'].iloc[-1]
            exit_signals, metrics = self.check_exit_signals(df, current_price)
            self.display_position_status(metrics, exit_signals)
            
            # Save metrics
            self.save_check_results(metrics, exit_signals)
        else:
            print(colored("❌ Failed to fetch data", 'red'))
    
    def save_check_results(self, metrics, exit_signals):
        """Save check results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'metrics': {
                'current_price': float(metrics['current_price']),
                'pnl_pct': float(metrics['pnl_pct']),
                'pnl_dollar': float(metrics['pnl_dollar']),
                'days_held': int(metrics['days_held'])
            },
            'hurst_analysis': metrics['hurst_analysis'] if metrics['hurst_analysis'] else None,
            'exit_signals': exit_signals,
            'alert_count': len(exit_signals),
            'max_urgency': max([s['urgency'] for s in exit_signals], default='NONE')
        }
        
        filename = f'tlry_monitor_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to {filename}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TLRY Position Monitor with Hurst Integration')
    parser.add_argument('--continuous', '-c', action='store_true', 
                       help='Run continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=300,
                       help='Check interval in seconds (default: 300)')
    parser.add_argument('--update-position', '-u', action='store_true',
                       help='Update position details')
    
    args = parser.parse_args()
    
    monitor = TLRYHurstMonitor()
    
    if args.update_position:
        # Interactive position update
        print("Update TLRY position:")
        monitor.position['entry_price'] = float(input(f"Entry price [{monitor.position['entry_price']}]: ") or monitor.position['entry_price'])
        monitor.position['shares'] = int(input(f"Shares [{monitor.position['shares']}]: ") or monitor.position['shares'])
        monitor.save_position()
        print("✅ Position updated!")
    
    if args.continuous:
        monitor.run_continuous_monitor(check_interval=args.interval)
    else:
        monitor.run_single_check()


if __name__ == "__main__":
    main()