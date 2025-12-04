#!/usr/bin/env python
"""
🌟 UNIVERSAL POSITION MONITOR WITH HURST ANALYSIS
Monitors all open positions from the database with technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import argparse
from termcolor import colored
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.position_tracker import PositionTracker
from src.fractal_alpha.indicators.fractals.hurst_exponent import HurstExponentIndicator
from enhanced_position_alerts import AlertManager, AlertType, AlertPriority, create_smart_alerts_from_exit_signals


class UniversalPositionMonitor:
    """Monitor all open positions with technical analysis"""
    
    def __init__(self):
        self.tracker = PositionTracker()
        self.hurst_indicator = HurstExponentIndicator(
            lookback_periods=100,
            rolling_window=50
        )
        self.alert_manager = AlertManager()
        
        # Thresholds
        self.profit_target = 0.20  # 20% profit
        self.stop_loss = -0.10    # 10% loss
        self.regime_exit_threshold = 0.45  # Exit if Hurst < 0.45
        self.rsi_overbought = 70
        self.volume_spike_multiplier = 3
        
    def get_market_data(self, symbol: str, days: int = 150) -> pd.DataFrame:
        """Fetch market data for a symbol"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                return None
                
            # Fix multi-level column issue if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure lowercase columns
            df.columns = df.columns.str.lower()
            
            # Calculate indicators
            self._add_technical_indicators(df)
            
            return df
        except Exception as e:
            print(colored(f"❌ Error fetching {symbol}: {str(e)}", 'red'))
            return None
            
    def _add_technical_indicators(self, df: pd.DataFrame):
        """Add technical indicators to dataframe"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR for volatility-based stops
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
    def calculate_hurst_regime(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate Hurst exponent and determine regime"""
        if len(df) < 100:
            return None
            
        try:
            result = self.hurst_indicator.calculate(df, symbol)
            
            if result and 'hurst_value' in result.metadata:
                return {
                    'hurst': result.metadata['hurst_value'],
                    'regime': result.metadata.get('market_regime', 'unknown'),
                    'confidence': result.confidence,
                    'interpretation': result.metadata.get('interpretation', ''),
                    'signal': result.signal.value
                }
        except Exception as e:
            print(f"Hurst calculation error for {symbol}: {e}")
            
        return None
        
    def check_exit_signals(self, position: Dict, df: pd.DataFrame, current_price: float) -> Tuple[List, Dict]:
        """Check for exit signals on a position"""
        exit_signals = []
        
        # Calculate position metrics
        entry_price = position['entry_price']
        shares = position['shares']
        # Handle both date formats
        entry_date_str = str(position['entry_date'])
        try:
            entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')
        days_held = (datetime.now() - entry_date).days
        pnl_pct = (current_price - entry_price) / entry_price
        pnl_dollar = (current_price - entry_price) * shares
        
        # 1. Price-based exits
        if pnl_pct >= self.profit_target:
            exit_signals.append({
                'type': 'PROFIT_TARGET',
                'urgency': 'HIGH',
                'reason': f'Reached {pnl_pct:.1%} profit target',
                'action': 'SELL_75%'
            })
            
        if pnl_pct <= self.stop_loss:
            exit_signals.append({
                'type': 'STOP_LOSS',
                'urgency': 'CRITICAL',
                'reason': f'Hit {pnl_pct:.1%} stop loss',
                'action': 'SELL_ALL'
            })
            
        # 2. Volatility-based trailing stop
        if len(df) > 14 and 'atr' in df.columns and pnl_pct > 0.05:  # Only if in profit > 5%
            atr = df['atr'].iloc[-1]
            trail_stop = current_price - (2.5 * atr)  # 2.5x ATR trailing stop
            
            # Only alert if trailing stop is above entry price (protecting profits)
            # and current price is getting close to the trailing stop
            if trail_stop > entry_price and (current_price - trail_stop) / current_price < 0.05:
                exit_signals.append({
                    'type': 'TRAILING_STOP',
                    'urgency': 'MEDIUM',
                    'reason': f'Price near trailing stop at ${trail_stop:.2f} (protecting +{(trail_stop - entry_price) / entry_price:.1%} gain)',
                    'action': 'SET_STOP'
                })
        
        # 3. Hurst regime analysis
        hurst_analysis = self.calculate_hurst_regime(df, position['symbol'])
        if hurst_analysis:
            hurst_value = hurst_analysis['hurst']
            regime = hurst_analysis['regime']
            
            if hurst_value < self.regime_exit_threshold and pnl_pct > 0:
                exit_signals.append({
                    'type': 'REGIME_CHANGE',
                    'urgency': 'HIGH',
                    'reason': f'Hurst {hurst_value:.3f} indicates {regime} regime',
                    'action': 'SELL_50%',
                    'hurst_analysis': hurst_analysis
                })
                
        # 4. Technical indicator exits
        if len(df) > 0:
            latest = df.iloc[-1]
            
            # RSI overbought
            if latest['rsi'] > self.rsi_overbought and pnl_pct > 0:
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
                    'reason': f'High volume ({latest["volume_ratio"]:.1f}x avg) selling',
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
        
        # 5. Time-based exit
        if days_held > 30 and pnl_pct < 0.05:
            exit_signals.append({
                'type': 'TIME_EXIT',
                'urgency': 'LOW',
                'reason': f'Held {days_held} days with minimal gain ({pnl_pct:.1%})',
                'action': 'SELL_50%'
            })
            
        metrics = {
            'current_price': current_price,
            'entry_price': entry_price,
            'shares': shares,
            'days_held': days_held,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'hurst_analysis': hurst_analysis
        }
        
        return exit_signals, metrics
        
    def display_position_status(self, position: Dict, metrics: Dict, exit_signals: List):
        """Display status for a single position"""
        symbol = position['symbol']
        
        print(f"\n{'='*60}")
        print(colored(f"📊 {symbol} - Position #{position['id']}", 'cyan', attrs=['bold']))
        print(f"Entry: ${metrics['entry_price']:.2f} | Current: ${metrics['current_price']:.2f} | Shares: {metrics['shares']}")
        print(f"Days Held: {metrics['days_held']}")
        
        # P&L
        pnl_color = 'green' if metrics['pnl_pct'] > 0 else 'red'
        print(colored(f"P&L: {metrics['pnl_pct']:+.1%} (${metrics['pnl_dollar']:+,.2f})", pnl_color, attrs=['bold']))
        
        # Hurst Analysis
        if metrics['hurst_analysis']:
            hurst = metrics['hurst_analysis']
            print(colored(f"Regime: {hurst['regime']} (Hurst={hurst['hurst']:.3f}, Confidence={hurst['confidence']:.0f}%)", 'yellow'))
            
        # Exit signals
        if exit_signals:
            print(colored(f"\n🚨 {len(exit_signals)} Exit Signals:", 'red'))
            
            # Sort by urgency
            urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            exit_signals.sort(key=lambda x: urgency_order.get(x['urgency'], 4))
            
            for signal in exit_signals[:3]:  # Show top 3 signals
                urgency_color = {
                    'CRITICAL': 'red',
                    'HIGH': 'yellow',
                    'MEDIUM': 'blue',
                    'LOW': 'white'
                }.get(signal['urgency'], 'white')
                
                print(f"  [{colored(signal['urgency'], urgency_color)}] {signal['reason']}")
        else:
            print(colored("✅ No exit signals - HOLD", 'green'))
            
    def monitor_all_positions(self):
        """Monitor all open positions"""
        # Get open positions
        open_positions = self.tracker.get_open_positions()
        
        if open_positions.empty:
            print(colored("❌ No open positions to monitor", 'yellow'))
            return
            
        print(colored(f"\n🌟 MONITORING {len(open_positions)} OPEN POSITIONS", 'cyan', attrs=['bold']))
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_value = 0
        total_pnl = 0
        alerts_summary = []
        
        for _, position in open_positions.iterrows():
            symbol = position['symbol']
            
            # Get market data
            df = self.get_market_data(symbol)
            
            if df is not None and len(df) > 0:
                current_price = df['close'].iloc[-1]
                
                # Check exit signals
                exit_signals, metrics = self.check_exit_signals(position.to_dict(), df, current_price)
                
                # Create enhanced alerts from exit signals
                if exit_signals:
                    smart_alerts = create_smart_alerts_from_exit_signals(
                        symbol, exit_signals, metrics, self.alert_manager
                    )
                
                # Display position
                self.display_position_status(position.to_dict(), metrics, exit_signals)
                
                # Track totals
                current_value = current_price * position['shares']
                total_value += current_value
                total_pnl += metrics['pnl_dollar']
                
                # Collect critical alerts
                critical_signals = [s for s in exit_signals if s['urgency'] in ['CRITICAL', 'HIGH']]
                if critical_signals:
                    alerts_summary.append({
                        'symbol': symbol,
                        'signals': critical_signals,
                        'pnl_pct': metrics['pnl_pct']
                    })
            else:
                print(colored(f"\n❌ Failed to fetch data for {symbol}", 'red'))
                
        # Portfolio summary
        print(f"\n{'='*80}")
        print(colored("💼 PORTFOLIO SUMMARY", 'cyan', attrs=['bold']))
        print(f"Total Value: ${total_value:,.2f}")
        pnl_color = 'green' if total_pnl > 0 else 'red'
        print(colored(f"Total P&L: ${total_pnl:+,.2f}", pnl_color))
        
        # Critical alerts summary
        if alerts_summary:
            print(colored(f"\n⚠️ POSITIONS REQUIRING ATTENTION ({len(alerts_summary)}):", 'red', attrs=['bold']))
            for alert in alerts_summary:
                print(f"  • {alert['symbol']}: {len(alert['signals'])} signals, P&L: {alert['pnl_pct']:+.1%}")
        
        # Show recent alerts summary
        recent_alerts = self.alert_manager.get_recent_alerts(hours=1)
        if recent_alerts:
            print(colored(f"\n📋 RECENT ALERTS ({len(recent_alerts)} in last hour):", 'yellow'))
            for alert in recent_alerts[-3:]:  # Show last 3
                print(f"  • {alert.symbol}: {alert.title} ({alert.priority.value})")
                
        print(f"\n{'='*80}")
        
        return alerts_summary
        
    def run_continuous_monitor(self, check_interval: int = 300):
        """Run continuous monitoring"""
        print(colored("🚀 Starting Universal Position Monitor", 'cyan', attrs=['bold']))
        print(f"Checking every {check_interval} seconds...")
        
        while True:
            try:
                alerts = self.monitor_all_positions()
                
                # Send notifications for critical alerts
                if alerts:
                    print("\n" + colored("🔔 ALERT: Positions need attention!", 'red', attrs=['bold', 'blink']))
                    
                # Wait for next check
                print(f"\nNext check in {check_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print(colored("\n\n👋 Monitoring stopped by user", 'yellow'))
                break
            except Exception as e:
                print(colored(f"\n❌ Error: {str(e)}", 'red'))
                time.sleep(check_interval)
                
    def save_monitor_results(self, alerts_summary: List):
        """Save monitoring results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'positions_monitored': len(self.tracker.get_open_positions()),
            'alerts': alerts_summary,
            'alert_count': len(alerts_summary)
        }
        
        filename = f'position_monitor_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Universal Position Monitor')
    parser.add_argument('--continuous', '-c', action='store_true', 
                       help='Run continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=300,
                       help='Check interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    monitor = UniversalPositionMonitor()
    
    if args.continuous:
        monitor.run_continuous_monitor(check_interval=args.interval)
    else:
        # Single check
        alerts = monitor.monitor_all_positions()
        monitor.save_monitor_results(alerts)


if __name__ == "__main__":
    main()