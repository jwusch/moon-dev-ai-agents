#!/usr/bin/env python3
"""
🚨 TLRY Automated Exit Monitor
Continuously monitors TLRY for AEGS exit signals and alerts
"""

import time
import datetime
import yfinance as yf
from tlry_exit_tracker import TLRYExitTracker
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class TLRYAutoMonitor:
    def __init__(self, entry_price=None, check_interval_minutes=15):
        self.tracker = TLRYExitTracker()
        self.entry_price = entry_price
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.alert_history = []
        
    def send_alert(self, message, level='info'):
        """Send alert (can be extended to email/SMS/Discord)"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if level == 'critical':
            print(colored(f"\n🚨🚨🚨 CRITICAL ALERT - {timestamp} 🚨🚨🚨", 'red', attrs=['bold', 'blink']))
            print(colored(message, 'red', attrs=['bold']))
        elif level == 'warning':
            print(colored(f"\n⚠️  WARNING - {timestamp}", 'yellow', attrs=['bold']))
            print(colored(message, 'yellow'))
        else:
            print(f"\n📊 Update - {timestamp}")
            print(message)
            
        self.alert_history.append({
            'time': timestamp,
            'level': level,
            'message': message
        })
        
    def check_critical_conditions(self):
        """Check for critical exit conditions that need immediate attention"""
        try:
            df_1h, df_daily = self.tracker.get_current_data()
            df_1h = self.tracker.calculate_indicators(df_1h)
            
            current_price = df_1h['Close'].iloc[-1]
            rsi = df_1h['RSI'].iloc[-1]
            bb_percent = df_1h['BB_%'].iloc[-1]
            hurst = df_1h['Hurst'].iloc[-1] if 'Hurst' in df_1h.columns else 0.5
            regime = self.tracker.interpret_hurst(hurst)
            
            critical_alerts = []
            
            # Critical RSI condition
            if rsi > 75:
                critical_alerts.append(f"RSI EXTREME: {rsi:.1f} (>75)")
                
            # At upper Bollinger Band
            if bb_percent > 0.98:
                critical_alerts.append(f"AT UPPER BB: {bb_percent:.1%}")
                
            # Check for rapid price spike
            if len(df_1h) > 4:
                price_change_4h = ((current_price - df_1h['Close'].iloc[-4]) / df_1h['Close'].iloc[-4]) * 100
                if price_change_4h > 15:
                    critical_alerts.append(f"RAPID SPIKE: +{price_change_4h:.1f}% in 4 hours")
            
            # Check for regime-based alerts
            if hurst < 0.4 and rsi > 60:
                critical_alerts.append(f"MEAN-REVERTING + HIGH RSI: H={hurst:.2f}")
            elif hurst < 0.45 and bb_percent > 0.8:
                critical_alerts.append(f"MEAN-REVERSION REGIME: Take profits")
                    
            # Check P&L if entry price provided
            if self.entry_price:
                pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                if pnl_percent > 30:
                    critical_alerts.append(f"PROFIT TARGET: +{pnl_percent:.1f}%")
                elif pnl_percent < -10:
                    critical_alerts.append(f"STOP LOSS: {pnl_percent:.1f}%")
                    
            return critical_alerts, current_price
            
        except Exception as e:
            print(f"Error checking conditions: {e}")
            return [], 0
            
    def run_continuous_monitoring(self):
        """Run continuous monitoring with alerts"""
        
        print(colored("🚨 TLRY AUTOMATED EXIT MONITOR 🚨", 'cyan', attrs=['bold']))
        print("="*60)
        print(f"Check Interval: {self.check_interval/60:.0f} minutes")
        if self.entry_price:
            print(f"Entry Price: ${self.entry_price:.2f}")
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        print("\nPress Ctrl+C to stop monitoring")
        
        last_exit_score = 0
        check_count = 0
        
        while True:
            try:
                check_count += 1
                print(f"\n--- Check #{check_count} ---")
                
                # Check critical conditions first
                critical_alerts, current_price = self.check_critical_conditions()
                
                if critical_alerts:
                    alert_msg = f"TLRY @ ${current_price:.2f}\n"
                    alert_msg += "\n".join(f"• {alert}" for alert in critical_alerts)
                    self.send_alert(alert_msg, 'critical')
                
                # Run full analysis
                df_1h, df_daily = self.tracker.get_current_data()
                df_1h = self.tracker.calculate_indicators(df_1h)
                df_daily = self.tracker.calculate_indicators(df_daily)
                
                exit_signals, exit_score = self.tracker.check_aegs_exit_signals(df_1h, df_daily)
                
                # Alert on score changes
                if exit_score > last_exit_score and exit_score >= 6:
                    alert_msg = f"Exit score increased: {last_exit_score} → {exit_score}\n"
                    if exit_signals['immediate']:
                        alert_msg += "Immediate signals: " + ", ".join(exit_signals['immediate'])
                    self.send_alert(alert_msg, 'warning')
                    
                # Regular update
                if check_count % 4 == 0:  # Every hour if checking every 15 min
                    hurst_val = df_1h['Hurst'].iloc[-1] if 'Hurst' in df_1h.columns else 0.5
                    regime_str = self.tracker.interpret_hurst(hurst_val)
                    
                    summary = f"TLRY @ ${current_price:.2f} | "
                    summary += f"RSI: {df_1h['RSI'].iloc[-1]:.1f} | "
                    summary += f"BB%: {df_1h['BB_%'].iloc[-1]:.1%} | "
                    summary += f"Hurst: {hurst_val:.2f} ({regime_str}) | "
                    summary += f"Exit Score: {exit_score}"
                    
                    if self.entry_price:
                        pnl = ((current_price - self.entry_price) / self.entry_price) * 100
                        summary += f" | P&L: {pnl:+.1f}%"
                        
                    self.send_alert(summary, 'info')
                    
                last_exit_score = exit_score
                
                # Sleep until next check
                print(f"\nNext check in {self.check_interval/60:.0f} minutes...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print(colored("\n\n🛑 Monitoring stopped by user", 'yellow'))
                break
            except Exception as e:
                print(colored(f"\n❌ Error: {e}", 'red'))
                print("Retrying in 1 minute...")
                time.sleep(60)
                
        # Show summary
        self.show_summary()
        
    def show_summary(self):
        """Show monitoring summary"""
        print("\n" + "="*60)
        print(colored("📊 MONITORING SUMMARY", 'cyan', attrs=['bold']))
        print("="*60)
        
        if self.alert_history:
            print(f"\nTotal Alerts: {len(self.alert_history)}")
            
            critical_count = sum(1 for a in self.alert_history if a['level'] == 'critical')
            warning_count = sum(1 for a in self.alert_history if a['level'] == 'warning')
            
            print(f"Critical: {critical_count}")
            print(f"Warnings: {warning_count}")
            
            print("\nRecent Alerts:")
            for alert in self.alert_history[-5:]:
                level_color = 'red' if alert['level'] == 'critical' else 'yellow' if alert['level'] == 'warning' else 'white'
                print(colored(f"{alert['time']} - {alert['message'][:80]}...", level_color))

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TLRY Automated Exit Monitor')
    parser.add_argument('--entry', type=float, help='Your entry price for P&L calculation')
    parser.add_argument('--interval', type=int, default=15, help='Check interval in minutes (default: 15)')
    
    args = parser.parse_args()
    
    monitor = TLRYAutoMonitor(entry_price=args.entry, check_interval_minutes=args.interval)
    monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()