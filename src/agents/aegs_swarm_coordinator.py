"""
🔥💎 AEGS SWARM COORDINATOR 💎🔥
Main orchestrator for the AEGS AI discovery swarm

Coordinates:
1. Discovery Agent - Finds candidates
2. Backtest Agent - Tests candidates  
3. Analysis Agent - Ranks results
4. Notification Agent - Alerts on goldmines
"""

import os
import sys
import time
import json
import schedule
from datetime import datetime, timedelta
from termcolor import colored

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.aegs_discovery_agent import AEGSDiscoveryAgent
from src.agents.aegs_backtest_agent import AEGSBacktestAgent
from src.agents.backtest_history import BacktestHistory

class AEGSSwarmCoordinator:
    """
    Coordinates the AEGS discovery swarm
    """
    
    def __init__(self):
        self.name = "AEGS Swarm Coordinator"
        self.run_count = 0
        self.total_goldmines = 0
        
        # Initialize agents
        self.discovery_agent = AEGSDiscoveryAgent()
        self.backtest_agent = AEGSBacktestAgent(max_workers=5)
        
        # Configuration
        self.config = {
            'discovery_schedule': 'daily',
            'discovery_time': '06:00',
            'max_candidates_per_run': 50,
            'min_excess_for_alert': 1000,  # Alert on goldmines
            'continuous_mode': False,
            'continuous_interval_hours': 4,  # Run every 4 hours by default
            'use_schedule': False  # Use fixed schedule vs interval
        }
        
    def run_discovery_cycle(self):
        """Run a complete discovery cycle"""
        
        self.run_count += 1
        
        print(colored(f"\n🔥💎 AEGS SWARM CYCLE #{self.run_count} STARTING 💎🔥", 'cyan', attrs=['bold']))
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Show backtest history summary
        history = BacktestHistory()
        summary = history.get_summary()
        print(f"\n📚 Backtest History:")
        print(f"   Total symbols tested: {summary['total_symbols_tested']}")
        print(f"   Tested today: {summary['tested_today']}")
        print(f"   Tested this month: {summary['tested_this_month']}")
        print(f"   Retest interval: {summary['retest_interval_days']} days")
        
        cycle_start = time.time()
        
        # Clean up old history records periodically
        if self.run_count % 10 == 0:  # Every 10th run
            history.cleanup_old_records()
        
        # Phase 1: Discovery
        print(colored("\n📡 PHASE 1: DISCOVERY", 'yellow', attrs=['bold']))
        candidates = self.discovery_agent.run()
        
        if not candidates:
            print("❌ No candidates discovered in this cycle")
            return
        
        print(f"✅ Discovered {len(candidates)} candidates")
        
        # Save discovery results (the agent already does this)
        discovery_file = self._get_latest_discovery_file()
        
        # Phase 2: Backtesting
        print(colored("\n🧪 PHASE 2: BACKTESTING", 'yellow', attrs=['bold']))
        results = self.backtest_agent.run(discovery_file)
        
        # Phase 3: Analysis
        print(colored("\n📊 PHASE 3: ANALYSIS", 'yellow', attrs=['bold']))
        goldmines, high_potential = self._analyze_results(results)
        
        # Phase 4: Notification
        if goldmines:
            self._send_goldmine_alert(goldmines)
        
        # Cycle complete
        cycle_time = time.time() - cycle_start
        self._print_cycle_summary(len(candidates), results, goldmines, cycle_time)
        
        # Update scanner registry (already done by backtest agent)
        print("\n✅ Registry automatically updated with new discoveries")
        
    def _get_latest_discovery_file(self):
        """Get the most recent discovery file"""
        import glob
        files = sorted(glob.glob("aegs_discoveries_*.json"))
        return files[-1] if files else None
    
    def _analyze_results(self, results):
        """Analyze backtest results"""
        
        if not results:
            return [], []
        
        goldmines = []
        high_potential = []
        
        for result in results:
            if result['excess_return'] > 1000:
                goldmines.append(result)
                self.total_goldmines += 1
            elif result['excess_return'] > 100:
                high_potential.append(result)
        
        # Sort by excess return
        goldmines.sort(key=lambda x: x['excess_return'], reverse=True)
        high_potential.sort(key=lambda x: x['excess_return'], reverse=True)
        
        return goldmines, high_potential
    
    def _send_goldmine_alert(self, goldmines):
        """Send alert for new goldmines"""
        
        print(colored("\n🚨 GOLDMINE ALERT! 🚨", 'red', attrs=['bold', 'blink']))
        print("=" * 80)
        
        for gm in goldmines:
            symbol = gm['symbol']
            excess = gm['excess_return']
            win_rate = gm['win_rate']
            
            print(colored(f"💎 {symbol}: {excess:+,.0f}% excess return!", 'red', attrs=['bold']))
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Discovery: {gm['discovery_reason']}")
            
            # Calculate potential
            investment = 10000
            potential = investment * (1 + gm['strategy_return']/100)
            print(colored(f"   💰 $10k → ${potential:,.0f}", 'cyan'))
        
        print("\n🎯 ACTION REQUIRED:")
        print("   1. Monitor these symbols for entry signals")
        print("   2. Run 'python aegs_enhanced_scanner.py' to check current signals")
        print("   3. Deploy capital on oversold conditions")
        
        # In production, this would send email/Discord/Telegram alerts
        self._save_alert(goldmines)
    
    def _save_alert(self, goldmines):
        """Save goldmine alert to file"""
        
        alert_data = {
            'alert_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'goldmine_discovery',
            'count': len(goldmines),
            'goldmines': goldmines
        }
        
        filename = f"aegs_goldmine_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    def _print_cycle_summary(self, candidates_count, results, goldmines, cycle_time):
        """Print cycle summary"""
        
        print("\n" + "=" * 80)
        print(colored("📈 CYCLE SUMMARY", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        print(f"Cycle #{self.run_count} Complete:")
        print(f"   ⏱️  Duration: {cycle_time/60:.1f} minutes")
        print(f"   🔍 Candidates discovered: {candidates_count}")
        print(f"   ✅ Successfully backtested: {len(results) if results else 0}")
        print(f"   💎 New goldmines: {len(goldmines)}")
        print(f"   🏆 Total goldmines found: {self.total_goldmines}")
        
        if goldmines:
            print(colored(f"\n🎉 This cycle discovered {len(goldmines)} new millionaire makers!", 'green', attrs=['bold']))
    
    def run_continuous(self):
        """Run in continuous mode with scheduling"""
        
        print(colored("🔄 AEGS SWARM - CONTINUOUS MODE", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        if self.config['use_schedule']:
            print(f"Schedule: {self.config['discovery_schedule']} at {self.config['discovery_time']}")
        else:
            print(f"Interval: Every {self.config['continuous_interval_hours']} hours")
        
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        if self.config['use_schedule']:
            # Schedule daily discovery
            schedule.every().day.at(self.config['discovery_time']).do(self.run_discovery_cycle)
            
            # Run once immediately
            self.run_discovery_cycle()
            
            # Keep running with schedule
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        else:
            # Interval-based continuous mode
            while True:
                try:
                    # Run discovery cycle
                    self.run_discovery_cycle()
                    
                    # Wait for next cycle
                    wait_hours = self.config['continuous_interval_hours']
                    wait_seconds = wait_hours * 3600
                    
                    print(colored(f"\n⏰ Waiting {wait_hours} hours until next discovery cycle...", 'yellow'))
                    print(f"Next run: {(datetime.now() + timedelta(hours=wait_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Sleep with countdown
                    for remaining in range(int(wait_seconds), 0, -60):
                        mins_left = remaining // 60
                        if mins_left % 30 == 0 and mins_left > 0:  # Update every 30 mins
                            print(f"   {mins_left} minutes remaining...")
                        time.sleep(min(60, remaining))
                        
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(colored(f"\n❌ Error in discovery cycle: {str(e)}", 'red'))
                    print("Waiting 5 minutes before retry...")
                    time.sleep(300)  # Wait 5 minutes on error
    
    def run_once(self):
        """Run a single discovery cycle"""
        self.run_discovery_cycle()


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='AEGS Swarm Coordinator')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('--time', default='06:00', help='Discovery time for continuous mode')
    
    args = parser.parse_args()
    
    # Initialize coordinator
    coordinator = AEGSSwarmCoordinator()
    
    if args.continuous:
        coordinator.config['discovery_time'] = args.time
        coordinator.config['continuous_mode'] = True
        coordinator.run_continuous()
    else:
        coordinator.run_once()


if __name__ == "__main__":
    main()