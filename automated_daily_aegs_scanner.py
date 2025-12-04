#!/usr/bin/env python3
"""
🤖 AUTOMATED DAILY AEGS SCANNER 🤖
Daily automated scanning across all exchanges with intelligent management
"""

import json
import time
import os
import schedule
from datetime import datetime, timedelta
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner
import smtplib
from email.mime.text import MimeText

class AutomatedDailyScanner:
    """Automated daily AEGS scanning system"""
    
    def __init__(self):
        self.scanner = MultiExchangeAEGSScanner()
        self.results_history = []
        self.notification_email = None  # Set this for email alerts
        
    def get_scan_targets(self):
        """Get intelligent scan targets based on market conditions"""
        
        targets = {
            'high_priority': [],      # Always scan these
            'rotation_pool': [],      # Rotate through these
            'discovery_pool': []      # New symbol discovery
        }
        
        # High priority: Current goldmine performers
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            # Always scan extreme goldmines and recent high performers
            for category in ['extreme_goldmines', 'high_potential']:
                targets['high_priority'].extend(list(registry['goldmine_symbols'][category].keys()))
            
            # Positive performers for rotation
            targets['rotation_pool'].extend(list(registry['goldmine_symbols']['positive'].keys()))
            
        except Exception as e:
            print(f"⚠️  Could not load goldmine: {e}")
        
        # Discovery pool: Major indices for new opportunities
        major_indices = self.scanner.get_major_index_symbols()
        for category, symbols in major_indices.items():
            targets['discovery_pool'].extend(symbols)
        
        print(f"📊 Scan Targets:")
        print(f"   High Priority: {len(targets['high_priority'])} symbols")
        print(f"   Rotation Pool: {len(targets['rotation_pool'])} symbols")
        print(f"   Discovery Pool: {len(targets['discovery_pool'])} symbols")
        
        return targets
    
    def create_daily_scan_plan(self, targets):
        """Create intelligent daily scan plan"""
        
        # Daily limits to avoid overwhelming
        daily_limits = {
            'high_priority': 50,      # Always scan these
            'rotation': 30,           # Rotate through positive performers  
            'discovery': 20           # Try new symbols
        }
        
        scan_plan = []
        
        # Always scan high priority
        scan_plan.extend(targets['high_priority'][:daily_limits['high_priority']])
        
        # Rotate through positive performers (different subset each day)
        day_of_year = datetime.now().timetuple().tm_yday
        rotation_start = (day_of_year * daily_limits['rotation']) % len(targets['rotation_pool'])
        rotation_symbols = targets['rotation_pool'][rotation_start:rotation_start + daily_limits['rotation']]
        scan_plan.extend(rotation_symbols)
        
        # Discovery: New symbols based on day of week
        discovery_start = (day_of_year * daily_limits['discovery']) % len(targets['discovery_pool'])
        discovery_symbols = targets['discovery_pool'][discovery_start:discovery_start + daily_limits['discovery']]
        scan_plan.extend(discovery_symbols)
        
        # Remove duplicates while preserving order
        unique_scan_plan = []
        seen = set()
        for symbol in scan_plan:
            if symbol not in seen:
                unique_scan_plan.append(symbol)
                seen.add(symbol)
        
        print(f"🎯 Today's Scan Plan: {len(unique_scan_plan)} symbols")
        return unique_scan_plan
    
    def run_daily_scan(self):
        """Execute the daily scan"""
        
        print("🤖 AUTOMATED DAILY AEGS SCAN STARTING")
        print("=" * 60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get scan targets
        targets = self.get_scan_targets()
        scan_plan = self.create_daily_scan_plan(targets)
        
        if not scan_plan:
            print("❌ No symbols to scan today!")
            return
        
        # Execute scan
        start_time = time.time()
        results = self.scanner.scan_multi_exchange(custom_symbols=scan_plan)
        scan_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"daily_aegs_scan_{timestamp}.json"
        
        # Enhanced results with scan context
        daily_results = {
            'scan_date': datetime.now().isoformat(),
            'scan_type': 'automated_daily',
            'scan_duration_minutes': scan_time / 60,
            'symbols_scanned': len(scan_plan),
            'cache_stats': self.scanner.cache.cache_stats,
            'targets_breakdown': {
                'high_priority_count': len(targets['high_priority']),
                'rotation_count': len(targets['rotation_pool']),
                'discovery_count': len(targets['discovery_pool'])
            },
            'results': results,
            'profitable_count': len(results['profitable']),
            'new_discoveries': self.identify_new_discoveries(results['profitable'])
        }
        
        with open(filename, 'w') as f:
            json.dump(daily_results, f, indent=2)
        
        # Update history
        self.results_history.append({
            'date': datetime.now().isoformat(),
            'filename': filename,
            'profitable_count': len(results['profitable']),
            'scan_duration': scan_time / 60
        })
        
        # Generate and send report
        self.generate_daily_report(daily_results, filename)
        
        print(f"✅ Daily scan complete! Results saved to: {filename}")
        return daily_results
    
    def identify_new_discoveries(self, profitable_symbols):
        """Identify newly discovered profitable symbols"""
        
        # Load existing goldmine to compare
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            existing_symbols = set()
            for category in registry['goldmine_symbols']:
                existing_symbols.update(registry['goldmine_symbols'][category].keys())
            
            # Find new profitable symbols
            new_discoveries = []
            for symbol_data in profitable_symbols:
                if symbol_data['symbol'] not in existing_symbols:
                    new_discoveries.append(symbol_data)
            
            return new_discoveries
            
        except:
            return profitable_symbols  # If no existing registry, all are "new"
    
    def generate_daily_report(self, results, filename):
        """Generate comprehensive daily report"""
        
        profitable = results['results']['profitable']
        new_discoveries = results['new_discoveries']
        
        report = f"""
🤖 DAILY AEGS SCAN REPORT
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

📊 SCAN SUMMARY:
   Symbols Scanned: {results['symbols_scanned']}
   Scan Duration: {results['scan_duration_minutes']:.1f} minutes
   Cache Stats: {results['cache_stats']}
   
💰 PROFITABILITY:
   Profitable Symbols: {len(profitable)}
   New Discoveries: {len(new_discoveries)}
   Success Rate: {len(profitable)/results['symbols_scanned']*100:.1f}%

"""
        
        if profitable:
            # Sort by return
            profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
            
            report += "🏆 TOP PERFORMERS TODAY:\n"
            for i, result in enumerate(profitable[:10], 1):
                symbol = result['symbol']
                ret = result['strategy_return']
                trades = result['total_trades']
                report += f"   {i:2}. {symbol:<5} +{ret:6.1f}% ({trades} trades)\n"
        
        if new_discoveries:
            report += "\n🔥 NEW DISCOVERIES:\n"
            for discovery in new_discoveries[:5]:
                symbol = discovery['symbol']
                ret = discovery['strategy_return']
                report += f"   🆕 {symbol:<5} +{ret:6.1f}%\n"
        
        report += f"\n💾 Full results: {filename}\n"
        report += f"📈 Registry updated with new discoveries\n"
        
        print(report)
        
        # Save report to file
        report_filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        # Send email notification if configured
        if self.notification_email:
            self.send_email_notification(report)
    
    def send_email_notification(self, report):
        """Send email notification (configure SMTP settings)"""
        # Implementation depends on your email provider
        # This is a template - configure with your SMTP settings
        pass
    
    def update_goldmine_with_discoveries(self, new_discoveries):
        """Automatically update goldmine with new discoveries"""
        if not new_discoveries:
            return
            
        print(f"🔄 Auto-updating goldmine with {len(new_discoveries)} new discoveries...")
        
        try:
            # Load current goldmine
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            # Add new discoveries
            for discovery in new_discoveries:
                symbol = discovery['symbol']
                strategy_return = discovery['strategy_return']
                
                # Categorize
                if strategy_return >= 100:
                    category = "extreme_goldmines"
                elif strategy_return >= 30:
                    category = "high_potential"
                else:
                    category = "positive"
                
                # Add to registry
                registry['goldmine_symbols'][category][symbol] = {
                    'strategy_return': discovery['strategy_return'],
                    'total_trades': discovery['total_trades'],
                    'win_rate': discovery['win_rate'],
                    'excess_return': discovery['excess_return'],
                    'added_date': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'automated_daily_scan',
                    'auto_discovered': True
                }
            
            # Update metadata
            registry['metadata']['last_auto_update'] = datetime.now().isoformat()
            registry['metadata']['auto_discoveries'] = len(new_discoveries)
            
            # Save updated registry
            with open('aegs_goldmine_registry.json', 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"✅ Goldmine updated with {len(new_discoveries)} auto-discoveries")
            
        except Exception as e:
            print(f"❌ Error updating goldmine: {e}")
    
    def setup_scheduler(self, run_time="09:30"):
        """Setup automated daily scheduling"""
        
        print(f"⏰ Setting up daily scan at {run_time}")
        
        # Schedule daily scan
        schedule.every().day.at(run_time).do(self.run_daily_scan)
        
        # Also schedule weekend discovery scans
        schedule.every().saturday.at("10:00").do(self.run_weekend_discovery_scan)
        
        print("✅ Scheduler configured!")
        return schedule
    
    def run_weekend_discovery_scan(self):
        """Weekend scan for broader symbol discovery"""
        print("🔍 WEEKEND DISCOVERY SCAN")
        
        # More aggressive discovery on weekends
        major_indices = self.scanner.get_major_index_symbols()
        all_discovery_symbols = []
        
        for category, symbols in major_indices.items():
            all_discovery_symbols.extend(symbols[:50])  # Larger sample on weekends
        
        # Remove duplicates
        unique_symbols = list(set(all_discovery_symbols))
        
        results = self.scanner.scan_multi_exchange(custom_symbols=unique_symbols)
        
        # Save weekend results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"weekend_discovery_scan_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'scan_date': datetime.now().isoformat(),
                'scan_type': 'weekend_discovery',
                'results': results
            }, f, indent=2)
        
        print(f"✅ Weekend discovery complete: {filename}")

def main():
    """Setup and run automated daily scanner"""
    
    print("🤖 AUTOMATED DAILY AEGS SCANNER SETUP")
    print("=" * 60)
    
    scanner = AutomatedDailyScanner()
    
    print("\n🎯 Options:")
    print("1. Run single daily scan now")
    print("2. Setup automated daily scheduling")
    print("3. Run weekend discovery scan")
    print("4. View recent scan history")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Run single scan
        results = scanner.run_daily_scan()
        
        # Auto-update goldmine if new discoveries
        if results['new_discoveries']:
            scanner.update_goldmine_with_discoveries(results['new_discoveries'])
    
    elif choice == "2":
        # Setup scheduling
        run_time = input("Enter daily scan time (HH:MM, default 09:30): ").strip()
        if not run_time:
            run_time = "09:30"
        
        schedule_obj = scanner.setup_scheduler(run_time)
        
        print(f"🚀 Daily scanner running... Press Ctrl+C to stop")
        print(f"Next scan scheduled for {run_time}")
        
        try:
            while True:
                schedule_obj.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n🛑 Automated scanner stopped")
    
    elif choice == "3":
        # Weekend discovery
        scanner.run_weekend_discovery_scan()
    
    elif choice == "4":
        # Show history
        print("\n📈 Recent Scan History:")
        for entry in scanner.results_history[-10:]:
            date = entry['date'][:10]
            count = entry['profitable_count']
            duration = entry['scan_duration']
            print(f"   {date}: {count} profitable ({duration:.1f}min)")
    
    print("\n✅ Automated daily scanner ready!")

if __name__ == "__main__":
    main()