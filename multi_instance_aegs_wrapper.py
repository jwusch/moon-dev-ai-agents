#!/usr/bin/env python3
"""
🔒 MULTI-INSTANCE AEGS WRAPPER 🔒
Safe wrapper for running AEGS backtesting with multiple Claude Code instances
Implements file locking, instance separation, and collision detection
"""

import os
import json
import fcntl
import time
import hashlib
from datetime import datetime
from pathlib import Path
import tempfile
import psutil
from aegs_working_symbols_comprehensive import AEGSWorkingSymbolsBacktester
from yfinance_proxy_wrapper import YFinanceProxyWrapper

class MultiInstanceAEGSWrapper:
    """Thread-safe wrapper for AEGS backtesting across multiple instances"""
    
    def __init__(self, instance_name=None):
        self.instance_name = instance_name or self._generate_instance_name()
        self.base_dir = Path("/mnt/c/Users/jwusc/moon-dev-ai-agents")
        self.lock_dir = self.base_dir / "aegs_locks"
        self.instance_dir = self.base_dir / f"aegs_instances/{self.instance_name}"
        self.shared_cache_dir = self.base_dir / "working_symbols_cache"
        
        # Create directories
        self.lock_dir.mkdir(exist_ok=True)
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔒 MULTI-INSTANCE AEGS WRAPPER")
        print(f"   Instance: {self.instance_name}")
        print(f"   Instance Dir: {self.instance_dir}")
        print(f"   Lock Dir: {self.lock_dir}")
        print(f"   Shared Cache: {self.shared_cache_dir}")
        
        # Track active instances
        self.instance_registry = self.base_dir / "aegs_active_instances.json"
        self._register_instance()
    
    def _generate_instance_name(self):
        """Generate unique instance name based on PID and timestamp"""
        pid = os.getpid()
        timestamp = datetime.now().strftime('%H%M%S')
        return f"claude_{pid}_{timestamp}"
    
    def _register_instance(self):
        """Register this instance in the active registry"""
        try:
            # Load existing registry
            if self.instance_registry.exists():
                with open(self.instance_registry, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"instances": {}}
            
            # Add this instance
            registry["instances"][self.instance_name] = {
                "pid": os.getpid(),
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "instance_dir": str(self.instance_dir)
            }
            
            # Save registry
            with open(self.instance_registry, 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"📝 Registered instance: {self.instance_name}")
            
        except Exception as e:
            print(f"⚠️  Failed to register instance: {e}")
    
    def _cleanup_stale_instances(self):
        """Clean up instances from dead processes"""
        try:
            if not self.instance_registry.exists():
                return
            
            with open(self.instance_registry, 'r') as f:
                registry = json.load(f)
            
            active_pids = [p.pid for p in psutil.process_iter()]
            cleaned = 0
            
            for instance_name, info in list(registry["instances"].items()):
                if info["pid"] not in active_pids:
                    print(f"🧹 Cleaning stale instance: {instance_name} (PID {info['pid']})")
                    del registry["instances"][instance_name]
                    cleaned += 1
            
            if cleaned > 0:
                with open(self.instance_registry, 'w') as f:
                    json.dump(registry, f, indent=2)
                print(f"✅ Cleaned {cleaned} stale instances")
            
        except Exception as e:
            print(f"⚠️  Failed to cleanup stale instances: {e}")
    
    def get_active_instances(self):
        """Get list of currently active instances"""
        self._cleanup_stale_instances()
        
        try:
            if self.instance_registry.exists():
                with open(self.instance_registry, 'r') as f:
                    registry = json.load(f)
                return list(registry["instances"].keys())
            return [self.instance_name]
        except:
            return [self.instance_name]
    
    def acquire_symbol_lock(self, symbol, timeout=300):
        """Acquire exclusive lock for a symbol"""
        lock_file = self.lock_dir / f"{symbol}.lock"
        
        try:
            # Create lock file
            lock_fd = open(lock_file, 'w')
            
            # Try to acquire exclusive lock with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Write lock info
                    lock_fd.write(json.dumps({
                        "instance": self.instance_name,
                        "pid": os.getpid(),
                        "symbol": symbol,
                        "locked_at": datetime.now().isoformat()
                    }))
                    lock_fd.flush()
                    
                    print(f"🔒 Acquired lock for {symbol}")
                    return lock_fd
                    
                except BlockingIOError:
                    # Lock is held by another process
                    time.sleep(1)
                    continue
            
            print(f"⏰ Timeout acquiring lock for {symbol}")
            lock_fd.close()
            return None
            
        except Exception as e:
            print(f"❌ Lock error for {symbol}: {e}")
            return None
    
    def release_symbol_lock(self, lock_fd, symbol):
        """Release symbol lock"""
        try:
            if lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
                
                # Remove lock file
                lock_file = self.lock_dir / f"{symbol}.lock"
                if lock_file.exists():
                    lock_file.unlink()
                
                print(f"🔓 Released lock for {symbol}")
                
        except Exception as e:
            print(f"⚠️  Error releasing lock for {symbol}: {e}")
    
    def get_instance_cache_dir(self, timeframe):
        """Get instance-specific cache directory for timeframe"""
        cache_dir = self.instance_dir / "cache" / timeframe
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_shared_cache_file(self, symbol, timeframe, days_back):
        """Get shared cache file path"""
        return self.shared_cache_dir / timeframe / f"{symbol}_{timeframe}_{days_back}d.pkl"
    
    def safe_backtest_symbol(self, symbol, strategies):
        """Safely backtest a symbol with multi-instance protection"""
        
        # Try to acquire lock for this symbol
        lock_fd = self.acquire_symbol_lock(symbol)
        if not lock_fd:
            print(f"⏭️  Skipping {symbol} - locked by another instance")
            return {}
        
        try:
            print(f"\n🔒 [{self.instance_name}] Processing {symbol}...")
            
            # Create instance-specific backtester
            backtester = AEGSWorkingSymbolsBacktester()
            
            # Override cache directories to use shared cache for reads, instance cache for writes
            original_cache_dir = backtester.cache_dir
            
            symbol_results = {}
            
            # Fetch data for all timeframes (using shared cache)
            data_cache = {}
            for timeframe in ["1m", "15m", "1h"]:
                print(f"   📊 Fetching {timeframe} data...")
                data_cache[timeframe] = backtester.fetch_timeframe_data(symbol, timeframe)
            
            # Test each strategy
            for strategy_name, params in strategies.items():
                timeframe = strategy_name.split('_')[0]
                df = data_cache.get(timeframe)
                
                if df is None or df.empty:
                    symbol_results[strategy_name] = {'status': 'no_data', 'trades': 0}
                    continue
                
                # Run backtest
                trades = backtester.backtest_strategy(symbol, df.copy(), strategy_name, params)
                
                if trades:
                    total_return = sum([t['return_pct'] for t in trades])
                    win_rate = len([t for t in trades if t['return_pct'] > 0]) / len(trades) * 100
                    
                    symbol_results[strategy_name] = {
                        'status': 'success',
                        'trades': len(trades),
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'avg_return': total_return / len(trades)
                    }
                    
                    print(f"    🎯 {strategy_name}: {len(trades)} trades, {total_return:+.1f}%, {win_rate:.0f}% wins")
                else:
                    symbol_results[strategy_name] = {'status': 'no_trades', 'trades': 0}
            
            return symbol_results
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            return {}
        
        finally:
            # Always release the lock
            self.release_symbol_lock(lock_fd, symbol)
    
    def run_distributed_backtest(self, symbols=None, strategies=None, max_symbols_per_instance=20):
        """Run backtest distributed across instances"""
        
        print(f"\n🚀 DISTRIBUTED AEGS BACKTEST - INSTANCE: {self.instance_name}")
        print("=" * 70)
        
        # Use default symbols if none provided
        if symbols is None:
            backtester = AEGSWorkingSymbolsBacktester()
            symbols = backtester.working_symbols
        
        if strategies is None:
            backtester = AEGSWorkingSymbolsBacktester()
            strategies = backtester.strategies
        
        # Check active instances
        active_instances = self.get_active_instances()
        print(f"🔍 Active instances: {len(active_instances)} ({', '.join(active_instances)})")
        
        # Distribute symbols among instances
        my_index = active_instances.index(self.instance_name) if self.instance_name in active_instances else 0
        my_symbols = symbols[my_index::len(active_instances)][:max_symbols_per_instance]
        
        print(f"📊 My symbols ({len(my_symbols)}): {', '.join(my_symbols[:10])}{'...' if len(my_symbols) > 10 else ''}")
        
        # Process symbols safely
        all_results = {}
        start_time = time.time()
        
        for i, symbol in enumerate(my_symbols, 1):
            print(f"\n[{i:2}/{len(my_symbols)}] Processing {symbol}...")
            
            symbol_results = self.safe_backtest_symbol(symbol, strategies)
            all_results[symbol] = symbol_results
            
            # Rate limiting
            if i < len(my_symbols):
                time.sleep(1)
        
        # Save instance results
        self.save_instance_results(all_results, start_time)
        
        return all_results
    
    def save_instance_results(self, results, start_time):
        """Save results for this instance"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate summary stats
        all_trades = []
        for symbol_data in results.values():
            for strategy_data in symbol_data.values():
                if isinstance(strategy_data, dict) and strategy_data.get('trades', 0) > 0:
                    # Reconstruct trade data for summary
                    trades = strategy_data.get('trades', 0)
                    total_return = strategy_data.get('total_return', 0)
                    for _ in range(trades):
                        all_trades.append({'return_pct': total_return / trades})
        
        summary = {
            'instance': self.instance_name,
            'timestamp': timestamp,
            'runtime_minutes': (time.time() - start_time) / 60,
            'symbols_processed': len(results),
            'total_trades': len(all_trades),
            'total_return': sum([t['return_pct'] for t in all_trades]) if all_trades else 0,
            'win_rate': len([t for t in all_trades if t['return_pct'] > 0]) / len(all_trades) * 100 if all_trades else 0
        }
        
        # Save to instance directory
        instance_results = {
            'summary': summary,
            'results': results,
            'processed_at': datetime.now().isoformat()
        }
        
        results_file = self.instance_dir / f"aegs_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(instance_results, f, indent=2, default=str)
        
        print(f"\n💾 Instance results saved: {results_file}")
        print(f"   📊 {summary['symbols_processed']} symbols, {summary['total_trades']} trades")
        print(f"   💰 Total return: {summary['total_return']:+.1f}%")
        print(f"   🏆 Win rate: {summary['win_rate']:.1f}%")
    
    def aggregate_all_results(self):
        """Aggregate results from all instances"""
        
        print(f"\n📊 AGGREGATING RESULTS FROM ALL INSTANCES")
        print("=" * 50)
        
        all_instance_results = []
        
        # Find all instance directories
        instances_base = self.base_dir / "aegs_instances"
        if instances_base.exists():
            for instance_dir in instances_base.iterdir():
                if instance_dir.is_dir():
                    # Find latest results file
                    result_files = list(instance_dir.glob("aegs_results_*.json"))
                    if result_files:
                        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            with open(latest_file, 'r') as f:
                                instance_data = json.load(f)
                                all_instance_results.append(instance_data)
                                print(f"   ✅ {instance_dir.name}: {instance_data['summary']['symbols_processed']} symbols")
                        except Exception as e:
                            print(f"   ❌ {instance_dir.name}: Error loading results - {e}")
        
        if not all_instance_results:
            print("❌ No instance results found!")
            return None
        
        # Aggregate statistics
        total_symbols = sum([r['summary']['symbols_processed'] for r in all_instance_results])
        total_trades = sum([r['summary']['total_trades'] for r in all_instance_results])
        total_return = sum([r['summary']['total_return'] for r in all_instance_results])
        
        # Weighted average win rate
        total_winning_trades = 0
        for r in all_instance_results:
            instance_trades = r['summary']['total_trades']
            instance_win_rate = r['summary']['win_rate']
            total_winning_trades += (instance_trades * instance_win_rate / 100)
        
        overall_win_rate = total_winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Create aggregated summary
        aggregated_summary = {
            'aggregated_at': datetime.now().isoformat(),
            'instances_count': len(all_instance_results),
            'total_symbols': total_symbols,
            'total_trades': total_trades,
            'total_return': total_return,
            'overall_win_rate': overall_win_rate,
            'instances': [r['summary'] for r in all_instance_results]
        }
        
        # Save aggregated results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        agg_file = self.base_dir / f"aegs_aggregated_results_{timestamp}.json"
        with open(agg_file, 'w') as f:
            json.dump(aggregated_summary, f, indent=2, default=str)
        
        print(f"\n🎯 AGGREGATED MULTI-INSTANCE RESULTS:")
        print(f"   📊 Instances: {len(all_instance_results)}")
        print(f"   📈 Symbols: {total_symbols}")
        print(f"   🎯 Trades: {total_trades}")
        print(f"   💰 Total Return: {total_return:+.1f}%")
        print(f"   🏆 Win Rate: {overall_win_rate:.1f}%")
        print(f"   💾 Saved: {agg_file}")
        
        return aggregated_summary
    
    def cleanup_instance(self):
        """Clean up this instance"""
        try:
            # Remove from registry
            if self.instance_registry.exists():
                with open(self.instance_registry, 'r') as f:
                    registry = json.load(f)
                
                if self.instance_name in registry["instances"]:
                    del registry["instances"][self.instance_name]
                    
                    with open(self.instance_registry, 'w') as f:
                        json.dump(registry, f, indent=2)
                    
                    print(f"🧹 Cleaned up instance: {self.instance_name}")
            
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

def main():
    """Demo multi-instance AEGS wrapper"""
    
    print("🔒🚀 MULTI-INSTANCE AEGS BACKTEST DEMO 🚀🔒")
    print("=" * 80)
    print("🎯 Safe multi-instance AEGS backtesting with file locking")
    print("⚡ Automatic symbol distribution and collision detection")
    print("💎 Shared cache with instance-specific results")
    
    # Create wrapper
    wrapper = MultiInstanceAEGSWrapper()
    
    try:
        # Run distributed backtest
        results = wrapper.run_distributed_backtest(
            max_symbols_per_instance=10  # Limit for demo
        )
        
        # Wait a moment for other instances to finish
        print(f"\n⏳ Waiting 10s for other instances to complete...")
        time.sleep(10)
        
        # Aggregate all results
        if results:
            aggregated = wrapper.aggregate_all_results()
            
            if aggregated:
                print(f"\n✅ MULTI-INSTANCE BACKTEST COMPLETE!")
                print(f"   🎯 {aggregated['instances_count']} instances processed {aggregated['total_symbols']} symbols")
                print(f"   📊 {aggregated['total_trades']} total trades across all instances")
    
    finally:
        # Clean up
        wrapper.cleanup_instance()

if __name__ == "__main__":
    main()