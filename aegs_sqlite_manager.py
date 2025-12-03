#!/usr/bin/env python3
"""
AEGS SQLite Database Manager
Solves all concurrency issues with proper database transactions
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager
from termcolor import colored
import threading

class AEGSDatabaseManager:
    def __init__(self, db_path='aegs_data.db'):
        self.db_path = db_path
        self.local = threading.local()
        self._init_database()
        
    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(
                self.db_path,
                isolation_level='IMMEDIATE',  # Prevents write conflicts
                timeout=30.0  # Wait up to 30 seconds for locks
            )
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    def _init_database(self):
        """Initialize database schema"""
        with open('aegs_database_schema.sql', 'r') as f:
            schema = f.read()
            
        with self.transaction() as conn:
            conn.executescript(schema)
    
    def add_invalid_symbol(self, symbol, reason, error_type='backtest_error'):
        """Add or update invalid symbol (thread-safe)"""
        with self.transaction() as conn:
            # Use INSERT OR REPLACE with proper handling
            conn.execute("""
                INSERT INTO invalid_symbols (symbol, reason, error_type, fail_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(symbol) DO UPDATE SET
                    fail_count = fail_count + 1,
                    last_failed = CURRENT_TIMESTAMP,
                    reason = excluded.reason
            """, (symbol, reason, error_type))
            
            # Check if it was promoted to permanent
            result = conn.execute(
                "SELECT is_permanent, fail_count FROM invalid_symbols WHERE symbol = ?",
                (symbol,)
            ).fetchone()
            
            if result and result['is_permanent']:
                print(colored(f"⚠️ {symbol} permanently excluded after {result['fail_count']} failures", 'yellow'))
    
    def is_symbol_excluded(self, symbol):
        """Check if symbol is invalid or permanently excluded"""
        conn = self._get_connection()
        result = conn.execute("""
            SELECT symbol FROM invalid_symbols 
            WHERE symbol = ? AND (is_permanent = TRUE OR fail_count > 0)
        """, (symbol,)).fetchone()
        return result is not None
    
    def get_excluded_symbols(self):
        """Get all symbols that should be skipped"""
        conn = self._get_connection()
        results = conn.execute("""
            SELECT symbol FROM invalid_symbols 
            WHERE is_permanent = TRUE OR fail_count >= 5
        """).fetchall()
        return {row['symbol'] for row in results}
    
    def add_discovery(self, symbol, discovery_type, metrics):
        """Add a new discovery"""
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO discoveries 
                (symbol, discovery_type, excess_return, strategy_return, win_rate)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol, 
                discovery_type,
                metrics.get('excess_return'),
                metrics.get('strategy_return'),
                metrics.get('win_rate')
            ))
    
    def cache_backtest_result(self, symbol, result, ttl_hours=24):
        """Cache backtest results"""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backtest_cache 
                (symbol, result_json, cached_at, expires_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """, (symbol, json.dumps(result), expires_at))
    
    def get_cached_backtest(self, symbol):
        """Get cached backtest if not expired"""
        conn = self._get_connection()
        result = conn.execute("""
            SELECT result_json FROM backtest_cache
            WHERE symbol = ? AND expires_at > CURRENT_TIMESTAMP
        """, (symbol,)).fetchone()
        
        if result:
            return json.loads(result['result_json'])
        return None
    
    def get_statistics(self):
        """Get database statistics"""
        conn = self._get_connection()
        stats = {}
        
        # Invalid symbols
        stats['total_invalid'] = conn.execute(
            "SELECT COUNT(*) as cnt FROM invalid_symbols"
        ).fetchone()['cnt']
        
        # Permanent exclusions
        stats['permanent_exclusions'] = conn.execute(
            "SELECT COUNT(*) as cnt FROM invalid_symbols WHERE is_permanent = TRUE"
        ).fetchone()['cnt']
        
        # Top failures
        stats['top_failures'] = conn.execute("""
            SELECT symbol, fail_count, reason 
            FROM invalid_symbols 
            ORDER BY fail_count DESC 
            LIMIT 10
        """).fetchall()
        
        # Recent discoveries
        stats['recent_discoveries'] = conn.execute("""
            SELECT symbol, discovery_type, excess_return 
            FROM discoveries 
            ORDER BY discovered_at DESC 
            LIMIT 10
        """).fetchall()
        
        return stats
    
    def migrate_from_json(self, json_files):
        """Migrate existing JSON data to database"""
        migrated = 0
        
        for json_file in json_files:
            if not os.path.exists(json_file):
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'invalid_symbols' in data:
                # Migrate invalid symbols
                with self.transaction() as conn:
                    for symbol, info in data['invalid_symbols'].items():
                        conn.execute("""
                            INSERT OR REPLACE INTO invalid_symbols
                            (symbol, reason, error_type, fail_count, first_failed, last_failed)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            symbol,
                            info.get('reason', 'Unknown'),
                            info.get('error_type', 'Unknown'),
                            info.get('fail_count', 1),
                            info.get('first_failed', datetime.now().isoformat()),
                            info.get('last_failed', datetime.now().isoformat())
                        ))
                        migrated += 1
                        
            if 'excluded_symbols' in data:
                # Migrate permanent exclusions
                with self.transaction() as conn:
                    for symbol, info in data['excluded_symbols'].items():
                        conn.execute("""
                            UPDATE invalid_symbols 
                            SET is_permanent = TRUE 
                            WHERE symbol = ?
                        """, (symbol,))
                        migrated += 1
        
        print(colored(f"✅ Migrated {migrated} records to database", 'green'))
        return migrated


def demonstrate_benefits():
    """Show how the database solves concurrency issues"""
    db = AEGSDatabaseManager()
    
    print(colored("\n🔒 DATABASE BENEFITS DEMONSTRATION", 'cyan', attrs=['bold']))
    print("="*60)
    
    # 1. No more concurrent modification errors
    print(colored("\n1. Concurrent Access (No Errors!):", 'yellow'))
    
    # Simulate multiple threads trying to update same symbol
    import concurrent.futures
    
    def update_symbol(thread_id, symbol):
        for i in range(10):
            db.add_invalid_symbol(symbol, f"Test from thread {thread_id}")
        return f"Thread {thread_id} completed"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(update_symbol, i, 'TEST') for i in range(5)]
        for future in concurrent.futures.as_completed(futures):
            print(f"  ✅ {future.result()}")
    
    # 2. Atomic operations
    print(colored("\n2. Atomic Fail Count:", 'yellow'))
    conn = db._get_connection()
    result = conn.execute(
        "SELECT fail_count FROM invalid_symbols WHERE symbol = 'TEST'"
    ).fetchone()
    print(f"  Final count after 50 concurrent updates: {result['fail_count']}")
    
    # 3. Performance
    print(colored("\n3. Query Performance:", 'yellow'))
    import time
    
    # Add many symbols
    start = time.time()
    for i in range(1000):
        db.add_invalid_symbol(f'PERF{i}', 'Performance test')
    print(f"  Added 1000 symbols in {time.time() - start:.2f}s")
    
    # Check exclusions
    start = time.time()
    excluded = db.get_excluded_symbols()
    print(f"  Retrieved {len(excluded)} exclusions in {time.time() - start:.4f}s")
    
    # 4. Statistics
    print(colored("\n4. Rich Statistics:", 'yellow'))
    stats = db.get_statistics()
    print(f"  Total invalid symbols: {stats['total_invalid']}")
    print(f"  Permanent exclusions: {stats['permanent_exclusions']}")
    print(f"  Top failure: {stats['top_failures'][0] if stats['top_failures'] else 'None'}")


if __name__ == "__main__":
    # Demonstrate the benefits
    demonstrate_benefits()
    
    # Offer migration
    print(colored("\n💡 MIGRATION AVAILABLE", 'cyan'))
    print("Run with --migrate to import existing JSON data:")
    print("  python aegs_sqlite_manager.py --migrate")
    
    import sys
    if '--migrate' in sys.argv:
        db = AEGSDatabaseManager()
        db.migrate_from_json([
            'aegs_invalid_symbols.json',
            'aegs_permanent_exclusions.json'
        ])
        
        stats = db.get_statistics()
        print(f"\n📊 Database now contains:")
        print(f"  {stats['total_invalid']} invalid symbols")
        print(f"  {stats['permanent_exclusions']} permanent exclusions")