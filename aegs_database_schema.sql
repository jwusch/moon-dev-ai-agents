-- SQLite schema for AEGS symbol tracking
-- This eliminates all concurrency issues

-- Invalid symbols with automatic fail count tracking
CREATE TABLE IF NOT EXISTS invalid_symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    reason TEXT NOT NULL,
    error_type VARCHAR(50) NOT NULL,
    fail_count INTEGER DEFAULT 1,
    first_failed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_failed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_permanent BOOLEAN DEFAULT FALSE
);

-- Permanent exclusions (symbols that should never be retried)
CREATE TABLE IF NOT EXISTS permanent_exclusions (
    symbol VARCHAR(20) PRIMARY KEY,
    reason TEXT NOT NULL,
    fail_count INTEGER NOT NULL,
    excluded_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES invalid_symbols(symbol)
);

-- Discovery results
CREATE TABLE IF NOT EXISTS discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    discovery_type VARCHAR(50) NOT NULL,
    excess_return DECIMAL(10,2),
    strategy_return DECIMAL(10,2),
    win_rate DECIMAL(5,2),
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes separately (SQLite syntax)
CREATE INDEX IF NOT EXISTS idx_symbol ON discoveries(symbol);
CREATE INDEX IF NOT EXISTS idx_discovery_time ON discoveries(discovered_at);

-- Backtest results cache
CREATE TABLE IF NOT EXISTS backtest_cache (
    symbol VARCHAR(20) PRIMARY KEY,
    result_json TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Atomic operations for fail count
CREATE TRIGGER IF NOT EXISTS update_fail_count
AFTER INSERT ON invalid_symbols
FOR EACH ROW
WHEN (SELECT COUNT(*) FROM invalid_symbols WHERE symbol = NEW.symbol) > 0
BEGIN
    UPDATE invalid_symbols 
    SET fail_count = fail_count + 1,
        last_failed = CURRENT_TIMESTAMP
    WHERE symbol = NEW.symbol;
END;

-- Auto-promote to permanent after threshold
CREATE TRIGGER IF NOT EXISTS auto_permanent_exclusion
AFTER UPDATE ON invalid_symbols
FOR EACH ROW
WHEN NEW.fail_count >= 10 AND NEW.is_permanent = FALSE
BEGIN
    UPDATE invalid_symbols SET is_permanent = TRUE WHERE symbol = NEW.symbol;
    INSERT OR IGNORE INTO permanent_exclusions (symbol, reason, fail_count)
    VALUES (NEW.symbol, NEW.reason, NEW.fail_count);
END;