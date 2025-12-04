-- Position Tracking Database Schema
-- For tracking all trades, current positions, and historical performance

-- Main positions table for all trades (open and closed)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL,
    entry_date DATETIME NOT NULL,
    entry_price DECIMAL(10, 4) NOT NULL,
    shares INTEGER NOT NULL,
    position_size DECIMAL(12, 2),
    
    -- Exit details (NULL if position still open)
    exit_date DATETIME,
    exit_price DECIMAL(10, 4),
    exit_value DECIMAL(12, 2),
    
    -- P&L calculations
    profit_loss DECIMAL(12, 2),
    profit_loss_pct DECIMAL(8, 4),
    
    -- Status and metadata
    status VARCHAR(10) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
    strategy TEXT,
    entry_reason TEXT,
    exit_reason TEXT,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes separately
CREATE INDEX idx_symbol ON positions(symbol);
CREATE INDEX idx_status ON positions(status);
CREATE INDEX idx_entry_date ON positions(entry_date);
CREATE INDEX idx_exit_date ON positions(exit_date);

-- Table for tracking current holdings (view of open positions)
CREATE VIEW current_holdings AS
SELECT 
    id,
    symbol,
    shares,
    entry_price,
    entry_date,
    position_size,
    JULIANDAY('now') - JULIANDAY(entry_date) as days_held,
    strategy
FROM positions
WHERE status = 'OPEN'
ORDER BY entry_date DESC;

-- Table for additional position metadata
CREATE TABLE IF NOT EXISTS position_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    note_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    note_type VARCHAR(20), -- 'analysis', 'alert', 'adjustment', etc
    note TEXT,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Performance summary view
CREATE VIEW performance_summary AS
SELECT 
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
    SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_positions,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(SUM(profit_loss), 2) as total_profit_loss,
    ROUND(AVG(profit_loss_pct), 2) as avg_return_pct,
    ROUND(MAX(profit_loss), 2) as best_trade,
    ROUND(MIN(profit_loss), 2) as worst_trade
FROM positions
GROUP BY symbol;

-- Monthly performance view
CREATE VIEW monthly_performance AS
SELECT 
    strftime('%Y-%m', exit_date) as month,
    COUNT(*) as trades_closed,
    SUM(profit_loss) as monthly_pnl,
    AVG(profit_loss_pct) as avg_return_pct,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses
FROM positions
WHERE status = 'CLOSED'
GROUP BY strftime('%Y-%m', exit_date)
ORDER BY month DESC;

-- Triggers for updating timestamps
CREATE TRIGGER update_position_timestamp 
AFTER UPDATE ON positions
BEGIN
    UPDATE positions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Insert TLRY closed position
INSERT INTO positions (
    symbol, entry_date, entry_price, shares,
    exit_date, exit_price, status, strategy,
    entry_reason, exit_reason
) VALUES (
    'TLRY', '2025-12-02', 7.18, 100,
    '2025-12-03', 7.70, 'CLOSED', 'Overnight swing trade',
    'AEGS signal', 'Profit target reached - 7.2% gain'
);