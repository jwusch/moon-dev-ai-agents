"""
🌙 Moon Dev's Nice Functions for Robinhood
US stock/options/crypto trading utilities
Built with love by Moon Dev 🚀
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from termcolor import colored, cprint
import robin_stocks.robinhood as rs
from datetime import datetime, timedelta
import time
from src.agents.robinhood_adapter import RobinhoodAdapter

# Initialize adapter (will be done on first use)
_adapter = None

def _get_adapter() -> RobinhoodAdapter:
    """Get or create Robinhood adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = RobinhoodAdapter()
        # Authenticate using environment variables
        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')
        mfa_token = os.getenv('ROBINHOOD_MFA_TOKEN')
        
        if username and password:
            _adapter.authenticate(username, password, mfa_token)
        else:
            cprint("⚠️ Robinhood credentials not found in environment", "yellow")
    
    return _adapter

def market_buy(symbol: str, usd_amount: float, account=None) -> Dict:
    """
    Execute market buy order with proper position sizing
    
    Args:
        symbol: Stock/crypto symbol
        usd_amount: USD amount to buy
        account: Not used (for compatibility)
        
    Returns:
        Order result dictionary
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            cprint("❌ Not authenticated to Robinhood", "red")
            return {'success': False, 'error': 'Not authenticated'}
        
        # Check if market is open for stocks
        if not adapter._is_crypto(symbol) and not adapter.is_market_open():
            cprint(f"⚠️ Market is closed. Order will be queued for next open.", "yellow")
        
        # Get current price
        current_price = adapter.get_price(symbol)
        if not current_price:
            cprint(f"❌ Could not get price for {symbol}", "red")
            return {'success': False, 'error': 'Price unavailable'}
        
        # Calculate quantity
        if adapter._is_crypto(symbol):
            # For crypto, robin_stocks takes USD amount directly
            quantity = usd_amount
        else:
            # For stocks, calculate number of shares
            quantity = int(usd_amount / current_price)
            if quantity < 1:
                cprint(f"❌ USD amount ${usd_amount} too small for {symbol} @ ${current_price}", "red")
                return {'success': False, 'error': 'Insufficient funds for 1 share'}
        
        cprint(f"🎯 Buying {quantity} {'USD of' if adapter._is_crypto(symbol) else 'shares of'} {symbol} @ ${current_price:,.2f}", "cyan")
        
        # Place order
        result = adapter.place_market_order(symbol, quantity, 'buy')
        
        if result['success']:
            cprint(f"✅ Successfully bought {symbol}", "green")
            if not adapter._is_crypto(symbol):
                cprint(f"   Shares: {quantity}, Total: ${quantity * current_price:,.2f}", "cyan")
        else:
            cprint(f"❌ Failed to buy {symbol}: {result.get('error', 'Unknown error')}", "red")
        
        return result
        
    except Exception as e:
        cprint(f"❌ Market buy error: {str(e)}", "red")
        return {'success': False, 'error': str(e)}

def market_sell(symbol: str, percentage: float, account=None) -> Dict:
    """
    Execute market sell order
    
    Args:
        symbol: Stock/crypto symbol
        percentage: Percentage of position to sell (0-100)
        account: Not used (for compatibility)
        
    Returns:
        Order result dictionary
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            cprint("❌ Not authenticated to Robinhood", "red")
            return {'success': False, 'error': 'Not authenticated'}
        
        # Get current position
        position_info = get_position(symbol)
        if not position_info['has_position']:
            cprint(f"❌ No position in {symbol} to sell", "red")
            return {'success': False, 'error': 'No position'}
        
        # Calculate quantity to sell
        total_quantity = position_info['quantity']
        sell_quantity = total_quantity * (percentage / 100)
        
        if not adapter._is_crypto(symbol):
            # For stocks, round down to whole shares
            sell_quantity = int(sell_quantity)
            if sell_quantity < 1:
                cprint(f"❌ Sell percentage too small, would sell 0 shares", "red")
                return {'success': False, 'error': 'Quantity too small'}
        
        cprint(f"🎯 Selling {sell_quantity} {'of' if adapter._is_crypto(symbol) else 'shares of'} {symbol} ({percentage}% of position)", "cyan")
        
        # Place order
        result = adapter.place_market_order(symbol, sell_quantity, 'sell')
        
        if result['success']:
            cprint(f"✅ Successfully sold {percentage}% of {symbol}", "green")
        else:
            cprint(f"❌ Failed to sell {symbol}: {result.get('error', 'Unknown error')}", "red")
        
        return result
        
    except Exception as e:
        cprint(f"❌ Market sell error: {str(e)}", "red")
        return {'success': False, 'error': str(e)}

def get_position(symbol: str, account=None) -> Dict:
    """
    Get position details with normalized format
    
    Args:
        symbol: Stock/crypto symbol
        account: Not used (for compatibility)
        
    Returns:
        Position dictionary with standard format
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return {
                'has_position': False,
                'quantity': 0,
                'symbol': symbol,
                'avg_cost': 0,
                'current_price': 0,
                'market_value': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0
            }
        
        # Get all positions
        positions = adapter.get_positions()
        
        if positions.empty:
            return {
                'has_position': False,
                'quantity': 0,
                'symbol': symbol,
                'avg_cost': 0,
                'current_price': 0,
                'market_value': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0
            }
        
        # Find specific position
        position = positions[positions['symbol'] == symbol]
        
        if position.empty:
            return {
                'has_position': False,
                'quantity': 0,
                'symbol': symbol,
                'avg_cost': 0,
                'current_price': 0,
                'market_value': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0
            }
        
        # Get first (should be only) row
        pos = position.iloc[0]
        
        return {
            'has_position': True,
            'quantity': pos['quantity'],
            'symbol': symbol,
            'avg_cost': pos['avg_cost'],
            'current_price': pos['current_price'],
            'market_value': pos['market_value'],
            'unrealized_pnl': pos['unrealized_pnl'],
            'unrealized_pnl_pct': pos['unrealized_pnl_pct']
        }
        
    except Exception as e:
        print(f"❌ Error getting position: {e}")
        return {
            'has_position': False,
            'quantity': 0,
            'symbol': symbol,
            'avg_cost': 0,
            'current_price': 0,
            'market_value': 0,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0
        }

def get_account_value(account=None) -> float:
    """
    Get total account value including positions
    
    Args:
        account: Not used (for compatibility)
        
    Returns:
        Total account value in USD
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return 0.0
        
        account_info = adapter.get_account_info()
        return account_info.get('portfolio_value', 0.0)
        
    except Exception as e:
        print(f"❌ Error getting account value: {e}")
        return 0.0

def get_balance(account=None) -> float:
    """
    Get available buying power
    
    Args:
        account: Not used (for compatibility)
        
    Returns:
        Available buying power in USD
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return 0.0
        
        account_info = adapter.get_account_info()
        return account_info.get('buying_power', 0.0)
        
    except Exception as e:
        print(f"❌ Error getting balance: {e}")
        return 0.0

def get_all_positions(account=None) -> List[Dict]:
    """
    Get all positions in normalized format
    
    Args:
        account: Not used (for compatibility)
        
    Returns:
        List of position dictionaries
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return []
        
        positions_df = adapter.get_positions()
        
        if positions_df.empty:
            return []
        
        # Convert DataFrame to list of dicts
        positions = []
        for _, row in positions_df.iterrows():
            positions.append({
                'symbol': row['symbol'],
                'quantity': row['quantity'],
                'avg_cost': row['avg_cost'],
                'current_price': row['current_price'],
                'market_value': row['market_value'],
                'unrealized_pnl': row['unrealized_pnl'],
                'unrealized_pnl_pct': row['unrealized_pnl_pct'],
                'type': row['type']  # 'stock' or 'crypto'
            })
        
        return positions
        
    except Exception as e:
        print(f"❌ Error getting all positions: {e}")
        return []

def get_watchlist_data(watchlist_name: str = "Default") -> pd.DataFrame:
    """
    Get price data for watchlist symbols
    
    Args:
        watchlist_name: Name of Robinhood watchlist
        
    Returns:
        DataFrame with symbol price data
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return pd.DataFrame()
        
        # Get watchlist symbols
        symbols = adapter.get_watchlist(watchlist_name)
        
        if not symbols:
            cprint(f"⚠️ No symbols in watchlist '{watchlist_name}'", "yellow")
            return pd.DataFrame()
        
        # Get price data for each symbol
        data = []
        for symbol in symbols:
            price = adapter.get_price(symbol)
            if price:
                # Get additional data
                stats = adapter.get_24h_stats(symbol)
                
                data.append({
                    'symbol': symbol,
                    'price': price,
                    'change_24h': stats.get('change_24h', 0),
                    'volume_24h': stats.get('volume_24h', 0),
                    'high_24h': stats.get('high_24h', price),
                    'low_24h': stats.get('low_24h', price)
                })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"❌ Error getting watchlist data: {e}")
        return pd.DataFrame()

def place_trailing_stop(symbol: str, trail_percentage: float) -> Dict:
    """
    Place trailing stop order
    
    Args:
        symbol: Stock symbol (crypto not supported)
        trail_percentage: Trailing stop percentage
        
    Returns:
        Order result dictionary
    """
    try:
        adapter = _get_adapter()
        
        if not adapter.authenticated:
            return {'success': False, 'error': 'Not authenticated'}
        
        if adapter._is_crypto(symbol):
            cprint(f"❌ Trailing stops not supported for crypto", "red")
            return {'success': False, 'error': 'Not supported for crypto'}
        
        # Get current position
        position_info = get_position(symbol)
        if not position_info['has_position']:
            cprint(f"❌ No position in {symbol} for trailing stop", "red")
            return {'success': False, 'error': 'No position'}
        
        quantity = position_info['quantity']
        
        # Place trailing stop order
        order = rs.orders.order_sell_trailing_stop(
            symbol, quantity, trail_percentage
        )
        
        if order:
            cprint(f"✅ Trailing stop placed for {symbol} with {trail_percentage}% trail", "green")
            return {
                'success': True,
                'order_id': order.get('id'),
                'symbol': symbol,
                'quantity': quantity,
                'trail_percentage': trail_percentage,
                'raw_response': order
            }
        else:
            return {'success': False, 'error': 'Order failed'}
            
    except Exception as e:
        cprint(f"❌ Trailing stop error: {str(e)}", "red")
        return {'success': False, 'error': str(e)}

def get_market_hours() -> Dict:
    """Get current market status and hours"""
    try:
        adapter = _get_adapter()
        return adapter.get_market_hours()
    except Exception as e:
        print(f"❌ Error getting market hours: {e}")
        return {}

def is_market_open() -> bool:
    """Check if market is open for trading"""
    try:
        adapter = _get_adapter()
        return adapter.is_market_open()
    except Exception as e:
        print(f"❌ Error checking market status: {e}")
        return False

def kill_switch(symbol: str, account=None) -> Dict:
    """
    Emergency close position (alias for 100% market sell)
    
    Args:
        symbol: Stock/crypto symbol
        account: Not used (for compatibility)
        
    Returns:
        Order result dictionary
    """
    cprint(f"🚨 KILL SWITCH activated for {symbol}", "red")
    return market_sell(symbol, 100, account)

def chunk_kill(symbol: str, max_order_size: float = 1000, slippage: float = 50) -> Dict:
    """
    Close position in chunks (for large positions)
    Note: Robinhood typically handles large orders well, so this just sells 100%
    
    Args:
        symbol: Stock/crypto symbol
        max_order_size: Not used (for compatibility)
        slippage: Not used (for compatibility)
        
    Returns:
        Order result dictionary
    """
    cprint(f"📦 Chunk kill for {symbol} (Robinhood handles sizing automatically)", "cyan")
    return market_sell(symbol, 100)

def get_ohlcv_data(symbol: str, timeframe: str = '1H', days_back: int = 3) -> pd.DataFrame:
    """
    Get OHLCV data for analysis
    
    Args:
        symbol: Stock/crypto symbol
        timeframe: Time interval (5m, 1h, 1d, etc.)
        days_back: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        adapter = _get_adapter()
        
        # Calculate limit based on timeframe and days
        if timeframe in ['5m', '5minute']:
            limit = days_back * 24 * 12  # 12 bars per hour
        elif timeframe in ['1h', '1H', 'hour']:
            limit = days_back * 24
        elif timeframe in ['1d', '1D', 'day']:
            limit = days_back
        else:
            limit = 100
        
        return adapter.get_ohlcv_data(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"❌ Error getting OHLCV data: {e}")
        return pd.DataFrame()

def set_leverage(symbol: str, leverage: int) -> Dict:
    """
    Set leverage (not applicable for Robinhood - stocks are 1x only)
    
    Args:
        symbol: Stock symbol
        leverage: Requested leverage
        
    Returns:
        Status dictionary
    """
    cprint(f"ℹ️ Robinhood does not support leverage trading (stocks are 1x only)", "yellow")
    return {
        'success': True,
        'message': 'Leverage not supported on Robinhood',
        'actual_leverage': 1
    }

def get_data(symbol: str, days_back: int, timeframe: str) -> pd.DataFrame:
    """
    Get data (alias for get_ohlcv_data for compatibility)
    """
    return get_ohlcv_data(symbol, timeframe, days_back)

# Test functions
if __name__ == "__main__":
    print("\n🧪 Testing Robinhood Nice Functions")
    print("=" * 50)
    
    # Test market status
    market_open = is_market_open()
    print(f"Market Open: {market_open}")
    
    # Test balance
    balance = get_balance()
    print(f"Buying Power: ${balance:,.2f}")
    
    # Test account value
    account_value = get_account_value()
    print(f"Account Value: ${account_value:,.2f}")
    
    # Test positions
    positions = get_all_positions()
    print(f"Open Positions: {len(positions)}")
    
    if positions:
        for pos in positions:
            print(f"  - {pos['symbol']}: {pos['quantity']} @ ${pos['current_price']:,.2f} (P&L: {pos['unrealized_pnl_pct']:.2f}%)")