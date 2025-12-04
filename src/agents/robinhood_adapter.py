"""
🏦 Robinhood Adapter for Moon Dev Trading System
Unofficial API integration for US stocks, options, and crypto
Built with love by Moon Dev 🌙

⚠️ DISCLAIMER: This uses an unofficial API. Use at your own risk.
"""

import os
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime, timedelta
import time
import robin_stocks.robinhood as rs
import pyotp
from termcolor import colored, cprint

class RobinhoodAdapter:
    """
    Robinhood adapter for US stock/options/crypto trading
    Uses robin_stocks library for unofficial API access
    """
    
    def __init__(self):
        self.name = "Robinhood"
        self.authenticated = False
        self.account_info = None
        self.rate_limit_delay = 0.5  # 500ms between requests
        self.last_request_time = 0
        
        cprint(f"🏦 {self.name} Adapter initialized", "cyan")
        cprint("⚠️  Using unofficial API - subject to change", "yellow")
        
    def _rate_limit(self):
        """Enforce rate limiting to avoid API blocks"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
            
        self.last_request_time = time.time()
    
    def authenticate(self, username: str, password: str, mfa_token: Optional[str] = None) -> bool:
        """
        Handle Robinhood login with MFA support
        
        Args:
            username: Robinhood username/email
            password: Account password
            mfa_token: Optional TOTP secret for MFA
            
        Returns:
            bool: Success status
        """
        try:
            if mfa_token:
                # Use TOTP for MFA
                totp = pyotp.TOTP(mfa_token).now()
                login_result = rs.login(username, password, mfa_code=totp)
            else:
                # Try regular login (will prompt for SMS code if needed)
                login_result = rs.login(username, password)
            
            if login_result:
                self.authenticated = True
                self.account_info = self._get_account_info()
                cprint(f"✅ Successfully authenticated to Robinhood", "green")
                cprint(f"   Account: {self.account_info.get('account_number', 'Unknown')}", "cyan")
                return True
            else:
                cprint(f"❌ Failed to authenticate to Robinhood", "red")
                return False
                
        except Exception as e:
            cprint(f"❌ Authentication error: {str(e)}", "red")
            return False
    
    def _get_account_info(self) -> Dict:
        """Get basic account information"""
        try:
            profile = rs.profiles.load_account_profile()
            return {
                'account_number': profile.get('account_number', ''),
                'buying_power': float(profile.get('buying_power', 0)),
                'cash': float(profile.get('cash', 0)),
                'type': profile.get('type', 'Unknown')
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a stock/crypto"""
        try:
            self._rate_limit()
            
            # Check if it's crypto
            if self._is_crypto(symbol):
                # Get crypto quote
                quote = rs.crypto.get_crypto_quote(symbol)
                if quote:
                    return float(quote['mark_price'])
            else:
                # Get stock quote
                quote = rs.stocks.get_latest_price(symbol)
                if quote and quote[0]:
                    return float(quote[0])
                    
            return None
            
        except Exception as e:
            print(f"❌ Price error for {symbol}: {e}")
            return None
    
    def get_ohlcv_data(self, symbol: str, interval: str = '1h', 
                      limit: int = 100) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock/crypto symbol
            interval: Time interval (5minute, 10minute, hour, day, week)
            limit: Number of data points
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit()
            
            # Convert interval to Robinhood format
            rh_interval = self._convert_interval(interval)
            
            # Calculate span based on interval and limit
            span = self._calculate_span(rh_interval, limit)
            
            if self._is_crypto(symbol):
                # Get crypto historicals
                historicals = rs.crypto.get_crypto_historicals(
                    symbol, interval=rh_interval, span=span
                )
            else:
                # Get stock historicals
                historicals = rs.stocks.get_stock_historicals(
                    symbol, interval=rh_interval, span=span
                )
            
            if not historicals:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(historicals)
            
            # Rename columns to match expected format
            column_map = {
                'begins_at': 'timestamp',
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Convert string prices to float
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
                    
            # Convert volume to int
            if 'volume' in df.columns:
                df['volume'] = df['volume'].astype(float).astype(int)
            
            # Select only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Limit to requested number of rows
            df = df.tail(limit)
            
            return df
            
        except Exception as e:
            print(f"❌ OHLCV error for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_market_order(self, symbol: str, quantity: float, 
                          side: str = 'buy') -> Dict:
        """
        Place market order
        
        Args:
            symbol: Stock/crypto symbol
            quantity: Number of shares/amount
            side: 'buy' or 'sell'
            
        Returns:
            Order result dictionary
        """
        try:
            self._rate_limit()
            
            if not self.authenticated:
                raise Exception("Not authenticated")
            
            if self._is_crypto(symbol):
                # Crypto order
                if side == 'buy':
                    # For crypto, quantity is in USD
                    order = rs.orders.order_buy_crypto_by_price(
                        symbol, quantity
                    )
                else:
                    # For crypto sell, quantity is amount of crypto
                    order = rs.orders.order_sell_crypto_by_quantity(
                        symbol, quantity
                    )
            else:
                # Stock order
                if side == 'buy':
                    order = rs.orders.order_buy_market(
                        symbol, quantity
                    )
                else:
                    order = rs.orders.order_sell_market(
                        symbol, quantity
                    )
            
            if order:
                cprint(f"✅ {side.upper()} order placed for {symbol}", "green")
                return {
                    'success': True,
                    'order_id': order.get('id'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'type': 'market',
                    'raw_response': order
                }
            else:
                return {
                    'success': False,
                    'error': 'Order failed'
                }
                
        except Exception as e:
            cprint(f"❌ Market order error: {str(e)}", "red")
            return {
                'success': False,
                'error': str(e)
            }
    
    def place_limit_order(self, symbol: str, quantity: float, 
                         limit_price: float, side: str = 'buy') -> Dict:
        """
        Place limit order
        
        Args:
            symbol: Stock/crypto symbol
            quantity: Number of shares/amount
            limit_price: Limit price
            side: 'buy' or 'sell'
            
        Returns:
            Order result dictionary
        """
        try:
            self._rate_limit()
            
            if not self.authenticated:
                raise Exception("Not authenticated")
            
            if self._is_crypto(symbol):
                # Crypto doesn't support limit orders in robin_stocks
                cprint(f"⚠️ Crypto limit orders not supported, using market order", "yellow")
                return self.place_market_order(symbol, quantity, side)
            else:
                # Stock limit order
                if side == 'buy':
                    order = rs.orders.order_buy_limit(
                        symbol, quantity, limit_price
                    )
                else:
                    order = rs.orders.order_sell_limit(
                        symbol, quantity, limit_price
                    )
            
            if order:
                cprint(f"✅ LIMIT {side.upper()} order placed for {symbol} @ ${limit_price}", "green")
                return {
                    'success': True,
                    'order_id': order.get('id'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'limit_price': limit_price,
                    'type': 'limit',
                    'raw_response': order
                }
            else:
                return {
                    'success': False,
                    'error': 'Order failed'
                }
                
        except Exception as e:
            cprint(f"❌ Limit order error: {str(e)}", "red")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        try:
            self._rate_limit()
            
            if not self.authenticated:
                return pd.DataFrame()
            
            positions = []
            
            # Get stock positions
            stock_positions = rs.account.get_open_stock_positions()
            if stock_positions:
                for pos in stock_positions:
                    # Get symbol from instrument URL
                    instrument = rs.stocks.get_instrument_by_url(pos['instrument'])
                    symbol = instrument['symbol']
                    
                    positions.append({
                        'symbol': symbol,
                        'quantity': float(pos['quantity']),
                        'avg_cost': float(pos['average_buy_price']),
                        'current_price': self.get_price(symbol),
                        'type': 'stock',
                        'created_at': pos['created_at']
                    })
            
            # Get crypto positions
            crypto_positions = rs.crypto.get_crypto_positions()
            if crypto_positions:
                for pos in crypto_positions:
                    if float(pos['quantity']) > 0:
                        symbol = pos['currency']['code']
                        positions.append({
                            'symbol': symbol,
                            'quantity': float(pos['quantity']),
                            'avg_cost': float(pos['cost_bases'][0]['direct_cost_basis']) / float(pos['quantity']) if float(pos['quantity']) > 0 else 0,
                            'current_price': self.get_price(symbol),
                            'type': 'crypto',
                            'created_at': pos['created_at']
                        })
            
            df = pd.DataFrame(positions)
            
            # Calculate P&L if we have positions
            if not df.empty:
                df['market_value'] = df['quantity'] * df['current_price']
                df['cost_basis'] = df['quantity'] * df['avg_cost']
                df['unrealized_pnl'] = df['market_value'] - df['cost_basis']
                df['unrealized_pnl_pct'] = (df['unrealized_pnl'] / df['cost_basis'] * 100).round(2)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return pd.DataFrame()
    
    def get_account_info(self) -> Dict:
        """Get account balance and buying power"""
        try:
            self._rate_limit()
            
            if not self.authenticated:
                return {}
            
            # Get account profile
            profile = rs.profiles.load_account_profile()
            
            # Get portfolio data
            portfolio = rs.profiles.load_portfolio_profile()
            
            return {
                'account_number': profile.get('account_number', ''),
                'buying_power': float(profile.get('buying_power', 0)),
                'cash': float(profile.get('cash', 0)),
                'portfolio_value': float(portfolio.get('market_value', 0)),
                'total_return': float(portfolio.get('total_return_today', 0)),
                'total_return_pct': float(portfolio.get('total_return_today', 0)) / float(portfolio.get('market_value', 1)) * 100 if float(portfolio.get('market_value', 0)) > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return {}
    
    def get_options_chain(self, symbol: str, expiration_date: str) -> pd.DataFrame:
        """
        Get options chain data
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date (YYYY-MM-DD)
            
        Returns:
            DataFrame with options chain
        """
        try:
            self._rate_limit()
            
            if not self.authenticated:
                return pd.DataFrame()
            
            # Get options chain
            options = rs.options.find_options_by_expiration(
                symbol, expiration_date
            )
            
            if not options:
                return pd.DataFrame()
            
            # Process options data
            chain_data = []
            for opt in options:
                chain_data.append({
                    'symbol': symbol,
                    'type': opt['type'],
                    'strike': float(opt['strike_price']),
                    'expiration': opt['expiration_date'],
                    'bid': float(opt.get('bid_price', 0)),
                    'ask': float(opt.get('ask_price', 0)),
                    'last': float(opt.get('last_trade_price', 0)),
                    'volume': int(opt.get('volume', 0)),
                    'open_interest': int(opt.get('open_interest', 0)),
                    'implied_volatility': float(opt.get('implied_volatility', 0))
                })
            
            return pd.DataFrame(chain_data)
            
        except Exception as e:
            print(f"❌ Error getting options chain: {e}")
            return pd.DataFrame()
    
    def place_options_order(self, symbol: str, contract_details: Dict, 
                           quantity: int, order_type: str = 'market') -> Dict:
        """
        Place options order
        
        Args:
            symbol: Stock symbol
            contract_details: Dict with strike, expiration, type (call/put)
            quantity: Number of contracts
            order_type: 'market' or 'limit'
            
        Returns:
            Order result dictionary
        """
        try:
            self._rate_limit()
            
            if not self.authenticated:
                raise Exception("Not authenticated")
            
            # This is a placeholder - implement based on robin_stocks options trading
            cprint(f"⚠️ Options trading implementation pending", "yellow")
            
            return {
                'success': False,
                'error': 'Options trading not yet implemented'
            }
            
        except Exception as e:
            cprint(f"❌ Options order error: {str(e)}", "red")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_watchlist(self, name: str = "Default") -> List[str]:
        """Get symbols from a watchlist"""
        try:
            self._rate_limit()
            
            if not self.authenticated:
                return []
            
            # Get watchlist by name
            watchlists = rs.account.get_watchlist_by_name(name)
            
            if watchlists and 'results' in watchlists:
                symbols = []
                for item in watchlists['results']:
                    instrument = rs.stocks.get_instrument_by_url(item['instrument'])
                    if instrument:
                        symbols.append(instrument['symbol'])
                return symbols
            
            return []
            
        except Exception as e:
            print(f"❌ Error getting watchlist: {e}")
            return []
    
    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency"""
        # Common crypto symbols on Robinhood
        crypto_symbols = ['BTC', 'ETH', 'DOGE', 'SHIB', 'LTC', 'BCH', 'ETC', 
                         'BSV', 'COMP', 'LINK', 'UNI', 'AAVE', 'MATIC', 'SOL']
        return symbol.upper() in crypto_symbols
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to Robinhood format"""
        interval_map = {
            '5m': '5minute',
            '10m': '10minute',
            '1h': 'hour',
            '1d': 'day',
            '1w': 'week'
        }
        return interval_map.get(interval, 'hour')
    
    def _calculate_span(self, interval: str, limit: int) -> str:
        """Calculate span based on interval and limit"""
        if interval == '5minute':
            return 'day'
        elif interval == '10minute':
            return 'week'
        elif interval == 'hour':
            if limit <= 24:
                return 'day'
            elif limit <= 168:  # 7 days
                return 'week'
            else:
                return 'month'
        elif interval == 'day':
            if limit <= 30:
                return 'month'
            elif limit <= 90:
                return '3month'
            else:
                return 'year'
        else:
            return '5year'
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            self._rate_limit()
            market_data = rs.markets.get_market_today_hours('NYSE')
            if market_data:
                return market_data.get('is_open', False)
            return False
        except:
            return False
    
    def get_market_hours(self) -> Dict:
        """Get market hours for today"""
        try:
            self._rate_limit()
            return rs.markets.get_market_today_hours('NYSE')
        except:
            return {}


# Test the adapter
if __name__ == "__main__":
    adapter = RobinhoodAdapter()
    
    print("\n🧪 Testing Robinhood Adapter")
    print("=" * 50)
    
    # Get credentials from environment
    username = os.getenv('ROBINHOOD_USERNAME')
    password = os.getenv('ROBINHOOD_PASSWORD')
    mfa_token = os.getenv('ROBINHOOD_MFA_TOKEN')
    
    if username and password:
        # Test authentication
        if adapter.authenticate(username, password, mfa_token):
            print("✅ Authentication successful")
            
            # Test getting price
            aapl_price = adapter.get_price('AAPL')
            print(f"AAPL Price: ${aapl_price:,.2f}" if aapl_price else "Failed to get AAPL price")
            
            # Test positions
            positions = adapter.get_positions()
            print(f"Positions: {len(positions)} open positions")
            
            # Test account info
            account = adapter.get_account_info()
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
            
            # Test market hours
            market_open = adapter.is_market_open()
            print(f"Market Open: {market_open}")
            
            # Logout
            rs.authentication.logout()
    else:
        print("⚠️ Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env file")