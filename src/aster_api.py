"""
🌙 Moon Dev's Custom Aster DEX API Implementation
Built with love by Moon Dev 🚀

This is our own implementation of the Aster DEX API client.
No external dependencies needed!
"""

import requests
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import os
from termcolor import cprint


class AsterAPI:
    """Custom Aster DEX API client implementation"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """Initialize Aster API client
        
        Args:
            api_key: Your Aster API key
            api_secret: Your Aster API secret
            testnet: Whether to use testnet (default: False for mainnet)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Set base URL based on testnet flag
        if testnet:
            self.base_url = "https://api.testnet.aster.markets"
            cprint("⚠️  Using Aster TESTNET", "yellow")
        else:
            self.base_url = "https://api.aster.markets"
            cprint("✅ Using Aster MAINNET", "green")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        })
        
    def _sign_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Sign request with HMAC SHA256
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Signed headers
        """
        timestamp = str(int(time.time() * 1000))
        
        # Create signature payload
        if params:
            params_string = json.dumps(params, separators=(',', ':'))
        else:
            params_string = ""
            
        signature_payload = f"{timestamp}{method}{endpoint}{params_string}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-Timestamp': timestamp,
            'X-Signature': signature
        }
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Aster API
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Request parameters
            
        Returns:
            JSON response
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add signature headers
        headers = self._sign_request(method, endpoint, params)
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(url, json=params, headers=headers)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            cprint(f"❌ API request error: {e}", "red")
            if hasattr(e.response, 'text'):
                cprint(f"Response: {e.response.text}", "red")
            raise
    
    # Market Data Methods
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information including all trading pairs"""
        return self._make_request("GET", "/v1/exchange-info")
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of levels to return (default: 20)
            
        Returns:
            Order book data with bids and asks
        """
        params = {"symbol": symbol, "limit": limit}
        return self._make_request("GET", "/v1/orderbook", params)
    
    def get_ask_bid(self, symbol: str) -> Tuple[float, float, float]:
        """Get best ask/bid prices for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (best_ask, best_bid, mid_price)
        """
        orderbook = self.get_orderbook(symbol, limit=1)
        
        best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
        best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
        mid_price = (best_ask + best_bid) / 2 if best_ask and best_bid else 0
        
        return best_ask, best_bid, mid_price
    
    # Account Methods
    
    def get_account_info(self) -> Dict:
        """Get account information including balances"""
        return self._make_request("GET", "/v1/account")
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position data or None if no position
        """
        positions = self._make_request("GET", "/v1/positions")
        
        # Find position for the symbol
        for pos in positions.get('positions', []):
            if pos['symbol'] == symbol:
                return {
                    'symbol': pos['symbol'],
                    'position_amount': float(pos['position_amount']),
                    'entry_price': float(pos['entry_price']),
                    'mark_price': float(pos.get('mark_price', 0)),
                    'pnl': float(pos.get('unrealized_pnl', 0)),
                    'pnl_percentage': float(pos.get('pnl_percentage', 0)),
                    'is_long': float(pos['position_amount']) > 0,
                    'leverage': int(pos.get('leverage', 1)),
                    'margin': float(pos.get('margin', 0))
                }
        
        return None
    
    # Trading Methods
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change leverage for a symbol
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (1-100)
            
        Returns:
            Confirmation response
        """
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return self._make_request("POST", "/v1/leverage", params)
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                   size: float, price: float = None, 
                   reduce_only: bool = False,
                   post_only: bool = False,
                   client_order_id: str = None) -> Dict:
        """Place an order
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT' or 'MARKET'
            size: Order size in base currency
            price: Limit price (required for LIMIT orders)
            reduce_only: Whether order can only reduce position
            post_only: Whether order must be maker only
            client_order_id: Custom order ID
            
        Returns:
            Order response with order_id
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "size": size,
            "reduce_only": reduce_only,
            "post_only": post_only
        }
        
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params["price"] = price
            
        if client_order_id:
            params["client_order_id"] = client_order_id
            
        return self._make_request("POST", "/v1/order", params)
    
    def get_order(self, symbol: str, order_id: str = None, 
                  client_order_id: str = None) -> Dict:
        """Get order status
        
        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID
            client_order_id: Custom order ID
            
        Returns:
            Order details
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        params = {"symbol": symbol}
        if order_id:
            params["order_id"] = order_id
        if client_order_id:
            params["client_order_id"] = client_order_id
            
        return self._make_request("GET", "/v1/order", params)
    
    def cancel_order(self, symbol: str, order_id: str = None,
                    client_order_id: str = None) -> Dict:
        """Cancel an order
        
        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID
            client_order_id: Custom order ID
            
        Returns:
            Cancellation confirmation
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        params = {"symbol": symbol}
        if order_id:
            params["order_id"] = order_id
        if client_order_id:
            params["client_order_id"] = client_order_id
            
        return self._make_request("DELETE", "/v1/order", params)
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all open orders
        
        Args:
            symbol: Optional symbol to cancel orders for (None = all symbols)
            
        Returns:
            Cancellation summary
        """
        endpoint = "/v1/orders/cancel-all"
        params = {"symbol": symbol} if symbol else {}
        return self._make_request("POST", endpoint, params)


class AsterFuncs:
    """Helper functions for Aster DEX operations"""
    
    def __init__(self, api: AsterAPI):
        """Initialize with API instance"""
        self.api = api
        
    def calculate_position_size(self, account_balance: float, risk_percent: float,
                               entry_price: float, stop_loss: float,
                               leverage: int = 1) -> float:
        """Calculate optimal position size based on risk
        
        Args:
            account_balance: Total account balance
            risk_percent: Risk per trade (e.g., 1 for 1%)
            entry_price: Entry price
            stop_loss: Stop loss price
            leverage: Leverage to use
            
        Returns:
            Position size in base currency
        """
        risk_amount = account_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        # Position size = Risk Amount / Price Difference
        # Adjust for leverage
        position_size = (risk_amount / price_diff) * leverage
        
        return position_size
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Symbol details including min/max values
        """
        exchange_info = self.api.get_exchange_info()
        
        for sym_info in exchange_info.get('symbols', []):
            if sym_info['symbol'] == symbol:
                return sym_info
                
        raise ValueError(f"Symbol {symbol} not found")
    
    def round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to valid tick size
        
        Args:
            price: Price to round
            tick_size: Minimum price increment
            
        Returns:
            Rounded price
        """
        return round(price / tick_size) * tick_size
    
    def format_symbol_info(self, sym_info: Dict) -> Dict:
        """Format symbol information for compatibility
        
        Args:
            sym_info: Raw symbol info from API
            
        Returns:
            Formatted symbol info
        """
        return {
            'symbol': sym_info.get('symbol'),
            'base_asset': sym_info.get('baseAsset'),
            'quote_asset': sym_info.get('quoteAsset'),
            'min_quantity': float(sym_info.get('minQty', 0.001)),
            'max_quantity': float(sym_info.get('maxQty', 1000000)),
            'tick_size': float(sym_info.get('tickSize', 0.01)),
            'step_size': float(sym_info.get('stepSize', 0.001)),
            'min_notional': float(sym_info.get('minNotional', 5.0)),
            'max_leverage': int(sym_info.get('maxLeverage', 100))
        }


# Export both classes for compatibility
__all__ = ['AsterAPI', 'AsterFuncs']