"""
🌙 Public Data API - Alternative to Moon Dev API
Free data sources for liquidations, funding, and OI
Built with love by Moon Dev community 🚀
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import ccxt
from typing import Optional, Dict, List
import json
import traceback
import numpy as np
from src.agents.binance_proxy import create_binance_session

class PublicDataAPI:
    """
    Alternative data sources to Moon Dev API using free public endpoints
    """
    
    def __init__(self):
        self.coingecko_key = os.getenv('COINGECKO_API_KEY')
        
        # Use specialized Binance session with proxy support
        try:
            self.binance_session = create_binance_session()
            print("✅ Binance proxy session initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Binance session: {e}")
            self.binance_session = None
            
        # Regular session for non-Binance APIs
        self.session = requests.Session()
        
        # Initialize exchange connections
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        print("🌙 Public Data API initialized!")
        print("📊 Using real data sources only - no fake data allowed")
    
    
    
    def get_liquidation_data(self, symbol='BTCUSDT', limit=1000):
        """
        Get liquidation data from Binance
        
        Args:
            symbol: Trading pair (default BTCUSDT)
            limit: Number of records to fetch
            
        Returns:
            DataFrame with liquidation data
        """
        try:
            print(f"🔍 Fetching liquidation data for {symbol}...")
            
            # Binance liquidation endpoint
            url = f"https://fapi.binance.com/fapi/v1/allForceOrders"
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            print(f"📡 API Request: GET {url}")
            print(f"📦 Parameters: {params}")
            
            # Use Binance proxy session if available
            if self.binance_session:
                response = self.binance_session.get(url, params=params)
            else:
                raise Exception("❌ Binance session not initialized. Configure proxy in .env")
            print(f"📨 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API returned error status: {response.status_code}")
                print(f"📄 Response content: {response.text[:500]}...")
                # Geo-restriction detected - fail instead of using fake data
                if response.status_code == 451:
                    raise Exception("❌ Binance API blocked (451 geo-restriction). Please configure PIA proxy in .env file")
                return pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size'])
            
            data = response.json()
            print(f"📊 Response type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
            
            if data and isinstance(data, list) and len(data) > 0:
                print(f"🎯 Processing {len(data)} liquidation records...")
                print(f"📝 First record sample: {data[0] if data else 'None'}")
                
                df = pd.DataFrame(data)
                print(f"📊 DataFrame columns: {df.columns.tolist()}")
                
                # Check required columns
                required_cols = ['time', 'executedQty', 'price', 'side']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"❌ Missing required columns: {missing_cols}")
                    print(f"📋 Available columns: {df.columns.tolist()}")
                    return pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size'])
                
                # Convert to Moon Dev format
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                df['size'] = df['executedQty'].astype(float) * df['price'].astype(float)
                df['side'] = df['side']
                
                # Add long/short columns
                df['long_size'] = df.apply(lambda x: x['size'] if x['side'] == 'SELL' else 0, axis=1)
                df['short_size'] = df.apply(lambda x: x['size'] if x['side'] == 'BUY' else 0, axis=1)
                df['total_size'] = df['long_size'] + df['short_size']
                
                print(f"✅ Fetched {len(df)} liquidation records")
                print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"💰 Total liquidation value: ${df['total_size'].sum():,.2f}")
                return df[['timestamp', 'long_size', 'short_size', 'total_size']]
            else:
                print("⚠️ No liquidation data returned from API")
                print(f"📋 Response data: {data}")
                return pd.DataFrame(columns=['timestamp', 'long_size', 'short_size', 'total_size'])
            
        except Exception as e:
            print(f"❌ Error fetching liquidations: {type(e).__name__}: {e}")
            print(f"📋 Full error traceback:")
            traceback.print_exc()
            # Fail instead of using fake data
            raise Exception(f"❌ Failed to fetch real liquidation data: {e}")
    
    def get_funding_data(self):
        """
        Get funding rates from multiple sources
        
        Returns:
            DataFrame with funding rates
        """
        try:
            print("🔍 Fetching funding rates...")
            
            # Get Binance funding rates
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            print(f"📡 API Request: GET {url}")
            
            # Use Binance proxy session if available
            if self.binance_session:
                response = self.binance_session.get(url)
            else:
                raise Exception("❌ Binance session not initialized. Configure proxy in .env")
            print(f"📨 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API returned error status: {response.status_code}")
                print(f"📄 Response content: {response.text[:500]}...")
                return pd.DataFrame(columns=['symbol', 'funding_rate', 'next_funding_time', 'exchange'])
            
            data = response.json()
            print(f"📊 Response type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
            
            funding_data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'symbol' in item:
                        funding_data.append({
                            'symbol': item['symbol'],
                            'funding_rate': float(item.get('lastFundingRate', 0)),
                            'next_funding_time': pd.to_datetime(item.get('nextFundingTime', 0), unit='ms'),
                            'exchange': 'Binance'
                        })
            
            df = pd.DataFrame(funding_data)
            print(f"✅ Fetched funding rates for {len(df)} markets")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching funding rates: {type(e).__name__}: {e}")
            print(f"📋 Full error traceback:")
            traceback.print_exc()
            return pd.DataFrame(columns=['symbol', 'funding_rate', 'next_funding_time', 'exchange'])
    
    def get_oi_data(self, symbols=['BTCUSDT', 'ETHUSDT']):
        """
        Get open interest data from Binance
        
        Args:
            symbols: List of symbols to fetch OI for
            
        Returns:
            DataFrame with open interest data
        """
        try:
            print("🔍 Fetching open interest data...")
            
            oi_data = []
            for symbol in symbols:
                url = f"https://fapi.binance.com/fapi/v1/openInterest"
                params = {'symbol': symbol}
                
                print(f"📡 Fetching OI for {symbol}: GET {url}")
                
                # Use Binance proxy session if available
                if self.binance_session:
                    response = self.binance_session.get(url, params=params)
                else:
                    raise Exception("❌ Binance session not initialized. Configure proxy in .env")
                print(f"📨 Response Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"❌ API error for {symbol}: {response.status_code}")
                    print(f"📄 Response: {response.text[:200]}...")
                    continue
                
                data = response.json()
                print(f"📊 {symbol} OI response: {data}")
                
                if 'openInterest' in data:
                    oi_value = float(data['openInterest'])
                    oi_data.append({
                        'symbol': symbol,
                        'open_interest': oi_value,
                        'timestamp': datetime.now(),
                        'exchange': 'Binance'
                    })
                    print(f"✅ {symbol} OI: {oi_value:,.2f}")
                else:
                    print(f"⚠️ No 'openInterest' field in response for {symbol}")
                    print(f"📋 Available fields: {list(data.keys())}")
                
                time.sleep(0.1)  # Rate limiting
            
            df = pd.DataFrame(oi_data)
            print(f"✅ Fetched OI data for {len(df)} markets")
            
            # Fail instead of using fake data
            if df.empty:
                raise Exception("❌ No OI data from Binance. Please configure PIA proxy in .env file")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching OI data: {type(e).__name__}: {e}")
            print(f"📋 Full error traceback:")
            traceback.print_exc()
            # Fail instead of using fake data
            if not oi_data:
                raise Exception("❌ Failed to fetch real OI data. Please configure PIA proxy in .env file")
            return pd.DataFrame(oi_data)
    
    def get_coingecko_derivatives(self):
        """
        Get derivatives data from CoinGecko (requires API key)
        
        Returns:
            DataFrame with derivatives data including OI and funding
        """
        if not self.coingecko_key:
            print("⚠️ CoinGecko API key not found")
            return None
            
        try:
            print("🔍 Fetching CoinGecko derivatives data...")
            
            headers = {'x-cg-pro-api-key': self.coingecko_key}
            url = "https://api.coingecko.com/api/v3/derivatives/tickers"
            
            print(f"📡 API Request: GET {url}")
            response = self.session.get(url, headers=headers)
            print(f"📨 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API returned error status: {response.status_code}")
                print(f"📄 Response content: {response.text[:500]}...")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'tickers' in data:
                df = pd.DataFrame(data['tickers'])
                print(f"✅ Fetched {len(df)} derivatives from CoinGecko")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error fetching CoinGecko data: {type(e).__name__}: {e}")
            print(f"📋 Full error traceback:")
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_defi_llama_data(self):
        """
        Get DeFi data from DeFiLlama (completely free)
        
        Returns:
            Dict with various DeFi metrics
        """
        try:
            print("🔍 Fetching DeFiLlama data...")
            
            # Get derivatives overview
            url = "https://api.llama.fi/overview/derivatives"
            print(f"📡 API Request: GET {url}")
            response = self.session.get(url)
            print(f"📨 Derivatives Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Derivatives API error: {response.status_code}")
                print(f"📄 Response: {response.text[:500]}...")
                derivatives = {}
            else:
                derivatives = response.json()
            
            # Get fees overview  
            url = "https://api.llama.fi/overview/fees"
            print(f"📡 API Request: GET {url}")
            response = self.session.get(url)
            print(f"📨 Fees Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Fees API error: {response.status_code}")
                print(f"📄 Response: {response.text[:500]}...")
                fees = {}
            else:
                fees = response.json()
            
            print("✅ Fetched DeFiLlama data")
            return {
                'derivatives': derivatives,
                'fees': fees
            }
            
        except Exception as e:
            print(f"❌ Error fetching DeFiLlama data: {type(e).__name__}: {e}")
            print(f"📋 Full error traceback:")
            traceback.print_exc()
            return {'derivatives': {}, 'fees': {}}
    
    def _get_alternative_liquidation_data(self, symbol='BTCUSDT', limit=1000):
        """
        NO FAKE DATA - Always fail if we can't get real data
        """
        raise Exception("❌ No fake/synthetic data allowed. Configure PIA proxy to access real Binance data")
    
    def get_cryptocompare_data(self, symbol='BTC', convert='USD'):
        """
        Get data from CryptoCompare API (globally available)
        """
        try:
            print(f"🔍 Fetching CryptoCompare data for {symbol}...")
            
            # CryptoCompare API endpoints (free tier)
            base_url = "https://min-api.cryptocompare.com/data"
            
            # Get current price
            url = f"{base_url}/price"
            params = {'fsym': symbol, 'tsyms': convert}
            
            print(f"📡 API Request: GET {url}")
            response = self.session.get(url, params=params)
            print(f"📨 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ CryptoCompare price data: {data}")
                return data
            else:
                print(f"❌ CryptoCompare API error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching CryptoCompare data: {e}")
            return None
    
    def _get_alternative_oi_data(self, symbols):
        """
        NO FAKE DATA - Always fail if we can't get real data
        """
        raise Exception("❌ No fake/synthetic data allowed. Configure PIA proxy to access real Binance data")

# Example usage
if __name__ == "__main__":
    api = PublicDataAPI()
    
    # Test liquidation data
    print("\n📊 Testing Liquidation Data:")
    liq_data = api.get_liquidation_data(limit=100)
    if liq_data is not None and not liq_data.empty:
        print(liq_data.head())
    else:
        print("No liquidation data available")
    
    # Test funding rates
    print("\n📊 Testing Funding Rates:")
    funding = api.get_funding_data()
    if funding is not None and not funding.empty:
        print(funding.head())
    else:
        print("No funding data available")
    
    # Test OI data
    print("\n📊 Testing Open Interest:")
    oi = api.get_oi_data()
    if oi is not None and not oi.empty:
        print(oi.head())
    else:
        print("No OI data available")
    
    # Test CryptoCompare
    print("\n📊 Testing CryptoCompare:")
    crypto_data = api.get_cryptocompare_data('BTC')
    if crypto_data:
        print(f"BTC Price: ${crypto_data.get('USD', 'N/A'):,.2f}")