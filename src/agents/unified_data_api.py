"""
🌙 Unified Data API
Smart data source selector that automatically chooses the best available API
Supports Binance, TradingView, and other sources
"""

import os
import pandas as pd
from typing import Optional, Dict, List, Union
from datetime import datetime

class UnifiedDataAPI:
    """
    Unified API that automatically selects the best data source
    Priority order:
    1. Moon Dev API (if available)
    2. Binance API (with proxy if needed)
    3. TradingView API (always available)
    4. Other fallback sources
    """
    
    def __init__(self, preferred_source: Optional[str] = None):
        """
        Initialize with automatic source selection
        
        Args:
            preferred_source: Force a specific source ('binance', 'tradingview', 'moondev')
        """
        self.apis = {}
        self.available_sources = []
        self.preferred_source = preferred_source
        
        # Try to initialize each API
        self._init_apis()
        
        print("🌙 Unified Data API initialized!")
        print(f"📊 Available sources: {', '.join(self.available_sources)}")
        if self.preferred_source:
            print(f"⭐ Preferred source: {self.preferred_source}")
    
    def _init_apis(self):
        """Initialize all available APIs"""
        
        # 1. Try Moon Dev API (if key exists)
        if os.getenv('MOONDEV_API_KEY'):
            try:
                from src.agents.api import MoonDevAPI
                self.apis['moondev'] = MoonDevAPI()
                self.available_sources.append('moondev')
                print("✅ Moon Dev API available")
            except Exception as e:
                print(f"⚠️ Moon Dev API not available: {e}")
        
        # 2. Try TradingView first (no geo-restrictions!)
        try:
            from src.agents.tradingview_adapter import TradingViewAdapter
            self.apis['tradingview'] = TradingViewAdapter()
            self.available_sources.append('tradingview')
            print("✅ TradingView API available (primary)")
        except Exception as e:
            print(f"❌ TradingView API failed to initialize: {e}")
        
        # 3. Try Binance as fallback
        try:
            from src.agents.public_data_api import PublicDataAPI
            self.apis['binance'] = PublicDataAPI()
            # Test if Binance is actually accessible
            test_data = self.apis['binance'].get_liquidation_data(symbol='BTCUSDT', limit=1)
            if test_data is not None:
                self.available_sources.append('binance')
                print("✅ Binance API available (fallback)")
            else:
                print("⚠️ Binance API initialized but not accessible")
        except Exception as e:
            print(f"⚠️ Binance API not available: {e}")
        
        if not self.available_sources:
            raise Exception("❌ No data sources available!")
    
    def _get_api(self, method: str) -> tuple:
        """
        Get the best API for a specific method
        
        Returns:
            (api_instance, source_name)
        """
        # If preferred source is set and available, use it
        if self.preferred_source and self.preferred_source in self.apis:
            api = self.apis[self.preferred_source]
            if hasattr(api, method):
                return api, self.preferred_source
        
        # Otherwise, find first available API with the method
        for source in self.available_sources:
            api = self.apis.get(source)
            if api and hasattr(api, method):
                return api, source
        
        raise Exception(f"❌ No API available for method: {method}")
    
    def get_liquidation_data(self, symbol='BTCUSDT', limit=100) -> pd.DataFrame:
        """Get liquidation data from best available source"""
        try:
            api, source = self._get_api('get_liquidation_data')
            print(f"🔄 Getting liquidation data from {source}...")
            
            data = api.get_liquidation_data(symbol=symbol, limit=limit)
            
            if data is not None and not data.empty:
                data['source'] = source
                return data
            
            # If primary fails, try fallback
            if source != 'tradingview' and 'tradingview' in self.apis:
                print(f"⚠️ {source} failed, falling back to TradingView...")
                data = self.apis['tradingview'].get_liquidation_data(symbol=symbol, limit=limit)
                if data is not None and not data.empty:
                    data['source'] = 'tradingview'
                    return data
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting liquidation data: {e}")
            return pd.DataFrame()
    
    def get_funding_data(self) -> pd.DataFrame:
        """Get funding rates from best available source"""
        try:
            api, source = self._get_api('get_funding_data')
            print(f"🔄 Getting funding data from {source}...")
            
            data = api.get_funding_data()
            
            if data is not None and not data.empty:
                data['source'] = source
                return data
            
            # Fallback to TradingView
            if source != 'tradingview' and 'tradingview' in self.apis:
                print(f"⚠️ {source} failed, falling back to TradingView...")
                data = self.apis['tradingview'].get_funding_data()
                if data is not None and not data.empty:
                    data['source'] = 'tradingview'
                    return data
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting funding data: {e}")
            return pd.DataFrame()
    
    def get_oi_data(self, symbols=['BTCUSDT', 'ETHUSDT']) -> pd.DataFrame:
        """Get open interest data from best available source"""
        try:
            # Try to get from APIs that have this method
            if 'binance' in self.apis and hasattr(self.apis['binance'], 'get_oi_data'):
                print("🔄 Getting OI data from Binance...")
                data = self.apis['binance'].get_oi_data(symbols)
                if data is not None and not data.empty:
                    data['source'] = 'binance'
                    return data
            
            # Fallback to TradingView estimation
            if 'tradingview' in self.apis:
                print("🔄 Getting OI data from TradingView (estimated)...")
                data = self.apis['tradingview'].get_oi_data(symbols)
                if data is not None and not data.empty:
                    data['source'] = 'tradingview'
                    return data
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error getting OI data: {e}")
            return pd.DataFrame()
    
    def get_technical_analysis(self, symbol: str, interval: str = '1h') -> Optional[Dict]:
        """Get technical analysis (best from TradingView)"""
        try:
            # Prefer TradingView for technical analysis
            if 'tradingview' in self.apis:
                print(f"🔄 Getting technical analysis from TradingView...")
                return self.apis['tradingview'].get_technical_analysis(symbol, interval)
            
            # Otherwise return basic data from other sources
            print("⚠️ TradingView not available for technical analysis")
            return None
            
        except Exception as e:
            print(f"❌ Error getting technical analysis: {e}")
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price from any available source"""
        try:
            # Try TradingView first (most reliable for current price)
            if 'tradingview' in self.apis:
                analysis = self.apis['tradingview'].tv_api.get_analysis(symbol)
                if analysis:
                    return analysis['indicators'].get('close')
            
            # Try other sources
            for source, api in self.apis.items():
                if hasattr(api, 'get_ohlcv_data'):
                    data = api.get_ohlcv_data(symbol, '1h', 1)
                    if data is not None and not data.empty:
                        return data.iloc[-1]['close']
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return self.available_sources
    
    def set_preferred_source(self, source: str):
        """Set preferred data source"""
        if source in self.available_sources:
            self.preferred_source = source
            print(f"✅ Preferred source set to: {source}")
        else:
            print(f"❌ Source '{source}' not available. Available: {self.available_sources}")
    
    def get_source_status(self) -> Dict[str, bool]:
        """Get status of all data sources"""
        status = {}
        
        for source in ['moondev', 'binance', 'tradingview']:
            if source in self.apis:
                # Try a simple test
                try:
                    if source == 'tradingview':
                        test = self.apis[source].tv_api.get_recommendation('BTCUSDT')
                        status[source] = test is not None
                    elif hasattr(self.apis[source], 'get_funding_data'):
                        test = self.apis[source].get_funding_data()
                        status[source] = test is not None
                    else:
                        status[source] = True
                except:
                    status[source] = False
            else:
                status[source] = False
        
        return status

# Example usage
if __name__ == "__main__":
    print("🌙 Testing Unified Data API...")
    
    # Initialize unified API
    api = UnifiedDataAPI()
    
    # Show available sources
    print(f"\n📊 Available sources: {api.get_available_sources()}")
    
    # Get source status
    print("\n📊 Source Status:")
    for source, active in api.get_source_status().items():
        print(f"  {source}: {'✅ Active' if active else '❌ Inactive'}")
    
    # Test 1: Get price
    print("\n📊 Test 1: Get BTC Price")
    price = api.get_price('BTCUSDT')
    if price:
        print(f"BTC Price: ${price:,.2f}")
    
    # Test 2: Get liquidations
    print("\n📊 Test 2: Get Liquidations")
    liq_data = api.get_liquidation_data('BTCUSDT', limit=5)
    if not liq_data.empty:
        print(f"Got {len(liq_data)} liquidation records from {liq_data['source'].iloc[0]}")
    
    # Test 3: Get funding
    print("\n📊 Test 3: Get Funding Rates")
    funding = api.get_funding_data()
    if not funding.empty:
        print(f"Got {len(funding)} funding rates from {funding['source'].iloc[0]}")
    
    # Test 4: Get technical analysis
    print("\n📊 Test 4: Get Technical Analysis")
    ta = api.get_technical_analysis('BTCUSDT')
    if ta:
        print(f"BTC Recommendation: {ta['summary']['RECOMMENDATION']}")
    
    # Test 5: Force TradingView
    print("\n📊 Test 5: Force TradingView Source")
    api.set_preferred_source('tradingview')
    liq_data = api.get_liquidation_data('ETHUSDT', limit=3)
    if not liq_data.empty:
        print(f"Got {len(liq_data)} ETH liquidation estimates from {liq_data['source'].iloc[0]}")