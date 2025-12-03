"""
🌙 API Adapter - Drop-in replacement for MoonDevAPI
Seamlessly switch between Moon Dev API and free public sources
"""

import os
from src.agents.api import MoonDevAPI
from src.agents.public_data_api import PublicDataAPI
import pandas as pd

class APIAdapter:
    """
    Smart API adapter with intelligent source selection:
    Priority: Moon Dev API → TradingView → Binance
    TradingView is preferred for public data (no geo-restrictions!)
    """
    
    def __init__(self):
        self.moon_dev_key = os.getenv('MOONDEV_API_KEY')
        self.data_source = os.getenv('DATA_SOURCE', 'auto').lower()  # auto, moondev, binance, tradingview
        
        if self.data_source == 'yfinance':
            print("🚀 Using YFinance API (forced)")
            from src.agents.yfinance_adapter import YFinanceAdapter
            self.api = YFinanceAdapter()
            self.using_moon_dev = False
            self.source = 'yfinance'
        elif self.data_source == 'tradingview':
            print("📊 Using TradingView API (forced)")
            from src.agents.tradingview_adapter import TradingViewAdapter
            self.api = TradingViewAdapter()
            self.using_moon_dev = False
            self.source = 'tradingview'
        elif self.data_source == 'binance':
            print("🌍 Using Binance/Public Data API (forced)")
            self.api = PublicDataAPI()
            self.using_moon_dev = False
            self.source = 'binance'
        elif self.moon_dev_key and self.moon_dev_key != 'your_moondev_key_here' and self.data_source != 'public':
            print("🌙 Using Moon Dev API")
            self.api = MoonDevAPI()
            self.using_moon_dev = True
            self.source = 'moondev'
        else:
            # Auto mode - try YFinance first, then TradingView, fallback to Binance
            print("🌍 Using Public Data Sources (auto mode)")
            try:
                # Try YFinance first (no auth needed!)
                print("🚀 Attempting YFinance API...")
                from src.agents.yfinance_adapter import YFinanceAdapter
                self.api = YFinanceAdapter()
                # Quick test
                try:
                    test_price = self.api.get_price('BTCUSDT')
                    if test_price and test_price > 0:
                        self.using_moon_dev = False
                        self.source = 'yfinance'
                        print("✅ YFinance API is working (primary)")
                        print(f"   BTC Price: ${test_price:,.2f}")
                    else:
                        raise Exception("YFinance not responding")
                except Exception as e:
                    # If YFinance fails, try TradingView
                    raise Exception(f"YFinance test failed: {e}")
            except Exception as e:
                # Try TradingView as second option
                print("⚠️ YFinance not available, trying TradingView...")
                try:
                    from src.agents.tradingview_adapter import TradingViewAdapter
                    self.api = TradingViewAdapter()
                    # Quick test with rate limit protection
                    try:
                        test_price = self.api.get_price('BTCUSDT')
                        if test_price and test_price > 0:
                            self.using_moon_dev = False
                            self.source = 'tradingview'
                            print("✅ TradingView API is working (secondary)")
                            print(f"   BTC Price: ${test_price:,.2f}")
                        else:
                            raise Exception("TradingView not responding")
                    except Exception as e:
                        # If TradingView fails, we'll try Binance
                        raise Exception(f"TradingView test failed: {e}")
                except Exception as e:
                    # Fallback to Binance
                    import traceback
                    print("⚠️ TradingView not available, trying Binance...")
                    print(f"❌ TradingView error: {str(e)}")
                    print(f"📍 Error type: {type(e).__name__}")
                    print("📍 Traceback:")
                    traceback.print_exc()
                    try:
                        self.api = PublicDataAPI()
                        # Quick test to see if Binance works
                        test = self.api.get_funding_data()
                        if test is None or test.empty:
                            raise Exception("Binance not accessible")
                        self.using_moon_dev = False
                        self.source = 'binance'
                        print("✅ Binance API is working (fallback)")
                    except:
                        # Last resort - use YFinance even if test failed
                        print("⚠️ Binance also not accessible, defaulting to YFinance")
                        from src.agents.yfinance_adapter import YFinanceAdapter
                        self.api = YFinanceAdapter()
                        self.using_moon_dev = False
                        self.source = 'yfinance'
    
    def get_liquidation_data(self, limit=None):
        """Get liquidation data from available source"""
        if self.using_moon_dev:
            return self.api.get_liquidation_data(limit=limit)
        else:
            # Public API uses different params
            return self.api.get_liquidation_data(limit=limit if limit else 1000)
    
    def get_funding_data(self):
        """Get funding data from available source"""
        return self.api.get_funding_data()
    
    def get_oi_data(self):
        """Get OI data from available source"""
        if self.using_moon_dev:
            return self.api.get_oi_data()
        else:
            # For TradingView or Binance
            if hasattr(self.api, 'get_oi_data'):
                oi_data = self.api.get_oi_data(['BTCUSDT', 'ETHUSDT'])
                if oi_data is not None:
                    # Only try CoinGecko merge if not TradingView
                    if self.source != 'tradingview' and hasattr(self.api, 'get_coingecko_derivatives'):
                        cg_data = self.api.get_coingecko_derivatives()
                        if cg_data is not None:
                            # Merge data sources
                            return self._merge_oi_data(oi_data, cg_data)
                return oi_data
            return pd.DataFrame()
    
    def _merge_oi_data(self, binance_oi, coingecko_data):
        """Merge OI data from multiple sources"""
        # This is a simplified merge - enhance based on your needs
        return binance_oi
    
    # Fallback methods for Moon Dev specific endpoints
    def get_copybot_follow_list(self):
        """Not available in public API"""
        if self.using_moon_dev:
            return self.api.get_copybot_follow_list()
        else:
            print("⚠️ Copybot data only available with Moon Dev API")
            return pd.DataFrame()
    
    def get_whale_addresses(self):
        """Not available in public API - return empty"""
        if self.using_moon_dev:
            return self.api.get_whale_addresses()
        else:
            print("⚠️ Whale addresses only available with Moon Dev API")
            return pd.DataFrame()

# Usage example - just replace MoonDevAPI with APIAdapter
if __name__ == "__main__":
    # This works with or without Moon Dev API key!
    api = APIAdapter()
    
    # Get liquidation data
    liq_data = api.get_liquidation_data(limit=100)
    print(f"Got {len(liq_data) if liq_data is not None else 0} liquidation records")
    
    # Get funding data
    funding = api.get_funding_data()
    print(f"Got {len(funding) if funding is not None else 0} funding rates")