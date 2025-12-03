"""
🌙 Test Binance Proxy Configuration
Verifies that Binance API calls work through proxy without 451 errors
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.public_data_api import PublicDataAPI
from src.agents.api_adapter import APIAdapter

def test_binance_direct():
    """Test direct Binance API access through PublicDataAPI"""
    print("\n" + "="*60)
    print("🧪 Testing Binance API through PublicDataAPI")
    print("="*60)
    
    try:
        api = PublicDataAPI()
        
        # Test 1: Liquidation data
        print("\n📊 Test 1: Fetching liquidation data...")
        liq_data = api.get_liquidation_data(symbol='BTCUSDT', limit=10)
        if liq_data is not None and not liq_data.empty:
            print(f"✅ Success! Got {len(liq_data)} liquidation records")
            print(f"📈 Latest liquidation: {liq_data.iloc[0]['timestamp']}")
            print(f"💰 Total value: ${liq_data['total_size'].sum():,.2f}")
        else:
            print("❌ No liquidation data returned")
            
        # Test 2: Funding rates
        print("\n📊 Test 2: Fetching funding rates...")
        funding = api.get_funding_data()
        if funding is not None and not funding.empty:
            print(f"✅ Success! Got funding rates for {len(funding)} markets")
            btc_funding = funding[funding['symbol'] == 'BTCUSDT']
            if not btc_funding.empty:
                print(f"📈 BTC funding rate: {btc_funding.iloc[0]['funding_rate']:.4%}")
        else:
            print("❌ No funding data returned")
            
        # Test 3: Open Interest
        print("\n📊 Test 3: Fetching open interest...")
        oi_data = api.get_oi_data(['BTCUSDT', 'ETHUSDT'])
        if oi_data is not None and not oi_data.empty:
            print(f"✅ Success! Got OI data for {len(oi_data)} markets")
            for _, row in oi_data.iterrows():
                print(f"📈 {row['symbol']}: {row['open_interest']:,.2f} contracts")
        else:
            print("❌ No OI data returned")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        if "451" in str(e):
            print("\n⚠️ Geo-restriction detected! You need to configure proxy:")
            print("1. Add to your .env file:")
            print("   PIA_USERNAME=your_pia_username")
            print("   PIA_PASSWORD=your_pia_password")
            print("2. Make sure PIA VPN is connected")
        return False

def test_api_adapter():
    """Test API access through APIAdapter"""
    print("\n" + "="*60)
    print("🧪 Testing Binance API through APIAdapter")
    print("="*60)
    
    try:
        api = APIAdapter()
        
        # Quick test
        print("\n📊 Testing liquidation data...")
        liq_data = api.get_liquidation_data(limit=5)
        if liq_data is not None and not liq_data.empty:
            print(f"✅ Success! APIAdapter working correctly")
            return True
        else:
            print("❌ No data from APIAdapter")
            return False
            
    except Exception as e:
        print(f"❌ APIAdapter test failed: {e}")
        return False

def check_env_config():
    """Check current proxy configuration"""
    print("\n" + "="*60)
    print("🔧 Current Proxy Configuration")
    print("="*60)
    
    configs = [
        ('PIA_USERNAME', 'PIA Username'),
        ('PIA_PASSWORD', 'PIA Password'),
        ('PIA_HTTP_PROXY', 'PIA HTTP Proxy'),
        ('HTTPS_PROXY', 'HTTPS Proxy'),
        ('HTTP_PROXY', 'HTTP Proxy'),
        ('BINANCE_PROXY', 'Binance-specific Proxy')
    ]
    
    configured = False
    for env_var, display_name in configs:
        value = os.getenv(env_var)
        if value:
            if 'PASSWORD' in env_var:
                print(f"✅ {display_name}: {'*' * 8} (configured)")
            else:
                # Mask credentials in URLs
                if '@' in str(value):
                    parts = value.split('@')
                    masked = f"{parts[0].split('://')[0]}://***:***@{parts[1]}"
                    print(f"✅ {display_name}: {masked}")
                else:
                    print(f"✅ {display_name}: {value}")
            configured = True
        else:
            print(f"❌ {display_name}: Not configured")
    
    if not configured:
        print("\n⚠️ No proxy configuration found!")
        print("📝 Add proxy credentials to your .env file to access Binance from restricted regions")
    
    return configured

if __name__ == "__main__":
    print("🌙 Moon Dev Binance Proxy Test")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check configuration
    has_proxy = check_env_config()
    
    if not has_proxy:
        print("\n⚠️ Warning: No proxy configured")
        print("If you're in a geo-restricted region, Binance API calls will fail")
        response = input("\n🤔 Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("👋 Exiting. Configure proxy in .env file first.")
            sys.exit(0)
    
    # Run tests
    success1 = test_binance_direct()
    success2 = test_api_adapter()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Direct API Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"API Adapter Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Binance API is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check your proxy configuration.")
        print("\n📝 Quick fix:")
        print("1. Get PIA VPN account from privateinternetaccess.com")
        print("2. Add credentials to .env file:")
        print("   PIA_USERNAME=your_username")
        print("   PIA_PASSWORD=your_password")
        print("3. Run this test again")