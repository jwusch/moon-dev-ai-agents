"""
🌙 Get TSLA Data Using Basic TradingView API
Alternative method when session conflicts occur

Author: Claude (Anthropic)
"""

import sys
import os
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

from agents.tradingview_api import TradingViewAPI
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def get_tsla_with_basic_api():
    """Get TSLA data using the basic TradingView API"""
    print("🌙 Getting TSLA Data - Basic API Method")
    print("=" * 50)
    
    try:
        # Use the basic TradingView API
        tv_api = TradingViewAPI()
        
        print("📊 Getting current TSLA data from TradingView...")
        tsla_data = tv_api.get_indicators('TSLA', exchange='NASDAQ')
        
        if tsla_data:
            print("✅ Current TSLA Data:")
            print(f"   Price: ${tsla_data.get('close', 0):.2f}")
            print(f"   Open: ${tsla_data.get('open', 0):.2f}")
            print(f"   High: ${tsla_data.get('high', 0):.2f}")
            print(f"   Low: ${tsla_data.get('low', 0):.2f}")
            print(f"   Volume: {tsla_data.get('volume', 0):,.0f}")
        else:
            print("❌ No current data from TradingView basic API")
            
    except Exception as e:
        print(f"❌ TradingView basic API error: {e}")
        print("Trying alternative methods...")

def get_tsla_with_yfinance():
    """Get TSLA historical data using yfinance as backup"""
    print("\n📊 Getting TSLA Historical Data (yfinance backup)")
    print("-" * 50)
    
    try:
        # Get 6 months of TSLA data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        print(f"📅 Fetching data from {start_date.date()} to {end_date.date()}")
        
        tsla = yf.Ticker("TSLA")
        hist = tsla.history(start=start_date, end=end_date)
        
        if not hist.empty:
            print(f"✅ Retrieved {len(hist)} days of TSLA data")
            
            # Display summary
            latest = hist.iloc[-1]
            oldest = hist.iloc[0]
            
            print(f"\n📊 6-Month TSLA Summary:")
            print(f"   Current Price: ${latest['Close']:.2f}")
            print(f"   6-Month High: ${hist['High'].max():.2f}")
            print(f"   6-Month Low: ${hist['Low'].min():.2f}")
            
            # Calculate return
            start_price = oldest['Close']
            end_price = latest['Close']
            return_pct = ((end_price - start_price) / start_price) * 100
            print(f"   6-Month Return: {return_pct:+.1f}%")
            
            # Show recent data
            print(f"\n📋 Recent TSLA Data (last 5 days):")
            recent = hist.tail()
            for date, row in recent.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                print(f"   {date_str}: O=${row['Open']:.2f} H=${row['High']:.2f} L=${row['Low']:.2f} C=${row['Close']:.2f} V={row['Volume']:,.0f}")
            
            # Save to CSV
            filename = f"TSLA_6months_yfinance_{datetime.now().strftime('%Y%m%d')}.csv"
            hist.to_csv(filename)
            print(f"\n💾 Data saved to: {filename}")
            
            return hist
        else:
            print("❌ No data retrieved from yfinance")
            return None
            
    except Exception as e:
        print(f"❌ yfinance error: {e}")
        return None

def get_tsla_with_alpha_vantage():
    """Alternative using Alpha Vantage (if API key available)"""
    print("\n📊 Alpha Vantage Method")
    print("-" * 30)
    
    # Check if Alpha Vantage key exists
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not av_key:
        print("❌ No Alpha Vantage API key found in .env")
        print("💡 Add ALPHA_VANTAGE_API_KEY=your_key to .env for this method")
        return None
    
    try:
        import requests
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'TSLA',
            'apikey': av_key,
            'outputsize': 'compact'  # Last 100 days
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            print(f"✅ Retrieved {len(time_series)} days of data")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            
            # Show summary
            print(f"Latest Close: ${df['Close'].iloc[-1]:.2f}")
            print(f"6-Month High: ${df['High'].max():.2f}")
            print(f"6-Month Low: ${df['Low'].min():.2f}")
            
            return df
        else:
            print("❌ Alpha Vantage API error:", data.get('Error Message', 'Unknown'))
            return None
            
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        return None

def main():
    print("🌙 TSLA Data Retrieval - Multiple Methods")
    print("=" * 60)
    
    # Method 1: Basic TradingView API
    get_tsla_with_basic_api()
    
    # Method 2: Yahoo Finance (most reliable)
    tsla_data = get_tsla_with_yfinance()
    
    # Method 3: Alpha Vantage (if available)
    get_tsla_with_alpha_vantage()
    
    if tsla_data is not None:
        print("\n✅ Successfully retrieved TSLA data using backup methods!")
        print("\n💡 For TradingView session API:")
        print("1. Make sure you're completely logged out of tradingview.com")
        print("2. Try getting fresh session tokens")
        print("3. Restart the session server")
    else:
        print("\n❌ Could not retrieve TSLA data from any source")
        print("💡 Try installing yfinance: pip install yfinance")

if __name__ == "__main__":
    main()