"""
🚀 Demo: Using YFinance with Moon Dev Trading Agents
Shows how to use YFinance for backtesting and live trading

Author: Claude (Anthropic)
"""

import sys
sys.path.append('/mnt/c/Users/jwusc/moon-dev-ai-agents/src')

from agents.api_adapter import APIAdapter
from agents.yfinance_adapter import YFinanceAdapter
import pandas as pd
import os
from datetime import datetime, timedelta

def demo_backtesting_data():
    """Show how to get historical data for backtesting"""
    print("📊 Demo 1: Getting Backtesting Data")
    print("=" * 60)
    
    # Force YFinance
    os.environ['DATA_SOURCE'] = 'yfinance'
    
    yf_api = YFinanceAdapter()
    
    # Get different timeframes
    timeframes = {
        '1m': 'Last 7 days max',
        '5m': 'Last 60 days max',
        '15m': 'Last 60 days max',
        '1h': 'Last 730 days max',
        '1d': 'Unlimited'
    }
    
    symbol = 'TSLA'
    
    for interval, description in timeframes.items():
        print(f"\n⏱️ {interval} bars ({description}):")
        
        # Determine appropriate number of bars
        if interval in ['1m', '5m']:
            bars = 100
        elif interval in ['15m', '1h']:
            bars = 200
        else:
            bars = 365
            
        data = yf_api.get_ohlcv_data(symbol, interval, bars)
        
        if not data.empty:
            print(f"   Retrieved {len(data)} bars")
            print(f"   Date range: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
            print(f"   Latest: O=${data['open'].iloc[-1]:.2f} H=${data['high'].iloc[-1]:.2f} L=${data['low'].iloc[-1]:.2f} C=${data['close'].iloc[-1]:.2f}")

def demo_multi_symbol_analysis():
    """Show how to analyze multiple symbols"""
    print("\n\n🚀 Demo 2: Multi-Symbol Analysis")
    print("=" * 60)
    
    yf_api = YFinanceAdapter()
    
    # Tech giants + Crypto
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 
               'BTCUSDT', 'ETHUSDT', 'SPY', 'QQQ']
    
    print("📊 Getting current prices for portfolio...")
    
    analysis = []
    
    for symbol in symbols:
        try:
            # Get 24h stats
            stats = yf_api.get_24h_stats(symbol)
            
            if stats:
                analysis.append({
                    'Symbol': symbol,
                    'Price': stats.get('price', 0),
                    '24h Change': stats.get('change_24h', 0),
                    '24h High': stats.get('high_24h', 0),
                    '24h Low': stats.get('low_24h', 0),
                    '24h Volume': stats.get('volume_24h', 0)
                })
                
        except:
            pass
    
    # Create DataFrame for nice display
    df = pd.DataFrame(analysis)
    
    if not df.empty:
        # Sort by 24h change
        df = df.sort_values('24h Change', ascending=False)
        
        print("\n📈 Top Movers (24h):")
        print("-" * 60)
        
        for _, row in df.iterrows():
            symbol = row['Symbol']
            price = row['Price']
            change = row['24h Change']
            volume = row['24h Volume']
            
            # Color code based on change
            if change > 0:
                change_str = f"+{change:.2f}%"
            else:
                change_str = f"{change:.2f}%"
                
            print(f"   {symbol:<10} ${price:>10,.2f}  {change_str:>8}  Vol: ${volume/1e6:>6.1f}M")

def demo_trading_strategy():
    """Demo simple trading strategy with YFinance data"""
    print("\n\n🎯 Demo 3: Simple Moving Average Strategy")
    print("=" * 60)
    
    yf_api = YFinanceAdapter()
    
    symbol = 'TSLA'
    
    # Get hourly data for SMA calculation
    print(f"📊 Analyzing {symbol} with SMA strategy...")
    
    data = yf_api.get_ohlcv_data(symbol, '1h', 200)
    
    if not data.empty:
        # Calculate SMAs
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        
        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        current_price = latest['close']
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        
        print(f"\n📍 Current Analysis:")
        print(f"   Price: ${current_price:.2f}")
        print(f"   SMA 20: ${sma_20:.2f}")
        print(f"   SMA 50: ${sma_50:.2f}")
        
        # Check for crossover
        if prev['SMA_20'] <= prev['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
            print("\n🚀 GOLDEN CROSS detected! (Bullish signal)")
        elif prev['SMA_20'] >= prev['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
            print("\n🔻 DEATH CROSS detected! (Bearish signal)")
        elif latest['SMA_20'] > latest['SMA_50']:
            print("\n📈 Uptrend (SMA 20 > SMA 50)")
        else:
            print("\n📉 Downtrend (SMA 20 < SMA 50)")
        
        # Position relative to SMAs
        if current_price > sma_20 and current_price > sma_50:
            print("   ✅ Price above both SMAs (Strong)")
        elif current_price > sma_20:
            print("   ⚠️ Price above SMA 20 only (Moderate)")
        else:
            print("   ❌ Price below both SMAs (Weak)")

def demo_save_for_backtesting():
    """Show how to save data for backtesting.py"""
    print("\n\n💾 Demo 4: Save Data for Backtesting")
    print("=" * 60)
    
    yf_api = YFinanceAdapter()
    
    symbols = ['TSLA', 'AAPL', 'SPY']
    
    for symbol in symbols:
        print(f"\n📊 Saving {symbol} data...")
        
        # Get 6 months of daily data
        data = yf_api.get_ohlcv_data(symbol, '1d', 180)
        
        if not data.empty:
            # Format for backtesting.py
            bt_data = pd.DataFrame({
                'Open': data['open'],
                'High': data['high'],
                'Low': data['low'],
                'Close': data['close'],
                'Volume': data['volume']
            }, index=pd.to_datetime(data['timestamp']))
            
            # Save to CSV
            filename = f"{symbol}_backtesting_data.csv"
            bt_data.to_csv(filename)
            print(f"   ✅ Saved {len(bt_data)} days to {filename}")
            print(f"   Date range: {bt_data.index[0].date()} to {bt_data.index[-1].date()}")

def main():
    print("🚀 YFinance Integration Demo for Moon Dev AI Agents")
    print("=" * 70)
    print("No API keys required! Free, unlimited data for backtesting.\n")
    
    # Run all demos
    demo_backtesting_data()
    demo_multi_symbol_analysis()
    demo_trading_strategy()
    demo_save_for_backtesting()
    
    print("\n\n✅ YFinance Integration Complete!")
    print("=" * 70)
    print("💡 Key Benefits:")
    print("   • No authentication required")
    print("   • Works with stocks and crypto")
    print("   • Multiple timeframes (1m to daily)")
    print("   • Bulk data downloads")
    print("   • Perfect for backtesting")
    print("\n🚀 Your agents can now use YFinance as the primary data source!")

if __name__ == "__main__":
    main()