"""
🚀💎 QUICK PENNY VOLATILITY SCAN 💎🚀
Fast scan of top volatile penny stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from termcolor import colored
import json

def quick_volatility_scan():
    """Quick scan of known volatile pennies"""
    
    print(colored("🚀 QUICK PENNY VOLATILITY SCAN", 'cyan', attrs=['bold']))
    print("=" * 50)
    
    # Focus on most active volatile pennies
    hot_pennies = [
        'WKHS', 'CENN', 'MNTS', 'HYLN', 'SNDL', 'BNGO', 
        'BLNK', 'CHPT', 'ATAI', 'OGI', 'CGC', 'CAN',
        'BTBT', 'SOS', 'EBON', 'COSM', 'INDO', 'NAK',
        'BARK', 'TALK', 'NCTY', 'LAC', 'IONQ', 'ASTS'
    ]
    
    results = []
    
    for symbol in hot_pennies:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if len(hist) < 10:
                continue
                
            # Quick metrics
            current_price = hist['Close'].iloc[-1]
            
            # Skip if over $5
            if current_price > 5:
                continue
                
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Intraday range
            intraday = ((hist['High'] - hist['Low']) / hist['Low']).mean()
            
            # Recent momentum
            momentum_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            
            # Volume
            avg_volume = hist['Volume'].mean()
            
            if volatility > 0.05:  # 5%+ daily volatility
                results.append({
                    'symbol': symbol,
                    'price': current_price,
                    'volatility': volatility,
                    'intraday_range': intraday,
                    'momentum_5d': momentum_5d,
                    'volume': avg_volume
                })
                
                print(f"✅ {symbol}: ${current_price:.3f} | Vol: {volatility:.1%} | 5D: {momentum_5d:+.1f}%")
                
        except:
            continue
    
    # Sort by volatility
    results.sort(key=lambda x: x['volatility'], reverse=True)
    
    print(colored("\n🏆 TOP VOLATILITY PLAYS:", 'yellow', attrs=['bold']))
    
    for i, stock in enumerate(results[:5], 1):
        print(f"\n{i}. {stock['symbol']} @ ${stock['price']:.3f}")
        print(f"   Daily Volatility: {stock['volatility']:.1%}")
        print(f"   Avg Intraday Range: {stock['intraday_range']:.1%}")
        print(f"   5-Day Momentum: {stock['momentum_5d']:+.1f}%")
        
        # Calculate swing potential
        daily_swing = stock['price'] * stock['volatility']
        print(colored(f"   💰 Daily swing: ${daily_swing:.3f} per share", 'green'))
        
        # Show profit potential
        shares_1k = 1000 / stock['price']
        daily_profit = shares_1k * daily_swing
        print(f"   📊 $1,000 position = {shares_1k:.0f} shares")
        print(f"   💸 Potential daily profit: ${daily_profit:.0f}")
    
    # Save results
    filename = f"penny_volatility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({'results': results, 'scan_time': datetime.now().isoformat()}, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    quick_volatility_scan()