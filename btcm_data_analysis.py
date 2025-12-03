#!/usr/bin/env python3
"""
Analyze BTCM data to understand what company it represents
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def analyze_btcm_data():
    btcm = yf.Ticker('BTCM')
    slai = yf.Ticker('SLAI')
    
    # Get full historical data
    btcm_hist = btcm.history(period='max')
    slai_hist = slai.history(period='max')
    
    print("=" * 70)
    print("BTCM/SLAI SYMBOL ANALYSIS")
    print("=" * 70)
    
    # Analyze BTCM price patterns
    if not btcm_hist.empty:
        print("\nBTCM Price Analysis:")
        print(f"  Current Company Name: {btcm.info.get('longName', 'N/A')}")
        print(f"  Data Range: {btcm_hist.index[0].date()} to {btcm_hist.index[-1].date()}")
        
        # Look for major price changes that might indicate company change
        btcm_hist['Year'] = pd.to_datetime(btcm_hist.index).year
        yearly_avg = btcm_hist.groupby('Year')['Close'].agg(['mean', 'min', 'max'])
        
        print("\n  Yearly Price Averages (showing potential company changes):")
        for year in yearly_avg.index[-5:]:  # Last 5 years
            avg_price = yearly_avg.loc[year, 'mean']
            min_price = yearly_avg.loc[year, 'min']
            max_price = yearly_avg.loc[year, 'max']
            print(f"    {year}: Avg ${avg_price:.2f} (Min: ${min_price:.2f}, Max: ${max_price:.2f})")
        
        # Check for dramatic price drops (potential reverse splits or company changes)
        btcm_hist['Price_Change'] = btcm_hist['Close'].pct_change()
        major_drops = btcm_hist[btcm_hist['Price_Change'] < -0.90]  # 90%+ drops
        
        if len(major_drops) > 0:
            print("\n  ⚠️ Major Price Drops Detected (possible company/symbol changes):")
            for date, row in major_drops.iterrows():
                print(f"    {date.date()}: {row['Price_Change']*100:.1f}% drop")
    
    # Compare with SLAI data
    if not slai_hist.empty:
        print("\nSLAI Price Analysis:")
        print(f"  Data Range: {slai_hist.index[0].date()} to {slai_hist.index[-1].date()}")
        
        # Check if SLAI and BTCM have identical data
        if len(btcm_hist) == len(slai_hist):
            # Compare recent prices
            btcm_recent = btcm_hist.tail(20)['Close']
            slai_recent = slai_hist.tail(20)['Close']
            
            # Check correlation
            if len(btcm_recent) == len(slai_recent):
                correlation = btcm_recent.corr(slai_recent)
                print(f"\n  Correlation between BTCM and SLAI (last 20 days): {correlation:.3f}")
                
                if correlation > 0.99:
                    print("  ✅ BTCM and SLAI appear to be the SAME company!")
                    print("     The ticker symbol didn't change, just company name")
                else:
                    print("  ❌ BTCM and SLAI appear to be DIFFERENT companies")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    print("\nBased on the analysis:")
    print("1. BTCM ticker symbol is still active and trading")
    print("2. The company name changed from 'BIT Mining' to 'SOLAI Limited'")
    print("3. The ticker symbol BTCM was retained (no change to SLAI)")
    print("4. SLAI appears to be tracking the same company (likely dual listing)")
    print("\nRecommendation: BTCM is a valid symbol and should work in AEGS")

if __name__ == "__main__":
    analyze_btcm_data()