#!/usr/bin/env python3
"""
🌊 Fractal Alpha Dashboard Quick Viewer
View dashboard data in terminal with nice formatting
"""

import json
import glob
from datetime import datetime
from termcolor import colored
import os

def view_latest_dashboard():
    """Find and display the latest dashboard file"""
    
    # Find all dashboard JSON files
    json_files = glob.glob('fractal_alpha_dashboard_*.json')
    
    if not json_files:
        print(colored("❌ No dashboard files found!", 'red'))
        print("Run 'python fractal_alpha_dashboard.py' to generate data first")
        return
    
    # Get the latest file
    latest_file = max(json_files)
    print(colored(f"📊 Loading: {latest_file}", 'cyan'))
    print("=" * 80)
    
    # Load data
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Calculate summary stats
    symbols = list(data.values())
    total_symbols = len(symbols)
    
    if total_symbols == 0:
        print(colored("⚠️ No symbol data in file", 'yellow'))
        return
    
    # Market overview
    print(colored("\n🌊 MARKET OVERVIEW", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    avg_hurst = sum(s['hurst'] for s in symbols) / total_symbols
    avg_liquidity = sum(s['liquidity_score'] for s in symbols) / total_symbols
    avg_opportunity = sum(s['opportunity_score'] for s in symbols) / total_symbols
    
    print(f"📊 Total Symbols: {total_symbols}")
    print(f"🧠 Average Hurst: {avg_hurst:.3f} ({get_regime_name(avg_hurst)})")
    print(f"💧 Average Liquidity: {avg_liquidity:.1f}/100")
    print(f"🎯 Average Opportunity Score: {avg_opportunity:.1f}/100")
    
    # Regime distribution
    print(colored("\n🧠 REGIME DISTRIBUTION", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    regime_counts = {}
    for symbol in symbols:
        regime = symbol['regime']['hurst_regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_symbols) * 100
        regime_name = regime.replace('_', ' ').title()
        print(f"  {regime_name}: {count} symbols ({percentage:.1f}%)")
    
    # Top opportunities
    print(colored("\n🚀 TOP OPPORTUNITIES", 'green', attrs=['bold']))
    print("=" * 80)
    
    opportunities = sorted(symbols, key=lambda x: x['opportunity_score'], reverse=True)[:10]
    
    if opportunities[0]['opportunity_score'] < 40:
        print(colored("⏸️ No high-scoring opportunities detected", 'blue'))
    else:
        for i, opp in enumerate(opportunities, 1):
            if opp['opportunity_score'] < 40:
                break
                
            symbol = opp['symbol']
            score = opp['opportunity_score']
            price = opp['price']
            hurst = opp['hurst']
            risk = opp['risk_level']
            category = opp['category']
            
            # Color code by score
            if score >= 70:
                color = 'green'
                marker = '🚀'
            elif score >= 50:
                color = 'yellow'
                marker = '✅'
            else:
                color = 'white'
                marker = '⚡'
            
            print(colored(f"\n{marker} #{i}. {symbol} @ ${price:.2f}", color, attrs=['bold']))
            print(f"   Category: {category}")
            print(f"   Score: {score}/100 | Hurst: {hurst:.3f} | Risk: {risk}")
            print(f"   Regime: {opp['regime']['hurst_regime'].replace('_', ' ').title()}")
            print(f"   Liquidity: {opp['liquidity_score']:.1f}/100")
    
    # Risk summary
    print(colored("\n⚠️ RISK SUMMARY", 'yellow', attrs=['bold']))
    print("=" * 80)
    
    risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    for symbol in symbols:
        risk_counts[symbol['risk_level']] = risk_counts.get(symbol['risk_level'], 0) + 1
    
    print(f"🟢 Low Risk: {risk_counts.get('LOW', 0)} symbols")
    print(f"🟡 Medium Risk: {risk_counts.get('MEDIUM', 0)} symbols")
    print(f"🔴 High Risk: {risk_counts.get('HIGH', 0)} symbols")
    
    # Categories summary
    print(colored("\n📈 CATEGORY PERFORMANCE", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    categories = {}
    for symbol in symbols:
        cat = symbol['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(symbol['opportunity_score'])
    
    for cat, scores in sorted(categories.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg_score = sum(scores) / len(scores)
        high_scores = sum(1 for s in scores if s >= 50)
        print(f"  {cat}: Avg Score {avg_score:.1f}/100 | High Signals: {high_scores}/{len(scores)}")
    
    # File info
    print(colored("\n📁 FILE INFO", 'grey', attrs=['bold']))
    print("=" * 80)
    print(f"File: {latest_file}")
    print(f"Size: {os.path.getsize(latest_file) / 1024:.1f} KB")
    
    # Get timestamp from any symbol
    if symbols:
        timestamp_str = symbols[0].get('timestamp', 'Unknown')
        print(f"Generated: {timestamp_str}")

def get_regime_name(hurst):
    """Get regime name from Hurst value"""
    if hurst < 0.35:
        return "Extreme Mean Reversion"
    elif hurst < 0.45:
        return "Mean Reverting"
    elif hurst < 0.55:
        return "Random Walk"
    elif hurst < 0.65:
        return "Trending"
    else:
        return "Strong Trending"

def list_all_dashboards():
    """List all available dashboard files"""
    json_files = sorted(glob.glob('fractal_alpha_dashboard_*.json'))
    
    print(colored("\n📂 Available Dashboard Files:", 'cyan'))
    print("=" * 80)
    
    if not json_files:
        print("No dashboard files found")
    else:
        for f in json_files:
            size = os.path.getsize(f) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"  {f} ({size:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Show available files
    list_all_dashboards()
    
    # View the latest dashboard
    view_latest_dashboard()
    
    print(colored("\n✨ For interactive viewing, open fractal_dashboard_viewer.html in your browser!", 'green'))
    print(colored("Or run: python serve_dashboard.py", 'green'))