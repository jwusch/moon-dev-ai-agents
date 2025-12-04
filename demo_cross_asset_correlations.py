#!/usr/bin/env python3
"""
🌍 Cross-Asset Correlation Analysis Demo
Demonstrates VIX, Dollar, and Sector correlations for comprehensive market analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import yfinance as yf
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Import our correlation indicators
from src.fractal_alpha.indicators.correlations.vix_correlation import VIXCorrelationIndicator
from src.fractal_alpha.indicators.correlations.dollar_correlation import DollarCorrelationIndicator
from src.fractal_alpha.indicators.correlations.sector_correlation import SectorCorrelationIndicator


def fetch_market_data(symbols: list, days: int = 100) -> Dict[str, pd.DataFrame]:
    """Fetch market data for multiple symbols"""
    
    data = {}
    
    print(f"Fetching {days} days of data for analysis...")
    
    for symbol in symbols:
        print(f"  Downloading {symbol}...", end="", flush=True)
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d", interval="1d")
            
            if not hist.empty:
                # Clean column names
                hist.columns = [col.lower() for col in hist.columns]
                data[symbol] = hist
                print(" ✓")
            else:
                print(" ✗ (no data)")
        except Exception as e:
            print(f" ✗ ({str(e)})")
    
    return data


def analyze_correlations(asset_symbol: str = "SPY", asset_type: str = "equity"):
    """Perform comprehensive cross-asset correlation analysis"""
    
    print(colored(f"\n🌍 CROSS-ASSET CORRELATION ANALYSIS FOR {asset_symbol}", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Define symbols to fetch
    symbols = [asset_symbol, '^VIX', 'DX-Y.NYB']  # Asset, VIX, Dollar Index
    
    # Add sector ETFs
    sectors = ['XLK', 'XLF', 'XLE', 'XLU', 'XLY', 'XLV', 'XLRE', 'XLB']
    symbols.extend(sectors)
    
    # Fetch all data
    market_data = fetch_market_data(symbols, days=100)
    
    if asset_symbol not in market_data:
        print(colored(f"❌ Unable to fetch data for {asset_symbol}", "red"))
        return
    
    # Prepare asset data
    asset_data = market_data[asset_symbol]
    
    # 1. VIX Correlation Analysis
    print(colored("\n😱 1. VIX CORRELATION ANALYSIS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    vix_indicator = VIXCorrelationIndicator(
        lookback_days=20,
        correlation_window=20,
        detect_regime_shifts=True
    )
    
    if '^VIX' in market_data:
        vix_data_dict = {
            'asset': asset_data,
            'vix': market_data['^VIX']
        }
        vix_result = vix_indicator.calculate(vix_data_dict, asset_symbol)
    else:
        # Use simulated VIX
        print("  (Using simulated VIX based on realized volatility)")
        vix_result = vix_indicator.calculate(asset_data, asset_symbol)
    
    # Display VIX results
    print(f"VIX Level: {vix_result.metadata['vix_level']:.2f}")
    print(f"VIX Regime: {colored(vix_result.metadata['vix_regime'].upper(), 'red' if 'extreme' in vix_result.metadata['vix_regime'] else 'yellow')}")
    print(f"VIX Percentile: {vix_result.metadata['vix_percentile']:.1f}%")
    print(f"Current Correlation: {vix_result.metadata.get('current_correlation', 0):.3f}")
    
    if vix_result.metadata.get('correlation_anomaly', False):
        print(colored("⚠️ ANOMALOUS VIX CORRELATION DETECTED", "red"))
    
    if vix_result.metadata.get('correlation_breakdown', False):
        print(colored("🚨 VIX CORRELATION BREAKDOWN - REGIME CHANGE", "red", attrs=["bold"]))
    
    print(f"\nVIX Signal: {colored(vix_result.signal.value, 'green' if 'BUY' in vix_result.signal.value else 'red' if 'SELL' in vix_result.signal.value else 'white')}")
    print(f"VIX Confidence: {vix_result.confidence:.1f}%")
    
    # 2. Dollar Correlation Analysis
    print(colored("\n💵 2. DOLLAR CORRELATION ANALYSIS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    dollar_indicator = DollarCorrelationIndicator(
        lookback_days=20,
        correlation_window=20,
        detect_divergence=True,
        asset_type=asset_type
    )
    
    if 'DX-Y.NYB' in market_data:
        dollar_data_dict = {
            'asset': asset_data,
            'dollar': market_data['DX-Y.NYB']
        }
        dollar_result = dollar_indicator.calculate(dollar_data_dict, asset_symbol)
    else:
        # Use simulated dollar
        print("  (Using simulated Dollar Index)")
        dollar_result = dollar_indicator.calculate(asset_data, asset_symbol)
    
    # Display Dollar results
    print(f"Dollar Index Level: {dollar_result.metadata['dollar_level']:.2f}")
    print(f"Dollar Strength: {colored(dollar_result.metadata['dollar_strength'].upper(), 'green' if dollar_result.metadata['dollar_strength'] == 'strong' else 'red' if dollar_result.metadata['dollar_strength'] == 'weak' else 'yellow')}")
    print(f"Dollar Trend: {dollar_result.metadata['dollar_trend'].upper()}")
    print(f"Current Correlation: {dollar_result.metadata.get('current_correlation', 0):.3f}")
    print(f"Expected Correlation: {dollar_result.metadata.get('expected_correlation', 0):.3f}")
    
    if dollar_result.metadata.get('divergence_type', 'normal') != 'normal':
        div_type = dollar_result.metadata['divergence_type']
        print(colored(f"📊 {div_type.upper()} DETECTED", "yellow"))
    
    print(f"\nDollar Signal: {colored(dollar_result.signal.value, 'green' if 'BUY' in dollar_result.signal.value else 'red' if 'SELL' in dollar_result.signal.value else 'white')}")
    print(f"Dollar Confidence: {dollar_result.confidence:.1f}%")
    
    # 3. Sector Correlation Analysis
    print(colored("\n🏢 3. SECTOR CORRELATION ANALYSIS", "yellow", attrs=["bold"]))
    print("-" * 60)
    
    sector_indicator = SectorCorrelationIndicator(
        lookback_days=20,
        correlation_window=20,
        detect_rotation=True
    )
    
    # Prepare sector data
    sector_data = {s: market_data[s] for s in sectors if s in market_data}
    sector_data['asset'] = asset_data
    
    if len(sector_data) >= 4:  # Need at least a few sectors
        sector_result = sector_indicator.calculate(sector_data, asset_symbol)
        
        # Display Sector results
        print(f"Market Sentiment: {colored(sector_result.metadata['market_sentiment'].upper(), 'green' if 'risk_on' in sector_result.metadata['market_sentiment'] else 'red' if 'risk_off' in sector_result.metadata['market_sentiment'] else 'yellow')}")
        print(f"Market Condition: {sector_result.metadata['market_condition'].upper()}")
        print(f"Average Sector Correlation: {sector_result.metadata['avg_sector_correlation']:.3f}")
        print(f"Market Breadth: {sector_result.metadata.get('market_breadth', 0.5):.1%}")
        
        # Top sectors
        print("\nLeading Sectors:")
        for sector in sector_result.metadata.get('top_sectors', [])[:3]:
            color = 'green' if sector['return'] > 0 else 'red'
            print(f"  {sector['sector']}: {colored(f\"{sector['return']:+.1f}%\", color)}")
        
        # Rotation detection
        if sector_result.metadata.get('leadership_change', False):
            print(colored("\n🔄 SECTOR ROTATION DETECTED", "yellow"))
            print(f"Pattern: {sector_result.metadata.get('rotation_pattern', 'Unknown').upper()}")
        
        # Divergences
        divergences = sector_result.metadata.get('divergences', [])
        if divergences:
            print("\nSector Divergences:")
            for div in divergences:
                print(f"  ⚠️ {div.replace('_', ' ').title()}")
        
        print(f"\nSector Signal: {colored(sector_result.signal.value, 'green' if 'BUY' in sector_result.signal.value else 'red' if 'SELL' in sector_result.signal.value else 'white')}")
        print(f"Sector Confidence: {sector_result.confidence:.1f}%")
    else:
        print("Insufficient sector data for analysis")
        sector_result = None
    
    # Combined Analysis
    print(colored("\n🎯 COMBINED CORRELATION ANALYSIS", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Aggregate signals
    signals = []
    if vix_result:
        signals.append((vix_result.signal.value, vix_result.confidence, "VIX"))
    if dollar_result:
        signals.append((dollar_result.signal.value, dollar_result.confidence, "Dollar"))
    if sector_result:
        signals.append((sector_result.signal.value, sector_result.confidence, "Sector"))
    
    # Calculate consensus
    buy_votes = sum(1 for s, _, _ in signals if s == "BUY")
    sell_votes = sum(1 for s, _, _ in signals if s == "SELL")
    
    avg_buy_conf = np.mean([c for s, c, _ in signals if s == "BUY"]) if buy_votes > 0 else 0
    avg_sell_conf = np.mean([c for s, c, _ in signals if s == "SELL"]) if sell_votes > 0 else 0
    
    if buy_votes > sell_votes:
        consensus = "BUY"
        consensus_confidence = avg_buy_conf
    elif sell_votes > buy_votes:
        consensus = "SELL"
        consensus_confidence = avg_sell_conf
    else:
        consensus = "HOLD"
        consensus_confidence = 0
    
    print(f"Consensus Signal: {colored(consensus, 'green' if consensus == 'BUY' else 'red' if consensus == 'SELL' else 'yellow', attrs=['bold'])}")
    print(f"Consensus Confidence: {consensus_confidence:.1f}%")
    print(f"Signal Agreement: {max(buy_votes, sell_votes)}/{len(signals)}")
    
    print("\nIndividual Signals:")
    for sig, conf, name in signals:
        print(f"  {name}: {colored(sig, 'white')} ({conf:.1f}%)")
    
    # Market Regime Summary
    print(colored("\n📊 MARKET REGIME SUMMARY", "yellow"))
    print("-" * 60)
    
    regimes = []
    
    # VIX regime
    if vix_result and vix_result.metadata['vix_regime'] == 'extreme_fear':
        regimes.append("😱 Extreme Fear (Contrarian Buy)")
    elif vix_result and vix_result.metadata['vix_regime'] == 'complacency':
        regimes.append("😴 Complacency (Caution)")
    
    # Dollar regime
    if dollar_result and dollar_result.metadata['macro_regime'] == 'dollar_surge':
        regimes.append("💵 Dollar Surge (Risk-Off)")
    elif dollar_result and dollar_result.metadata['macro_regime'] == 'weak_dollar':
        regimes.append("💸 Weak Dollar (Risk-On)")
    
    # Sector regime
    if sector_result and sector_result.metadata['market_sentiment'] == 'risk_on':
        regimes.append("🚀 Risk-On Sentiment")
    elif sector_result and sector_result.metadata['market_sentiment'] == 'risk_off':
        regimes.append("🛡️ Risk-Off Sentiment")
    
    for regime in regimes:
        print(f"• {regime}")
    
    # Key Correlations Summary
    print(colored("\n🔗 KEY CORRELATIONS", "yellow"))
    print("-" * 60)
    
    if vix_result:
        vix_corr = vix_result.metadata.get('current_correlation', 0)
        print(f"Asset-VIX Correlation: {vix_corr:.3f} ", end="")
        if vix_corr > -0.3:
            print(colored("(BREAKDOWN - Unusual!)", "red"))
        elif vix_corr < -0.8:
            print(colored("(Strong inverse - Normal)", "green"))
        else:
            print("(Moderate inverse)")
    
    if dollar_result:
        dollar_corr = dollar_result.metadata.get('current_correlation', 0)
        expected = dollar_result.metadata.get('expected_correlation', 0)
        print(f"Asset-Dollar Correlation: {dollar_corr:.3f} (Expected: {expected:.3f})")
    
    if sector_result:
        avg_corr = sector_result.metadata.get('avg_sector_correlation', 0)
        print(f"Average Sector Correlation: {avg_corr:.3f} ", end="")
        if avg_corr > 0.8:
            print(colored("(Macro-driven market)", "yellow"))
        elif avg_corr < 0.3:
            print(colored("(Stock picking market)", "green"))
        else:
            print("(Normal)")
    
    # Trading Recommendations
    print(colored("\n💡 TRADING RECOMMENDATIONS", "green"))
    print("-" * 60)
    
    recommendations = []
    
    # VIX-based recommendations
    if vix_result:
        if vix_result.metadata['vix_regime'] == 'extreme_fear' and vix_result.metadata['vix_percentile'] > 90:
            recommendations.append("📈 VIX extreme suggests contrarian long opportunity")
        elif vix_result.metadata.get('correlation_breakdown', False):
            recommendations.append("⚠️ VIX correlation breakdown - expect volatility")
    
    # Dollar-based recommendations
    if dollar_result:
        if dollar_result.metadata.get('divergence_type') == 'bullish_divergence':
            recommendations.append("📈 Bullish dollar divergence detected")
        elif dollar_result.metadata['dollar_percentile'] > 90:
            recommendations.append("💵 Dollar at extremes - watch for reversal")
    
    # Sector-based recommendations
    if sector_result:
        if sector_result.metadata.get('rotation_pattern') == 'risk_off_to_risk_on':
            recommendations.append("🔄 Rotation to risk-on detected - bullish")
        elif sector_result.metadata.get('breadth_quality') == 'weak':
            recommendations.append("⚠️ Weak market breadth - be selective")
    
    if consensus == "BUY" and consensus_confidence > 65:
        recommendations.append("✅ Strong cross-asset BUY signal")
    elif consensus == "SELL" and consensus_confidence > 65:
        recommendations.append("❌ Strong cross-asset SELL signal")
    else:
        recommendations.append("⏸️ Mixed signals - wait for clarity")
    
    for rec in recommendations:
        print(f"• {rec}")


if __name__ == "__main__":
    import sys
    
    # Default to SPY if no symbol provided
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    # Determine asset type
    if symbol in ['GLD', 'SLV', 'USO', 'DBA']:
        asset_type = 'commodity'
    elif symbol in ['FXE', 'FXY', 'UUP']:
        asset_type = 'currency'
    elif symbol in ['BTC-USD', 'ETH-USD']:
        asset_type = 'crypto'
    else:
        asset_type = 'equity'
    
    try:
        analyze_correlations(symbol, asset_type)
    except Exception as e:
        print(colored(f"\n❌ Error: {e}", "red"))
        print("Make sure you have yfinance installed: pip install yfinance")