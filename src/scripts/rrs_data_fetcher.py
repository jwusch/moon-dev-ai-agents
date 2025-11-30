"""
🌙 Moon Dev's RRS (Real Relative Strength) Data Fetcher
Calculates Real Relative Strength comparing asset vs benchmark
Built with love by Moon Dev 🚀

WHAT IS RRS?
============
Real Relative Strength (RRS) measures how an asset performs RELATIVE to a benchmark,
normalized by volatility (ATR). Unlike RSI which is absolute momentum, RRS is comparative.

FORMULA:
    RRS = (Asset_Price_Change / Asset_ATR) - (Benchmark_Price_Change / Benchmark_ATR)

INTERPRETATION:
    RRS > 0  → Asset is OUTPERFORMING the benchmark (bullish)
    RRS < 0  → Asset is UNDERPERFORMING the benchmark (bearish)
    RRS rising → Asset gaining relative strength
    RRS falling → Asset losing relative strength

USAGE:
    python src/scripts/rrs_data_fetcher.py

    Or import and use programmatically:
        from src.scripts.rrs_data_fetcher import calculate_rrs, fetch_rrs_data
        rrs_data = fetch_rrs_data("SOL", benchmark="BTC", timeframe="15m", days_back=7)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import cprint
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import *
from src import nice_funcs as n

# Try to import hyperliquid functions (optional)
try:
    from src import nice_funcs_hyperliquid as hl
    HAS_HYPERLIQUID = True
except ImportError:
    HAS_HYPERLIQUID = False
    cprint("⚠️ HyperLiquid functions not available, using Solana/Birdeye only", "yellow")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default benchmark for crypto (use BTC as the "SPY" of crypto)
DEFAULT_CRYPTO_BENCHMARK = "BTC"

# RRS calculation periods
RRS_LOOKBACK = 12      # Bars to look back for price change
ATR_PERIOD = 14        # ATR calculation period
RRS_MA_PERIOD = 20     # Moving average period for RRS smoothing

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "rrs_data"

# ============================================================================
# CORE RRS CALCULATION
# ============================================================================

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR)

    ATR measures volatility by looking at the range of price movement.
    Used to normalize price changes in RRS calculation.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR lookback period (default 14)

    Returns:
        Series: ATR values
    """
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    close = df['Close'] if 'Close' in df.columns else df['close']

    # True Range = max of:
    # 1. Current High - Current Low
    # 2. abs(Current High - Previous Close)
    # 3. abs(Current Low - Previous Close)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the smoothed average of True Range
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_price_change(df, period=12):
    """
    Calculate rolling price change over N periods

    Args:
        df: DataFrame with Close column
        period: Lookback period for price change

    Returns:
        Series: Price change values
    """
    close = df['Close'] if 'Close' in df.columns else df['close']
    return close - close.shift(period)


def calculate_rrs(asset_df, benchmark_df, lookback=RRS_LOOKBACK, atr_period=ATR_PERIOD):
    """
    Calculate Real Relative Strength (RRS)

    THE CORE FORMULA:
        RRS = (Asset_Price_Change / Asset_ATR) - (Benchmark_Price_Change / Benchmark_ATR)

    This normalizes price changes by volatility, so a $100 move in BTC
    is comparable to a $1 move in a smaller altcoin.

    Args:
        asset_df: DataFrame with OHLCV for the asset
        benchmark_df: DataFrame with OHLCV for the benchmark
        lookback: Period for price change calculation
        atr_period: Period for ATR calculation

    Returns:
        DataFrame: Asset data with RRS columns added
    """
    cprint(f"📊 Calculating RRS with {lookback}-bar lookback and {atr_period}-period ATR", "cyan")

    # Make copies to avoid modifying originals
    asset = asset_df.copy()
    benchmark = benchmark_df.copy()

    # Ensure both DataFrames have the same length (align by index)
    min_len = min(len(asset), len(benchmark))
    asset = asset.tail(min_len).reset_index(drop=True)
    benchmark = benchmark.tail(min_len).reset_index(drop=True)

    # Calculate ATR for both
    asset['ATR'] = calculate_atr(asset, atr_period)
    benchmark['Benchmark_ATR'] = calculate_atr(benchmark, atr_period)

    # Calculate price changes
    asset['Price_Change'] = calculate_price_change(asset, lookback)
    benchmark['Benchmark_Price_Change'] = calculate_price_change(benchmark, lookback)

    # Calculate normalized price changes (price change / ATR)
    # This makes moves comparable across different volatility levels
    asset['Normalized_Change'] = asset['Price_Change'] / asset['ATR']
    benchmark['Benchmark_Normalized_Change'] = benchmark['Benchmark_Price_Change'] / benchmark['Benchmark_ATR']

    # THE RRS FORMULA
    # Positive = asset outperforming benchmark
    # Negative = asset underperforming benchmark
    asset['RRS'] = asset['Normalized_Change'] - benchmark['Benchmark_Normalized_Change']

    # Add smoothed RRS (moving average for trend identification)
    asset['RRS_MA'] = asset['RRS'].rolling(window=RRS_MA_PERIOD).mean()

    # Add RRS momentum (is RRS rising or falling?)
    asset['RRS_Momentum'] = asset['RRS'] - asset['RRS'].shift(1)

    # Add signal column for easy backtesting
    # 1 = RRS positive and rising (strong outperformance)
    # -1 = RRS negative and falling (strong underperformance)
    # 0 = neutral
    asset['RRS_Signal'] = 0
    asset.loc[(asset['RRS'] > 0) & (asset['RRS_Momentum'] > 0), 'RRS_Signal'] = 1
    asset.loc[(asset['RRS'] < 0) & (asset['RRS_Momentum'] < 0), 'RRS_Signal'] = -1

    # Add benchmark data for reference
    asset['Benchmark_Close'] = benchmark['Close'] if 'Close' in benchmark.columns else benchmark['close']

    cprint(f"✅ RRS calculation complete! {len(asset)} rows processed", "green")

    return asset


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_ohlcv_hyperliquid(symbol, timeframe="15m", days_back=7):
    """
    Fetch OHLCV data from HyperLiquid

    Args:
        symbol: Trading symbol (BTC, ETH, SOL, etc.)
        timeframe: Candle timeframe
        days_back: Days of historical data

    Returns:
        DataFrame: OHLCV data
    """
    if not HAS_HYPERLIQUID:
        cprint("❌ HyperLiquid not available", "red")
        return None

    cprint(f"🔄 Fetching {symbol} from HyperLiquid ({timeframe}, {days_back} days)", "cyan")

    # Calculate bars needed
    bars_per_day = {
        '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
        '1h': 24, '4h': 6, '1d': 1
    }
    bars = int(days_back * bars_per_day.get(timeframe, 96))

    # Convert timeframe format
    hl_timeframe = timeframe.replace('H', 'h').replace('D', 'd')

    data = hl.get_data(symbol=symbol, timeframe=hl_timeframe, bars=bars, add_indicators=False)

    if data is not None:
        cprint(f"✅ Got {len(data)} candles for {symbol}", "green")

    return data


def fetch_ohlcv_birdeye(address, timeframe="15m", days_back=7):
    """
    Fetch OHLCV data from Birdeye (Solana tokens)

    Args:
        address: Solana token address
        timeframe: Candle timeframe
        days_back: Days of historical data

    Returns:
        DataFrame: OHLCV data
    """
    cprint(f"🔄 Fetching {address[:8]}... from Birdeye ({timeframe}, {days_back} days)", "cyan")

    data = n.get_data(address, days_back, timeframe)

    if data is not None:
        cprint(f"✅ Got {len(data)} candles", "green")

    return data


def fetch_rrs_data(asset, benchmark=DEFAULT_CRYPTO_BENCHMARK, timeframe="15m",
                   days_back=7, exchange="HYPERLIQUID", save=True):
    """
    🌙 MAIN FUNCTION: Fetch data and calculate RRS

    This is the primary entry point for getting RRS data.

    Args:
        asset: Asset symbol (BTC, ETH, SOL) or Solana address
        benchmark: Benchmark symbol (default: BTC)
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        days_back: Days of historical data
        exchange: "HYPERLIQUID" or "SOLANA"
        save: Whether to save output to CSV

    Returns:
        DataFrame: OHLCV data with RRS columns

    Example:
        # Get SOL vs BTC relative strength
        rrs_data = fetch_rrs_data("SOL", benchmark="BTC", timeframe="15m", days_back=7)

        # Check if SOL is outperforming BTC
        latest_rrs = rrs_data['RRS'].iloc[-1]
        if latest_rrs > 0:
            print("SOL is outperforming BTC!")
    """
    cprint(f"\n🌙 Moon Dev's RRS Data Fetcher", "white", "on_blue")
    cprint(f"📈 Asset: {asset} | Benchmark: {benchmark}", "cyan")
    cprint(f"⏰ Timeframe: {timeframe} | Days: {days_back}", "cyan")

    # Fetch asset data
    if exchange == "HYPERLIQUID":
        asset_df = fetch_ohlcv_hyperliquid(asset, timeframe, days_back)
        benchmark_df = fetch_ohlcv_hyperliquid(benchmark, timeframe, days_back)
    else:
        # Assume Solana addresses
        asset_df = fetch_ohlcv_birdeye(asset, timeframe, days_back)
        # For Solana, need to provide BTC address or use a known one
        cprint("⚠️ For Solana tokens, benchmark should be a token address", "yellow")
        benchmark_df = fetch_ohlcv_birdeye(benchmark, timeframe, days_back)

    if asset_df is None or benchmark_df is None:
        cprint("❌ Failed to fetch data!", "red")
        return None

    # Calculate RRS
    rrs_data = calculate_rrs(asset_df, benchmark_df)

    # Save to file if requested
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{asset}_vs_{benchmark}_{timeframe}_{timestamp}.csv"
        filepath = OUTPUT_DIR / filename
        rrs_data.to_csv(filepath, index=False)
        cprint(f"💾 Saved to {filepath}", "green")

    return rrs_data


# ============================================================================
# SIGNAL GENERATION (for backtesting/live trading)
# ============================================================================

def get_rrs_signals(rrs_data):
    """
    Generate trading signals from RRS data

    SIGNAL LOGIC:
        BUY when:  RRS crosses above 0 (asset starting to outperform)
        SELL when: RRS crosses below 0 (asset starting to underperform)

    Additional filters:
        - RRS_MA slope for trend confirmation
        - RRS momentum for signal strength

    Args:
        rrs_data: DataFrame with RRS columns

    Returns:
        DataFrame: Signals with entry/exit points
    """
    df = rrs_data.copy()

    # Detect RRS zero crossovers
    df['RRS_Cross_Up'] = (df['RRS'] > 0) & (df['RRS'].shift(1) <= 0)
    df['RRS_Cross_Down'] = (df['RRS'] < 0) & (df['RRS'].shift(1) >= 0)

    # Detect RRS MA crossovers (RRS crossing its own moving average)
    df['RRS_MA_Cross_Up'] = (df['RRS'] > df['RRS_MA']) & (df['RRS'].shift(1) <= df['RRS_MA'].shift(1))
    df['RRS_MA_Cross_Down'] = (df['RRS'] < df['RRS_MA']) & (df['RRS'].shift(1) >= df['RRS_MA'].shift(1))

    # Generate signals
    # Entry: RRS crosses above 0 AND RRS > RRS_MA (confirmed strength)
    # Exit: RRS crosses below 0 OR RRS < RRS_MA (losing strength)
    df['Entry_Long'] = df['RRS_Cross_Up'] & (df['RRS'] > df['RRS_MA'])
    df['Exit_Long'] = df['RRS_Cross_Down'] | df['RRS_MA_Cross_Down']

    df['Entry_Short'] = df['RRS_Cross_Down'] & (df['RRS'] < df['RRS_MA'])
    df['Exit_Short'] = df['RRS_Cross_Up'] | df['RRS_MA_Cross_Up']

    return df


def print_latest_rrs(rrs_data, asset, benchmark):
    """
    Print a summary of the latest RRS values

    Args:
        rrs_data: DataFrame with RRS columns
        asset: Asset symbol
        benchmark: Benchmark symbol
    """
    latest = rrs_data.iloc[-1]

    cprint(f"\n{'='*50}", "yellow")
    cprint(f"🌙 RRS Summary: {asset} vs {benchmark}", "white", "on_blue")
    cprint(f"{'='*50}", "yellow")

    # Current RRS value
    rrs = latest['RRS']
    rrs_ma = latest['RRS_MA']
    momentum = latest['RRS_Momentum']

    # Determine status
    if rrs > 0:
        status = "OUTPERFORMING"
        color = "green"
    else:
        status = "UNDERPERFORMING"
        color = "red"

    if momentum > 0:
        trend = "↑ RISING"
    else:
        trend = "↓ FALLING"

    cprint(f"📊 RRS Value: {rrs:.4f}", color)
    cprint(f"📈 RRS MA(20): {rrs_ma:.4f}", "cyan")
    cprint(f"📉 Momentum: {momentum:.4f} ({trend})", "cyan")
    cprint(f"🎯 Status: {asset} is {status} {benchmark}", color)

    # Signal
    signal = latest['RRS_Signal']
    if signal == 1:
        cprint(f"✅ SIGNAL: LONG (strong outperformance)", "green")
    elif signal == -1:
        cprint(f"🔴 SIGNAL: SHORT (strong underperformance)", "red")
    else:
        cprint(f"⚪ SIGNAL: NEUTRAL", "white")

    cprint(f"{'='*50}\n", "yellow")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        cprint("\n🌙 Moon Dev's RRS Data Fetcher Starting!", "white", "on_magenta")

        # Example: Calculate RRS for SOL vs BTC
        # You can change these parameters as needed
        ASSET = "SOL"
        BENCHMARK = "BTC"
        TIMEFRAME = "15m"
        DAYS_BACK = 7

        cprint(f"\n📊 Calculating RRS for {ASSET} vs {BENCHMARK}", "cyan")
        cprint(f"⏰ Timeframe: {TIMEFRAME} | Days: {DAYS_BACK}\n", "cyan")

        # Fetch and calculate RRS
        rrs_data = fetch_rrs_data(
            asset=ASSET,
            benchmark=BENCHMARK,
            timeframe=TIMEFRAME,
            days_back=DAYS_BACK,
            exchange="HYPERLIQUID",
            save=True
        )

        if rrs_data is not None:
            # Print latest values
            print_latest_rrs(rrs_data, ASSET, BENCHMARK)

            # Generate signals
            signals = get_rrs_signals(rrs_data)

            # Show recent signals
            recent_entries = signals[signals['Entry_Long'] == True].tail(5)
            if len(recent_entries) > 0:
                cprint("📈 Recent LONG entry signals:", "green")
                for idx, row in recent_entries.iterrows():
                    dt = row.get('Datetime (UTC)', row.get('datetime', 'N/A'))
                    cprint(f"   {dt} | RRS: {row['RRS']:.4f}", "green")

            recent_exits = signals[signals['Exit_Long'] == True].tail(5)
            if len(recent_exits) > 0:
                cprint("📉 Recent LONG exit signals:", "red")
                for idx, row in recent_exits.iterrows():
                    dt = row.get('Datetime (UTC)', row.get('datetime', 'N/A'))
                    cprint(f"   {dt} | RRS: {row['RRS']:.4f}", "red")

        cprint("\n✨ Moon Dev's RRS Data Fetcher Complete!", "white", "on_green")

    except KeyboardInterrupt:
        cprint("\n👋 Moon Dev's RRS Fetcher shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red")
        import traceback
        traceback.print_exc()
