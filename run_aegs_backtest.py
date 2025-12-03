"""
🔥💎 RUN AEGS BACKTEST FOR ANY SYMBOL 💎🔥
Simple script to backtest any symbol with AEGS and auto-register if successful

Usage: python run_aegs_backtest.py SYMBOL [CATEGORY]
Example: python run_aegs_backtest.py AAPL "Tech Stock"
"""

import sys
from comprehensive_qqq_backtest import ComprehensiveBacktester
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

def run_aegs_backtest(symbol, category=None):
    """Run AEGS backtest on any symbol"""
    
    print(colored(f"\n🔥💎 AEGS BACKTEST FOR {symbol} 💎🔥", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    try:
        # Initialize backtester
        backtester = ComprehensiveBacktester(symbol)
        
        # Download data
        print(f"📊 Downloading data for {symbol}...")
        df = backtester.download_maximum_data()
        
        if df is None or len(df) < 500:
            print(colored(f"❌ Insufficient data for {symbol} (need 500+ days)", 'red'))
            return None
        
        print(f"✅ Downloaded {len(df)} days of data")
        print(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Run comprehensive backtest
        print(f"\n🚀 Running AEGS backtest...")
        results = backtester.comprehensive_backtest(df)
        
        # Display results
        print("\n" + "=" * 80)
        print(colored("📊 BACKTEST RESULTS", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        # Key metrics
        excess = results.excess_return_pct
        strategy_return = results.strategy_total_return_pct
        buyhold_return = results.buy_hold_total_return_pct
        win_rate = results.win_rate
        trades = results.total_trades
        sharpe = results.strategy_sharpe
        years = results.total_years
        
        # Determine success level
        if excess > 1000:
            status = "💎 EXTREME GOLDMINE!"
            color = 'red'
            emoji = "🔥"
        elif excess > 100:
            status = "🚀 HIGH POTENTIAL!"
            color = 'yellow'
            emoji = "⚡"
        elif excess > 10:
            status = "✅ POSITIVE"
            color = 'green'
            emoji = "📈"
        elif excess > 0:
            status = "📊 MARGINAL"
            color = 'blue'
            emoji = "➡️"
        else:
            status = "❌ UNDERPERFORMS"
            color = 'red'
            emoji = "📉"
        
        print(colored(f"\n{emoji} {status}", color, attrs=['bold']))
        print(f"\nExcess Return: {excess:+.1f}%")
        print(f"Strategy Return: {strategy_return:+.1f}%")
        print(f"Buy & Hold Return: {buyhold_return:+.1f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {trades}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Years Tested: {years:.1f}")
        
        # Investment calculation
        if strategy_return > 0:
            investment_result = 10000 * (1 + strategy_return/100)
            print(colored(f"\n💰 $10,000 → ${investment_result:,.0f}", 'cyan'))
        
        # Auto-register if successful
        if excess > 10:  # Only register if >10% excess return
            print(colored(f"\n🎯 Registering {symbol} in AEGS goldmine registry...", 'green'))
            
            # Use provided category or guess based on symbol
            if category is None:
                category = guess_category(symbol)
            
            success = backtester.auto_register_result(category)
            
            if success:
                print(colored(f"✅ {symbol} added to goldmine registry!", 'green', attrs=['bold']))
                print(f"📂 Category: {category}")
                print(f"🔍 Will be included in future scans")
            else:
                print("❌ Failed to register (may already exist)")
        else:
            print(colored(f"\n❌ {symbol} not added to registry (excess return too low)", 'yellow'))
        
        # Trading recommendation
        print("\n" + "=" * 80)
        print(colored("💡 TRADING RECOMMENDATION:", 'white', attrs=['bold']))
        
        if excess > 1000:
            print(f"🔥 DEPLOY CAPITAL IMMEDIATELY on {symbol} pullbacks!")
            print(f"   Target position: 3-5% of portfolio")
            print(f"   Entry strategy: Wait for RSI < 30 or -10% daily drop")
        elif excess > 100:
            print(f"🚀 ADD {symbol} to watch list for opportunities")
            print(f"   Target position: 1-3% of portfolio")
            print(f"   Entry strategy: Multiple oversold signals")
        elif excess > 0:
            print(f"📊 Consider {symbol} for small positions only")
            print(f"   Target position: 0.5-1% of portfolio")
        else:
            print(f"❌ AVOID {symbol} for mean reversion strategies")
            print(f"   Consider trend-following instead")
        
        return results
        
    except Exception as e:
        print(colored(f"❌ Error backtesting {symbol}: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()
        return None

def guess_category(symbol):
    """Guess category based on symbol patterns"""
    
    symbol_upper = symbol.upper()
    
    # Crypto patterns
    if '-USD' in symbol_upper or 'BTC' in symbol_upper or 'ETH' in symbol_upper:
        return 'Cryptocurrency'
    
    # Crypto mining
    if symbol_upper in ['MARA', 'RIOT', 'CLSK', 'CORZ', 'WULF', 'CIFR', 'BTDR']:
        return 'Crypto Mining'
    
    # Leveraged ETFs
    if any(x in symbol_upper for x in ['TQQQ', 'SQQQ', 'TNA', 'TZA', 'LABU', 'NUGT']):
        return 'Leveraged ETFs'
    
    # Inverse ETFs
    if symbol_upper in ['SH', 'PSQ', 'DOG', 'DXD', 'SDS', 'SPXU']:
        return 'Inverse/Defensive'
    
    # Energy
    if symbol_upper in ['XOM', 'CVX', 'COP', 'OXY', 'DVN', 'FANG', 'MRO', 'EQT']:
        return 'Energy Cycles'
    
    # Biotech
    if symbol_upper in ['MRNA', 'BNTX', 'SAVA', 'BIIB', 'EDIT', 'NTLA', 'CRSP']:
        return 'Biotech Events'
    
    # Cannabis
    if symbol_upper in ['TLRY', 'CGC', 'CRON', 'ACB', 'SNDL']:
        return 'Cannabis Cycles'
    
    # Default
    return 'Unknown'

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) < 2:
        print(colored("📋 USAGE:", 'yellow'))
        print("   python run_aegs_backtest.py SYMBOL [CATEGORY]")
        print("\n📋 EXAMPLES:")
        print("   python run_aegs_backtest.py AAPL")
        print("   python run_aegs_backtest.py TSLA \"EV Stock\"")
        print("   python run_aegs_backtest.py DOGE-USD \"Cryptocurrency\"")
        print("\n💡 TIP: Category is optional - script will guess if not provided")
        return
    
    symbol = sys.argv[1].upper()
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run backtest
    results = run_aegs_backtest(symbol, category)
    
    if results:
        print(colored(f"\n✅ AEGS backtest complete for {symbol}!", 'green'))
    else:
        print(colored(f"\n❌ AEGS backtest failed for {symbol}", 'red'))

if __name__ == "__main__":
    main()