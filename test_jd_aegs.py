#!/usr/bin/env python3
"""
🎯 TEST JD SYMBOL WITH AEGS SCANNER 🎯
Single symbol test to demonstrate the full AEGS analysis process
"""

from working_cached_aegs_scanner import WorkingCacheAEGS
import json
from datetime import datetime

def test_jd_symbol():
    """Test JD symbol with full AEGS analysis"""
    
    print("🎯 TESTING JD SYMBOL WITH AEGS SCANNER")
    print("=" * 60)
    print("📊 This will show you exactly what happens during AEGS analysis")
    
    # Initialize scanner
    scanner = WorkingCacheAEGS()
    
    symbol = "JD"
    print(f"\n🔍 ANALYZING SYMBOL: {symbol}")
    print("=" * 40)
    
    # Step 1: Get data
    print("📥 STEP 1: Fetching data...")
    df = scanner.get_cached_data(symbol)
    
    if df is None:
        print(f"❌ No data available for {symbol}")
        return None
    
    print(f"✅ Data retrieved: {len(df)} bars")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Current price: ${df.iloc[-1]['Close']:.2f}")
    
    # Step 2: Price filter
    current_price = df.iloc[-1]['Close']
    print(f"\n💰 STEP 2: Price filter...")
    print(f"   Current price: ${current_price:.2f}")
    
    if current_price < 1.0:
        print(f"❌ REJECTED: Price ${current_price:.2f} < $1.00 minimum")
        return None
    else:
        print(f"✅ PASSED: Price ${current_price:.2f} ≥ $1.00")
    
    # Step 3: Run backtest
    print(f"\n🧮 STEP 3: Running AEGS backtest...")
    print("   Calculating indicators:")
    print("   • RSI (14-period)")
    print("   • Bollinger Bands (20-period, 2 std)")  
    print("   • Volume SMA (20-period)")
    print("   • AEGS entry signals...")
    
    result = scanner.backtest_aegs_strategy(df.copy())
    
    if result is None:
        print(f"❌ BACKTEST FAILED: No profitable trades found")
        print(f"   Possible reasons:")
        print(f"   • No AEGS entry signals triggered")
        print(f"   • All trades were unprofitable") 
        print(f"   • Insufficient data for analysis")
        return None
    
    print(f"✅ BACKTEST COMPLETE!")
    
    # Step 4: Analyze results
    print(f"\n📊 STEP 4: AEGS Results Analysis")
    print("=" * 40)
    
    strategy_return = result['strategy_return']
    total_trades = result['total_trades']
    win_rate = result['win_rate']
    avg_win = result.get('avg_win', 0)
    avg_loss = result.get('avg_loss', 0)
    
    print(f"💰 STRATEGY PERFORMANCE:")
    print(f"   Strategy Return: {strategy_return:+.1f}%")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Average Win: {avg_win:+.1f}%")
    print(f"   Average Loss: {avg_loss:+.1f}%")
    
    # Step 5: Trade details
    if 'trades' in result and result['trades']:
        print(f"\n📈 TRADE HISTORY:")
        print("   Entry Date    Exit Date     Entry Price  Exit Price   Return   Days  Exit Reason")
        print("   " + "-" * 80)
        
        for i, trade in enumerate(result['trades'][:10], 1):  # Show first 10 trades
            entry_date = trade['entry_date']
            exit_date = trade['exit_date'] 
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            return_pct = trade['return_pct']
            days_held = trade['days_held']
            exit_reason = trade['exit_reason']
            
            print(f"   {entry_date:<12} {exit_date:<12} ${entry_price:<10.2f} ${exit_price:<10.2f} {return_pct:+6.1f}%  {days_held:3}   {exit_reason}")
        
        if len(result['trades']) > 10:
            print(f"   ... and {len(result['trades']) - 10} more trades")
    
    # Step 6: Final verdict
    print(f"\n🎯 FINAL AEGS VERDICT:")
    print("=" * 40)
    
    if strategy_return > 0:
        print(f"✅ PROFITABLE GOLDMINE!")
        print(f"   {symbol} would be added to goldmine registry")
        
        # Categorization
        if strategy_return >= 100:
            category = "EXTREME GOLDMINE"
            emoji = "💀"
        elif strategy_return >= 30:
            category = "HIGH POTENTIAL"
            emoji = "🔥"
        else:
            category = "POSITIVE"
            emoji = "✅"
        
        print(f"   Category: {emoji} {category}")
        print(f"   Strategy Return: {strategy_return:+.1f}%")
    else:
        print(f"❌ NOT PROFITABLE")
        print(f"   {symbol} would NOT be added to goldmine")
        print(f"   Strategy Return: {strategy_return:+.1f}%")
    
    # Step 7: Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"jd_aegs_analysis_{timestamp}.json"
    
    detailed_result = {
        'symbol': symbol,
        'analysis_date': datetime.now().isoformat(),
        'current_price': current_price,
        'data_bars': len(df),
        'date_range': {
            'start': df.index[0].isoformat(),
            'end': df.index[-1].isoformat()
        },
        'aegs_result': result
    }
    
    with open(filename, 'w') as f:
        json.dump(detailed_result, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved: {filename}")
    
    return result

def main():
    """Run JD AEGS test"""
    print("🚀 Starting JD AEGS Analysis...")
    
    result = test_jd_symbol()
    
    if result:
        print(f"\n🎉 JD ANALYSIS COMPLETE!")
        print(f"   This is exactly what happens for every symbol in our scans")
    else:
        print(f"\n📊 JD ANALYSIS COMPLETE!")
        print(f"   Symbol did not meet AEGS profitability criteria")
    
    print(f"\n💡 This demonstrates the full AEGS process:")
    print(f"   1. Data retrieval (cache or download)")
    print(f"   2. Price filtering")
    print(f"   3. Technical indicator calculation") 
    print(f"   4. Signal detection and backtesting")
    print(f"   5. Profitability analysis")
    print(f"   6. Goldmine categorization")

if __name__ == "__main__":
    main()