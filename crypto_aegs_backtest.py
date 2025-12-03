"""
🔥💎 CRYPTO AEGS BACKTEST - BTC & ETH 💎🔥
Testing Bitcoin and Ethereum with Alpha Ensemble Goldmine Strategy

Testing crypto assets for golden opportunities using AEGS
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored
from comprehensive_qqq_backtest import ComprehensiveBacktester, ComprehensiveBacktestResults

class CryptoAEGSBacktest:
    """Test BTC and ETH with AEGS strategy"""
    
    def __init__(self):
        # Test crypto symbols with USD pairs
        self.crypto_symbols = [
            'BTC-USD',  # Bitcoin
            'ETH-USD',  # Ethereum
        ]
        
        # Additional high-volume crypto if time permits
        self.additional_crypto = [
            'SOL-USD',  # Solana - high volatility
            'MATIC-USD',  # Polygon
            'AVAX-USD',  # Avalanche
            'LINK-USD',  # Chainlink
            'XRP-USD',  # Ripple
            'ADA-USD',  # Cardano
            'DOGE-USD',  # Dogecoin - meme volatility
            'SHIB-USD',  # Shiba Inu - extreme volatility
        ]
        
        # Also test crypto stocks for comparison
        self.crypto_stocks = [
            'COIN',  # Coinbase
            'MARA',  # Marathon Digital (already tested)
            'RIOT',  # Riot Platforms
            'CLSK',  # CleanSpark
            'MSTR',  # MicroStrategy - Bitcoin proxy
            'GBTC',  # Grayscale Bitcoin Trust
        ]
        
    def test_crypto_symbols(self):
        """Test crypto with AEGS"""
        
        print(colored("🔥💎 CRYPTO AEGS BACKTEST - BTC & ETH FOCUS 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        print("Testing if AEGS can generate massive returns on crypto...")
        print("=" * 80)
        
        all_results = []
        
        # Test primary crypto first (BTC, ETH)
        print(colored("\n📊 PRIMARY CRYPTO TEST (BTC & ETH):", 'yellow', attrs=['bold']))
        print("-" * 60)
        
        for symbol in self.crypto_symbols:
            result = self._test_symbol(symbol, "Primary Crypto")
            if result:
                all_results.append(result)
        
        # Test additional crypto if successful
        print(colored("\n📊 ADDITIONAL HIGH-VOLUME CRYPTO:", 'yellow'))
        print("-" * 60)
        
        for symbol in self.additional_crypto:
            result = self._test_symbol(symbol, "Alt Crypto")
            if result:
                all_results.append(result)
        
        # Test crypto stocks
        print(colored("\n📊 CRYPTO STOCK PROXIES:", 'yellow'))
        print("-" * 60)
        
        for symbol in self.crypto_stocks:
            result = self._test_symbol(symbol, "Crypto Stock")
            if result:
                all_results.append(result)
        
        # Analyze results
        self._analyze_crypto_results(all_results)
        
        return all_results
    
    def _test_symbol(self, symbol: str, category: str):
        """Test individual crypto symbol"""
        
        print(f"\n🔍 Testing {symbol} ({category})...", end='', flush=True)
        
        try:
            # Initialize backtester
            backtester = ComprehensiveBacktester(symbol)
            
            # Download maximum data
            df = backtester.download_maximum_data()
            
            if df is None or len(df) < 500:  # Need decent history
                print(f" ❌ Insufficient data (need 500+ days)")
                return None
            
            print(f" ✅ {len(df)} days of data")
            
            # Run comprehensive backtest
            results = backtester.comprehensive_backtest(df)
            
            # Calculate key metrics
            excess_return = results.excess_return_pct
            strategy_return = results.strategy_total_return_pct
            buy_hold_return = results.buy_hold_total_return_pct
            win_rate = results.win_rate
            total_trades = results.total_trades
            years = results.total_years
            sharpe = results.strategy_sharpe
            
            # Print immediate results
            if excess_return > 1000:
                print(colored(f"   🔥💎 GOLDMINE: {excess_return:+,.0f}% excess!", 'red', attrs=['bold']))
                print(colored(f"   💰 $10k → ${10000 * (1 + strategy_return/100):,.0f}", 'green'))
            elif excess_return > 100:
                print(colored(f"   🚀 HIGH: {excess_return:+.0f}% excess", 'yellow'))
            elif excess_return > 0:
                print(colored(f"   ✅ Positive: {excess_return:+.1f}% excess", 'green'))
            else:
                print(colored(f"   ❌ Negative: {excess_return:.1f}% excess", 'red'))
            
            # Additional metrics for winners
            print(f"   📊 Strategy: {strategy_return:+.0f}% | Buy&Hold: {buy_hold_return:+.0f}%")
            print(f"   📊 Win Rate: {win_rate:.1f}% | Trades: {total_trades} | Years: {years:.1f}")
            
            # Special analysis for BTC/ETH
            if symbol in ['BTC-USD', 'ETH-USD']:
                print(colored(f"\n   💡 {symbol} DEEP ANALYSIS:", 'cyan'))
                
                # Check if mean reversion works on crypto
                if excess_return > 0:
                    print(f"      ✅ Mean reversion WORKS on {symbol}!")
                    print(f"      📈 Despite crypto's trending nature, AEGS found {total_trades} profitable reversions")
                else:
                    print(f"      ❌ Mean reversion struggles with {symbol}'s strong trends")
                    print(f"      💡 Consider trend-following for crypto instead")
                
                # Volatility analysis
                annual_vol = results.strategy_volatility
                print(f"      📊 Annual Volatility: {annual_vol:.1f}%")
                
                if annual_vol > 100:
                    print(f"      🔥 EXTREME volatility - perfect for AEGS!")
                elif annual_vol > 50:
                    print(f"      ⚡ High volatility - good for mean reversion")
                else:
                    print(f"      📉 Lower volatility than expected")
            
            return {
                'symbol': symbol,
                'category': category,
                'excess_return': excess_return,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'years': years,
                'sharpe': sharpe,
                'volatility': results.strategy_volatility,
                'results_object': results
            }
            
        except Exception as e:
            print(f" ❌ Error: {str(e)[:50]}...")
            return None
    
    def _analyze_crypto_results(self, results):
        """Comprehensive analysis of crypto results"""
        
        if not results:
            print("\n❌ No successful crypto backtests")
            return
        
        print("\n" + "=" * 80)
        print(colored("💎 CRYPTO AEGS ANALYSIS - FINAL RESULTS", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        # Sort by excess return
        results.sort(key=lambda x: x['excess_return'], reverse=True)
        
        # Find goldmines
        goldmines = [r for r in results if r['excess_return'] > 1000]
        high_potential = [r for r in results if 100 < r['excess_return'] <= 1000]
        positive = [r for r in results if 0 < r['excess_return'] <= 100]
        negative = [r for r in results if r['excess_return'] <= 0]
        
        # Primary crypto results
        btc_result = next((r for r in results if r['symbol'] == 'BTC-USD'), None)
        eth_result = next((r for r in results if r['symbol'] == 'ETH-USD'), None)
        
        print(colored("\n🔥 PRIMARY CRYPTO RESULTS:", 'cyan', attrs=['bold']))
        print("-" * 60)
        
        if btc_result:
            self._print_crypto_result("BITCOIN (BTC-USD)", btc_result)
        
        if eth_result:
            self._print_crypto_result("ETHEREUM (ETH-USD)", eth_result)
        
        # Category summary
        print(colored("\n📊 CATEGORY BREAKDOWN:", 'yellow'))
        print(f"   💎 Goldmines (>1000%): {len(goldmines)}")
        print(f"   🚀 High Potential (100-1000%): {len(high_potential)}")
        print(f"   ✅ Positive (<100%): {len(positive)}")
        print(f"   ❌ Negative: {len(negative)}")
        
        # Top performers
        if goldmines:
            print(colored("\n🔥💎 CRYPTO GOLDMINES DISCOVERED:", 'red', attrs=['bold']))
            for r in goldmines:
                print(f"   {r['symbol']}: {r['excess_return']:+,.0f}% excess | "
                      f"${10000 * (1 + r['strategy_return']/100):,.0f} from $10k")
        
        if high_potential:
            print(colored("\n🚀 HIGH POTENTIAL CRYPTO:", 'yellow'))
            for r in high_potential[:5]:
                print(f"   {r['symbol']}: {r['excess_return']:+.0f}% excess | "
                      f"Win Rate: {r['win_rate']:.1f}%")
        
        # Crypto vs Stocks comparison
        crypto_direct = [r for r in results if r['category'] in ['Primary Crypto', 'Alt Crypto']]
        crypto_stocks = [r for r in results if r['category'] == 'Crypto Stock']
        
        if crypto_direct and crypto_stocks:
            avg_crypto = np.mean([r['excess_return'] for r in crypto_direct])
            avg_stocks = np.mean([r['excess_return'] for r in crypto_stocks])
            
            print(colored("\n💡 CRYPTO vs CRYPTO STOCKS:", 'cyan'))
            print(f"   Direct Crypto Avg: {avg_crypto:+.0f}% excess")
            print(f"   Crypto Stocks Avg: {avg_stocks:+.0f}% excess")
            
            if avg_stocks > avg_crypto:
                print(colored("   🎯 Crypto STOCKS outperform direct crypto for mean reversion!", 'green'))
            else:
                print(colored("   🎯 Direct crypto provides better opportunities!", 'green'))
        
        # Final recommendations
        print(colored("\n🎯 CRYPTO TRADING RECOMMENDATIONS:", 'white', attrs=['bold']))
        
        if btc_result and btc_result['excess_return'] > 0:
            print("   ✅ BTC mean reversion IS profitable with AEGS")
            print(f"   💰 Deploy on BTC dips of -10% or more")
        else:
            print("   ❌ BTC trends too strongly for mean reversion")
            print("   💡 Use trend-following for BTC instead")
        
        if eth_result and eth_result['excess_return'] > 0:
            print("   ✅ ETH shows mean reversion potential")
            print(f"   💰 Target ETH oversold conditions (RSI < 30)")
        
        best_crypto = results[0] if results else None
        if best_crypto and best_crypto['excess_return'] > 100:
            print(f"\n   🔥 BEST CRYPTO PLAY: {best_crypto['symbol']}")
            print(f"   💎 {best_crypto['excess_return']:+.0f}% excess return potential")
            print(f"   📊 {best_crypto['total_trades']} trades over {best_crypto['years']:.1f} years")
        
        # Summary stats
        all_excess = [r['excess_return'] for r in results]
        print(colored(f"\n📈 OVERALL CRYPTO PERFORMANCE:", 'cyan'))
        print(f"   Symbols Tested: {len(results)}")
        print(f"   Average Excess: {np.mean(all_excess):+.0f}%")
        print(f"   Best Performer: {results[0]['symbol']} ({results[0]['excess_return']:+.0f}%)")
        
        if goldmines or high_potential:
            print(colored("\n🚀 CRYPTO AEGS VERDICT: SUCCESS!", 'green', attrs=['bold']))
            print("   Mean reversion WORKS on volatile crypto assets!")
        else:
            print(colored("\n📊 CRYPTO AEGS VERDICT: MIXED", 'yellow'))
            print("   Some opportunities exist but crypto trends limit mean reversion")
    
    def _print_crypto_result(self, name: str, result: dict):
        """Print detailed crypto result"""
        
        print(f"\n{name}:")
        
        excess = result['excess_return']
        strategy = result['strategy_return']
        buyhold = result['buy_hold_return']
        
        if excess > 0:
            print(colored(f"   ✅ AEGS BEATS Buy & Hold by {excess:+.0f}%!", 'green'))
        else:
            print(colored(f"   ❌ Buy & Hold wins by {abs(excess):.0f}%", 'red'))
        
        print(f"   📊 Strategy Return: {strategy:+.0f}%")
        print(f"   📊 Buy & Hold Return: {buyhold:+.0f}%")
        print(f"   📊 Win Rate: {result['win_rate']:.1f}% | Trades: {result['total_trades']}")
        print(f"   📊 Years Tested: {result['years']:.1f} | Sharpe: {result['sharpe']:.2f}")
        print(f"   📊 Annual Volatility: {result['volatility']:.1f}%")
        
        # Investment calculation
        if strategy > 0:
            print(colored(f"   💰 $10,000 → ${10000 * (1 + strategy/100):,.0f}", 'green'))


def main():
    """Run crypto AEGS backtest"""
    
    print("🚀 Starting Crypto AEGS Backtest...")
    print("🎯 Focus: Bitcoin (BTC) and Ethereum (ETH)")
    print("💎 Testing if mean reversion works on crypto...\n")
    
    tester = CryptoAEGSBacktest()
    results = tester.test_crypto_symbols()
    
    print("\n✅ Crypto AEGS backtest complete!")
    print("💡 Check results for crypto trading opportunities!")


if __name__ == "__main__":
    main()