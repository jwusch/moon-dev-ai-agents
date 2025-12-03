"""
🚀💎 PENNY STOCK EXTREME VOLATILITY SCANNER 💎🚀
Discover micro-caps with explosive volatility patterns

Focuses on:
- Stocks under $5
- Daily volatility >5%
- Volume spikes
- Momentum reversals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import concurrent.futures
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class PennyVolatilityScanner:
    """
    Scanner for extreme volatility penny stocks
    """
    
    def __init__(self):
        self.max_price = 5.0
        self.min_volume = 100000
        self.min_volatility = 0.05  # 5% daily
        self.max_workers = 10
        
    def get_penny_universe(self):
        """Get comprehensive penny stock universe"""
        
        print("🔍 Building penny stock universe...")
        
        penny_universe = [
            # Crypto/Blockchain
            'BTBT', 'CAN', 'SOS', 'EBON', 'NCTY', 'LGHL', 'BTCM', 'XNET', 'GRNQ',
            
            # EV/Clean Tech
            'WKHS', 'HYLN', 'GOEV', 'NKLA', 'CHPT', 'BLNK', 'EVGO', 'PTRA', 'LEV',
            'MVST', 'QS', 'FSR', 'ARVL', 'REE', 'PROTERRA', 'ELMS', 'SOLO', 'FUV',
            
            # Biotech/Healthcare
            'BNGO', 'ATER', 'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS', 'DRUG',
            'VERU', 'IMMP', 'OCGN', 'VXRT', 'INO', 'NVAX', 'SRNE', 'CYTK', 'KALA',
            
            # Tech/Space
            'ASTR', 'MNTS', 'BKSY', 'PL', 'ASTS', 'RKLB', 'IONQ', 'ARQQ', 'QBTS',
            'SPCE', 'LUNR', 'RDW', 'MAXN', 'VSAT', 'GILT', 'SWIR', 'IRDM',
            
            # Cannabis
            'SNDL', 'OGI', 'TLRY', 'CGC', 'CRON', 'ACB', 'HEXO', 'VFF', 'GRWG',
            
            # Mining/Commodities
            'NAK', 'LAC', 'MP', 'UUUU', 'DNN', 'NXE', 'CCJ', 'GOLD', 'HL', 'FSM',
            'BTG', 'AG', 'PAAS', 'CDE', 'EXK', 'GPL', 'PLM', 'NGD', 'AUY',
            
            # Recent Volatility Plays
            'CENN', 'COSM', 'INDO', 'MULN', 'FFIE', 'REV', 'APRN', 'BBBY', 'EXPR',
            'XELA', 'CEI', 'PROG', 'ATER', 'BBIG', 'NAKD', 'GNUS', 'OCUP', 'SESN',
            
            # Fintech/Digital
            'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'QFIN', 'PSFE', 'PAYO', 'FLNT',
            'MKTW', 'ML', 'VIRT', 'NRDS', 'TIGR', 'FUTU', 'COIN', 'MSTR',
            
            # SPACs/Recent IPOs
            'DWAC', 'PHUN', 'BENE', 'MARK', 'CFVI', 'TMTG', 'GREE', 'SPRT', 'IRNT',
            'OPAD', 'TMC', 'OPAL', 'HLBZ', 'ANGH', 'ARYD', 'CDLX', 'KSCP', 'LVRA',
            
            # Retail/Consumer
            'BARK', 'TALK', 'OPEN', 'WISH', 'JMIA', 'REAL', 'RVLV', 'QRTEA', 'OSTK',
            'W', 'CVNA', 'PRTS', 'VROOM', 'SFT', 'ROOT', 'MILE', 'LMND', 'HIPO',
            
            # Energy
            'INDO', 'IMPP', 'CEI', 'USWS', 'ENSV', 'REI', 'HUSA', 'SNMP', 'VTNR',
            'TRCH', 'MMAT', 'MDR', 'RIG', 'VAL', 'OIS', 'TDW', 'FTK', 'CRK'
        ]
        
        # Remove duplicates
        penny_universe = list(set(penny_universe))
        print(f"📊 Universe contains {len(penny_universe)} potential penny stocks")
        
        return penny_universe
    
    def analyze_stock(self, symbol):
        """Analyze a single stock for volatility patterns"""
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if len(hist) < 30:
                return None
            
            # Current price and volume
            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].tail(20).mean()
            
            # Price filter
            if current_price > self.max_price or avg_volume < self.min_volume:
                return None
            
            # Calculate volatility metrics
            returns = hist['Close'].pct_change().dropna()
            daily_volatility = returns.std()
            
            # Skip if not volatile enough
            if daily_volatility < self.min_volatility:
                return None
            
            # Advanced metrics
            # 1. Intraday volatility (High-Low range)
            hist['Intraday_Range'] = (hist['High'] - hist['Low']) / hist['Low']
            avg_intraday_vol = hist['Intraday_Range'].tail(20).mean()
            
            # 2. Volume volatility
            volume_volatility = hist['Volume'].pct_change().std()
            
            # 3. Gap analysis
            hist['Gap'] = hist['Open'] / hist['Close'].shift(1) - 1
            avg_gap = hist['Gap'].abs().mean()
            
            # 4. Momentum metrics
            momentum_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            momentum_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
            
            # 5. Price range from 52w high/low
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            range_position = (current_price - low_52w) / (high_52w - low_52w)
            
            # 6. Volume spike detection
            recent_volume = hist['Volume'].tail(5).mean()
            volume_spike = recent_volume / avg_volume
            
            # 7. Consecutive move detection
            last_5_days = hist['Close'].tail(5).pct_change().dropna()
            consecutive_ups = (last_5_days > 0).sum()
            consecutive_downs = (last_5_days < 0).sum()
            
            # Volatility score
            volatility_score = 0
            patterns = []
            
            # Extreme daily volatility
            if daily_volatility > 0.10:
                volatility_score += 40
                patterns.append(f"Extreme volatility {daily_volatility:.1%}")
            elif daily_volatility > 0.07:
                volatility_score += 25
                patterns.append(f"High volatility {daily_volatility:.1%}")
            else:
                volatility_score += 10
                patterns.append(f"Moderate volatility {daily_volatility:.1%}")
            
            # Intraday swings
            if avg_intraday_vol > 0.15:
                volatility_score += 20
                patterns.append(f"Huge intraday swings {avg_intraday_vol:.1%}")
            elif avg_intraday_vol > 0.10:
                volatility_score += 10
                patterns.append(f"Large daily ranges {avg_intraday_vol:.1%}")
            
            # Volume patterns
            if volume_spike > 3:
                volatility_score += 20
                patterns.append(f"Volume explosion {volume_spike:.1f}x")
            elif volume_spike > 2:
                volatility_score += 10
                patterns.append(f"Volume surge {volume_spike:.1f}x")
            
            # Momentum patterns
            if abs(momentum_5d) > 50:
                volatility_score += 15
                patterns.append(f"5-day rocket {momentum_5d:+.0f}%")
            elif abs(momentum_5d) > 25:
                volatility_score += 10
                patterns.append(f"Strong momentum {momentum_5d:+.0f}%")
            
            # Gap trading
            if avg_gap > 0.05:
                volatility_score += 10
                patterns.append(f"Gap trader {avg_gap:.1%} avg")
            
            # Position in range
            if range_position < 0.2:
                volatility_score += 5
                patterns.append("Near 52w low")
            elif range_position > 0.8:
                volatility_score += 5
                patterns.append("Near 52w high")
            
            if volatility_score >= 40:
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': avg_volume,
                    'daily_volatility': daily_volatility,
                    'intraday_volatility': avg_intraday_vol,
                    'volume_spike': volume_spike,
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    '52w_high': high_52w,
                    '52w_low': low_52w,
                    'range_position': range_position,
                    'avg_gap': avg_gap,
                    'volatility_score': volatility_score,
                    'patterns': patterns,
                    'last_updated': datetime.now().isoformat()
                }
            
        except Exception as e:
            return None
        
        return None
    
    def scan_universe(self):
        """Scan entire universe with parallel processing"""
        
        print("\n🚀 Scanning for extreme volatility patterns...")
        print("=" * 60)
        
        universe = self.get_penny_universe()
        results = []
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self.analyze_stock, symbol): symbol 
                                for symbol in universe}
            
            completed = 0
            total = len(universe)
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"[{completed}/{total}] ✅ {symbol}: Score {result['volatility_score']}")
                    else:
                        print(f"[{completed}/{total}] ⏭️  {symbol}: No match", end='\r')
                except:
                    print(f"[{completed}/{total}] ❌ {symbol}: Error", end='\r')
        
        # Sort by volatility score
        results.sort(key=lambda x: x['volatility_score'], reverse=True)
        
        return results
    
    def generate_report(self, results):
        """Generate comprehensive volatility report"""
        
        if not results:
            print("\n❌ No extreme volatility penny stocks found")
            return
        
        print(colored(f"\n🔥 EXTREME VOLATILITY REPORT 🔥", 'red', attrs=['bold']))
        print("=" * 80)
        print(f"Found {len(results)} explosive penny stocks")
        
        # Top 10 by score
        print(colored("\n🏆 TOP 10 VOLATILITY MONSTERS:", 'yellow', attrs=['bold']))
        
        for i, stock in enumerate(results[:10], 1):
            print(f"\n{i}. {stock['symbol']} @ ${stock['price']:.3f}")
            print(f"   Score: {stock['volatility_score']}/100")
            print(f"   Patterns: {', '.join(stock['patterns'][:3])}")
            print(f"   5-Day Move: {stock['momentum_5d']:+.1f}%")
            print(f"   Daily Vol: {stock['daily_volatility']:.1%} | Intraday: {stock['intraday_volatility']:.1%}")
            
            # Trading potential
            if stock['daily_volatility'] > 0.10:
                daily_swing = stock['price'] * stock['daily_volatility']
                print(colored(f"   💰 Daily swing potential: ${daily_swing:.3f} ({stock['daily_volatility']*100:.0f}%)", 'cyan'))
        
        # Category breakdown
        print(colored("\n📊 VOLATILITY CATEGORIES:", 'green', attrs=['bold']))
        
        extreme = [s for s in results if s['daily_volatility'] > 0.10]
        high = [s for s in results if 0.07 < s['daily_volatility'] <= 0.10]
        moderate = [s for s in results if s['daily_volatility'] <= 0.07]
        
        print(f"\n🔥 EXTREME (>10% daily): {len(extreme)} stocks")
        if extreme:
            symbols = [s['symbol'] for s in extreme[:5]]
            print(f"   Examples: {', '.join(symbols)}")
        
        print(f"\n⚡ HIGH (7-10% daily): {len(high)} stocks")
        if high:
            symbols = [s['symbol'] for s in high[:5]]
            print(f"   Examples: {', '.join(symbols)}")
        
        print(f"\n📈 MODERATE (5-7% daily): {len(moderate)} stocks")
        if moderate:
            symbols = [s['symbol'] for s in moderate[:5]]
            print(f"   Examples: {', '.join(symbols)}")
        
        # Trading strategies
        print(colored("\n💡 VOLATILITY TRADING STRATEGIES:", 'magenta', attrs=['bold']))
        print("\n1. DAY TRADING:")
        print("   - Focus on stocks with >15% intraday volatility")
        print("   - Use 15-min charts for entry/exit")
        print("   - Set stop loss at 5% below entry")
        print("   - Take profits at 10-20% gains")
        
        print("\n2. GAP TRADING:")
        gap_traders = [s for s in results if s['avg_gap'] > 0.05]
        print(f"   - {len(gap_traders)} stocks with >5% average gaps")
        print("   - Buy pre-market on gap downs")
        print("   - Sell into gap ups")
        
        print("\n3. MOMENTUM SCALPING:")
        momentum_plays = [s for s in results if abs(s['momentum_5d']) > 25]
        print(f"   - {len(momentum_plays)} stocks with >25% 5-day moves")
        print("   - Ride the trend with tight stops")
        print("   - Exit on first red day")
        
        # Save report
        report = {
            'scan_date': datetime.now().isoformat(),
            'total_found': len(results),
            'categories': {
                'extreme': len(extreme),
                'high': len(high),
                'moderate': len(moderate)
            },
            'top_volatility_stocks': results[:20]
        }
        
        filename = f"penny_volatility_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Full report saved to: {filename}")
        
        print(colored("\n⚠️  EXTREME RISK WARNING ⚠️", 'red', attrs=['bold', 'blink']))
        print("These stocks can move 50%+ in a day - IN EITHER DIRECTION!")
        print("Only trade with money you can afford to lose!")
        
        return report


def main():
    """Run penny stock volatility scanner"""
    
    print(colored("🚀💎 PENNY STOCK VOLATILITY SCANNER 💎🚀", 'cyan', attrs=['bold']))
    print("Hunting for explosive micro-cap opportunities...")
    print("=" * 60)
    
    scanner = PennyVolatilityScanner()
    
    # Scan universe
    results = scanner.scan_universe()
    
    # Generate report
    report = scanner.generate_report(results)
    
    print("\n✅ Scan complete!")


if __name__ == "__main__":
    main()