"""
🚀📉 IPO & SPAC BOOM/BUST CYCLE SCANNER 📉🚀
Find recent IPOs and SPACs in classic boom/bust patterns

Patterns to find:
1. IPO pop and drop (>50% decline from peak)
2. SPAC merger disasters 
3. Despac bounces
4. Redemption plays
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class IPOSPACScanner:
    """
    Scanner for IPO/SPAC boom bust cycles
    """
    
    def __init__(self):
        self.min_decline_from_peak = 0.50  # 50% decline
        self.ipo_lookback_days = 730  # 2 years
        
    def get_ipo_spac_universe(self):
        """Get recent IPOs and SPACs"""
        
        print("🔍 Building IPO/SPAC universe...")
        
        # 2023-2024 IPOs
        recent_ipos = [
            'ARM', 'KVUE', 'CART', 'SOLV', 'VNT', 'CAVA', 'INST',
            'KTTA', 'LNTH', 'MGIC', 'NUVL', 'PRMB', 'SMCI', 'TMDX',
            'VERV', 'VITL', 'WEAV', 'XMTR', 'ZETA', 'FIGS', 'DOCS',
            'RYAN', 'SGHC', 'TH', 'VTEX', 'YOU', 'BRZE', 'COUR'
        ]
        
        # Recent DeSPACs (completed mergers)
        despacs = [
            # EV/Mobility
            'LCID', 'RIVN', 'NKLA', 'GOEV', 'FSR', 'ARVL', 'REE',
            'PTRA', 'LEV', 'CHPT', 'BLNK', 'EVGO', 'VLTA', 'DCFC',
            'WKHS', 'HYLN', 'XL', 'AYRO', 'SOLO', 'FUV', 'KNDI',
            
            # Space/Aviation
            'SPCE', 'ASTR', 'RKLB', 'MNTS', 'BKSY', 'PL', 'ASTS',
            'LUNR', 'RDW', 'JOBY', 'LILM', 'ACHR', 'EVTL', 'BLADE',
            
            # Tech/Software
             'DWAC', 'CFVI', 'GRAB', 'HOOD', 'SOFI', 'PSFE', 'PAYO',
            'OPEN', 'OPFI', 'ML', 'VIRT', 'MAPS', 'BIGC', 'HIPO',
            
            # Healthcare/Biotech
            'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS', 'DRUG',
            'TALK', 'HIMS', 'NUVB', 'ACCD', 'SLGC', 'TCRX', 'PEAR',
            
            # Consumer/Retail
            'BARK', 'BIRD', 'TBLA', 'SEAT', 'SKIN', 'BOWL', 'BODY',
            'VIEW', 'PRTS', 'VROOM', 'SFT', 'ROOT', 'MILE', 'LMND',
            
            # Fintech/Crypto
            'COIN', 'BAKKT', 'EQOS', 'MKTX', 'PAYO', 'FLNT', 'KATX',
            
            # Energy/Sustainability
            'CHPT', 'STEM', 'RMO', 'QS', 'MVST', 'SES', 'FREY',
            'ORGN', 'ZEV', 'HYLN', 'NRGV', 'AMPS', 'PCT', 'TPIC'
        ]
        
        # Combine and remove duplicates
        universe = list(set(recent_ipos + despacs))
        
        print(f"📊 Universe contains {len(universe)} IPOs and SPACs")
        
        return universe
    
    def analyze_boom_bust(self, symbol):
        """Analyze for boom/bust patterns"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get max available data
            hist = ticker.history(period="2y")
            
            if len(hist) < 30:
                return None
            
            # Current metrics
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Find peaks and troughs
            all_time_high = hist['High'].max()
            all_time_low = hist['Low'].min()
            
            # Recent peaks (last 6 months)
            recent_hist = hist.tail(126)  # ~6 months
            recent_high = recent_hist['High'].max()
            recent_low = recent_hist['Low'].min()
            
            # Calculate declines
            decline_from_ath = (current_price - all_time_high) / all_time_high
            decline_from_recent = (current_price - recent_high) / recent_high
            
            # Position in range
            range_position = (current_price - all_time_low) / (all_time_high - all_time_low)
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Volume analysis
            avg_volume = hist['Volume'].tail(20).mean()
            volume_trend = hist['Volume'].tail(5).mean() / hist['Volume'].tail(20).mean()
            
            # Momentum
            momentum_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            momentum_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
            
            # RSI for oversold detection
            rsi = self._calculate_rsi(hist['Close'])
            
            # Boom/Bust Score
            boom_bust_score = 0
            patterns = []
            
            # Major decline from ATH
            if decline_from_ath < -0.80:
                boom_bust_score += 40
                patterns.append(f"Crashed {abs(decline_from_ath)*100:.0f}% from ATH")
            elif decline_from_ath < -0.60:
                boom_bust_score += 25
                patterns.append(f"Down {abs(decline_from_ath)*100:.0f}% from ATH")
            elif decline_from_ath < -0.40:
                boom_bust_score += 15
                patterns.append(f"Declined {abs(decline_from_ath)*100:.0f}% from peak")
            
            # Near bottom
            if range_position < 0.20:
                boom_bust_score += 20
                patterns.append("Near all-time lows")
            
            # Oversold RSI
            if rsi < 30:
                boom_bust_score += 15
                patterns.append(f"Oversold RSI {rsi:.0f}")
            elif rsi < 40:
                boom_bust_score += 10
                patterns.append(f"Low RSI {rsi:.0f}")
            
            # High volatility
            if volatility > 0.05:
                boom_bust_score += 10
                patterns.append(f"High volatility {volatility:.1%}")
            
            # Recent bounce
            if momentum_5d > 10:
                boom_bust_score += 15
                patterns.append(f"Bouncing +{momentum_5d:.0f}% (5d)")
            
            # Volume surge
            if volume_trend > 1.5:
                boom_bust_score += 10
                patterns.append(f"Volume surge {volume_trend:.1f}x")
            
            if boom_bust_score >= 40:
                # Get company info
                info = ticker.info
                name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
                
                # Determine type
                ipo_type = 'SPAC' if any(x in name.upper() for x in ['ACQUISITION', 'CORP', 'HOLDINGS']) else 'IPO'
                
                return {
                    'symbol': symbol,
                    'name': name,
                    'type': ipo_type,
                    'sector': sector,
                    'price': current_price,
                    'ath': all_time_high,
                    'atl': all_time_low,
                    'decline_from_ath': decline_from_ath * 100,
                    'range_position': range_position,
                    'rsi': rsi,
                    'volatility': volatility,
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    'volume_trend': volume_trend,
                    'boom_bust_score': boom_bust_score,
                    'patterns': patterns
                }
            
        except Exception as e:
            return None
        
        return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def scan_all(self):
        """Scan all IPOs and SPACs"""
        
        print("\n🚀 Scanning for boom/bust opportunities...")
        print("=" * 60)
        
        universe = self.get_ipo_spac_universe()
        results = []
        
        for i, symbol in enumerate(universe, 1):
            print(f"[{i}/{len(universe)}] Analyzing {symbol}...", end='\r')
            
            result = self.analyze_boom_bust(symbol)
            if result:
                results.append(result)
                print(f"[{i}/{len(universe)}] ✅ {symbol}: Score {result['boom_bust_score']}")
        
        # Sort by boom bust score
        results.sort(key=lambda x: x['boom_bust_score'], reverse=True)
        
        return results
    
    def generate_report(self, results):
        """Generate boom/bust cycle report"""
        
        if not results:
            print("\n❌ No boom/bust opportunities found")
            return
        
        print(colored(f"\n📉 BOOM/BUST CYCLE REPORT 📈", 'red', attrs=['bold']))
        print("=" * 80)
        print(f"Found {len(results)} potential reversal candidates")
        
        # Top opportunities
        print(colored("\n🏆 TOP BOOM/BUST OPPORTUNITIES:", 'yellow', attrs=['bold']))
        
        for i, stock in enumerate(results[:10], 1):
            print(f"\n{i}. {stock['symbol']} - {stock['name']}")
            print(f"   Type: {stock['type']} | Sector: {stock['sector']}")
            print(f"   Score: {stock['boom_bust_score']}/100")
            print(f"   Current: ${stock['price']:.2f} | ATH: ${stock['ath']:.2f} | ATL: ${stock['atl']:.2f}")
            print(colored(f"   Decline: {stock['decline_from_ath']:.0f}% from peak", 'red'))
            print(f"   Patterns: {', '.join(stock['patterns'][:3])}")
            
            # Recovery potential
            recovery_to_50pct = stock['ath'] * 0.5
            recovery_gain = (recovery_to_50pct - stock['price']) / stock['price'] * 100
            
            if recovery_gain > 100:
                print(colored(f"   💎 Recovery to 50% of ATH = {recovery_gain:.0f}% gain!", 'green'))
        
        # Category breakdown
        print(colored("\n📊 BREAKDOWN BY TYPE:", 'green', attrs=['bold']))
        
        ipos = [s for s in results if s['type'] == 'IPO']
        spacs = [s for s in results if s['type'] == 'SPAC']
        
        print(f"\n🆕 IPOs: {len(ipos)}")
        if ipos:
            top_ipo = max(ipos, key=lambda x: abs(x['decline_from_ath']))
            print(f"   Biggest crash: {top_ipo['symbol']} down {abs(top_ipo['decline_from_ath']):.0f}%")
        
        print(f"\n🏢 SPACs: {len(spacs)}")
        if spacs:
            top_spac = max(spacs, key=lambda x: abs(x['decline_from_ath']))
            print(f"   Biggest crash: {top_spac['symbol']} down {abs(top_spac['decline_from_ath']):.0f}%")
        
        # Sector analysis
        print(colored("\n🏭 SECTORS WITH MOST CASUALTIES:", 'magenta', attrs=['bold']))
        
        sector_counts = {}
        for stock in results:
            sector = stock['sector']
            if sector not in sector_counts:
                sector_counts[sector] = 0
            sector_counts[sector] += 1
        
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        
        for sector, count in sorted_sectors[:5]:
            print(f"   {sector}: {count} stocks")
        
        # Trading ideas
        print(colored("\n💡 TRADING STRATEGIES:", 'cyan', attrs=['bold']))
        
        print("\n1. DEAD CAT BOUNCE PLAYS:")
        bounce_candidates = [s for s in results if s['momentum_5d'] > 5 and s['decline_from_ath'] < -70]
        print(f"   - {len(bounce_candidates)} stocks bouncing from >70% decline")
        if bounce_candidates:
            examples = [s['symbol'] for s in bounce_candidates[:3]]
            print(f"   - Examples: {', '.join(examples)}")
        
        print("\n2. OVERSOLD REVERSALS:")
        oversold = [s for s in results if s['rsi'] < 30]
        print(f"   - {len(oversold)} stocks with RSI < 30")
        if oversold:
            examples = [f"{s['symbol']} (RSI {s['rsi']:.0f})" for s in oversold[:3]]
            print(f"   - Examples: {', '.join(examples)}")
        
        print("\n3. VOLUME ACCUMULATION:")
        volume_plays = [s for s in results if s['volume_trend'] > 1.5]
        print(f"   - {len(volume_plays)} stocks with volume surging")
        
        # Save report
        report = {
            'scan_date': datetime.now().isoformat(),
            'total_found': len(results),
            'categories': {
                'ipos': len(ipos),
                'spacs': len(spacs)
            },
            'top_opportunities': results[:20]
        }
        
        filename = f"ipo_spac_boom_bust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Full report saved to: {filename}")
        
        print(colored("\n⚠️  RISK WARNING ⚠️", 'red', attrs=['bold']))
        print("Many IPOs/SPACs go to ZERO. These are speculative plays!")
        print("Use stop losses and position sizing!")
        
        return report


def main():
    """Run IPO/SPAC boom bust scanner"""
    
    print(colored("🚀📉 IPO & SPAC BOOM/BUST SCANNER 📉🚀", 'cyan', attrs=['bold']))
    print("Finding crashed IPOs ready for reversal...")
    print("=" * 60)
    
    scanner = IPOSPACScanner()
    
    # Quick scan of known crashers
    quick_scan = [
        'LCID', 'RIVN', 'NKLA', 'GOEV', 'SPCE', 'HOOD', 
        'SOFI', 'GRAB', 'BIRD', 'OPEN', 'BARK', 'ASTR',
        'QS', 'CHPT', 'STEM', 'PSFE', 'WISH', 'CLOV'
    ]
    
    print("\n⚡ Quick scan of notorious boom/bust stocks...")
    
    results = []
    for symbol in quick_scan:
        result = scanner.analyze_boom_bust(symbol)
        if result:
            results.append(result)
            print(f"✅ {symbol}: {result['decline_from_ath']:.0f}% decline | Score: {result['boom_bust_score']}")
    
    # Generate report
    if results:
        results.sort(key=lambda x: x['boom_bust_score'], reverse=True)
        scanner.generate_report(results)
    
    print("\n✅ Scan complete!")


if __name__ == "__main__":
    main()