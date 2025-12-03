"""
🎬💎 ACTIVE PENNY STOCK SCANNER FOR HMNY-STYLE MOVES 💎🎬
Focus on currently trading penny stocks with massive potential
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

def get_active_penny_stocks():
    """Get currently active penny stocks under $5"""
    
    print("🔍 Finding active penny stocks...")
    
    # These are verified active penny stocks as of 2024
    active_pennies = [
        # Crypto/Blockchain pennies
        'BTBT', 'CAN', 'SOS', 'EBON', 'NCTY', 'LGHL', 'RIOT', 'MARA',
        # EV/Clean Energy
        'GOEV', 'NKLA', 'WKHS', 'HYLN', 'CHPT', 'BLNK', 'EVGO',
        # Biotech pennies
        'BNGO', 'ATER', 'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL',
        # Tech/Space
        'ASTR', 'MNTS', 'BKSY', 'PL', 'ASTS', 'RKLB', 'IONQ',
        # Cannabis
        'SNDL', 'OGI', 'TLRY', 'CGC', 'CRON', 'ACB',
        # Retail/Consumer
        'BBBY', 'CENN', 'BARK', 'TALK', 'OPEN', 'PSFE',
        # Mining/Commodities
        'BTG', 'HL', 'FSM', 'GPL', 'NAK', 'PLM',
        # Financial/Fintech
        'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'QFIN',
        # Recent volatility plays
        'COSM', 'VERU', 'BTCM', 'XELA', 'CEI', 'INDO'
    ]
    
    # Filter for actual penny stocks
    valid_pennies = []
    
    for symbol in active_pennies:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                avg_volume = hist['Volume'].mean()
                
                # Must be under $5 with 100k+ volume
                if current_price <= 5.0 and avg_volume > 100000:
                    valid_pennies.append({
                        'symbol': symbol,
                        'price': current_price,
                        'volume': avg_volume
                    })
                    print(f"   ✅ {symbol}: ${current_price:.2f} | Vol: {avg_volume:,.0f}")
                    
        except:
            continue
    
    print(f"\nFound {len(valid_pennies)} active penny stocks")
    return valid_pennies


def analyze_hmny_potential(penny_stocks):
    """Analyze penny stocks for HMNY-style explosive potential"""
    
    print(colored("\n🎬 ANALYZING HMNY EXPLOSION POTENTIAL 🎬", 'cyan', attrs=['bold']))
    print("=" * 60)
    
    hmny_candidates = []
    
    for stock in penny_stocks:
        symbol = stock['symbol']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get 6 months of data
            hist = ticker.history(period="6mo")
            
            if len(hist) < 60:
                continue
            
            # Current metrics
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Calculate key HMNY indicators
            
            # 1. Price compression (lower = more potential)
            price_52w_high = hist['High'].max()
            price_compression = current_price / price_52w_high
            
            # 2. Volume explosion potential
            avg_volume_60d = hist['Volume'].iloc[-60:].mean()
            avg_volume_10d = hist['Volume'].iloc[-10:].mean()
            volume_trend = avg_volume_10d / avg_volume_60d
            
            # 3. Volatility (need high volatility)
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # 4. Recent momentum
            momentum_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            momentum_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
            
            # 5. Volume spikes
            volume_spikes = (hist['Volume'] > avg_volume_60d * 3).sum()
            
            # HMNY Score Calculation
            score = 0
            factors = []
            
            # Ultra-cheap bonus (HMNY was $0.08)
            if current_price < 0.50:
                score += 40
                factors.append(f"Ultra-penny ${current_price:.2f}")
            elif current_price < 1.00:
                score += 25
                factors.append(f"Sub-$1 @ ${current_price:.2f}")
            elif current_price < 2.00:
                score += 15
                factors.append(f"Penny @ ${current_price:.2f}")
            
            # Crushed from highs (setup for reversal)
            if price_compression < 0.10:
                score += 30
                factors.append(f"Down {(1-price_compression)*100:.0f}% from high")
            elif price_compression < 0.25:
                score += 15
                factors.append(f"Beaten down {(1-price_compression)*100:.0f}%")
            
            # Volume increasing
            if volume_trend > 2.0:
                score += 25
                factors.append(f"Volume surge {volume_trend:.1f}x")
            elif volume_trend > 1.5:
                score += 10
                factors.append(f"Volume up {volume_trend:.1f}x")
            
            # High volatility
            if volatility > 2.0:
                score += 20
                factors.append(f"Extreme volatility {volatility:.0f}%")
            elif volatility > 1.0:
                score += 10
                factors.append(f"High volatility {volatility:.0f}%")
            
            # Recent pops
            if momentum_5d > 20:
                score += 15
                factors.append(f"5-day pop +{momentum_5d:.0f}%")
            
            # Multiple volume spikes
            if volume_spikes > 10:
                score += 15
                factors.append(f"{volume_spikes} volume spikes")
            
            if score >= 50:
                # Get company info
                info = ticker.info
                name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
                
                hmny_candidates.append({
                    'symbol': symbol,
                    'name': name,
                    'sector': sector,
                    'price': current_price,
                    'volume': current_volume,
                    '52w_high': price_52w_high,
                    'compression': price_compression,
                    'volatility': volatility,
                    'volume_trend': volume_trend,
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    'score': score,
                    'factors': factors
                })
                
                print(f"\n🎯 {symbol} - HMNY Score: {score}/100")
                print(f"   {name}")
                print(f"   Factors: {', '.join(factors)}")
                
        except Exception as e:
            continue
    
    return hmny_candidates


def calculate_moonshot_targets(candidates):
    """Calculate potential returns for HMNY-style moves"""
    
    if not candidates:
        print("\n❌ No HMNY-style candidates found")
        return
    
    print(colored("\n💰 MOONSHOT CALCULATIONS 💰", 'yellow', attrs=['bold']))
    print("=" * 60)
    print("HMNY went from $0.08 → $38.86 (48,475% gain)")
    print("=" * 60)
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    for i, c in enumerate(candidates[:5], 1):
        print(f"\n#{i}. {c['symbol']} @ ${c['price']:.3f}")
        print(f"   Score: {c['score']}/100")
        print(f"   52W High: ${c['52w_high']:.2f} (currently {c['compression']*100:.0f}% of high)")
        
        # Calculate various scenarios
        scenarios = [
            ("Back to 52W High", c['52w_high']),
            ("10x Move", c['price'] * 10),
            ("100x Move", c['price'] * 100),
            ("HMNY-Style (485x)", c['price'] * 485),
            ("1000x Moonshot", c['price'] * 1000)
        ]
        
        print("\n   📊 Price Targets & Returns:")
        for scenario, target in scenarios:
            gain = ((target - c['price']) / c['price']) * 100
            
            # Skip if target is less than current
            if target <= c['price']:
                continue
                
            print(f"\n   {scenario}: ${target:.2f} (+{gain:,.0f}%)")
            
            # Investment scenarios
            for investment in [100, 500, 1000, 5000]:
                returns = investment * (target / c['price'])
                if returns >= 1_000_000:
                    print(f"      ${investment:,} → ${returns:,.0f} 🚀 MILLIONAIRE!")
                elif returns >= 100_000:
                    print(f"      ${investment:,} → ${returns:,.0f} 💎")
                else:
                    print(f"      ${investment:,} → ${returns:,.0f}")
    
    # Save report
    report = {
        'scan_date': datetime.now().isoformat(),
        'strategy': 'HMNY Trillion Dollar Scanner',
        'total_candidates': len(candidates),
        'candidates': candidates
    }
    
    filename = f"hmny_moonshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Full report saved to: {filename}")
    
    print(colored("\n⚠️  EXTREME RISK WARNING ⚠️", 'red', attrs=['bold']))
    print("HMNY went from $38.86 → $0.02 (99.95% loss)")
    print("Most penny stocks go to ZERO. Only invest what you can lose!")
    
    return candidates


def main():
    """Run the HMNY moonshot scanner"""
    
    print(colored("🎬💎 HMNY MOONSHOT SCANNER 💎🎬", 'cyan', attrs=['bold']))
    print("Finding the next 48,475% runner...")
    print("=" * 60)
    
    # Get active penny stocks
    penny_stocks = get_active_penny_stocks()
    
    if penny_stocks:
        # Analyze for HMNY potential
        candidates = analyze_hmny_potential(penny_stocks)
        
        # Calculate moonshot targets
        calculate_moonshot_targets(candidates)
    else:
        print("\n❌ No active penny stocks found")


if __name__ == "__main__":
    main()