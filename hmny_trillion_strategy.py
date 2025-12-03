"""
🎬💎 HMNY TRILLION-DOLLAR STRATEGY 💎🎬
Hunt for the next MoviePass-style explosion

HMNY went from $0.08 to $38.86 (48,475% gain) in 2017
Then crashed to $0.02 (99.95% loss) by 2018

This strategy hunts for similar penny stocks with:
1. Disruptive business model hype
2. Massive retail interest
3. Short squeeze potential
4. Media attention catalyst
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class HMNYTrilliionStrategy:
    """
    Strategy to find the next HMNY-style mega runner
    """
    
    def __init__(self):
        self.penny_threshold = 5.0  # Under $5
        self.min_volume = 1000000   # 1M+ daily volume
        self.volatility_threshold = 0.10  # 10%+ daily volatility
        
    def find_hmny_candidates(self):
        """Find stocks with HMNY-like characteristics"""
        
        print(colored("🎬💎 HMNY TRILLION-DOLLAR HUNTER 💎🎬", 'cyan', attrs=['bold']))
        print("=" * 60)
        print("Hunting for the next 48,475% runner...")
        
        candidates = []
        
        # Penny stock universe with buzz potential
        penny_universe = [
            # Recent meme stocks under $5
            'MULN', 'FFIE', 'CENN', 'REV', 'APRN', 'BBBY', 'EXPR',
            # Crypto penny stocks
            'BTBT', 'CAN', 'SOS', 'EBON', 'LGHL', 'NCTY', 'BTCM',
            # EV/Battery penny stocks
            'GOEV', 'RIDE', 'NKLA', 'WKHS', 'HYLN', 'XL', 'AYRO',
            # Biotech penny stocks with catalyst potential
            'BNGO', 'JAGX', 'OCUP', 'PROG', 'SESN', 'ATER', 'GNUS',
            # Space/Tech penny stocks
            'ASTR', 'MNTS', 'BKSY', 'PL', 'ASTS', 'RKLB',
            # Cannabis penny stocks
            'SNDL', 'HEXO', 'OGI', 'HUGE', 'FIRE', 'WMD',
            # Recent IPO/SPAC disasters under $5
            'VIEW', 'OPEN', 'PSFE', 'BARK', 'BODY', 'TALK',
            # Squeeze candidates
            'COSM', 'VERU', 'PRTY', 'BLUE', 'RDBX', 'WEBR'
        ]
        
        print(f"\n🔍 Scanning {len(penny_universe)} penny stocks...")
        
        for symbol in penny_universe:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                
                if len(hist) < 20:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                avg_volume = hist['Volume'].tail(20).mean()
                
                # Must be under $5 with decent volume
                if current_price > self.penny_threshold or avg_volume < self.min_volume:
                    continue
                
                # Calculate volatility and momentum
                daily_returns = hist['Close'].pct_change()
                volatility = daily_returns.std()
                
                # Volume surge detection (like HMNY had)
                recent_volume = hist['Volume'].tail(5).mean()
                volume_spike = recent_volume / avg_volume
                
                # Price momentum
                week_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]
                month_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]
                
                # Scoring system for HMNY potential
                score = 0
                reasons = []
                
                # Ultra penny stock bonus
                if current_price < 1.0:
                    score += 30
                    reasons.append(f"Sub-$1 stock (${current_price:.2f})")
                elif current_price < 2.0:
                    score += 20
                    reasons.append(f"Sub-$2 stock (${current_price:.2f})")
                
                # Volume surge (critical for HMNY-style moves)
                if volume_spike > 5:
                    score += 40
                    reasons.append(f"Volume explosion {volume_spike:.1f}x")
                elif volume_spike > 2:
                    score += 20
                    reasons.append(f"Volume surge {volume_spike:.1f}x")
                
                # Volatility (need wild swings)
                if volatility > 0.15:
                    score += 30
                    reasons.append(f"Extreme volatility {volatility:.1%}")
                elif volatility > 0.10:
                    score += 15
                    reasons.append(f"High volatility {volatility:.1%}")
                
                # Recent momentum
                if week_return > 0.50:
                    score += 25
                    reasons.append(f"Weekly surge +{week_return:.0%}")
                elif week_return > 0.20:
                    score += 10
                    reasons.append(f"Weekly gain +{week_return:.0%}")
                
                # Beaten down with reversal potential
                if month_return < -0.50 and week_return > 0:
                    score += 20
                    reasons.append("Reversal from lows")
                
                if score >= 50:
                    # Get additional info
                    info = ticker.info
                    market_cap = info.get('marketCap', 0)
                    
                    candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'volume': avg_volume,
                        'volume_spike': volume_spike,
                        'volatility': volatility,
                        'week_return': week_return,
                        'month_return': month_return,
                        'market_cap': market_cap,
                        'score': score,
                        'reasons': reasons
                    })
                    
                    print(f"   ✅ {symbol}: Score {score}/100 - {reasons[0]}")
                    
            except Exception as e:
                continue
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
    
    def calculate_hmny_potential(self, candidates):
        """Calculate potential returns if these hit HMNY-style moves"""
        
        print(colored("\n💰 TRILLION-DOLLAR POTENTIAL CALCULATIONS 💰", 'yellow', attrs=['bold']))
        print("=" * 60)
        
        for i, c in enumerate(candidates[:5], 1):
            symbol = c['symbol']
            price = c['price']
            
            # HMNY scenarios
            scenarios = {
                'Conservative (100x)': price * 100,
                'Moderate (500x)': price * 500,
                'HMNY-style (48,475%)': price * 485.75,
                'Trillion-maker (10,000x)': price * 10000
            }
            
            print(f"\n#{i}. {symbol} @ ${price:.3f}")
            print(f"   Score: {c['score']}/100")
            print(f"   Reasons: {', '.join(c['reasons'])}")
            print("\n   📈 Potential Targets:")
            
            for scenario, target in scenarios.items():
                gain_pct = ((target - price) / price) * 100
                
                # Calculate investment returns
                investments = [100, 1000, 10000]
                print(f"\n   {scenario}: ${target:,.2f} (+{gain_pct:,.0f}%)")
                
                for inv in investments:
                    final_value = inv * (target / price)
                    if final_value >= 1_000_000:
                        print(f"      ${inv:,} → ${final_value:,.0f} 💎 MILLIONAIRE!")
                    else:
                        print(f"      ${inv:,} → ${final_value:,.0f}")
        
        return candidates
    
    def generate_trading_plan(self, candidates):
        """Generate specific trading plan for HMNY hunters"""
        
        if not candidates:
            return
        
        print(colored("\n🎯 HMNY HUNTING STRATEGY 🎯", 'green', attrs=['bold']))
        print("=" * 60)
        
        print("\n📋 TRADING RULES:")
        print("1. Position Sizing:")
        print("   - Risk only what you can afford to lose (99% loss possible)")
        print("   - Suggested: $100-$1,000 per position")
        print("   - Never more than 5% of portfolio")
        
        print("\n2. Entry Signals:")
        print("   - Volume spike >5x average ✅")
        print("   - Price under $2 preferred ✅")
        print("   - Breaking above 20-day high ✅")
        print("   - Social media buzz increasing ✅")
        
        print("\n3. Exit Strategy:")
        print("   - Take 25% off at 2x")
        print("   - Take 25% off at 5x")
        print("   - Take 25% off at 10x")
        print("   - Let 25% ride for moon shot")
        
        print("\n4. Risk Management:")
        print("   - Set stop loss at -50% (or none for true YOLO)")
        print("   - Never add to losing positions")
        print("   - If it goes 10x, your stop becomes break-even")
        
        # Save candidates
        report = {
            'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': 'HMNY Trillion Dollar Hunter',
            'candidates_found': len(candidates),
            'top_candidates': candidates[:10]
        }
        
        filename = f"hmny_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {filename}")
        
        return filename


def main():
    """Run the HMNY hunter"""
    
    print("WARNING: This strategy hunts extreme penny stocks!")
    print("99% of these will go to zero. Only invest what you can lose!\n")
    
    strategy = HMNYTrilliionStrategy()
    
    # Find candidates
    candidates = strategy.find_hmny_candidates()
    
    if candidates:
        # Calculate potential
        strategy.calculate_hmny_potential(candidates)
        
        # Generate trading plan
        strategy.generate_trading_plan(candidates)
        
        print(colored("\n🚀 HMNY HUNT COMPLETE! 🚀", 'cyan', attrs=['bold']))
        print(f"Found {len(candidates)} potential rockets")
        print("\nRemember: HMNY went from $0.08 to $38.86 to $0.02")
        print("These are lottery tickets, not investments!")
    else:
        print("\n❌ No HMNY-style candidates found today")
        print("The market might be too rational right now...")


if __name__ == "__main__":
    main()