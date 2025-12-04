"""
🔥💎 AEGS REGIME-AWARE SCANNER 💎🔥
Enhanced AEGS scanner with Hurst Exponent regime detection for adaptive signals

Combines traditional AEGS signals with market regime analysis for improved accuracy:
- Trending regimes (H > 0.5): Momentum signals prioritized
- Mean-reverting regimes (H < 0.5): Reversal signals prioritized  
- Regime transitions: Enhanced opportunity detection
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from termcolor import colored
import sys
import os

# Add path for fractal alpha indicators
sys.path.append('src')

try:
    from fractal_alpha.indicators.multifractal.hurst_exponent import HurstExponentIndicator
except ImportError:
    print("Warning: Could not import HurstExponentIndicator. Using simplified implementation.")
    HurstExponentIndicator = None


class HurstAEGSScanner:
    """
    Regime-aware AEGS scanner using Hurst Exponent analysis
    """
    
    def __init__(self):
        # Enhanced symbol list with regime-sensitive categories
        self.symbols = {
            'High Beta (Trend-Sensitive)': ['WULF', 'RIOT', 'MARA', 'SAVA', 'LCID'],
            'Momentum Stocks': ['GME', 'AMC', 'BB', 'SOFI', 'RIVN'],
            'Mean-Reversion Candidates': ['NOK', 'EQT', 'WKHS', 'BIIB', 'EDIT'],
            'Volatility ETFs': ['UVXY', 'VXX', 'SVXY', 'VIXY'],
            'Leveraged ETFs': ['TNA', 'SQQQ', 'LABU', 'NUGT', 'FNGU']
        }
        
        self.all_symbols = []
        for symbols in self.symbols.values():
            self.all_symbols.extend(symbols)
        self.all_symbols = list(set(self.all_symbols))
        
        # Results storage
        self.regime_signals = []
        self.traditional_signals = []
        
        # Hurst indicator
        self.hurst_indicator = HurstExponentIndicator() if HurstExponentIndicator else None
    
    def calculate_hurst_simple(self, prices, max_lag=50):
        """Simplified Hurst calculation when full indicator unavailable"""
        if len(prices) < max_lag * 2:
            return 0.5  # Neutral
            
        prices = np.array(prices)
        lags = range(2, min(max_lag, len(prices) // 2))
        
        rs_values = []
        for lag in lags:
            # Split into chunks
            chunks = [prices[i:i+lag] for i in range(0, len(prices)-lag, lag)]
            if len(chunks) < 2:
                continue
                
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                    
                mean_val = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_val)
                
                if len(deviations) == 0:
                    continue
                    
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk)
                
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))
        
        if len(rs_values) < 3:
            return 0.5
            
        # Linear regression of log(R/S) vs log(lag)
        lags_arr = np.array([rs[0] for rs in rs_values])
        rs_arr = np.array([rs[1] for rs in rs_values if rs[1] > 0])
        
        if len(rs_arr) < 3:
            return 0.5
            
        try:
            log_lags = np.log(lags_arr[:len(rs_arr)])
            log_rs = np.log(rs_arr)
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            return max(0.1, min(0.9, hurst))  # Bound between 0.1 and 0.9
        except:
            return 0.5
    
    def calculate_traditional_indicators(self, data):
        """Calculate traditional AEGS indicators"""
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Moving averages
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['Distance_Pct'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(10).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
        
        # Price changes
        df['Daily_Change'] = df['Close'].pct_change() * 100
        
        return df
    
    def analyze_regime_signals(self, symbol, data, hurst):
        """Generate regime-aware signals based on Hurst analysis"""
        
        latest = data.iloc[-1]
        
        # Traditional signal components
        trad_score = 0
        signals = []
        
        # RSI signals
        if pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                trad_score += 30
                signals.append(f"RSI={latest['RSI']:.1f}")
            elif latest['RSI'] < 35:
                trad_score += 15
                signals.append(f"RSI={latest['RSI']:.1f}")
        
        # BB signals
        if pd.notna(latest['BB_Position']):
            if latest['BB_Position'] < 0.2:
                trad_score += 25
                signals.append("BB_Oversold")
        
        # Distance from SMA
        if pd.notna(latest['Distance_Pct']):
            if latest['Distance_Pct'] < -5:
                trad_score += 20
                signals.append(f"Below_SMA={latest['Distance_Pct']:.1f}%")
        
        # Volume spike
        if latest['Volume_Ratio'] > 1.5:
            trad_score += 15
            signals.append(f"Vol={latest['Volume_Ratio']:.1f}x")
        
        # Price drop
        if latest['Daily_Change'] < -5:
            trad_score += 20
            signals.append(f"Drop={latest['Daily_Change']:.1f}%")
        
        # REGIME-AWARE ADJUSTMENT
        regime_multiplier = 1.0
        regime_type = "neutral"
        regime_confidence = 0
        
        if hurst < 0.4:
            # Strong mean-reverting regime
            regime_type = "mean_reverting"
            regime_multiplier = 1.4  # Boost reversal signals
            regime_confidence = 80
        elif hurst < 0.45:
            # Moderate mean-reversion
            regime_type = "weak_mean_reverting"
            regime_multiplier = 1.2
            regime_confidence = 60
        elif hurst > 0.6:
            # Strong trending regime
            regime_type = "trending"
            # For trending markets, we want momentum signals, not reversals
            if latest['Daily_Change'] > 0 and latest['RSI'] > 50:
                # Trending up - reduce contrarian signals
                regime_multiplier = 0.6
            else:
                regime_multiplier = 1.1
            regime_confidence = 70
        elif hurst > 0.55:
            regime_type = "weak_trending"
            regime_multiplier = 1.0
            regime_confidence = 40
        
        # Apply regime adjustment
        final_score = int(trad_score * regime_multiplier)
        
        return {
            'symbol': symbol,
            'traditional_score': trad_score,
            'regime_adjusted_score': final_score,
            'hurst': hurst,
            'regime_type': regime_type,
            'regime_confidence': regime_confidence,
            'signals': signals,
            'price': latest['Close'],
            'change': latest['Daily_Change'],
            'rsi': latest['RSI'],
            'distance': latest['Distance_Pct'],
            'volume': latest['Volume_Ratio']
        }
    
    def interpret_regime(self, hurst):
        """Interpret Hurst value for regime classification"""
        if hurst < 0.3:
            return "EXTREME Mean Reversion", "red"
        elif hurst < 0.4:
            return "STRONG Mean Reversion", "yellow"
        elif hurst < 0.45:
            return "Moderate Mean Reversion", "green"
        elif hurst < 0.55:
            return "Random Walk", "blue"
        elif hurst < 0.6:
            return "Weak Trending", "cyan"
        elif hurst < 0.7:
            return "STRONG Trending", "magenta"
        else:
            return "EXTREME Trending", "red"
    
    def scan_symbol(self, symbol, category):
        """Scan individual symbol with regime analysis"""
        
        try:
            print(f"   🔍 {symbol} ({category})...", end='', flush=True)
            
            # Download data
            data = yf.download(symbol, period='200d', progress=False, interval='1d')
            
            if data.empty or len(data) < 50:
                print(" ❌ Insufficient data")
                return None
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == symbol else col[1] for col in data.columns]
            
            # Calculate traditional indicators
            data = self.calculate_traditional_indicators(data)
            
            # Calculate Hurst Exponent
            if self.hurst_indicator:
                try:
                    hurst_result = self.hurst_indicator.calculate(data, symbol)
                    hurst = hurst_result.metadata.get('hurst_exponent', 0.5)
                except:
                    hurst = self.calculate_hurst_simple(data['Close'].values)
            else:
                hurst = self.calculate_hurst_simple(data['Close'].values)
            
            # Generate regime-aware signals
            analysis = self.analyze_regime_signals(symbol, data, hurst)
            analysis['category'] = category
            
            # Display result
            regime_desc, regime_color = self.interpret_regime(hurst)
            
            if analysis['regime_adjusted_score'] >= 70:
                print(colored(f" 🚀 REGIME BUY! Score: {analysis['regime_adjusted_score']} (was {analysis['traditional_score']})", 'green', attrs=['bold']))
                self.regime_signals.append(analysis)
            elif analysis['regime_adjusted_score'] >= 50:
                print(colored(f" ✅ Regime Signal: {analysis['regime_adjusted_score']} (was {analysis['traditional_score']})", 'green'))
                self.regime_signals.append(analysis)
            elif analysis['traditional_score'] >= 50:
                print(colored(f" ⚡ Traditional: {analysis['traditional_score']} | Regime: {regime_desc[:15]}", 'yellow'))
                self.traditional_signals.append(analysis)
            else:
                print(f" ⏸️ No signal | H={hurst:.2f} ({regime_desc[:10]})")
                
        except Exception as e:
            print(f" ❌ Error: {str(e)[:30]}")
    
    def run_regime_scan(self):
        """Run comprehensive regime-aware scan"""
        
        print(colored("🔥💎 AEGS REGIME-AWARE SCANNER 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 80)
        print("🧠 Using Hurst Exponent for market regime detection")
        print(f"📊 Scanning {len(self.all_symbols)} symbols with adaptive signals...")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Scan all symbols by category
        for category, symbols in self.symbols.items():
            print(f"\n🎯 Scanning {category}...")
            for symbol in symbols:
                self.scan_symbol(symbol, category)
        
        # Display comprehensive results
        self._display_regime_results()
    
    def _display_regime_results(self):
        """Display comprehensive regime analysis results"""
        
        print("\n" + "=" * 80)
        print(colored("🧠 REGIME-AWARE SCAN RESULTS", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if self.regime_signals:
            # Sort by regime-adjusted score
            self.regime_signals.sort(key=lambda x: x['regime_adjusted_score'], reverse=True)
            
            print(colored(f"\n🚀 REGIME-ENHANCED SIGNALS ({len(self.regime_signals)}):", 'green', attrs=['bold']))
            print("=" * 80)
            
            for i, signal in enumerate(self.regime_signals[:10], 1):  # Top 10
                symbol = signal['symbol']
                trad_score = signal['traditional_score']
                regime_score = signal['regime_adjusted_score']
                hurst = signal['hurst']
                regime_type = signal['regime_type']
                price = signal['price']
                change = signal['change']
                
                # Enhancement indicator
                enhancement = regime_score - trad_score
                if enhancement > 10:
                    enhance_text = colored(f"↑+{enhancement}", 'green', attrs=['bold'])
                elif enhancement > 0:
                    enhance_text = colored(f"↑+{enhancement}", 'green')
                elif enhancement < -10:
                    enhance_text = colored(f"↓{enhancement}", 'red', attrs=['bold'])
                elif enhancement < 0:
                    enhance_text = colored(f"↓{enhancement}", 'red')
                else:
                    enhance_text = "→0"
                
                # Regime description
                regime_desc, regime_color = self.interpret_regime(hurst)
                
                print(colored(f"\n#{i}. {symbol} @ ${price:.2f} ({change:+.1f}%)", 'white', attrs=['bold']))
                print(f"   📊 Score: {regime_score}/100 (Traditional: {trad_score}) {enhance_text}")
                print(colored(f"   🧠 Regime: {regime_desc} (H={hurst:.3f})", regime_color))
                print(f"   🎯 Category: {signal['category']}")
                print(f"   🔥 Triggers: {', '.join(signal['signals'][:3])}")
                
                # Regime-specific recommendations
                if regime_type == "mean_reverting":
                    print(colored("   💡 Strategy: Mean reversion - Buy dips, quick exits", 'cyan'))
                elif regime_type == "trending":
                    print(colored("   💡 Strategy: Momentum - Wait for breakouts", 'cyan'))
                else:
                    print(colored("   💡 Strategy: Standard AEGS approach", 'cyan'))
            
            # Best opportunity
            best = self.regime_signals[0]
            print("\n" + "=" * 80)
            print(colored("🎯 TOP REGIME-ENHANCED OPPORTUNITY:", 'red', attrs=['bold']))
            print(colored(f"   {best['symbol']} @ ${best['price']:.2f}", 'red', attrs=['bold']))
            print(colored(f"   Regime Score: {best['regime_adjusted_score']}/100", 'red'))
            print(colored(f"   Regime: {self.interpret_regime(best['hurst'])[0]}", 'red'))
            print(colored("   🚨 DEPLOY CAPITAL IMMEDIATELY", 'red', attrs=['bold', 'blink']))
        
        else:
            print(colored("\n⏸️ NO REGIME-ENHANCED SIGNALS", 'blue'))
        
        # Traditional signals that didn't get regime boost
        if self.traditional_signals:
            print(colored(f"\n⚡ TRADITIONAL SIGNALS (No Regime Boost) ({len(self.traditional_signals)}):", 'yellow'))
            
            self.traditional_signals.sort(key=lambda x: x['traditional_score'], reverse=True)
            
            for signal in self.traditional_signals[:5]:  # Top 5
                hurst = signal['hurst']
                regime_desc = self.interpret_regime(hurst)[0]
                print(f"   {signal['symbol']}: Score {signal['traditional_score']} | {regime_desc} (H={hurst:.3f})")
        
        # Market regime summary
        print("\n" + "=" * 80)
        print(colored("📈 MARKET REGIME ANALYSIS:", 'cyan'))
        print("=" * 80)
        
        all_hursts = []
        regime_counts = {}
        
        for signal in self.regime_signals + self.traditional_signals:
            hurst = signal['hurst']
            all_hursts.append(hurst)
            regime_type = signal['regime_type']
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        if all_hursts:
            avg_hurst = np.mean(all_hursts)
            regime_desc, _ = self.interpret_regime(avg_hurst)
            
            print(f"   📊 Average Market Hurst: {avg_hurst:.3f} ({regime_desc})")
            print(f"   🎯 Dominant Regime: {max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'Unknown'}")
            
            print(f"\n   📈 Regime Distribution:")
            for regime, count in regime_counts.items():
                pct = (count / len(all_hursts)) * 100
                print(f"      {regime.replace('_', ' ').title()}: {count} symbols ({pct:.1f}%)")
        
        # Trading recommendations
        print(f"\n" + "=" * 80)
        print(colored("💡 REGIME-AWARE TRADING STRATEGY:", 'white', attrs=['bold']))
        print("=" * 80)
        
        if self.regime_signals:
            print("1. 🚀 IMMEDIATE: Deploy capital on regime-enhanced signals")
            print("2. 🧠 REGIME PRIORITY: Focus on mean-reversion signals in current market")
            print("3. ⚡ QUICK EXITS: Set tight stops in mean-reverting regimes")
            print("4. 📊 MONITOR: Watch for regime transitions (Hurst changes)")
            print("5. 🎯 SIZE: Larger positions in high-confidence regime signals")
        else:
            print("1. ⏸️ WAIT: No regime-enhanced opportunities present")
            print("2. 📊 MONITOR: Market in transition or low-signal environment")
            print("3. 🧠 PREPARE: Analyze regime patterns for next opportunity")
        
        print(f"\n🔥 AEGS + Hurst = Adaptive Alpha Generation!")
        print(f"⏰ Regime Scan completed: {datetime.now().strftime('%H:%M:%S')}")


def main():
    """Run regime-aware AEGS scan"""
    
    print("🚀 Starting AEGS Regime-Aware Scanner...")
    print("🧠 Integrating Hurst Exponent for market regime detection...")
    print("🎯 Hunting for regime-enhanced opportunities...\n")
    
    scanner = HurstAEGSScanner()
    scanner.run_regime_scan()
    
    print("\n✅ Regime-aware AEGS scan complete!")
    print("💎 Deploy on regime-enhanced signals for maximum alpha!")


if __name__ == "__main__":
    main()