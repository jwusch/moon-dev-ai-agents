"""
🌊💎 FRACTAL ALPHA MARKET REGIME DASHBOARD 💎🌊
Comprehensive real-time dashboard combining all fractal indicators for complete market analysis

Features:
- Real-time regime detection across multiple assets
- Microstructure liquidity monitoring
- Fractal pattern recognition
- Integrated opportunity scoring
- Risk management guidance
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
import json
import time
from typing import Dict, List, Tuple, Optional


class FractalAlphaDashboard:
    """
    Comprehensive market regime dashboard integrating all fractal indicators
    """
    
    def __init__(self):
        # Asset universe organized by category
        self.asset_universe = {
            'Major Indices': ['SPY', 'QQQ', 'IWM', 'VTI'],
            'Volatility': ['VIX', 'UVXY', 'VXX', 'SVXY'],
            'Crypto Mining': ['WULF', 'RIOT', 'MARA', 'CLSK'],
            'High Beta': ['GME', 'AMC', 'BB', 'TSLA'],
            'Biotech': ['SAVA', 'EDIT', 'VKTX', 'BIIB'], 
            'Energy': ['EQT', 'XOM', 'CVX', 'SLB'],
            'Tech Growth': ['SOFI', 'PLTR', 'NVDA', 'AMD'],
            'Commodities': ['GLD', 'SLV', 'DBA', 'USO']
        }
        
        # Flatten for convenience
        self.all_symbols = []
        for symbols in self.asset_universe.values():
            self.all_symbols.extend(symbols)
        
        # Dashboard state
        self.dashboard_data = {}
        self.regime_summary = {}
        self.opportunities = []
        self.market_stress_level = 0
        
    def calculate_hurst_robust(self, prices: np.ndarray) -> Tuple[float, Dict]:
        """Multi-method Hurst calculation for reliability"""
        
        if len(prices) < 30:
            return 0.5, {'method': 'insufficient_data', 'confidence': 0}
        
        methods = {}
        
        # Method 1: R/S Analysis
        try:
            hurst_rs = self._hurst_rs(prices)
            methods['rs'] = hurst_rs
        except:
            methods['rs'] = None
            
        # Method 2: Variance Ratio
        try:
            hurst_var = self._hurst_variance(prices)
            methods['variance'] = hurst_var
        except:
            methods['variance'] = None
            
        # Method 3: DFA (simplified)
        try:
            hurst_dfa = self._hurst_dfa_simple(prices)
            methods['dfa'] = hurst_dfa
        except:
            methods['dfa'] = None
        
        # Calculate consensus
        valid_methods = [h for h in methods.values() if h is not None and 0.1 <= h <= 0.9]
        
        if not valid_methods:
            return 0.5, {'methods': methods, 'confidence': 0, 'consensus': 'failed'}
        
        consensus_hurst = np.median(valid_methods)
        confidence = 1.0 - (np.std(valid_methods) if len(valid_methods) > 1 else 0)
        
        return consensus_hurst, {
            'methods': methods,
            'confidence': confidence,
            'valid_count': len(valid_methods),
            'consensus': 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.5 else 'weak'
        }
    
    def _hurst_rs(self, prices: np.ndarray, max_lag: int = 30) -> float:
        """R/S analysis method"""
        returns = np.diff(np.log(prices))
        
        lags = range(2, min(max_lag, len(returns) // 4))
        rs_values = []
        
        for lag in lags:
            n_chunks = len(returns) // lag
            if n_chunks < 2:
                continue
                
            chunk_rs = []
            for i in range(n_chunks):
                chunk = returns[i*lag:(i+1)*lag]
                mean_return = np.mean(chunk)
                cumsum = np.cumsum(chunk - mean_return)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(chunk)
                
                if S > 0 and R > 0:
                    chunk_rs.append(R / S)
            
            if chunk_rs:
                rs_values.append((lag, np.mean(chunk_rs)))
        
        if len(rs_values) < 3:
            return 0.5
        
        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values])
        
        hurst = np.polyfit(lags_log, rs_log, 1)[0]
        return max(0.1, min(0.9, hurst))
    
    def _hurst_variance(self, prices: np.ndarray) -> float:
        """Variance ratio method"""
        returns = np.diff(np.log(prices))
        
        lags = [1, 2, 4, 8]
        lags = [l for l in lags if l < len(returns) // 3]
        
        if len(lags) < 3:
            return 0.5
        
        variances = []
        for lag in lags:
            n_periods = len(returns) // lag
            aggregated = []
            
            for i in range(n_periods):
                period_return = np.sum(returns[i*lag:(i+1)*lag])
                aggregated.append(period_return)
            
            if len(aggregated) > 1:
                var = np.var(aggregated) / lag
                variances.append((lag, var))
        
        if len(variances) < 3:
            return 0.5
        
        log_lags = np.log([x[0] for x in variances])
        log_vars = np.log([x[1] for x in variances])
        
        slope = np.polyfit(log_lags, log_vars, 1)[0]
        hurst = (slope + 2) / 2
        return max(0.1, min(0.9, hurst))
    
    def _hurst_dfa_simple(self, prices: np.ndarray) -> float:
        """Simplified DFA"""
        returns = np.diff(np.log(prices))
        y = np.cumsum(returns - np.mean(returns))
        
        boxes = [4, 8, 16, 32]
        boxes = [b for b in boxes if b < len(y) // 3]
        
        if len(boxes) < 3:
            return 0.5
        
        fluctuations = []
        for box_size in boxes:
            n_boxes = len(y) // box_size
            box_fluct = []
            
            for i in range(n_boxes):
                box_data = y[i*box_size:(i+1)*box_size]
                x = np.arange(len(box_data))
                trend = np.polyval(np.polyfit(x, box_data, 1), x)
                detrended = box_data - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                box_fluct.append(fluctuation)
            
            if box_fluct:
                fluctuations.append((box_size, np.mean(box_fluct)))
        
        if len(fluctuations) < 3:
            return 0.5
        
        log_boxes = np.log([x[0] for x in fluctuations])
        log_fluct = np.log([x[1] for x in fluctuations])
        
        hurst = np.polyfit(log_boxes, log_fluct, 1)[0]
        return max(0.1, min(0.9, hurst))
    
    def calculate_microstructure_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate key microstructure indicators"""
        
        # Simplified effective spread estimation
        high_low_spread = ((data['High'] - data['Low']) / data['Close']).rolling(20).mean().iloc[-1]
        
        # Volume-weighted price impact proxy
        volume_price_impact = (data['Volume'].rolling(5).std() / data['Volume'].rolling(5).mean()).iloc[-1]
        
        # Price momentum
        momentum_1d = data['Close'].pct_change(1).iloc[-1]
        momentum_5d = data['Close'].pct_change(5).iloc[-1]
        
        # Volatility regime
        volatility_20d = (data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()).iloc[-1]
        
        # Liquidity proxy (inverse of spread)
        liquidity_score = max(0, min(100, 100 - (high_low_spread * 10000)))  # Convert to bps
        
        return {
            'effective_spread_proxy': high_low_spread,
            'price_impact_proxy': volume_price_impact,
            'momentum_1d': momentum_1d,
            'momentum_5d': momentum_5d,
            'volatility_20d': volatility_20d,
            'liquidity_score': liquidity_score
        }
    
    def calculate_fractal_signals(self, data: pd.DataFrame) -> Dict:
        """Calculate fractal-based trading signals"""
        
        # Williams Fractals (simplified)
        highs = data['High'].values
        lows = data['Low'].values
        
        fractal_highs = []
        fractal_lows = []
        
        for i in range(2, len(highs) - 2):
            # High fractal: higher than 2 bars before and after
            if (highs[i] > highs[i-2] and highs[i] > highs[i-1] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                fractal_highs.append((i, highs[i]))
                
            # Low fractal: lower than 2 bars before and after  
            if (lows[i] < lows[i-2] and lows[i] < lows[i-1] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                fractal_lows.append((i, lows[i]))
        
        # Recent fractal analysis
        current_price = data['Close'].iloc[-1]
        
        fractal_support = None
        fractal_resistance = None
        
        if fractal_lows:
            recent_lows = [f for f in fractal_lows if f[0] > len(lows) - 50]  # Last 50 bars
            if recent_lows:
                fractal_support = max(recent_lows, key=lambda x: x[0])[1]  # Most recent
        
        if fractal_highs:
            recent_highs = [f for f in fractal_highs if f[0] > len(highs) - 50]
            if recent_highs:
                fractal_resistance = max(recent_highs, key=lambda x: x[0])[1]
        
        # Fractal signal strength
        signal_strength = 0
        signals = []
        
        if fractal_support and current_price <= fractal_support * 1.02:  # Within 2% of support
            signal_strength += 30
            signals.append(f"Near_Fractal_Support@{fractal_support:.2f}")
        
        if fractal_resistance and current_price >= fractal_resistance * 0.98:  # Within 2% of resistance
            signal_strength -= 20  # Negative signal
            signals.append(f"Near_Fractal_Resistance@{fractal_resistance:.2f}")
        
        return {
            'fractal_support': fractal_support,
            'fractal_resistance': fractal_resistance,
            'signal_strength': signal_strength,
            'signals': signals,
            'fractal_high_count': len(fractal_highs),
            'fractal_low_count': len(fractal_lows)
        }
    
    def analyze_symbol_comprehensive(self, symbol: str, category: str) -> Dict:
        """Comprehensive analysis of single symbol"""
        
        try:
            # Download extended data
            data = yf.download(symbol, period='6mo', progress=False, interval='1d')
            
            if data.empty or len(data) < 50:
                return None
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == symbol else col[1] for col in data.columns]
            
            # Calculate all metrics
            hurst, hurst_info = self.calculate_hurst_robust(data['Close'].values)
            microstructure = self.calculate_microstructure_metrics(data, symbol)
            fractals = self.calculate_fractal_signals(data)
            
            # Traditional indicators
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['BB_Position'] = self._calculate_bb_position(data['Close'])
            
            # Current market state
            latest = data.iloc[-1]
            
            # Regime classification
            regime = self._classify_regime(hurst, microstructure, fractals)
            
            # Overall opportunity score
            opportunity_score = self._calculate_opportunity_score(
                hurst, hurst_info, microstructure, fractals, latest, data
            )
            
            return {
                'symbol': symbol,
                'category': category,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'timestamp': datetime.now(),
                
                # Regime analysis
                'hurst': hurst,
                'hurst_info': hurst_info,
                'regime': regime,
                
                # Microstructure
                'microstructure': microstructure,
                
                # Fractals
                'fractals': fractals,
                
                # Traditional
                'rsi': latest['RSI'] if pd.notna(latest['RSI']) else 50,
                'bb_position': latest['BB_Position'] if pd.notna(latest['BB_Position']) else 0.5,
                
                # Overall assessment
                'opportunity_score': opportunity_score,
                'risk_level': self._assess_risk(microstructure, hurst_info, fractals)
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)[:50]}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 0.0001)
        return 100 - (100 / (1 + rs))
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (prices - lower) / ((upper - lower) + 0.0001)
    
    def _classify_regime(self, hurst: float, microstructure: Dict, fractals: Dict) -> Dict:
        """Classify overall market regime"""
        
        # Hurst-based regime
        if hurst < 0.35:
            hurst_regime = "extreme_mean_reverting"
        elif hurst < 0.45:
            hurst_regime = "mean_reverting"
        elif hurst < 0.55:
            hurst_regime = "random_walk"
        elif hurst < 0.65:
            hurst_regime = "trending"
        else:
            hurst_regime = "strong_trending"
        
        # Liquidity regime
        liquidity_score = microstructure['liquidity_score']
        if liquidity_score > 80:
            liquidity_regime = "high_liquidity"
        elif liquidity_score > 60:
            liquidity_regime = "normal_liquidity"
        elif liquidity_score > 40:
            liquidity_regime = "low_liquidity"
        else:
            liquidity_regime = "stressed_liquidity"
        
        # Volatility regime
        volatility = microstructure['volatility_20d']
        if volatility < 0.15:
            volatility_regime = "low_vol"
        elif volatility < 0.25:
            volatility_regime = "normal_vol"
        elif volatility < 0.40:
            volatility_regime = "high_vol"
        else:
            volatility_regime = "extreme_vol"
        
        return {
            'hurst_regime': hurst_regime,
            'liquidity_regime': liquidity_regime,
            'volatility_regime': volatility_regime,
            'composite_regime': f"{hurst_regime}_{liquidity_regime}_{volatility_regime}"
        }
    
    def _calculate_opportunity_score(self, hurst: float, hurst_info: Dict, 
                                   microstructure: Dict, fractals: Dict, 
                                   latest: pd.Series, data: pd.DataFrame) -> int:
        """Calculate comprehensive opportunity score"""
        
        score = 0
        
        # Hurst-based scoring
        if hurst < 0.4:  # Mean-reverting - look for oversold
            if latest['RSI'] < 30:
                score += 30
            if latest['BB_Position'] < 0.2:
                score += 25
            score += int(hurst_info['confidence'] * 20)  # Confidence bonus
        elif hurst > 0.6:  # Trending - look for momentum
            if microstructure['momentum_5d'] > 0.02:  # 2% momentum
                score += 20
            if latest['RSI'] > 50:
                score += 15
        
        # Fractal scoring
        score += max(-20, min(30, fractals['signal_strength']))
        
        # Liquidity bonus
        if microstructure['liquidity_score'] > 70:
            score += 15
        elif microstructure['liquidity_score'] < 30:
            score -= 10
        
        # Volume confirmation
        if microstructure['price_impact_proxy'] < 0.5:  # Low impact = good liquidity
            score += 10
        
        return max(0, min(100, score))
    
    def _assess_risk(self, microstructure: Dict, hurst_info: Dict, fractals: Dict) -> str:
        """Assess overall risk level"""
        
        risk_score = 0
        
        # Volatility risk
        if microstructure['volatility_20d'] > 0.3:
            risk_score += 2
        elif microstructure['volatility_20d'] > 0.2:
            risk_score += 1
        
        # Liquidity risk
        if microstructure['liquidity_score'] < 30:
            risk_score += 2
        elif microstructure['liquidity_score'] < 50:
            risk_score += 1
        
        # Regime uncertainty risk
        if hurst_info['confidence'] < 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def run_comprehensive_scan(self) -> None:
        """Run comprehensive market scan across all assets"""
        
        print(colored("🌊💎 FRACTAL ALPHA MARKET REGIME DASHBOARD 💎🌊", 'cyan', attrs=['bold']))
        print("=" * 90)
        print("🧠 Comprehensive fractal analysis across market regimes")
        print("📊 Microstructure liquidity + Hurst regime + Fractal patterns")
        print(f"🎯 Analyzing {len(self.all_symbols)} symbols across {len(self.asset_universe)} categories")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)
        
        # Analyze all symbols
        for category, symbols in self.asset_universe.items():
            print(f"\n🔍 {category}...")
            
            category_data = []
            for symbol in symbols:
                print(f"   📊 {symbol}...", end='', flush=True)
                
                analysis = self.analyze_symbol_comprehensive(symbol, category)
                if analysis:
                    self.dashboard_data[symbol] = analysis
                    category_data.append(analysis)
                    
                    score = analysis['opportunity_score']
                    risk = analysis['risk_level']
                    regime = analysis['regime']['hurst_regime']
                    
                    if score >= 70:
                        print(colored(f" 🚀 {score}/100 ({risk}) {regime[:8]}", 'green', attrs=['bold']))
                    elif score >= 50:
                        print(colored(f" ✅ {score}/100 ({risk}) {regime[:8]}", 'green'))
                    elif score >= 30:
                        print(colored(f" ⚡ {score}/100 ({risk}) {regime[:8]}", 'yellow'))
                    else:
                        print(f" ⏸️ {score}/100 ({risk}) {regime[:8]}")
                else:
                    print(" ❌ Failed")
        
        # Generate comprehensive dashboard
        self._generate_dashboard_summary()
    
    def _generate_dashboard_summary(self) -> None:
        """Generate comprehensive dashboard summary"""
        
        print("\n" + "=" * 90)
        print(colored("🌊 FRACTAL ALPHA DASHBOARD SUMMARY", 'yellow', attrs=['bold']))
        print("=" * 90)
        
        if not self.dashboard_data:
            print(colored("⚠️ No data available for analysis", 'red'))
            return
        
        # Extract opportunities
        opportunities = []
        for analysis in self.dashboard_data.values():
            if analysis['opportunity_score'] >= 40:
                opportunities.append(analysis)
        
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Market regime overview
        self._display_regime_overview()
        
        # Top opportunities
        self._display_top_opportunities(opportunities)
        
        # Market stress assessment
        self._display_market_stress()
        
        # Category analysis
        self._display_category_analysis()
    
    def _display_regime_overview(self) -> None:
        """Display market regime overview"""
        
        print(colored("\n🧠 MARKET REGIME OVERVIEW:", 'cyan', attrs=['bold']))
        print("=" * 90)
        
        # Collect regime statistics
        all_hursts = [data['hurst'] for data in self.dashboard_data.values()]
        regime_counts = {}
        liquidity_scores = []
        volatilities = []
        
        for data in self.dashboard_data.values():
            regime = data['regime']['hurst_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            liquidity_scores.append(data['microstructure']['liquidity_score'])
            volatilities.append(data['microstructure']['volatility_20d'])
        
        if all_hursts:
            avg_hurst = np.mean(all_hursts)
            avg_liquidity = np.mean(liquidity_scores)
            avg_volatility = np.mean(volatilities)
            
            print(f"📊 Market Hurst: {avg_hurst:.3f} ({self._get_regime_name(avg_hurst)})")
            print(f"💧 Average Liquidity: {avg_liquidity:.1f}/100")
            print(f"⚡ Average Volatility: {avg_volatility:.1%}")
            
            # Regime distribution
            print(f"\n🎯 Regime Distribution:")
            total_symbols = len(self.dashboard_data)
            for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_symbols) * 100
                print(f"   {regime.replace('_', ' ').title()}: {count} symbols ({pct:.1f}%)")
    
    def _display_top_opportunities(self, opportunities: List[Dict]) -> None:
        """Display top trading opportunities"""
        
        print(colored(f"\n🚀 TOP FRACTAL ALPHA OPPORTUNITIES ({len(opportunities)}):", 'green', attrs=['bold']))
        print("=" * 90)
        
        if not opportunities:
            print(colored("⏸️ No significant opportunities detected", 'blue'))
            return
        
        for i, opp in enumerate(opportunities[:10], 1):  # Top 10
            symbol = opp['symbol']
            price = opp['price']
            score = opp['opportunity_score']
            risk = opp['risk_level']
            regime = opp['regime']['hurst_regime']
            category = opp['category']
            
            hurst = opp['hurst']
            liquidity = opp['microstructure']['liquidity_score']
            
            print(colored(f"\n#{i}. {symbol} @ ${price:.2f} ({category})", 'white', attrs=['bold']))
            print(f"   📊 Opportunity Score: {score}/100 | Risk: {risk}")
            print(f"   🧠 Regime: {regime.replace('_', ' ').title()} (H={hurst:.3f})")
            print(f"   💧 Liquidity: {liquidity:.1f}/100 | RSI: {opp['rsi']:.1f}")
            
            # Fractal signals
            if opp['fractals']['signals']:
                print(f"   🔺 Fractals: {', '.join(opp['fractals']['signals'])}")
            
            # Strategy recommendation
            strategy = self._get_strategy_recommendation(opp)
            print(colored(f"   💡 Strategy: {strategy}", 'cyan'))
        
        # Highlight best opportunity
        if opportunities:
            best = opportunities[0]
            print("\n" + "=" * 90)
            print(colored("🎯 PRIME FRACTAL ALPHA OPPORTUNITY:", 'red', attrs=['bold']))
            print(colored(f"   {best['symbol']} @ ${best['price']:.2f}", 'red', attrs=['bold']))
            print(colored(f"   Score: {best['opportunity_score']}/100", 'red'))
            print(colored("   🚨 MAXIMUM ALPHA POTENTIAL", 'red', attrs=['bold']))
    
    def _display_market_stress(self) -> None:
        """Display market stress indicators"""
        
        print(colored(f"\n⚠️ MARKET STRESS ASSESSMENT:", 'yellow', attrs=['bold']))
        print("=" * 90)
        
        # Calculate stress indicators
        high_vol_count = sum(1 for data in self.dashboard_data.values() 
                           if data['microstructure']['volatility_20d'] > 0.3)
        
        low_liquidity_count = sum(1 for data in self.dashboard_data.values()
                                if data['microstructure']['liquidity_score'] < 40)
        
        high_risk_count = sum(1 for data in self.dashboard_data.values()
                            if data['risk_level'] == 'HIGH')
        
        total_symbols = len(self.dashboard_data)
        
        # Stress score calculation
        stress_score = ((high_vol_count + low_liquidity_count + high_risk_count) / 
                       (total_symbols * 3)) * 100
        
        if stress_score > 60:
            stress_level = "HIGH STRESS"
            stress_color = 'red'
        elif stress_score > 30:
            stress_level = "MODERATE STRESS"
            stress_color = 'yellow'
        else:
            stress_level = "LOW STRESS"
            stress_color = 'green'
        
        print(colored(f"📊 Market Stress Level: {stress_level} ({stress_score:.1f}/100)", stress_color, attrs=['bold']))
        print(f"⚡ High Volatility Assets: {high_vol_count}/{total_symbols}")
        print(f"💧 Low Liquidity Assets: {low_liquidity_count}/{total_symbols}")  
        print(f"⚠️ High Risk Assets: {high_risk_count}/{total_symbols}")
    
    def _display_category_analysis(self) -> None:
        """Display analysis by asset category"""
        
        print(colored(f"\n📈 CATEGORY PERFORMANCE ANALYSIS:", 'cyan', attrs=['bold']))
        print("=" * 90)
        
        category_stats = {}
        
        for data in self.dashboard_data.values():
            category = data['category']
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(data)
        
        # Calculate category metrics
        for category, assets in category_stats.items():
            avg_score = np.mean([a['opportunity_score'] for a in assets])
            avg_hurst = np.mean([a['hurst'] for a in assets])
            avg_liquidity = np.mean([a['microstructure']['liquidity_score'] for a in assets])
            high_opportunity_count = sum(1 for a in assets if a['opportunity_score'] >= 50)
            
            print(f"\n🎯 {category}:")
            print(f"   Avg Opportunity: {avg_score:.1f}/100 | Hurst: {avg_hurst:.3f}")
            print(f"   Avg Liquidity: {avg_liquidity:.1f}/100 | Signals: {high_opportunity_count}/{len(assets)}")
    
    def _get_regime_name(self, hurst: float) -> str:
        """Get regime name from Hurst value"""
        if hurst < 0.4:
            return "Mean Reverting"
        elif hurst < 0.55:
            return "Random Walk"
        else:
            return "Trending"
    
    def _get_strategy_recommendation(self, analysis: Dict) -> str:
        """Get strategy recommendation"""
        
        regime = analysis['regime']['hurst_regime']
        score = analysis['opportunity_score']
        risk = analysis['risk_level']
        
        if score >= 70:
            if 'mean_reverting' in regime:
                return "AGGRESSIVE reversal - Quick entry/exit"
            else:
                return "STRONG momentum - Trend following"
        elif score >= 50:
            if 'mean_reverting' in regime:
                return "Moderate reversal - Wait for confirmation"
            else:
                return "Trend continuation - Monitor breakouts"
        else:
            return "Watch list - Wait for better setup"
    
    def save_dashboard_data(self, filename: str = None) -> None:
        """Save dashboard data to JSON file"""
        
        if filename is None:
            filename = f"fractal_alpha_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for JSON serialization
        export_data = {}
        for symbol, data in self.dashboard_data.items():
            export_data[symbol] = {
                'symbol': data['symbol'],
                'category': data['category'],
                'price': float(data['price']),
                'opportunity_score': data['opportunity_score'],
                'hurst': float(data['hurst']),
                'regime': data['regime'],
                'liquidity_score': float(data['microstructure']['liquidity_score']),
                'risk_level': data['risk_level'],
                'timestamp': data['timestamp'].isoformat()
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Dashboard data saved to: {filename}")


def main():
    """Run the Fractal Alpha Dashboard"""
    
    print("🚀 Initializing Fractal Alpha Market Dashboard...")
    print("🧠 Preparing comprehensive regime analysis...")
    print("📊 Loading multi-dimensional indicators...\n")
    
    dashboard = FractalAlphaDashboard()
    dashboard.run_comprehensive_scan()
    
    # Save results
    dashboard.save_dashboard_data()
    
    print("\n✅ Fractal Alpha Dashboard scan complete!")
    print("🌊 Market regime analysis ready for deployment!")


if __name__ == "__main__":
    main()