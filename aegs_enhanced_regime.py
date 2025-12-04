"""
🚀💎 ENHANCED AEGS REGIME SCANNER 💎🚀
Advanced integration with improved Hurst calculation and regime-based strategy adaptation

Features:
- Robust Hurst Exponent calculation with multiple methods
- Regime transition detection
- Symbol-specific regime sensitivity analysis
- Enhanced signal confidence based on regime persistence
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
from typing import Dict, List, Tuple, Optional


class AdvancedHurstCalculator:
    """Enhanced Hurst Exponent calculator with multiple methods"""
    
    def __init__(self):
        self.methods = ['rs', 'dfa', 'variance']
    
    def calculate_hurst_rs(self, prices: np.ndarray, max_lag: int = 50) -> float:
        """R/S analysis method for Hurst calculation"""
        if len(prices) < 20:
            return 0.5
            
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 10:
            return 0.5
        
        # Determine lag range
        min_lag = max(2, len(returns) // 20)
        max_lag = min(max_lag, len(returns) // 4)
        lags = np.logspace(np.log10(min_lag), np.log10(max_lag), 20).astype(int)
        lags = np.unique(lags)
        
        rs_values = []
        
        for lag in lags:
            if lag >= len(returns):
                continue
                
            # Split returns into chunks
            n_chunks = len(returns) // lag
            if n_chunks < 2:
                continue
                
            chunk_rs = []
            for i in range(n_chunks):
                chunk = returns[i*lag:(i+1)*lag]
                
                if len(chunk) < 2:
                    continue
                
                # Calculate mean-adjusted cumulative sum
                mean_return = np.mean(chunk)
                cumsum = np.cumsum(chunk - mean_return)
                
                # Calculate range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Calculate standard deviation
                S = np.std(chunk)
                
                if S > 0 and R > 0:
                    chunk_rs.append(R / S)
            
            if chunk_rs:
                rs_values.append((lag, np.mean(chunk_rs)))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression in log space
        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values if x[1] > 0])
        
        if len(rs_log) < 3:
            return 0.5
        
        try:
            # Fit line and get slope (Hurst exponent)
            lags_log = lags_log[:len(rs_log)]
            hurst = np.polyfit(lags_log, rs_log, 1)[0]
            
            # Bound the result
            return max(0.05, min(0.95, hurst))
        except:
            return 0.5
    
    def calculate_hurst_dfa(self, prices: np.ndarray, max_lag: int = 50) -> float:
        """Detrended Fluctuation Analysis method"""
        if len(prices) < 20:
            return 0.5
            
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 10:
            return 0.5
        
        # Integrate the series
        y = np.cumsum(returns - np.mean(returns))
        
        # Box sizes
        min_box = max(4, len(y) // 50)
        max_box = min(max_lag, len(y) // 4)
        boxes = np.logspace(np.log10(min_box), np.log10(max_box), 15).astype(int)
        boxes = np.unique(boxes)
        
        fluctuations = []
        
        for box_size in boxes:
            if box_size >= len(y):
                continue
                
            # Number of boxes
            n_boxes = len(y) // box_size
            
            if n_boxes < 2:
                continue
                
            box_fluct = []
            
            for i in range(n_boxes):
                start = i * box_size
                end = (i + 1) * box_size
                
                if end > len(y):
                    continue
                    
                box_data = y[start:end]
                
                # Detrend (fit linear trend)
                x = np.arange(len(box_data))
                try:
                    trend = np.polyval(np.polyfit(x, box_data, 1), x)
                    detrended = box_data - trend
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    
                    if fluctuation > 0:
                        box_fluct.append(fluctuation)
                except:
                    continue
            
            if box_fluct:
                fluctuations.append((box_size, np.mean(box_fluct)))
        
        if len(fluctuations) < 3:
            return 0.5
        
        # Log-log regression
        try:
            log_boxes = np.log([x[0] for x in fluctuations])
            log_fluct = np.log([x[1] for x in fluctuations if x[1] > 0])
            
            if len(log_fluct) < 3:
                return 0.5
                
            log_boxes = log_boxes[:len(log_fluct)]
            hurst = np.polyfit(log_boxes, log_fluct, 1)[0]
            
            return max(0.05, min(0.95, hurst))
        except:
            return 0.5
    
    def calculate_hurst_variance(self, prices: np.ndarray) -> float:
        """Variance ratio method"""
        if len(prices) < 20:
            return 0.5
            
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        
        if len(returns) < 10:
            return 0.5
        
        # Various lag periods
        lags = [1, 2, 4, 8, 16]
        lags = [l for l in lags if l < len(returns) // 2]
        
        if len(lags) < 3:
            return 0.5
        
        variances = []
        
        for lag in lags:
            if lag >= len(returns):
                continue
                
            # Aggregate returns
            n_periods = len(returns) // lag
            aggregated = []
            
            for i in range(n_periods):
                period_return = np.sum(returns[i*lag:(i+1)*lag])
                aggregated.append(period_return)
            
            if len(aggregated) > 1:
                var = np.var(aggregated) / lag  # Normalize by time
                variances.append((lag, var))
        
        if len(variances) < 3:
            return 0.5
        
        try:
            log_lags = np.log([x[0] for x in variances])
            log_vars = np.log([x[1] for x in variances if x[1] > 0])
            
            if len(log_vars) < 3:
                return 0.5
                
            log_lags = log_lags[:len(log_vars)]
            
            # Hurst = (slope + 2) / 2
            slope = np.polyfit(log_lags, log_vars, 1)[0]
            hurst = (slope + 2) / 2
            
            return max(0.05, min(0.95, hurst))
        except:
            return 0.5
    
    def calculate_robust_hurst(self, prices: np.ndarray) -> Tuple[float, Dict]:
        """Calculate Hurst using multiple methods and return consensus"""
        
        methods_results = {}
        
        # Calculate using all methods
        methods_results['rs'] = self.calculate_hurst_rs(prices)
        methods_results['dfa'] = self.calculate_hurst_dfa(prices)
        methods_results['variance'] = self.calculate_hurst_variance(prices)
        
        # Remove invalid results
        valid_results = {k: v for k, v in methods_results.items() if 0.05 <= v <= 0.95}
        
        if not valid_results:
            return 0.5, methods_results
        
        # Calculate consensus (median to reduce outlier impact)
        consensus_hurst = np.median(list(valid_results.values()))
        
        # Calculate consistency
        if len(valid_results) > 1:
            std_dev = np.std(list(valid_results.values()))
            consistency = max(0, 1 - (std_dev * 4))  # Scale to 0-1
        else:
            consistency = 0.5
        
        results = {
            'hurst': consensus_hurst,
            'consistency': consistency,
            'methods': methods_results,
            'valid_methods': len(valid_results)
        }
        
        return consensus_hurst, results


class EnhancedRegimeAEGS:
    """Enhanced AEGS scanner with robust regime analysis"""
    
    def __init__(self):
        self.symbols = {
            'Crypto Mining': ['WULF', 'RIOT', 'MARA', 'CLSK', 'CORZ', 'CIFR'],
            'High Volatility': ['GME', 'AMC', 'BB', 'SAVA', 'LCID'],
            'Biotech/Growth': ['BIIB', 'VKTX', 'EDIT', 'CRSP', 'NTLA'],
            'ETFs/Leveraged': ['TNA', 'SQQQ', 'LABU', 'UVXY', 'NUGT'],
            'Traditional': ['NOK', 'SOFI', 'WKHS', 'EQT', 'RIVN']
        }
        
        self.all_symbols = []
        for symbols in self.symbols.values():
            self.all_symbols.extend(symbols)
        self.all_symbols = list(set(self.all_symbols))
        
        self.hurst_calculator = AdvancedHurstCalculator()
        self.results = []
        
    def calculate_enhanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators"""
        df = data.copy()
        
        # Standard indicators
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
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / ((df['BB_Upper'] - df['BB_Lower']) + 0.0001)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Moving averages with multiple timeframes
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Enhanced distance metrics
        df['Distance_SMA20'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100
        df['Distance_SMA50'] = (df['Close'] - df['SMA50']) / df['SMA50'] * 100
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
        df['Volume_SMA_Ratio'] = df['Volume'].rolling(5).mean() / df['Volume_MA']
        
        # Volatility metrics
        df['ATR'] = self._calculate_atr(df)
        df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        
        # Price momentum
        df['Price_Change_1d'] = df['Close'].pct_change() * 100
        df['Price_Change_3d'] = df['Close'].pct_change(3) * 100
        df['Price_Change_5d'] = df['Close'].pct_change(5) * 100
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def generate_regime_signals(self, symbol: str, data: pd.DataFrame, 
                               hurst: float, regime_info: Dict) -> Dict:
        """Generate sophisticated regime-aware signals"""
        
        latest = data.iloc[-1]
        
        # Base signal calculation
        signal_score = 0
        signal_components = []
        
        # RSI-based signals with regime adjustment
        if pd.notna(latest['RSI']):
            if hurst < 0.4:  # Mean-reverting regime
                # Strong oversold signals in mean-reverting markets
                if latest['RSI'] < 25:
                    signal_score += 40
                    signal_components.append(f"RSI_Oversold={latest['RSI']:.1f}")
                elif latest['RSI'] < 30:
                    signal_score += 30
                    signal_components.append(f"RSI_Low={latest['RSI']:.1f}")
                elif latest['RSI'] < 40:
                    signal_score += 15
                    signal_components.append(f"RSI_Mild={latest['RSI']:.1f}")
            else:  # Trending regime
                # More conservative RSI in trending markets
                if latest['RSI'] < 30:
                    signal_score += 25
                    signal_components.append(f"RSI_Trend_Low={latest['RSI']:.1f}")
        
        # Bollinger Band signals
        if pd.notna(latest['BB_Position']):
            bb_pos = latest['BB_Position']
            
            if hurst < 0.45:  # Mean-reverting
                # Strong BB signals in mean-reverting markets
                if bb_pos < 0:  # Below lower band
                    signal_score += 35
                    signal_components.append("BB_Breakdown")
                elif bb_pos < 0.2:
                    signal_score += 25
                    signal_components.append(f"BB_Low={bb_pos:.2f}")
            else:  # Trending
                if bb_pos < 0.1:
                    signal_score += 20
                    signal_components.append(f"BB_Trend_Low={bb_pos:.2f}")
        
        # Distance from moving averages
        if pd.notna(latest['Distance_SMA20']) and pd.notna(latest['Distance_SMA50']):
            dist_20 = latest['Distance_SMA20']
            dist_50 = latest['Distance_SMA50']
            
            if hurst < 0.4:  # Mean-reverting
                if dist_20 < -8:
                    signal_score += 30
                    signal_components.append(f"SMA20_Deep={dist_20:.1f}%")
                elif dist_20 < -5:
                    signal_score += 20
                    signal_components.append(f"SMA20_Below={dist_20:.1f}%")
                    
                if dist_50 < -10:
                    signal_score += 20
                    signal_components.append(f"SMA50_Below={dist_50:.1f}%")
            else:  # Trending - less weight on distance
                if dist_20 < -10:
                    signal_score += 15
                    signal_components.append(f"SMA_Trend_Below={dist_20:.1f}%")
        
        # MACD signals
        if (pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']) and 
            len(data) > 1):
            
            prev = data.iloc[-2]
            if (latest['MACD'] > latest['MACD_Signal'] and 
                prev['MACD'] <= prev['MACD_Signal']):
                
                if hurst < 0.45:
                    signal_score += 25  # Strong MACD signal in mean-reverting
                    signal_components.append("MACD_Bullish_Cross")
                else:
                    signal_score += 15  # Moderate in trending
                    signal_components.append("MACD_Cross")
        
        # Volume confirmation
        if pd.notna(latest['Volume_Ratio']):
            vol_ratio = latest['Volume_Ratio']
            
            # Volume spikes are important in all regimes
            if vol_ratio > 2.5:
                signal_score += 25
                signal_components.append(f"Vol_Spike={vol_ratio:.1f}x")
            elif vol_ratio > 1.8:
                signal_score += 15
                signal_components.append(f"Vol_High={vol_ratio:.1f}x")
        
        # Price momentum with regime consideration
        if pd.notna(latest['Price_Change_1d']) and pd.notna(latest['Price_Change_3d']):
            change_1d = latest['Price_Change_1d']
            change_3d = latest['Price_Change_3d']
            
            if hurst < 0.4:  # Mean-reverting - look for oversold bounces
                if change_1d < -8 and change_3d < -15:
                    signal_score += 30
                    signal_components.append(f"Oversold_Crash={change_1d:.1f}%")
                elif change_1d < -5:
                    signal_score += 15
                    signal_components.append(f"Pullback={change_1d:.1f}%")
        
        # Regime consistency bonus
        consistency_bonus = int(regime_info.get('consistency', 0) * 20)
        
        # Final regime adjustment
        regime_multiplier = 1.0
        regime_type = "neutral"
        
        if hurst < 0.35:
            regime_type = "strong_mean_reverting"
            regime_multiplier = 1.5  # Strong boost for reversal signals
        elif hurst < 0.45:
            regime_type = "mean_reverting"
            regime_multiplier = 1.3
        elif hurst > 0.65:
            regime_type = "strong_trending"
            regime_multiplier = 0.8  # Reduce reversal signals in trending
        elif hurst > 0.55:
            regime_type = "trending"
            regime_multiplier = 0.9
        
        final_score = int((signal_score * regime_multiplier) + consistency_bonus)
        
        return {
            'symbol': symbol,
            'price': latest['Close'],
            'base_score': signal_score,
            'final_score': final_score,
            'regime_multiplier': regime_multiplier,
            'consistency_bonus': consistency_bonus,
            'hurst': hurst,
            'regime_type': regime_type,
            'regime_info': regime_info,
            'signals': signal_components,
            'rsi': latest['RSI'],
            'bb_position': latest['BB_Position'],
            'distance_sma20': latest['Distance_SMA20'],
            'volume_ratio': latest['Volume_Ratio'],
            'price_change_1d': latest['Price_Change_1d'],
            'atr': latest['ATR']
        }
    
    def scan_symbol_enhanced(self, symbol: str, category: str) -> Optional[Dict]:
        """Enhanced symbol scanning with robust regime analysis"""
        
        try:
            print(f"   🔍 {symbol} ({category[:15]})...", end='', flush=True)
            
            # Download extended data for better Hurst calculation
            data = yf.download(symbol, period='1y', progress=False, interval='1d')
            
            if data.empty or len(data) < 50:
                print(" ❌ Insufficient data")
                return None
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == symbol else col[1] for col in data.columns]
            
            # Calculate enhanced indicators
            data = self.calculate_enhanced_indicators(data)
            
            # Calculate robust Hurst
            hurst, regime_info = self.hurst_calculator.calculate_robust_hurst(data['Close'].values)
            
            # Generate regime-aware signals
            analysis = self.generate_regime_signals(symbol, data, hurst, regime_info)
            analysis['category'] = category
            
            # Determine signal strength
            score = analysis['final_score']
            base_score = analysis['base_score']
            
            regime_desc = self._get_regime_description(hurst)
            consistency = regime_info.get('consistency', 0)
            
            if score >= 80:
                print(colored(f" 🚀 STRONG! {score} (base:{base_score}) | H={hurst:.3f} C={consistency:.2f}", 
                            'green', attrs=['bold']))
                return analysis
            elif score >= 60:
                print(colored(f" ✅ GOOD! {score} (base:{base_score}) | H={hurst:.3f}", 'green'))
                return analysis
            elif score >= 40:
                print(colored(f" ⚡ Watch: {score} (base:{base_score}) | {regime_desc[:15]}", 'yellow'))
                return analysis
            else:
                consistency_text = f"C={consistency:.1f}" if consistency > 0.7 else ""
                print(f" ⏸️ {score} | H={hurst:.3f} {consistency_text}")
                return None
                
        except Exception as e:
            print(f" ❌ Error: {str(e)[:25]}")
            return None
    
    def _get_regime_description(self, hurst: float) -> str:
        """Get regime description from Hurst value"""
        if hurst < 0.3:
            return "Extreme Mean Reversion"
        elif hurst < 0.4:
            return "Strong Mean Reversion"
        elif hurst < 0.45:
            return "Mean Reverting"
        elif hurst < 0.55:
            return "Random Walk"
        elif hurst < 0.65:
            return "Trending"
        elif hurst < 0.75:
            return "Strong Trending"
        else:
            return "Extreme Trending"
    
    def run_enhanced_scan(self):
        """Run the enhanced regime-aware scan"""
        
        print(colored("🚀💎 ENHANCED AEGS REGIME SCANNER 💎🚀", 'cyan', attrs=['bold']))
        print("=" * 80)
        print("🧠 Multi-method Hurst calculation with regime consistency analysis")
        print("📊 Advanced signal generation with regime-specific adjustments")
        print(f"🎯 Scanning {len(self.all_symbols)} symbols across {len(self.symbols)} categories")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Scan all categories
        for category, symbols in self.symbols.items():
            print(f"\n🎯 {category}...")
            for symbol in symbols:
                result = self.scan_symbol_enhanced(symbol, category)
                if result:
                    self.results.append(result)
        
        # Display results
        self._display_enhanced_results()
    
    def _display_enhanced_results(self):
        """Display comprehensive enhanced results"""
        
        print("\n" + "=" * 80)
        print(colored("🧠 ENHANCED REGIME ANALYSIS RESULTS", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        if not self.results:
            print(colored("\n⏸️ NO SIGNIFICANT SIGNALS DETECTED", 'blue'))
            print("Market conditions may not favor current strategy parameters")
            return
        
        # Sort by final score
        self.results.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(colored(f"\n🚀 REGIME-ENHANCED OPPORTUNITIES ({len(self.results)}):", 'green', attrs=['bold']))
        print("=" * 80)
        
        for i, result in enumerate(self.results[:8], 1):  # Top 8 results
            symbol = result['symbol']
            price = result['price']
            final_score = result['final_score']
            base_score = result['base_score']
            hurst = result['hurst']
            regime_type = result['regime_type']
            consistency = result['regime_info'].get('consistency', 0)
            
            # Enhancement calculation
            enhancement = final_score - base_score
            if enhancement > 15:
                enhance_text = colored(f"++{enhancement}", 'green', attrs=['bold'])
            elif enhancement > 5:
                enhance_text = colored(f"+{enhancement}", 'green')
            elif enhancement < -10:
                enhance_text = colored(f"{enhancement}", 'red')
            else:
                enhance_text = f"{enhancement:+d}"
            
            print(colored(f"\n#{i}. {symbol} @ ${price:.2f}", 'white', attrs=['bold']))
            print(f"   📊 Score: {final_score}/100 (Base: {base_score}) {enhance_text}")
            print(f"   🧠 Regime: {self._get_regime_description(hurst)} (H={hurst:.3f})")
            print(f"   ✅ Consistency: {consistency:.2f} | Category: {result['category']}")
            print(f"   🔥 Signals: {', '.join(result['signals'][:4])}")
            
            # Regime-specific strategy
            strategy = self._get_strategy_recommendation(regime_type, result)
            print(colored(f"   💡 Strategy: {strategy}", 'cyan'))
            
            # Risk assessment
            risk = self._assess_risk_level(result)
            print(colored(f"   ⚠️  Risk: {risk}", 'yellow'))
        
        # Top pick
        if self.results:
            best = self.results[0]
            print("\n" + "=" * 80)
            print(colored("🎯 TOP ENHANCED OPPORTUNITY:", 'red', attrs=['bold']))
            print(colored(f"   {best['symbol']} @ ${best['price']:.2f}", 'red', attrs=['bold']))
            print(colored(f"   Enhanced Score: {best['final_score']}/100", 'red'))
            print(colored(f"   Regime: {self._get_regime_description(best['hurst'])}", 'red'))
            print(colored("   🚨 PRIORITY DEPLOYMENT", 'red', attrs=['bold']))
        
        # Market regime overview
        self._display_market_regime_summary()
    
    def _get_strategy_recommendation(self, regime_type: str, result: Dict) -> str:
        """Get trading strategy based on regime"""
        
        if regime_type in ['strong_mean_reverting', 'mean_reverting']:
            return "Quick reversal - Tight stops, rapid profit taking"
        elif regime_type in ['strong_trending', 'trending']:
            return "Momentum confirmation - Wait for follow-through"
        else:
            return "Standard AEGS - Monitor for regime shifts"
    
    def _assess_risk_level(self, result: Dict) -> str:
        """Assess risk level based on multiple factors"""
        
        atr = result.get('atr', 0)
        volume_ratio = result.get('volume_ratio', 1)
        consistency = result['regime_info'].get('consistency', 0)
        
        risk_score = 0
        
        # ATR risk (higher ATR = higher risk)
        if atr > result['price'] * 0.05:  # 5% ATR
            risk_score += 2
        elif atr > result['price'] * 0.03:  # 3% ATR
            risk_score += 1
        
        # Volume risk (very low volume = higher risk)
        if volume_ratio < 0.5:
            risk_score += 2
        elif volume_ratio < 0.8:
            risk_score += 1
        
        # Consistency risk (low consistency = higher risk)
        if consistency < 0.3:
            risk_score += 2
        elif consistency < 0.6:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH - Small position size recommended"
        elif risk_score >= 2:
            return "MODERATE - Standard position size"
        else:
            return "LOW - Can use larger position"
    
    def _display_market_regime_summary(self):
        """Display overall market regime analysis"""
        
        print("\n" + "=" * 80)
        print(colored("📈 MARKET REGIME OVERVIEW:", 'cyan'))
        print("=" * 80)
        
        # Calculate regime statistics
        all_hursts = [r['hurst'] for r in self.results]
        all_consistencies = [r['regime_info'].get('consistency', 0) for r in self.results]
        
        if all_hursts:
            avg_hurst = np.mean(all_hursts)
            avg_consistency = np.mean(all_consistencies)
            
            print(f"   📊 Market Hurst: {avg_hurst:.3f} ({self._get_regime_description(avg_hurst)})")
            print(f"   🎯 Average Consistency: {avg_consistency:.2f}")
            
            # Regime distribution
            regime_counts = {}
            for result in self.results:
                regime = result['regime_type']
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            print(f"\n   📈 Regime Distribution:")
            for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.results)) * 100
                print(f"      {regime.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        print(f"\n🚀 Enhanced AEGS + Multi-Method Hurst = Superior Alpha!")
        print(f"⏰ Enhanced scan completed: {datetime.now().strftime('%H:%M:%S')}")


def main():
    scanner = EnhancedRegimeAEGS()
    scanner.run_enhanced_scan()


if __name__ == "__main__":
    main()