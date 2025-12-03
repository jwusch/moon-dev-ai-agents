"""
⚡ VXX 5-Minute Noise-Filtered Strategy
Advanced noise reduction techniques for 5-minute VXX trading signals

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class NoiseFilterResult:
    filter_name: str
    time_limit_periods: int
    time_limit_minutes: int
    total_trades: int
    win_rate: float
    total_return_pct: float
    avg_return_per_trade: float
    profit_factor: float
    sharpe_ratio: float
    trades_per_day: float
    noise_reduction_score: float

class VXX5MinNoiseFilter:
    """
    Enhanced 5-minute strategy with multiple noise reduction techniques
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 profit_target_pct: float = 5.0,
                 stop_loss_pct: float = 7.5,
                 commission: float = 1.0):
        
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
        
    def prepare_data_with_noise_filters(self, symbol: str = "VXX", period: str = "60d") -> pd.DataFrame:
        """Prepare 5-minute data with comprehensive noise reduction"""
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, interval="5m", progress=False)
            print(f"📊 Downloaded {len(df)} bars of 5-minute {symbol} data")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
        
        if len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Ensure proper column structure
        if df.columns.nlevels > 1:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # ===== NOISE REDUCTION TECHNIQUES =====
        
        # 1. PRICE SMOOTHING - Kalman Filter approximation
        print("   🔧 Applying price smoothing...")
        df['Close_Smooth'] = self.apply_kalman_smoothing(df['Close'])
        
        # 2. VOLUME-WEIGHTED PRICE for better signal quality
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['VWAP_5'] = (df['Close'] * df['Volume']).rolling(60).sum() / df['Volume'].rolling(60).sum()
            df['Price_Final'] = df['VWAP_5'].fillna(df['Close_Smooth'])
        else:
            df['Price_Final'] = df['Close_Smooth']
        
        # 3. ADAPTIVE INDICATORS - adjust periods based on volatility
        print("   🔧 Calculating adaptive indicators...")
        df['ATR_Short'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 20)
        df['Volatility_Regime'] = self.classify_volatility_regime(df['ATR_Short'])
        
        # Adaptive SMA: longer period in high volatility
        df['SMA_Period'] = 60 + (df['Volatility_Regime'] * 20)  # 60-120 periods
        df['SMA'] = df['Price_Final'].rolling(80).mean()  # Use fixed 80 for stability
        df['Distance_Pct'] = (df['Price_Final'] - df['SMA']) / df['SMA'] * 100
        
        # 4. NOISE-FILTERED RSI
        df['RSI_Raw'] = talib.RSI(df['Price_Final'].values, 42)
        df['RSI'] = df['RSI_Raw'].rolling(3).mean()  # 3-period smoothing
        
        # 5. BOLLINGER BANDS for volatility context
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Price_Final'].values, 40, 2, 2)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle * 100
        df['BB_Position'] = (df['Price_Final'] - bb_lower) / (bb_upper - bb_lower)
        
        # 6. TREND FILTER - Only trade in appropriate market conditions
        df['Trend_Fast'] = df['Price_Final'].rolling(20).mean()
        df['Trend_Slow'] = df['Price_Final'].rolling(60).mean()
        df['Trend_Direction'] = np.where(df['Trend_Fast'] > df['Trend_Slow'], 1, -1)
        
        # 7. VOLUME ANALYSIS - Enhanced volume filtering
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(60).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['Volume_Spike'] = df['Volume_Ratio'] > 1.5
            # Only use volume data if it's meaningful
            df['Volume_Quality'] = np.where(df['Volume'].rolling(20).std() > df['Volume'].rolling(20).mean() * 0.1, 1, 0)
        else:
            df['Volume_Ratio'] = 1.0
            df['Volume_Spike'] = False
            df['Volume_Quality'] = 0
        
        # 8. MOMENTUM FILTERING - Multi-timeframe momentum
        df['ROC_1'] = talib.ROC(df['Price_Final'].values, 1)
        df['ROC_5'] = talib.ROC(df['Price_Final'].values, 5)
        df['ROC_20'] = talib.ROC(df['Price_Final'].values, 20)
        
        # Momentum consistency score
        df['Momentum_Consistency'] = self.calculate_momentum_consistency(df)
        
        # 9. MARKET MICROSTRUCTURE - Time-of-day filters
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Time_Score'] = self.calculate_time_score(df['Hour'], df['Minute'])
        
        # 10. REGIME DETECTION - Market state classification
        df['Market_Regime'] = self.detect_market_regime(df)
        
        print(f"   ✅ Applied {10} noise reduction techniques")
        return df.dropna()
    
    def apply_kalman_smoothing(self, prices: pd.Series) -> pd.Series:
        """Simple Kalman filter approximation for price smoothing"""
        smoothed = prices.copy()
        variance = prices.rolling(20).var()
        
        for i in range(1, len(prices)):
            if pd.notna(variance.iloc[i]) and variance.iloc[i] > 0:
                # Simple Kalman gain approximation
                gain = variance.iloc[i] / (variance.iloc[i] + prices.rolling(5).var().iloc[i])
                if pd.notna(gain):
                    smoothed.iloc[i] = smoothed.iloc[i-1] + gain * (prices.iloc[i] - smoothed.iloc[i-1])
        
        return smoothed
    
    def classify_volatility_regime(self, atr: pd.Series) -> pd.Series:
        """Classify volatility into regimes (0=low, 1=medium, 2=high)"""
        atr_ma = atr.rolling(100).mean()
        regime = pd.Series(1, index=atr.index)  # Default medium
        
        regime[atr < atr_ma * 0.8] = 0  # Low volatility
        regime[atr > atr_ma * 1.2] = 2  # High volatility
        
        return regime
    
    def calculate_momentum_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum consistency across timeframes"""
        consistency = pd.Series(0, index=df.index)
        
        # Check if short, medium, and long-term momentum align
        if 'ROC_1' in df.columns and 'ROC_5' in df.columns and 'ROC_20' in df.columns:
            roc1_pos = df['ROC_1'] > 0
            roc5_pos = df['ROC_5'] > 0  
            roc20_pos = df['ROC_20'] > 0
            
            # Count aligned momentum signals
            consistency = (roc1_pos == roc5_pos).astype(int) + (roc5_pos == roc20_pos).astype(int)
        
        return consistency
    
    def calculate_time_score(self, hours: pd.Series, minutes: pd.Series) -> pd.Series:
        """Score time periods based on market activity"""
        time_score = pd.Series(0.5, index=hours.index)  # Default neutral
        
        # Market open (high activity)
        open_mask = (hours == 9) & (minutes >= 30) | (hours == 10)
        time_score[open_mask] = 1.0
        
        # Lunch time (lower activity)
        lunch_mask = (hours >= 12) & (hours <= 13)
        time_score[lunch_mask] = 0.3
        
        # Market close (high activity)
        close_mask = (hours >= 15) & (hours < 16)
        time_score[close_mask] = 1.0
        
        # Extended hours
        extended_mask = (hours < 9) | (hours >= 16)
        time_score[extended_mask] = 0.1
        
        return time_score
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime: 0=ranging, 1=trending_up, 2=trending_down, 3=volatile"""
        regime = pd.Series(0, index=df.index)
        
        if 'BB_Width' in df.columns and 'Trend_Direction' in df.columns:
            bb_width_ma = df['BB_Width'].rolling(20).mean()
            
            # Trending markets
            trending_mask = df['BB_Width'] > bb_width_ma * 1.2
            regime[trending_mask & (df['Trend_Direction'] == 1)] = 1  # Uptrend
            regime[trending_mask & (df['Trend_Direction'] == -1)] = 2  # Downtrend
            
            # Volatile but directionless
            volatile_mask = df['BB_Width'] > bb_width_ma * 1.5
            regime[volatile_mask] = 3
        
        return regime
    
    def calculate_enhanced_quality_score(self, df: pd.DataFrame, idx: int) -> int:
        """Enhanced quality scoring with noise reduction"""
        score = 0
        
        try:
            # 1. Distance from SMA (0-20 points) - more conservative
            distance = df['Distance_Pct'].iloc[idx]
            if distance < -2.5:
                score += 20
            elif distance < -2.0:
                score += 15
            elif distance < -1.5:
                score += 10
            elif distance < -1.0:
                score += 5
            
            # 2. RSI level (0-20 points) - smoothed RSI
            rsi = df['RSI'].iloc[idx]
            if rsi < 20:
                score += 20
            elif rsi < 25:
                score += 15
            elif rsi < 30:
                score += 10
            elif rsi < 35:
                score += 5
            
            # 3. Bollinger Band position (0-15 points)
            if 'BB_Position' in df.columns:
                bb_pos = df['BB_Position'].iloc[idx]
                if pd.notna(bb_pos):
                    if bb_pos < 0.2:  # Near lower band
                        score += 15
                    elif bb_pos < 0.3:
                        score += 10
                    elif bb_pos < 0.4:
                        score += 5
            
            # 4. Volume confirmation (0-15 points) - only if quality volume
            if df['Volume_Quality'].iloc[idx] == 1:
                volume_ratio = df['Volume_Ratio'].iloc[idx]
                if pd.notna(volume_ratio):
                    if volume_ratio > 2.0:
                        score += 15
                    elif volume_ratio > 1.5:
                        score += 10
                    elif volume_ratio > 1.2:
                        score += 5
            
            # 5. Momentum consistency (0-10 points)
            momentum_consistency = df['Momentum_Consistency'].iloc[idx]
            if pd.notna(momentum_consistency):
                score += int(momentum_consistency * 5)  # 0-2 scale to 0-10
            
            # 6. Time of day (0-10 points)
            time_score = df['Time_Score'].iloc[idx]
            if pd.notna(time_score):
                score += int(time_score * 10)
            
            # 7. Market regime bonus/penalty (0-10 points)
            market_regime = df['Market_Regime'].iloc[idx]
            if pd.notna(market_regime):
                if market_regime == 0:  # Ranging market (good for mean reversion)
                    score += 10
                elif market_regime in [1, 2]:  # Trending (bad for mean reversion)
                    score -= 5
                elif market_regime == 3:  # Volatile (neutral)
                    score += 2
            
        except (KeyError, IndexError):
            pass  # Gracefully handle missing data
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def run_enhanced_backtest(self, df: pd.DataFrame, max_hold_periods: int,
                            min_quality_score: int = 60) -> Dict:
        """Run backtest with enhanced noise filtering"""
        trades = []
        current_capital = self.initial_capital
        position = None
        equity_curve = []
        
        # Start after sufficient data for indicators
        start_idx = max(150, len(df) // 4)
        
        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_price = df['Price_Final'].iloc[i]  # Use filtered price
            
            # Market hours filter
            if (current_time.weekday() >= 5 or  
                current_time.hour < 9 or 
                (current_time.hour == 9 and current_time.minute < 30) or
                current_time.hour >= 16):
                continue
            
            # Check for new entry
            if position is None:
                distance = df['Distance_Pct'].iloc[i]
                rsi = df['RSI'].iloc[i]
                
                # Enhanced entry conditions
                base_entry = distance < -1.0 and rsi < 40
                
                # Additional filters
                regime_ok = df['Market_Regime'].iloc[i] in [0, 3]  # Ranging or volatile
                time_ok = df['Time_Score'].iloc[i] > 0.4  # Decent time of day
                
                if base_entry and regime_ok and time_ok:
                    score = self.calculate_enhanced_quality_score(df, i)
                    
                    if score >= min_quality_score:
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'entry_idx': i,
                            'score': score
                        }
            
            # Check for exit
            elif position is not None:
                periods_held = i - position['entry_idx']
                minutes_held = periods_held * 5
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                exit_reason = None
                
                # Enhanced exit conditions
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = 'Profit Target'
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'Stop Loss'
                elif df['Distance_Pct'].iloc[i] > 0.2:  # More conservative mean reversion exit
                    exit_reason = 'Mean Reversion'
                elif periods_held >= max_hold_periods:
                    exit_reason = 'Time Limit'
                # Additional exit: trend change
                elif df['Trend_Direction'].iloc[i] != df['Trend_Direction'].iloc[position['entry_idx']]:
                    exit_reason = 'Trend Change'
                
                if exit_reason:
                    position_size = current_capital * 0.95
                    shares = position_size / position['entry_price']
                    pnl_dollars = shares * (current_price - position['entry_price']) - (2 * self.commission)
                    current_capital += pnl_dollars
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'minutes_held': minutes_held,
                        'periods_held': periods_held,
                        'exit_reason': exit_reason,
                        'entry_score': position['score'],
                        'win': pnl_pct > 0
                    })
                    position = None
            
            equity_curve.append({
                'time': current_time,
                'equity': current_capital
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Analyze results
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_return_pct = (current_capital / self.initial_capital - 1) * 100
        
        # Time metrics
        start_time = trades_df['entry_time'].min()
        end_time = trades_df['exit_time'].max()
        total_days = (end_time - start_time).total_seconds() / (24 * 3600)
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        wins = trades_df[trades_df['win']]['pnl_pct']
        losses = trades_df[~trades_df['win']]['pnl_pct']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            trade_frequency_per_year = trades_per_day * 252
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trade_frequency_per_year)
        else:
            sharpe = 0
        
        # Noise reduction effectiveness (higher score = better filtering)
        avg_entry_score = trades_df['entry_score'].mean()
        noise_reduction_score = avg_entry_score / 100  # Normalize to 0-1
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'avg_return_per_trade': total_return_pct / total_trades if total_trades > 0 else 0,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'trades_per_day': trades_per_day,
            'avg_hold_minutes': trades_df['minutes_held'].mean(),
            'noise_reduction_score': noise_reduction_score,
            'trades_df': trades_df
        }
    
    def test_noise_filters(self, symbol: str = "VXX") -> List[NoiseFilterResult]:
        """Test different noise filtering configurations"""
        print(f"🎯 Testing noise-filtered 5-minute strategy for {symbol}...")
        
        # Prepare enhanced data
        df = self.prepare_data_with_noise_filters(symbol)
        if df is None:
            return []
        
        # Test different quality score thresholds and time limits
        test_configs = [
            {"name": "Conservative", "min_score": 70, "time_limit": 18},   # 1.5h
            {"name": "Balanced", "min_score": 60, "time_limit": 12},       # 1.0h  
            {"name": "Aggressive", "min_score": 50, "time_limit": 9},      # 0.75h
            {"name": "Ultra-Clean", "min_score": 80, "time_limit": 24},    # 2.0h
            {"name": "Quick-Exit", "min_score": 65, "time_limit": 6},      # 0.5h
        ]
        
        results = []
        
        for config in test_configs:
            print(f"   Testing {config['name']} filter (score≥{config['min_score']}, {config['time_limit']*5}min limit)...", end="")
            
            try:
                result = self.run_enhanced_backtest(df, config['time_limit'], config['min_score'])
                
                if 'error' not in result:
                    filter_result = NoiseFilterResult(
                        filter_name=config['name'],
                        time_limit_periods=config['time_limit'],
                        time_limit_minutes=config['time_limit'] * 5,
                        total_trades=result['total_trades'],
                        win_rate=result['win_rate'],
                        total_return_pct=result['total_return_pct'],
                        avg_return_per_trade=result['avg_return_per_trade'],
                        profit_factor=result['profit_factor'],
                        sharpe_ratio=result['sharpe_ratio'],
                        trades_per_day=result['trades_per_day'],
                        noise_reduction_score=result['noise_reduction_score']
                    )
                    results.append(filter_result)
                    print(f" ✅ {result['total_trades']} trades, {result['win_rate']:.1f}% win, {result['total_return_pct']:+.1f}% return")
                else:
                    print(f" ❌ {result['error']}")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        return results

def main():
    """Test noise-filtered 5-minute VXX strategy"""
    print("⚡ VXX 5-MINUTE NOISE-FILTERED STRATEGY")
    print("=" * 60)
    print("Testing advanced noise reduction techniques on 5-minute timeframe")
    print("Goal: Clean up signals to match or exceed 15-minute performance")
    
    optimizer = VXX5MinNoiseFilter(
        initial_capital=10000,
        profit_target_pct=5.0,
        stop_loss_pct=7.5,
        commission=1.0
    )
    
    print(f"\n{'='*20} NOISE REDUCTION TESTING {'='*20}")
    
    try:
        results = optimizer.test_noise_filters("VXX")
        
        if results:
            print(f"\n🏆 NOISE-FILTERED 5-MINUTE RESULTS:")
            print("=" * 90)
            print(f"{'Filter':<12} {'Return%':<8} {'Win%':<6} {'Trades':<7} {'Tr/Day':<7} {'Sharpe':<7} {'Quality':<8}")
            print("-" * 90)
            
            for result in results:
                print(f"{result.filter_name:<12} {result.total_return_pct:<8.1f} "
                      f"{result.win_rate:<6.1f} {result.total_trades:<7} "
                      f"{result.trades_per_day:<7.1f} {result.sharpe_ratio:<7.2f} "
                      f"{result.noise_reduction_score:<8.2f}")
            
            # Find best results
            best_return = max(results, key=lambda x: x.total_return_pct)
            best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
            best_win_rate = max(results, key=lambda x: x.win_rate)
            
            print(f"\n🎯 BEST FILTERED RESULTS:")
            print(f"• Best Return: {best_return.filter_name} → {best_return.total_return_pct:+.1f}% ({best_return.win_rate:.1f}% win)")
            print(f"• Best Sharpe: {best_sharpe.filter_name} → {best_sharpe.sharpe_ratio:.2f}")
            print(f"• Best Win Rate: {best_win_rate.filter_name} → {best_win_rate.win_rate:.1f}%")
            
            # Compare to original results
            print(f"\n📊 NOISE REDUCTION COMPARISON:")
            print(f"• Original 5-min best: +6.8% (43% win rate, noisy)")
            print(f"• Filtered 5-min best: {best_return.total_return_pct:+.1f}% ({best_return.win_rate:.1f}% win rate)")
            print(f"• 15-minute benchmark: +7.9% (60% win rate)")
            
            improvement_vs_original = best_return.total_return_pct - 6.8
            improvement_vs_15min = best_return.total_return_pct - 7.9
            print(f"• Improvement vs original 5-min: {improvement_vs_original:+.1f}%")
            print(f"• Performance vs 15-min: {improvement_vs_15min:+.1f}%")
            
            # Quality assessment
            print(f"\n💡 NOISE REDUCTION INSIGHTS:")
            avg_quality = np.mean([r.noise_reduction_score for r in results])
            print(f"• Average signal quality: {avg_quality:.2f} (0=noisy, 1=clean)")
            print(f"• Best filter trades/day: {best_return.trades_per_day:.1f} (vs 0.4 original)")
            print(f"• Signal improvement: {best_return.noise_reduction_score:.2f} quality score")
            
            if best_return.total_return_pct > 7.9:
                print(f"✅ SUCCESS: Noise filtering beats 15-minute benchmark!")
            elif best_return.win_rate > 50:
                print(f"✅ PARTIAL SUCCESS: Improved win rate, competitive returns")
            else:
                print(f"⚠️ MIXED: Some improvement but still room for optimization")
                
    except Exception as e:
        print(f"❌ Error testing noise filters: {e}")
    
    print(f"\n✅ Noise-filtered 5-minute strategy testing complete!")

if __name__ == "__main__":
    main()