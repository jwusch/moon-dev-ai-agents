"""
🔥💎 AEGS ENHANCED STRATEGY WITH TREND FILTERING 💎🔥
Adds multiple trend filters to avoid counter-trend trades

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class AEGSTrendFiltered:
    """Enhanced AEGS with comprehensive trend filtering"""
    
    def __init__(self):
        self.trend_filters = {
            'sma_trend': True,      # Simple moving average trend
            'ema_trend': True,      # Exponential moving average trend  
            'adx_filter': True,     # ADX trend strength
            'market_regime': True,  # Market regime detection
            'volume_trend': True,   # Volume trend confirmation
            'momentum_filter': True # Price momentum filter
        }
        
    def calculate_indicators(self, df):
        """Calculate all technical indicators including trend filters"""
        
        # Basic AEGS indicators
        df = self._calculate_basic_indicators(df)
        
        # Trend filtering indicators
        df = self._calculate_trend_indicators(df)
        
        return df
    
    def _calculate_basic_indicators(self, df):
        """Original AEGS indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Daily change
        df['Daily_Change'] = df['Close'].pct_change()
        
        return df
    
    def _calculate_trend_indicators(self, df):
        """Calculate trend filtering indicators"""
        
        # 1. Moving Average Trends
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        
        # 2. ADX for trend strength
        df = self._calculate_adx(df)
        
        # 3. Price momentum
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # 4. Volume trend
        df['Volume_MA50'] = df['Volume'].rolling(50).mean()
        df['Volume_Trend'] = df['Volume_MA20'] / df['Volume_MA50'] - 1
        
        # 5. Market regime detection
        df = self._detect_market_regime(df)
        
        return df
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Directional Movement
        df['DM_Plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        
        df['DM_Minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Smoothed values
        df['TR_Smooth'] = df['TR'].rolling(period).sum()
        df['DM_Plus_Smooth'] = df['DM_Plus'].rolling(period).sum()
        df['DM_Minus_Smooth'] = df['DM_Minus'].rolling(period).sum()
        
        # Directional Indicators
        df['DI_Plus'] = 100 * df['DM_Plus_Smooth'] / df['TR_Smooth']
        df['DI_Minus'] = 100 * df['DM_Minus_Smooth'] / df['TR_Smooth']
        
        # ADX
        df['DX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = df['DX'].rolling(period).mean()
        
        return df
    
    def _detect_market_regime(self, df):
        """Detect market regime (trending vs ranging)"""
        
        # Calculate regime score
        regime_score = 0
        
        # Price vs moving averages
        if len(df) > 50:
            recent_closes = df['Close'].tail(10)
            sma50_recent = df['SMA50'].tail(10)
            
            # Count how many recent closes are above SMA50
            above_sma = (recent_closes > sma50_recent).sum()
            regime_score += (above_sma - 5) * 10  # +/- 50 points max
        
        # Trend consistency
        if len(df) > 20:
            price_changes = df['Close'].pct_change().tail(20)
            positive_days = (price_changes > 0).sum()
            regime_score += (positive_days - 10) * 5  # +/- 50 points max
        
        # Volatility regime
        if len(df) > 20:
            recent_vol = df['Close'].pct_change().tail(20).std()
            historical_vol = df['Close'].pct_change().std()
            if recent_vol < historical_vol * 0.8:
                regime_score += 20  # Low volatility = trending
            elif recent_vol > historical_vol * 1.2:
                regime_score -= 20  # High volatility = ranging
        
        # Classify regime
        if regime_score > 50:
            regime = "BULL_TREND"
        elif regime_score > 20:
            regime = "WEAK_BULL"
        elif regime_score > -20:
            regime = "RANGING"
        elif regime_score > -50:
            regime = "WEAK_BEAR"
        else:
            regime = "BEAR_TREND"
        
        df['Market_Regime'] = regime
        df['Regime_Score'] = regime_score
        
        return df
    
    def check_trend_filters(self, row):
        """Check all trend filters for trade approval"""
        
        filters_passed = {}
        filters_passed['total_score'] = 0
        
        # 1. SMA Trend Filter (30 points)
        if pd.notna(row.get('SMA50')) and pd.notna(row.get('SMA200')):
            if row['Close'] > row['SMA50'] > row['SMA200']:
                filters_passed['sma_trend'] = 30
                filters_passed['total_score'] += 30
            elif row['Close'] > row['SMA50']:
                filters_passed['sma_trend'] = 15
                filters_passed['total_score'] += 15
            else:
                filters_passed['sma_trend'] = -20
                filters_passed['total_score'] -= 20
        
        # 2. EMA Trend Filter (25 points)
        if pd.notna(row.get('EMA21')) and pd.notna(row.get('EMA50')):
            if row['EMA21'] > row['EMA50']:
                filters_passed['ema_trend'] = 25
                filters_passed['total_score'] += 25
            else:
                filters_passed['ema_trend'] = -15
                filters_passed['total_score'] -= 15
        
        # 3. ADX Trend Strength Filter (20 points)
        if pd.notna(row.get('ADX')) and pd.notna(row.get('DI_Plus')):
            if row['ADX'] > 25 and row['DI_Plus'] > row.get('DI_Minus', 0):
                filters_passed['adx_filter'] = 20
                filters_passed['total_score'] += 20
            elif row['ADX'] < 20:
                filters_passed['adx_filter'] = -10  # Weak trend
                filters_passed['total_score'] -= 10
        
        # 4. Market Regime Filter (40 points)
        regime = row.get('Market_Regime', 'RANGING')
        if regime in ['BULL_TREND', 'WEAK_BULL']:
            regime_points = 40 if regime == 'BULL_TREND' else 25
            filters_passed['market_regime'] = regime_points
            filters_passed['total_score'] += regime_points
        elif regime == 'RANGING':
            filters_passed['market_regime'] = 0
        else:  # BEAR or WEAK_BEAR
            regime_points = -30 if regime == 'BEAR_TREND' else -15
            filters_passed['market_regime'] = regime_points
            filters_passed['total_score'] += regime_points
        
        # 5. Volume Trend Filter (15 points)
        if pd.notna(row.get('Volume_Trend')):
            if row['Volume_Trend'] > 0.1:  # Volume increasing
                filters_passed['volume_trend'] = 15
                filters_passed['total_score'] += 15
            elif row['Volume_Trend'] < -0.2:  # Volume declining
                filters_passed['volume_trend'] = -10
                filters_passed['total_score'] -= 10
        
        # 6. Momentum Filter (20 points)
        if pd.notna(row.get('Momentum_20')):
            if row['Momentum_20'] > 0.05:  # Strong positive momentum
                filters_passed['momentum_filter'] = 20
                filters_passed['total_score'] += 20
            elif row['Momentum_20'] > 0:  # Weak positive momentum
                filters_passed['momentum_filter'] = 10
                filters_passed['total_score'] += 10
            elif row['Momentum_20'] < -0.1:  # Strong negative momentum
                filters_passed['momentum_filter'] = -25
                filters_passed['total_score'] -= 25
        
        return filters_passed
    
    def generate_signals(self, df, min_trend_score=50):
        """Generate AEGS signals with trend filtering"""
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Trend_Score'] = 0
        df['Trade_Approved'] = False
        
        for i in range(50, len(df)):  # Start after all indicators are calculated
            row = df.iloc[i]
            
            # Calculate basic AEGS signal strength
            signal_strength = 0
            
            # RSI oversold
            if pd.notna(row['RSI']):
                if row['RSI'] < 30:
                    signal_strength += 35
                elif row['RSI'] < 35:
                    signal_strength += 20
            
            # Bollinger Band position
            if pd.notna(row['BB_Position']):
                if row['BB_Position'] < 0:  # Below lower band
                    signal_strength += 35
                elif row['BB_Position'] < 0.2:
                    signal_strength += 20
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
                if row['Volume_Ratio'] > 2.0 and row['Daily_Change'] < -0.02:
                    signal_strength += 30
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
            
            # Price drop magnitude
            if pd.notna(row['Daily_Change']):
                daily_change_pct = row['Daily_Change'] * 100
                if daily_change_pct < -10:
                    signal_strength += 35
                elif daily_change_pct < -5:
                    signal_strength += 20
            
            # Check trend filters
            trend_filters = self.check_trend_filters(row)
            trend_score = trend_filters['total_score']
            
            # Store values
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            df.iloc[i, df.columns.get_loc('Trend_Score')] = trend_score
            
            # Only approve trades if both AEGS signal AND trend filters pass
            if signal_strength >= 70 and trend_score >= min_trend_score:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                df.iloc[i, df.columns.get_loc('Trade_Approved')] = True
        
        return df


def backtest_trend_filtered_aegs(symbol, period='2y', min_trend_score=50):
    """Backtest AEGS strategy with trend filtering"""
    
    print(f'🔥💎 TREND-FILTERED AEGS BACKTEST: {symbol} 💎🔥')
    print('=' * 60)
    print(f'Minimum Trend Score Required: {min_trend_score}/150')
    print('=' * 60)
    
    # Get data
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if len(df) < 200:
            print(f'❌ Insufficient data for {symbol}')
            return None
            
    except Exception as e:
        print(f'❌ Error fetching {symbol}: {e}')
        return None
    
    # Initialize strategy
    strategy = AEGSTrendFiltered()
    
    # Calculate indicators and signals
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df, min_trend_score)
    
    # Backtest performance
    df['Position'] = 0
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = 0
    
    position = 0
    entry_price = 0
    trades = []
    rejected_signals = 0
    
    for i in range(len(df)):
        # Check for rejected signals (AEGS signal but trend filter failed)
        if df.iloc[i]['Signal_Strength'] >= 70 and not df.iloc[i]['Trade_Approved']:
            rejected_signals += 1
        
        if df.iloc[i]['Signal'] == 1 and position == 0:
            # Enter long position
            position = 1
            entry_price = df.iloc[i]['Close']
            df.iloc[i, df.columns.get_loc('Position')] = 1
            
        elif position == 1:
            df.iloc[i, df.columns.get_loc('Position')] = 1
            
            # Exit conditions
            current_price = df.iloc[i]['Close']
            returns = (current_price - entry_price) / entry_price
            
            # Take profit at 50% or stop loss at -20%
            if returns >= 0.5 or returns <= -0.2:
                # Exit position
                position = 0
                exit_price = current_price
                trade_return = returns
                
                # Get trade details
                trade_date = df.index[i].strftime('%Y-%m-%d')
                regime = df.iloc[i].get('Market_Regime', 'Unknown')
                trend_score = df.iloc[i].get('Trend_Score', 0)
                
                trades.append({
                    'entry_date': df.index[i-1].strftime('%Y-%m-%d'),
                    'exit_date': trade_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return * 100,
                    'market_regime': regime,
                    'trend_score': trend_score,
                    'days_held': 1
                })
                
                df.iloc[i, df.columns.get_loc('Position')] = 0
    
    # Calculate strategy returns
    for i in range(1, len(df)):
        if df.iloc[i]['Position'] == 1:
            df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
    
    # Performance metrics
    total_return = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    
    winning_trades = [t for t in trades if t['return_pct'] > 0]
    losing_trades = [t for t in trades if t['return_pct'] < 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
    
    # Display results
    print(f'📊 TREND-FILTERED BACKTEST RESULTS:')
    print(f'   Period: {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}')
    print(f'   Total Trades: {len(trades)}')
    print(f'   Rejected Signals: {rejected_signals}')
    print(f'   Signal Selectivity: {rejected_signals/(len(trades)+rejected_signals)*100:.1f}%' if (len(trades)+rejected_signals) > 0 else 0)
    print(f'   Win Rate: {win_rate:.1f}%')
    print(f'   Average Win: {avg_win:.1f}%')
    print(f'   Average Loss: {avg_loss:.1f}%')
    print(f'   Strategy Return: {total_return*100:.1f}%')
    print(f'   Buy & Hold Return: {buy_hold_return*100:.1f}%')
    print(f'   Excess Return: {(total_return - buy_hold_return)*100:.1f}%')
    
    if trades:
        print(f'\n🎯 RECENT TRADES (with trend context):')
        for trade in trades[-5:]:  # Last 5 trades
            color = '🟢' if trade['return_pct'] > 0 else '🔴'
            regime_color = '🟢' if 'BULL' in trade['market_regime'] else '🔴' if 'BEAR' in trade['market_regime'] else '🟡'
            print(f'   {color} {trade["entry_date"]}: {trade["return_pct"]:+.1f}% | {regime_color} {trade["market_regime"]} | Trend: {trade["trend_score"]}/150')
    
    # Regime analysis
    if trades:
        regime_performance = {}
        for trade in trades:
            regime = trade['market_regime']
            if regime not in regime_performance:
                regime_performance[regime] = {'trades': 0, 'total_return': 0, 'wins': 0}
            regime_performance[regime]['trades'] += 1
            regime_performance[regime]['total_return'] += trade['return_pct']
            if trade['return_pct'] > 0:
                regime_performance[regime]['wins'] += 1
        
        print(f'\n📈 PERFORMANCE BY MARKET REGIME:')
        for regime, stats in regime_performance.items():
            avg_return = stats['total_return'] / stats['trades']
            win_rate = stats['wins'] / stats['trades'] * 100
            print(f'   {regime}: {stats["trades"]} trades, {avg_return:+.1f}% avg, {win_rate:.0f}% wins')
    
    # Save results
    results = {
        'symbol': symbol,
        'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'min_trend_score': min_trend_score,
        'total_trades': len(trades),
        'rejected_signals': rejected_signals,
        'signal_selectivity': rejected_signals/(len(trades)+rejected_signals)*100 if (len(trades)+rejected_signals) > 0 else 0,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'strategy_return': total_return * 100,
        'buy_hold_return': buy_hold_return * 100,
        'excess_return': (total_return - buy_hold_return) * 100,
        'trades': trades[-10:]  # Last 10 trades
    }
    
    return results, df


if __name__ == '__main__':
    print("🔥💎 TESTING TREND-FILTERED AEGS STRATEGY 💎🔥\n")
    
    # Test TLRY with trend filtering
    symbol = 'TLRY'
    min_trend_score = 50  # Moderate trend filter
    
    print(f"{'='*80}")
    print(f"TESTING TREND-FILTERED AEGS ON {symbol}")
    print(f"Minimum Trend Score: {min_trend_score}/150")
    print(f"{'='*80}")
    
    results, df = backtest_trend_filtered_aegs(symbol, period='2y', min_trend_score=min_trend_score)
    
    if results:
        filename = f'aegs_trend_filtered_{symbol}_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n💾 Results saved to {filename}')