"""
🌐 Web-Based Quality Score Dashboard
Real-time VXX Mean Reversion 15 monitoring with modern UI

Author: Claude (Anthropic)
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
from threading import Thread, Lock
import pandas as pd
import sys
import os

# Import our existing monitoring system
from improved_entry_timing import ImprovedEntryTiming
from claude_datetime_tool import ClaudeDateTimeTool

# Suppress output during data fetching
class QuietImprover(ImprovedEntryTiming):
    def prepare_data(self, symbol, period="5d"):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            result = super().prepare_data(symbol, period)
            return result
        finally:
            sys.stdout = original_stdout

app = Flask(__name__)
CORS(app)

# Global data storage
data_lock = Lock()
market_data = {}
last_update = None

# Initialize systems
improver = QuietImprover()
datetime_tool = ClaudeDateTimeTool()

# Default symbols
SYMBOLS = [
    'VXX', 'SQQQ', 'AMD', 'NVDA', 'TSLA', 'VIXY',
    'UVXY', 'SPXS', 'SPY', 'QQQ', 'TQQQ', 'XLF',
    'BTC-USD', 'ETH-USD', 'SOL-USD'
]

def get_symbol_data(symbol):
    """Get current data for a symbol"""
    try:
        df = improver.prepare_data(symbol, period="5d")
        if df is None or len(df) < 50:
            return None
            
        # Get last valid bar
        idx = len(df) - 1
        
        # Skip if outside market hours for non-crypto
        if not symbol.endswith('-USD'):
            current_time = df.index[idx]
            if current_time.hour < 9 or current_time.hour >= 16:
                for i in range(len(df)-1, max(0, len(df)-10), -1):
                    if 9 <= df.index[i].hour < 16:
                        idx = i
                        break
        
        # Get values
        price = float(df['Close'].iloc[idx])
        sma = float(df['SMA20'].iloc[idx]) if not pd.isna(df['SMA20'].iloc[idx]) else price
        distance = float(df['Distance%'].iloc[idx]) if not pd.isna(df['Distance%'].iloc[idx]) else 0
        rsi = float(df['RSI'].iloc[idx]) if not pd.isna(df['RSI'].iloc[idx]) else 50
        volume_ratio = float(df['Volume_Ratio'].iloc[idx]) if 'Volume_Ratio' in df and not pd.isna(df['Volume_Ratio'].iloc[idx]) else 1.0
        
        # Check for signal
        signal = None
        score = 0
        details = {}
        
        if not pd.isna(distance) and not pd.isna(rsi):
            if distance < -1.0 and rsi < 40:
                signal = 'LONG'
                score, details = improver.evaluate_entry_quality(df, idx)
            elif distance > 1.0 and rsi > 60:
                signal = 'SHORT'
                score, details = improver.evaluate_entry_quality(df, idx)
        
        return {
            'symbol': symbol,
            'price': round(price, 2),
            'sma': round(sma, 2),
            'distance': round(distance, 2),
            'rsi': round(rsi, 1),
            'volume_ratio': round(volume_ratio, 2),
            'signal': signal,
            'score': score,
            'details': details,
            'timestamp': df.index[idx].isoformat()
        }
        
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")
        return None

def update_market_data():
    """Background thread to update market data"""
    global market_data, last_update
    
    while True:
        try:
            new_data = {}
            
            for symbol in SYMBOLS:
                data = get_symbol_data(symbol)
                if data:
                    new_data[symbol] = data
            
            with data_lock:
                market_data = new_data
                last_update = datetime.now()
            
            time.sleep(10)  # Update every 10 seconds
            
        except Exception as e:
            print(f"Update error: {e}")
            time.sleep(30)

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint for current market data"""
    with data_lock:
        # Get datetime info
        dt_info = datetime_tool.get_current_datetime("standard")
        
        # Check market status
        market_hours = {}
        for market in ['NYSE', 'CRYPTO']:
            status = datetime_tool.check_market_hours(market)
            market_hours[market] = status
        
        return jsonify({
            'data': market_data,
            'last_update': last_update.isoformat() if last_update else None,
            'datetime_info': dt_info,
            'market_hours': market_hours
        })

@app.route('/api/symbols')
def get_symbols():
    """Get list of monitored symbols"""
    return jsonify({'symbols': SYMBOLS})

# Create the HTML template
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VXX Mean Reversion 15 - Live Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .quality-bar {
            transition: width 0.3s ease;
        }
        .signal-card {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.8; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-blue-400 mb-2">VXX Mean Reversion 15</h1>
            <p class="text-gray-400">Real-Time Quality Score Monitor</p>
            <div id="datetime" class="mt-2 text-sm text-gray-500"></div>
        </div>

        <!-- Market Status -->
        <div id="market-status" class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <!-- Populated by JS -->
        </div>

        <!-- Active Signals -->
        <div id="signals-section" class="mb-8">
            <h2 class="text-2xl font-bold text-green-400 mb-4">🎯 Active Signals</h2>
            <div id="active-signals" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Populated by JS -->
            </div>
        </div>

        <!-- All Symbols Table -->
        <div class="bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 class="text-2xl font-bold text-yellow-400 mb-4">📊 Market Overview</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead>
                        <tr class="border-b border-gray-700">
                            <th class="px-4 py-2">Symbol</th>
                            <th class="px-4 py-2">Price</th>
                            <th class="px-4 py-2">Distance</th>
                            <th class="px-4 py-2">RSI</th>
                            <th class="px-4 py-2">Volume</th>
                            <th class="px-4 py-2">Signal</th>
                            <th class="px-4 py-2">Score</th>
                        </tr>
                    </thead>
                    <tbody id="symbols-table">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <div class="text-center mt-8 text-sm text-gray-500">
            <p>Updates every 10 seconds | Entry: Distance &lt;-1% + RSI &lt;40 (LONG) | Distance &gt;1% + RSI &gt;60 (SHORT)</p>
        </div>
    </div>

    <script>
        let lastData = {};

        function formatTime(isoString) {
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        }

        function getQualityColor(score) {
            if (score >= 70) return 'bg-green-500';
            if (score >= 50) return 'bg-yellow-500';
            return 'bg-red-500';
        }

        function getDistanceColor(distance) {
            if (distance < -2) return 'text-green-400';
            if (distance > 2) return 'text-red-400';
            return 'text-gray-400';
        }

        function getRSIColor(rsi) {
            if (rsi < 30) return 'text-green-400';
            if (rsi > 70) return 'text-red-400';
            return 'text-gray-400';
        }

        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    // Update datetime
                    document.getElementById('datetime').innerHTML = `
                        ${data.datetime_info.local_time} | 
                        <span class="${data.datetime_info.is_market_hours ? 'text-green-400' : 'text-red-400'}">
                            US Markets: ${data.datetime_info.is_market_hours ? 'OPEN' : 'CLOSED'}
                        </span>
                    `;

                    // Update market status
                    const marketStatus = document.getElementById('market-status');
                    marketStatus.innerHTML = `
                        <div class="bg-gray-800 rounded p-4">
                            <h3 class="font-bold text-sm mb-1">NYSE</h3>
                            <p class="${data.market_hours.NYSE.is_open ? 'text-green-400' : 'text-red-400'}">
                                ${data.market_hours.NYSE.is_open ? 'OPEN' : 'CLOSED'}
                            </p>
                        </div>
                        <div class="bg-gray-800 rounded p-4">
                            <h3 class="font-bold text-sm mb-1">CRYPTO</h3>
                            <p class="text-green-400">24/7</p>
                        </div>
                    `;

                    // Filter and sort signals
                    const signals = Object.values(data.data)
                        .filter(d => d.signal)
                        .sort((a, b) => b.score - a.score);

                    // Update active signals
                    const signalsDiv = document.getElementById('active-signals');
                    if (signals.length > 0) {
                        signalsDiv.innerHTML = signals.map(sig => `
                            <div class="bg-gray-800 rounded-lg p-6 signal-card">
                                <div class="flex justify-between items-center mb-4">
                                    <h3 class="text-xl font-bold">${sig.symbol}</h3>
                                    <span class="${sig.signal === 'LONG' ? 'text-green-400' : 'text-red-400'} font-bold">
                                        ${sig.signal} ${sig.signal === 'LONG' ? '↑' : '↓'}
                                    </span>
                                </div>
                                <div class="mb-4">
                                    <p class="text-2xl font-bold">$${sig.price}</p>
                                    <p class="text-sm text-gray-400">Distance: ${sig.distance}% | RSI: ${sig.rsi}</p>
                                </div>
                                <div class="mb-2">
                                    <div class="flex justify-between text-sm mb-1">
                                        <span>Quality Score</span>
                                        <span>${sig.score}/100</span>
                                    </div>
                                    <div class="bg-gray-700 rounded-full h-4 overflow-hidden">
                                        <div class="quality-bar ${getQualityColor(sig.score)} h-full" 
                                             style="width: ${sig.score}%"></div>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        signalsDiv.innerHTML = '<p class="text-gray-500">No active signals</p>';
                    }

                    // Update symbols table
                    const tableBody = document.getElementById('symbols-table');
                    tableBody.innerHTML = Object.values(data.data)
                        .sort((a, b) => b.score - a.score)
                        .map(d => `
                            <tr class="border-b border-gray-700 hover:bg-gray-750">
                                <td class="px-4 py-2 font-bold">${d.symbol}</td>
                                <td class="px-4 py-2">$${d.price}</td>
                                <td class="px-4 py-2 ${getDistanceColor(d.distance)}">${d.distance}%</td>
                                <td class="px-4 py-2 ${getRSIColor(d.rsi)}">${d.rsi}</td>
                                <td class="px-4 py-2">${d.volume_ratio}x</td>
                                <td class="px-4 py-2">
                                    ${d.signal ? `<span class="${d.signal === 'LONG' ? 'text-green-400' : 'text-red-400'}">${d.signal}</span>` : '-'}
                                </td>
                                <td class="px-4 py-2">
                                    ${d.score > 0 ? `<span class="${getQualityColor(d.score)} px-2 py-1 rounded text-xs">${d.score}</span>` : '-'}
                                </td>
                            </tr>
                        `).join('');

                    lastData = data;
                })
                .catch(error => console.error('Error:', error));
        }

        // Update every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""

# Create templates directory and save HTML
import os
os.makedirs('templates', exist_ok=True)
with open('templates/dashboard.html', 'w') as f:
    f.write(dashboard_html)

def main():
    print("🌐 Starting VXX Mean Reversion 15 Web Dashboard")
    print("=" * 60)
    
    # Start background data updater
    updater = Thread(target=update_market_data, daemon=True)
    updater.start()
    
    print("✅ Dashboard running at: http://localhost:5000")
    print("✅ Updates every 10 seconds")
    print("\nPress Ctrl+C to stop")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()