"""
🌙 Moon Dev's Strategy Marketplace Dashboard
Web interface for browsing, analyzing, and downloading trading strategies
Built with love by Moon Dev 🚀
"""

import os
import sys
import json
from flask import Flask, render_template_string, jsonify, request, send_file
from datetime import datetime
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.strategy_registry_agent import StrategyRegistryAgent
from src.marketplace.analytics import StrategyAnalytics
from src.marketplace.exporter import StrategyExporter

app = Flask(__name__)
registry = StrategyRegistryAgent()
analytics = StrategyAnalytics()
exporter = StrategyExporter(registry)

# HTML Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌙 Moon Dev Strategy Marketplace</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #333;
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .search-section {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .search-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        
        input, select, button {
            padding: 10px;
            border: 1px solid #333;
            background: #0a0a0a;
            color: white;
            border-radius: 5px;
            width: 100%;
        }
        
        button {
            background: #667eea;
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #764ba2;
        }
        
        .strategy-grid {
            display: grid;
            gap: 20px;
        }
        
        .strategy-card {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
            transition: all 0.3s;
        }
        
        .strategy-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }
        
        .strategy-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }
        
        .strategy-title {
            font-size: 1.5em;
            color: #667eea;
        }
        
        .strategy-meta {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        
        .tag {
            padding: 5px 10px;
            background: #333;
            border-radius: 15px;
            font-size: 0.8em;
        }
        
        .tag.category {
            background: #2d3748;
        }
        
        .tag.risk-low {
            background: #48bb78;
            color: black;
        }
        
        .tag.risk-medium {
            background: #ed8936;
            color: black;
        }
        
        .tag.risk-high {
            background: #e53e3e;
        }
        
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        
        .perf-item {
            background: #0a0a0a;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        .perf-label {
            font-size: 0.8em;
            color: #888;
        }
        
        .perf-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .rating {
            display: flex;
            gap: 5px;
            align-items: center;
        }
        
        .star {
            color: #ffd700;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
        }
        
        .modal-content {
            position: relative;
            background: #1a1a2e;
            margin: 50px auto;
            padding: 30px;
            width: 80%;
            max-width: 800px;
            border-radius: 10px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close {
            position: absolute;
            right: 20px;
            top: 20px;
            font-size: 30px;
            cursor: pointer;
            color: #888;
        }
        
        .close:hover {
            color: #fff;
        }
        
        @media (max-width: 768px) {
            .strategy-header {
                flex-direction: column;
            }
            
            .action-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌙 Moon Dev Strategy Marketplace</h1>
            <p>Discover, analyze, and share proven trading strategies</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-strategies">0</div>
                <div>Total Strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="active-strategies">0</div>
                <div>Active Strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="total-authors">0</div>
                <div>Contributors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avg-return">0%</div>
                <div>Avg. Return</div>
            </div>
        </div>
        
        <div class="search-section">
            <h2>🔍 Search Strategies</h2>
            <div class="search-grid">
                <input type="text" id="search-query" placeholder="Search by name or description...">
                <select id="filter-category">
                    <option value="">All Categories</option>
                    <option value="momentum">Momentum</option>
                    <option value="mean_reversion">Mean Reversion</option>
                    <option value="technical">Technical</option>
                    <option value="ml_based">ML Based</option>
                </select>
                <select id="filter-risk">
                    <option value="">All Risk Levels</option>
                    <option value="low">Low Risk</option>
                    <option value="medium">Medium Risk</option>
                    <option value="high">High Risk</option>
                </select>
                <select id="sort-by">
                    <option value="rating">Sort by Rating</option>
                    <option value="return">Sort by Return</option>
                    <option value="downloads">Sort by Downloads</option>
                    <option value="recent">Sort by Recent</option>
                </select>
            </div>
            <button onclick="searchStrategies()">Search</button>
        </div>
        
        <div id="strategy-list" class="strategy-grid">
            <!-- Strategies will be loaded here -->
        </div>
    </div>
    
    <!-- Strategy Detail Modal -->
    <div id="strategy-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modal-body">
                <!-- Strategy details will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        // Load marketplace data on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadStrategies();
        });
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('total-strategies').textContent = stats.total_strategies;
                document.getElementById('active-strategies').textContent = stats.active_strategies;
                document.getElementById('total-authors').textContent = stats.total_authors;
                document.getElementById('avg-return').textContent = stats.avg_return + '%';
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        async function loadStrategies() {
            try {
                const response = await fetch('/api/strategies');
                const strategies = await response.json();
                displayStrategies(strategies);
            } catch (error) {
                console.error('Error loading strategies:', error);
            }
        }
        
        async function searchStrategies() {
            const query = document.getElementById('search-query').value;
            const category = document.getElementById('filter-category').value;
            const risk = document.getElementById('filter-risk').value;
            
            const params = new URLSearchParams({
                query: query,
                category: category,
                risk_level: risk
            });
            
            try {
                const response = await fetch(`/api/strategies/search?${params}`);
                const strategies = await response.json();
                displayStrategies(strategies);
            } catch (error) {
                console.error('Error searching strategies:', error);
            }
        }
        
        function displayStrategies(strategies) {
            const container = document.getElementById('strategy-list');
            container.innerHTML = '';
            
            if (strategies.length === 0) {
                container.innerHTML = '<p style="text-align: center; padding: 40px;">No strategies found</p>';
                return;
            }
            
            strategies.forEach(strategy => {
                const card = createStrategyCard(strategy);
                container.appendChild(card);
            });
        }
        
        function createStrategyCard(strategy) {
            const card = document.createElement('div');
            card.className = 'strategy-card';
            
            const perf = strategy.performance_summary || {};
            const rating = strategy.rating || {average: 0, count: 0};
            
            card.innerHTML = `
                <div class="strategy-header">
                    <div>
                        <h3 class="strategy-title">${strategy.name}</h3>
                        <p style="color: #888; margin: 5px 0;">by ${strategy.author}</p>
                        <div class="rating">
                            ${getStarRating(rating.average)}
                            <span>(${rating.count} reviews)</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="tag risk-${strategy.risk_level}">${strategy.risk_level.toUpperCase()} RISK</span>
                    </div>
                </div>
                
                <p style="margin-bottom: 15px;">${strategy.description}</p>
                
                <div class="strategy-meta">
                    ${strategy.category.map(cat => `<span class="tag category">${cat}</span>`).join('')}
                </div>
                
                ${perf.total_return ? `
                <div class="performance-grid">
                    <div class="perf-item">
                        <div class="perf-label">Return</div>
                        <div class="perf-value">${perf.total_return || 'N/A'}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Sharpe</div>
                        <div class="perf-value">${perf.sharpe_ratio || 'N/A'}</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Win Rate</div>
                        <div class="perf-value">${perf.win_rate || 'N/A'}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Max DD</div>
                        <div class="perf-value">${perf.max_drawdown || 'N/A'}%</div>
                    </div>
                </div>
                ` : '<p style="color: #888; text-align: center; padding: 20px;">Performance data not available</p>'}
                
                <div class="action-buttons">
                    <button onclick="viewStrategy('${strategy.strategy_id}')">View Details</button>
                    <button onclick="downloadStrategy('${strategy.strategy_id}')">Download</button>
                    ${strategy.status === 'under_review' ? '<button disabled>Under Review</button>' : ''}
                </div>
            `;
            
            return card;
        }
        
        function getStarRating(rating) {
            const fullStars = Math.floor(rating);
            const halfStar = rating % 1 >= 0.5;
            let stars = '';
            
            for (let i = 0; i < fullStars; i++) {
                stars += '<span class="star">★</span>';
            }
            
            if (halfStar) {
                stars += '<span class="star">☆</span>';
            }
            
            for (let i = fullStars + (halfStar ? 1 : 0); i < 5; i++) {
                stars += '<span style="color: #444;">★</span>';
            }
            
            return stars;
        }
        
        async function viewStrategy(strategyId) {
            try {
                const response = await fetch(`/api/strategies/${strategyId}`);
                const strategy = await response.json();
                showStrategyModal(strategy);
            } catch (error) {
                console.error('Error viewing strategy:', error);
            }
        }
        
        function showStrategyModal(strategy) {
            const modal = document.getElementById('strategy-modal');
            const modalBody = document.getElementById('modal-body');
            
            const perf = strategy.performance_summary || {};
            
            modalBody.innerHTML = `
                <h2>${strategy.name}</h2>
                <p style="color: #888; margin-bottom: 20px;">by ${strategy.author} • Version ${strategy.version}</p>
                
                <h3>Description</h3>
                <p style="margin-bottom: 20px;">${strategy.description}</p>
                
                <h3>Requirements</h3>
                <ul style="margin-bottom: 20px;">
                    <li>Minimum Capital: $${strategy.min_capital}</li>
                    <li>Timeframes: ${strategy.timeframes.join(', ')}</li>
                    <li>Instruments: ${strategy.instruments.join(', ')}</li>
                    <li>Dependencies: ${strategy.dependencies.length > 0 ? strategy.dependencies.join(', ') : 'None'}</li>
                </ul>
                
                ${perf.total_return ? `
                <h3>Performance Metrics</h3>
                <div class="performance-grid" style="margin-bottom: 20px;">
                    <div class="perf-item">
                        <div class="perf-label">Total Return</div>
                        <div class="perf-value">${perf.total_return}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Annual Return</div>
                        <div class="perf-value">${perf.annual_return || 'N/A'}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Sharpe Ratio</div>
                        <div class="perf-value">${perf.sharpe_ratio}</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Sortino Ratio</div>
                        <div class="perf-value">${perf.sortino_ratio || 'N/A'}</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Win Rate</div>
                        <div class="perf-value">${perf.win_rate}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Profit Factor</div>
                        <div class="perf-value">${perf.profit_factor || 'N/A'}</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Max Drawdown</div>
                        <div class="perf-value">${perf.max_drawdown}%</div>
                    </div>
                    <div class="perf-item">
                        <div class="perf-label">Total Trades</div>
                        <div class="perf-value">${perf.total_trades || 'N/A'}</div>
                    </div>
                </div>
                ` : ''}
                
                <div class="action-buttons">
                    <button onclick="downloadStrategy('${strategy.strategy_id}')">Download Strategy</button>
                    <button onclick="rateStrategy('${strategy.strategy_id}')">Rate Strategy</button>
                    <button onclick="closeModal()">Close</button>
                </div>
            `;
            
            modal.style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('strategy-modal').style.display = 'none';
        }
        
        async function downloadStrategy(strategyId) {
            try {
                window.location.href = `/api/strategies/${strategyId}/download`;
            } catch (error) {
                console.error('Error downloading strategy:', error);
            }
        }
        
        async function rateStrategy(strategyId) {
            const rating = prompt('Rate this strategy (1-5 stars):');
            if (rating && rating >= 1 && rating <= 5) {
                try {
                    const response = await fetch(`/api/strategies/${strategyId}/rate`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({rating: parseFloat(rating)})
                    });
                    
                    if (response.ok) {
                        alert('Thank you for rating!');
                        loadStrategies();
                    }
                } catch (error) {
                    console.error('Error rating strategy:', error);
                }
            }
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('strategy-modal');
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/stats')
def get_stats():
    """Get marketplace statistics"""
    strategies = registry.registry['strategies']
    active = sum(1 for s in strategies.values() if s['status'] == 'active')
    
    # Calculate average return
    returns = []
    for strategy in strategies.values():
        if strategy.get('performance_summary', {}).get('total_return'):
            returns.append(strategy['performance_summary']['total_return'])
    
    avg_return = round(sum(returns) / len(returns), 2) if returns else 0
    
    return jsonify({
        'total_strategies': len(strategies),
        'active_strategies': active,
        'total_authors': len(registry.registry['authors']),
        'avg_return': avg_return
    })

@app.route('/api/strategies')
def get_all_strategies():
    """Get all strategies"""
    strategies = list(registry.registry['strategies'].values())
    return jsonify(strategies)

@app.route('/api/strategies/search')
def search_strategies():
    """Search strategies with filters"""
    query = request.args.get('query', '')
    category = request.args.get('category', None)
    risk_level = request.args.get('risk_level', None)
    
    results = registry.search_strategies(
        query=query,
        category=category,
        risk_level=risk_level
    )
    
    return jsonify(results)

@app.route('/api/strategies/<strategy_id>')
def get_strategy(strategy_id):
    """Get a specific strategy"""
    strategy = registry.get_strategy(strategy_id)
    if strategy:
        return jsonify(strategy)
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/download')
def download_strategy(strategy_id):
    """Download strategy package"""
    try:
        # Create package
        package_path = exporter.export_strategy_package(
            strategy_id,
            '/tmp/',
            include_performance=True
        )
        
        # Increment download counter
        registry.increment_downloads(strategy_id)
        
        # Send file
        return send_file(package_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>/rate', methods=['POST'])
def rate_strategy(strategy_id):
    """Rate a strategy"""
    data = request.get_json()
    rating = data.get('rating', 0)
    
    if 1 <= rating <= 5:
        registry.update_rating(strategy_id, rating)
        return jsonify({'success': True})
    
    return jsonify({'error': 'Invalid rating'}), 400

@app.route('/api/strategies', methods=['POST'])
def submit_strategy():
    """Submit a new strategy (placeholder)"""
    # This would handle strategy submission
    # For now, return not implemented
    return jsonify({'error': 'Not implemented yet'}), 501


if __name__ == '__main__':
    print("🌙 Starting Moon Dev Strategy Marketplace Dashboard...")
    print("📊 Access at: http://localhost:8002")
    app.run(host='0.0.0.0', port=8002, debug=True)