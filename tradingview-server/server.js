/**
 * 🌙 TradingView API Server
 * Provides authenticated access to TradingView data via HTTP endpoints
 */

require('dotenv').config();
const express = require('express');
const TradingView = require('../tradingview-api/main');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// Debug path
console.log('Server directory:', __dirname);
console.log('Public path:', path.join(__dirname, 'public'));
console.log('Does public exist?', require('fs').existsSync(path.join(__dirname, 'public')));

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Also add a specific route for root
app.get('/', (req, res) => {
  // Try multiple possible locations
  const possiblePaths = [
    path.join(__dirname, 'public', 'index.html'),
    path.join(__dirname, 'tradingview-login.html'),
    path.join(process.cwd(), 'tradingview-server', 'public', 'index.html'),
    path.join(process.cwd(), 'public', 'index.html')
  ];
  
  console.log('Looking for login UI in:', possiblePaths);
  
  for (const indexPath of possiblePaths) {
    if (require('fs').existsSync(indexPath)) {
      console.log('Found at:', indexPath);
      return res.sendFile(indexPath);
    }
  }
  
  // If not found, return a simple inline HTML login page
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>TradingView Login</title>
      <style>
        body { 
          background: #1a1a1a; 
          color: white; 
          font-family: sans-serif; 
          display: flex; 
          justify-content: center; 
          align-items: center; 
          height: 100vh; 
          margin: 0;
        }
        .container { 
          background: #2a2a2a; 
          padding: 40px; 
          border-radius: 10px; 
          text-align: center;
          max-width: 400px;
        }
        h1 { font-size: 24px; margin-bottom: 20px; }
        .info { margin: 20px 0; line-height: 1.6; }
        .status { 
          padding: 10px; 
          background: #333; 
          border-radius: 5px; 
          margin: 10px 0;
        }
        a { color: #4a9eff; text-decoration: none; }
        .code { 
          background: #000; 
          padding: 10px; 
          border-radius: 5px; 
          font-family: monospace;
          margin: 10px 0;
          display: block;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>🌙 TradingView Login</h1>
        <div class="status">Server is running on port ${PORT}</div>
        <div class="info">
          <p>The login UI file was not found in the expected location.</p>
          <p>You can still use the API endpoints directly:</p>
          <div class="code">POST /login<br>Body: { "username": "email", "password": "pass" }</div>
          <p>Or check the server console for the file path issue.</p>
          <p><a href="/health">Check Health Status</a></p>
        </div>
      </div>
    </body>
    </html>
  `);
});

// Configuration
const PORT = process.env.TV_SERVER_PORT || 8888;
const TV_USERNAME = process.env.TV_USERNAME;
const TV_PASSWORD = process.env.TV_PASSWORD;

// Global state
let tvClient = null;
let userSession = null;
let charts = new Map(); // Store active charts

// Middleware to ensure authentication
const requireAuth = async (req, res, next) => {
  if (!userSession) {
    return res.status(401).json({ error: 'Not authenticated. Call /login first.' });
  }
  next();
};

// Check login status
app.get('/login-status', (req, res) => {
  res.json({ 
    authenticated: !!userSession,
    user: userSession?.user || null
  });
});

// Get config (for prepopulating form)
app.get('/config', (req, res) => {
  res.json({
    hasUsername: !!TV_USERNAME,
    hasPassword: !!TV_PASSWORD,
    // Only send username for prepopulation, never send password to frontend
    username: TV_USERNAME || '',
    // Indicate if password is configured
    passwordConfigured: !!TV_PASSWORD
  });
});

// Manual login endpoint - returns TradingView login URL
app.post('/manual-login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    console.log('🔐 Manual login requested...');
    
    // Return TradingView login URL
    const loginUrl = 'https://www.tradingview.com/accounts/signin/';
    
    // Start a background process to check for successful login
    if (username && password) {
      // Store credentials for later attempt
      process.env.MANUAL_USERNAME = username;
      process.env.MANUAL_PASSWORD = password;
      
      // Try manual login in background
      setTimeout(async () => {
        try {
          console.log('🔄 Attempting background login...');
          userSession = await TradingView.loginUser(username, password, true); // true = manual mode
          
          if (userSession) {
            tvClient = new TradingView.Client({
              token: userSession.session,
              signature: userSession.signature,
            });
            console.log('✅ Manual login successful!');
          }
        } catch (error) {
          console.log('⏳ Waiting for manual login completion...');
        }
      }, 5000);
    }
    
    res.json({ 
      success: true,
      loginUrl: loginUrl,
      message: 'Please complete login in the browser/iframe'
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Login endpoint
app.post('/login', async (req, res) => {
  try {
    // Use provided credentials or fall back to env variables
    const username = req.body.username || TV_USERNAME;
    const password = req.body.password || TV_PASSWORD;
    
    // If username matches env username and no password provided, use env password
    if (username === TV_USERNAME && !req.body.password && TV_PASSWORD) {
      // Using env password
      console.log('🔐 Using credentials from environment variables');
    }
    
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }
    
    console.log('🔐 Attempting TradingView login...');
    console.log(`📧 Username: ${username}`);
    console.log(`🔑 Password length: ${password ? password.length : 0} chars`);
    console.log(`📍 Password from env: ${!req.body.password}`);
    
    userSession = await TradingView.loginUser(username, password, false);
    
    // Create authenticated client
    tvClient = new TradingView.Client({
      token: userSession.session,
      signature: userSession.signature,
    });
    
    console.log('✅ Login successful!');
    res.json({ 
      success: true, 
      user: userSession.user,
      message: 'Authenticated successfully'
    });
    
  } catch (error) {
    console.error('❌ Login failed:', error);
    res.status(401).json({ error: error.message });
  }
});

// Get real-time chart data
app.post('/chart', requireAuth, async (req, res) => {
  try {
    const { symbol, timeframe = 'D', exchange = 'BINANCE' } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    const chartId = `${exchange}:${symbol}`;
    
    // Create new chart session
    const chart = new tvClient.Session.Chart();
    charts.set(chartId, chart);
    
    // Store data as it comes in
    let latestData = null;
    
    chart.onError((err) => {
      console.error('Chart error:', err);
    });
    
    chart.onSymbolLoaded(() => {
      console.log(`Market "${chart.infos.description}" loaded`);
    });
    
    chart.onUpdate(() => {
      if (chart.periods && chart.periods[0]) {
        latestData = {
          symbol: chartId,
          description: chart.infos.description,
          currency: chart.infos.currency_id,
          ...chart.periods[0],
          indicators: {}
        };
      }
    });
    
    // Set the market
    chart.setMarket(chartId, { timeframe });
    
    // Wait for data
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Close chart to save resources
    chart.delete();
    charts.delete(chartId);
    
    if (latestData) {
      res.json(latestData);
    } else {
      res.status(404).json({ error: 'No data received' });
    }
    
  } catch (error) {
    console.error('Chart error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get indicator values
app.post('/indicator', requireAuth, async (req, res) => {
  try {
    const { 
      symbol, 
      indicator,
      timeframe = '60',
      exchange = 'BINANCE',
      options = {}
    } = req.body;
    
    if (!symbol || !indicator) {
      return res.status(400).json({ error: 'Symbol and indicator required' });
    }
    
    const chartId = `${exchange}:${symbol}`;
    
    // Create chart session
    const chart = new tvClient.Session.Chart();
    
    // Store results
    let indicatorData = null;
    
    chart.onError((err) => {
      console.error('Chart error:', err);
    });
    
    // Set market
    chart.setMarket(chartId, { timeframe });
    
    // Wait for market to load
    await new Promise(resolve => {
      chart.onSymbolLoaded(() => resolve());
    });
    
    // Create indicator
    const indicatorInstance = new TradingView.BuiltInIndicator(indicator);
    
    // Apply any options
    Object.entries(options).forEach(([key, value]) => {
      indicatorInstance.setOption(key, value);
    });
    
    const study = new chart.Study(indicatorInstance);
    
    study.onUpdate(() => {
      indicatorData = study.periods;
    });
    
    // Wait for data
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Cleanup
    chart.delete();
    
    res.json({
      symbol: chartId,
      indicator: indicator,
      data: indicatorData || []
    });
    
  } catch (error) {
    console.error('Indicator error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get historical OHLCV data
app.post('/history', requireAuth, async (req, res) => {
  try {
    const { 
      symbol, 
      timeframe = '60',  // 1, 5, 15, 30, 60, 240, 1D, 1W, 1M
      exchange = 'BINANCE',
      bars = 100  // Number of historical bars to fetch
    } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    const chartId = `${exchange}:${symbol}`;
    
    // Create chart session
    const chart = new tvClient.Session.Chart();
    
    // Store historical data
    let historicalData = [];
    let chartLoaded = false;
    
    chart.onError((err) => {
      console.error('Chart error:', err);
    });
    
    chart.onSymbolLoaded(() => {
      console.log(`Historical data for "${chart.infos.description}" loading...`);
      chartLoaded = true;
    });
    
    chart.onUpdate(() => {
      // chart.periods contains all available candles
      if (chart.periods && chart.periods.length > 0) {
        historicalData = chart.periods.map(period => ({
          time: period.time,
          open: period.open,
          high: period.max,
          low: period.min,
          close: period.close,
          volume: period.volume
        }));
      }
    });
    
    // Set the market with the specified timeframe
    chart.setMarket(chartId, { 
      timeframe: timeframe,
      range: bars  // Request specific number of bars
    });
    
    // Wait for data to load
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Sort by time (oldest first)
    historicalData.sort((a, b) => a.time - b.time);
    
    // Limit to requested number of bars
    if (historicalData.length > bars) {
      historicalData = historicalData.slice(-bars);
    }
    
    // Cleanup
    chart.delete();
    
    res.json({
      symbol: chartId,
      timeframe: timeframe,
      bars: historicalData.length,
      data: historicalData
    });
    
  } catch (error) {
    console.error('History error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get multiple symbols at once
app.post('/batch', requireAuth, async (req, res) => {
  try {
    const { symbols, timeframe = '60', exchange = 'BINANCE' } = req.body;
    
    if (!symbols || !Array.isArray(symbols)) {
      return res.status(400).json({ error: 'Symbols array required' });
    }
    
    const results = [];
    
    for (const symbol of symbols) {
      const chartId = `${exchange}:${symbol}`;
      const chart = new tvClient.Session.Chart();
      
      let data = null;
      
      chart.onUpdate(() => {
        if (chart.periods && chart.periods[0]) {
          data = {
            symbol: symbol,
            exchange: exchange,
            close: chart.periods[0].close,
            open: chart.periods[0].open,
            high: chart.periods[0].max,
            low: chart.periods[0].min,
            volume: chart.periods[0].volume,
            time: chart.periods[0].time
          };
        }
      });
      
      chart.setMarket(chartId, { timeframe });
      
      // Wait for data
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      if (data) {
        results.push(data);
      }
      
      chart.delete();
    }
    
    res.json({ results });
    
  } catch (error) {
    console.error('Batch error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Search symbols
app.get('/search', async (req, res) => {
  try {
    const { query, type = 'crypto' } = req.query;
    
    if (!query) {
      return res.status(400).json({ error: 'Query required' });
    }
    
    const results = await TradingView.searchMarket(query, type);
    res.json(results);
    
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    authenticated: !!userSession,
    uptime: process.uptime()
  });
});

// Logout
app.post('/logout', (req, res) => {
  if (tvClient) {
    tvClient.end();
    tvClient = null;
  }
  userSession = null;
  charts.clear();
  
  res.json({ success: true, message: 'Logged out' });
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n👋 Shutting down TradingView server...');
  
  // Close all charts
  charts.forEach(chart => chart.delete());
  
  // Close client connection
  if (tvClient) {
    tvClient.end();
  }
  
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log('🌙 TradingView API Server');
  console.log(`📡 Running on http://localhost:${PORT}`);
  console.log('📊 Endpoints:');
  console.log('  POST /login - Authenticate with TradingView');
  console.log('  POST /chart - Get real-time chart data');
  console.log('  POST /indicator - Get indicator values');
  console.log('  POST /batch - Get multiple symbols');
  console.log('  GET  /search - Search symbols');
  console.log('  GET  /health - Check server status');
  console.log('  POST /logout - Close session');
  
  if (TV_USERNAME && TV_PASSWORD) {
    console.log('\n🔐 Auto-login enabled with environment credentials');
  } else {
    console.log('\n⚠️  No credentials in environment. Use /login endpoint.');
  }
});