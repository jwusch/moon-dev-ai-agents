/**
 * 🌙 TradingView API Server - Session Token Version
 * Uses pre-extracted session tokens instead of password login
 */

require('dotenv').config();
const express = require('express');
const TradingView = require('../tradingview-api/main');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Configuration
const PORT = process.env.TV_SESSION_PORT || 8889; // Different port
const TV_SESSION_ID = process.env.TV_SESSION_ID || process.env.TV_SESSION;
const TV_SESSION_SIGNATURE = process.env.TV_SESSION_SIGNATURE;

// Debug logging
console.log('\n🔍 Checking environment variables:');
console.log(`TV_SESSION_ID: ${TV_SESSION_ID ? `Found (${TV_SESSION_ID.length} chars)` : 'NOT FOUND'}`);
console.log(`TV_SESSION_SIGNATURE: ${TV_SESSION_SIGNATURE ? `Found (${TV_SESSION_SIGNATURE.length} chars)` : 'NOT FOUND'}`);
console.log(`TV_SESSION (alt): ${process.env.TV_SESSION ? `Found (${process.env.TV_SESSION.length} chars)` : 'NOT FOUND'}`);

// Show all env vars starting with TV_
console.log('\n📋 All TV_ variables in .env:');
Object.keys(process.env).filter(k => k.startsWith('TV_')).forEach(key => {
  console.log(`  ${key}: ${process.env[key] ? 'Set' : 'Not set'}`);
});

// Global client
let tvClient = null;

// Initialize client with session tokens
function initializeClient() {
  if (!TV_SESSION_ID || !TV_SESSION_SIGNATURE) {
    console.error('❌ TV_SESSION_ID and TV_SESSION_SIGNATURE must be set in .env');
    console.log('\n📝 How to get them:');
    console.log('1. Login to tradingview.com in your browser');
    console.log('2. Open DevTools (F12) → Application → Cookies');
    console.log('3. Copy values of "sessionid" and "sessionid_sign"');
    console.log('4. Add to .env:');
    console.log('   TV_SESSION_ID=your_sessionid_value');
    console.log('   TV_SESSION_SIGNATURE=your_sessionid_sign_value\n');
    return false;
  }

  try {
    console.log('🔄 Attempting to create TradingView client...');
    console.log(`📋 Using session ID: ${TV_SESSION_ID.substring(0, 8)}...`);
    console.log(`📋 Using signature: ${TV_SESSION_SIGNATURE.substring(0, 10)}...`);
    
    // Create client with existing session
    tvClient = new TradingView.Client({
      token: TV_SESSION_ID,
      signature: TV_SESSION_SIGNATURE,
    });
    
    console.log('✅ TradingView client initialized with session tokens!');
    
    // Test the client
    setTimeout(() => {
      console.log('🧪 Testing client connection...');
      testClientConnection();
    }, 2000);
    
    return true;
  } catch (error) {
    console.error('❌ Failed to initialize client:', error.message);
    console.error('📍 Full error:', error);
    return false;
  }
}

function testClientConnection() {
  try {
    console.log('📡 Creating test chart session...');
    const chart = new tvClient.Session.Chart();
    
    chart.onError((err) => {
      console.error('🚨 Chart error:', err);
    });
    
    chart.onSymbolLoaded(() => {
      console.log('✅ Chart loaded successfully - session is working!');
      chart.delete();
    });
    
    // Test with a simple symbol
    chart.setMarket('BINANCE:BTCUSDT', { timeframe: '1D' });
    
    setTimeout(() => {
      console.log('⏱️ Test timeout - checking if chart is still active...');
      if (chart) chart.delete();
    }, 5000);
    
  } catch (error) {
    console.error('❌ Client test failed:', error.message);
    console.log('💡 This might mean the session tokens are expired or invalid');
    console.log('💡 Try logging in to TradingView again and extracting fresh tokens');
  }
}

// Middleware to check client
const requireClient = (req, res, next) => {
  if (!tvClient) {
    return res.status(401).json({ 
      error: 'Client not initialized. Check session tokens in .env' 
    });
  }
  next();
};

// Get real-time chart data
app.post('/chart', requireClient, async (req, res) => {
  try {
    const { symbol, timeframe = 'D', exchange = 'BINANCE' } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    const chartId = `${exchange}:${symbol}`;
    const chart = new tvClient.Session.Chart();
    
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
          ...chart.periods[0]
        };
      }
    });
    
    chart.setMarket(chartId, { timeframe });
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    chart.delete();
    
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

// Get historical data
app.post('/history', requireClient, async (req, res) => {
  try {
    const { symbol, timeframe = '60', exchange = 'BINANCE', bars = 100 } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ error: 'Symbol required' });
    }
    
    const chartId = `${exchange}:${symbol}`;
    const chart = new tvClient.Session.Chart();
    
    let historicalData = [];
    
    chart.onUpdate(() => {
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
    
    chart.setMarket(chartId, { timeframe, range: bars });
    
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    historicalData.sort((a, b) => a.time - b.time);
    
    if (historicalData.length > bars) {
      historicalData = historicalData.slice(-bars);
    }
    
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

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    clientInitialized: !!tvClient,
    sessionConfigured: !!(TV_SESSION_ID && TV_SESSION_SIGNATURE),
    uptime: process.uptime()
  });
});

// Session info
app.get('/session', (req, res) => {
  res.json({
    hasSessionId: !!TV_SESSION_ID,
    hasSignature: !!TV_SESSION_SIGNATURE,
    sessionIdLength: TV_SESSION_ID ? TV_SESSION_ID.length : 0,
    signatureLength: TV_SESSION_SIGNATURE ? TV_SESSION_SIGNATURE.length : 0,
    clientActive: !!tvClient
  });
});

// Instructions page
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>TradingView Session API</title>
      <style>
        body { 
          background: #1a1a1a; 
          color: white; 
          font-family: monospace; 
          padding: 40px;
          line-height: 1.6;
        }
        h1 { color: #4a9eff; }
        .code { 
          background: #000; 
          padding: 20px; 
          border-radius: 5px;
          margin: 20px 0;
          overflow-x: auto;
        }
        .status { 
          padding: 10px; 
          background: ${tvClient ? '#1b5e20' : '#b71c1c'}; 
          border-radius: 5px; 
          display: inline-block;
          margin: 10px 0;
        }
      </style>
    </head>
    <body>
      <h1>🌙 TradingView Session API</h1>
      
      <div class="status">
        ${tvClient ? '✅ Client Active' : '❌ Client Not Initialized'}
      </div>
      
      <h2>Setup Instructions</h2>
      
      <ol>
        <li>Login to <a href="https://www.tradingview.com" target="_blank">tradingview.com</a></li>
        <li>Open Developer Tools (F12)</li>
        <li>Go to Application → Cookies → tradingview.com</li>
        <li>Find and copy:
          <ul>
            <li><code>sessionid</code> cookie value</li>
            <li><code>sessionid_sign</code> cookie value</li>
          </ul>
        </li>
        <li>Add to your .env file:</li>
      </ol>
      
      <div class="code">
TV_SESSION_ID=your_sessionid_value_here
TV_SESSION_SIGNATURE=your_sessionid_sign_value_here
      </div>
      
      <h2>Quick Console Method</h2>
      <p>Run this in browser console on tradingview.com:</p>
      
      <div class="code">
const cookies = document.cookie.split('; ');
const sessionid = cookies.find(c => c.startsWith('sessionid='))?.split('=')[1];
const sessionid_sign = cookies.find(c => c.startsWith('sessionid_sign='))?.split('=')[1];
console.log('TV_SESSION_ID=' + sessionid);
console.log('TV_SESSION_SIGNATURE=' + sessionid_sign);
      </div>
      
      <h2>API Endpoints</h2>
      <div class="code">
POST /chart      - Get real-time data
POST /history    - Get historical OHLCV
GET  /health     - Check server status
GET  /session    - Check session info
      </div>
      
      <h2>Test Command</h2>
      <div class="code">
curl -X POST http://localhost:${PORT}/chart \\
  -H "Content-Type: application/json" \\
  -d '{"symbol":"BTCUSDT","exchange":"BINANCE"}'
      </div>
    </body>
    </html>
  `);
});

// Initialize client on startup
const clientInitialized = initializeClient();

// Start server
app.listen(PORT, () => {
  console.log('🌙 TradingView Session API Server');
  console.log(`📡 Running on http://localhost:${PORT}`);
  console.log(`🔐 Session mode: ${clientInitialized ? 'Active' : 'Not configured'}`);
  
  if (!clientInitialized) {
    console.log('\n⚠️  Please configure session tokens in .env');
    console.log('📝 Visit http://localhost:' + PORT + ' for instructions');
  }
});