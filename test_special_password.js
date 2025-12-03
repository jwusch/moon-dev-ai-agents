/**
 * Test TradingView login with special character password
 */

require('dotenv').config();
const TradingView = require('./tradingview-api/main');

console.log('🔍 Testing TradingView Login with Special Password\n');

// Your actual password
const actualPassword = '.N2KpbEL)6.N5m#';
const username = process.env.TV_USERNAME || 'jwusch@gmail.com';

console.log('Password details:');
console.log(`- Length: ${actualPassword.length} characters`);
console.log(`- Contains dot: ${actualPassword.includes('.')}`);
console.log(`- Contains parentheses: ${actualPassword.includes(')')}`);
console.log(`- Contains hash: ${actualPassword.includes('#')}`);

console.log('\nPassword from .env:');
const envPassword = process.env.TV_PASSWORD;
console.log(`- Length: ${envPassword ? envPassword.length : 0} characters`);
console.log(`- Matches actual: ${envPassword === actualPassword}`);

if (envPassword !== actualPassword) {
    console.log('\n⚠️  WARNING: Password in .env does not match!');
    console.log(`ENV: "${envPassword}"`);
    console.log(`Expected: "${actualPassword}"`);
}

async function testLogin(password, description) {
    console.log(`\n📝 ${description}`);
    console.log(`Testing with password: ${password.length} chars`);
    
    try {
        const userSession = await TradingView.loginUser(username, password, false);
        console.log('✅ LOGIN SUCCESSFUL!');
        console.log(`User: ${userSession.user}`);
        return true;
    } catch (error) {
        console.log('❌ LOGIN FAILED!');
        console.log(`Error: ${error.message}`);
        return false;
    }
}

// Run tests
(async () => {
    // Test 1: Direct password
    await testLogin(actualPassword, 'Test 1: Using actual password directly');
    
    // Test 2: From env
    if (envPassword) {
        await testLogin(envPassword, 'Test 2: Using password from .env');
    }
    
    console.log('\n\n📋 Update your .env file:');
    console.log('==========================');
    console.log('Make sure your .env has exactly this line:');
    console.log(`TV_PASSWORD=${actualPassword}`);
    console.log('\nOr with quotes:');
    console.log(`TV_PASSWORD="${actualPassword}"`);
    
    process.exit(0);
})();