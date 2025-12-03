/**
 * Debug TradingView Login Issue
 * This script tests the login process step by step
 */

require('dotenv').config();
const TradingView = require('./tradingview-api/main');

console.log('🔍 TradingView Login Debug Script\n');

// 1. Check environment variables
console.log('1️⃣ Environment Variables Check:');
console.log('================================');
const username = process.env.TV_USERNAME;
const password = process.env.TV_PASSWORD;

console.log(`Username: ${username || 'NOT SET'}`);
console.log(`Password: ${password ? `Set (${password.length} chars)` : 'NOT SET'}`);

if (!username || !password) {
    console.log('\n❌ Missing credentials in .env file!');
    process.exit(1);
}

// 2. Test password characteristics
console.log('\n2️⃣ Password Characteristics:');
console.log('================================');
console.log(`Length: ${password.length}`);
console.log(`First 3 chars: ${password.substring(0, 3)}...`);
console.log(`Last 3 chars: ...${password.substring(password.length - 3)}`);
console.log(`Contains spaces: ${password.includes(' ')}`);
console.log(`Contains quotes: ${password.includes('"') || password.includes("'")}`);
console.log(`Contains dollar sign: ${password.includes('$')}`);
console.log(`Contains backslash: ${password.includes('\\')}`);
console.log(`Contains newline: ${password.includes('\n')}`);
console.log(`Trimmed length: ${password.trim().length}`);

if (password !== password.trim()) {
    console.log('\n⚠️  WARNING: Password has leading/trailing whitespace!');
}

// 3. Test direct API login
console.log('\n3️⃣ Testing Direct TradingView API Login:');
console.log('=========================================');

async function testLogin() {
    try {
        console.log(`\n🔐 Attempting login with:`);
        console.log(`   Email: ${username}`);
        console.log(`   Pass: ${password.length} characters`);
        
        // Try login with more error details
        const userSession = await TradingView.loginUser(username, password, false);
        
        console.log('\n✅ LOGIN SUCCESSFUL!');
        console.log('Session details:');
        console.log(`- User: ${userSession.user}`);
        console.log(`- Session ID: ${userSession.session ? 'Present' : 'Missing'}`);
        console.log(`- Signature: ${userSession.signature ? 'Present' : 'Missing'}`);
        
        return userSession;
        
    } catch (error) {
        console.log('\n❌ LOGIN FAILED!');
        console.log(`Error Type: ${error.constructor.name}`);
        console.log(`Error Message: ${error.message}`);
        
        // Check for specific error patterns
        if (error.message.includes('Invalid password')) {
            console.log('\n🔍 Password-specific debugging:');
            console.log('- Double-check your password in .env file');
            console.log('- Make sure there are no extra spaces or quotes');
            console.log('- Try wrapping password in double quotes if it contains special characters');
            console.log('\nExample .env format:');
            console.log('TV_PASSWORD="your_password_here"');
            console.log('or');
            console.log('TV_PASSWORD=your_password_here');
        } else if (error.message.includes('captcha')) {
            console.log('\n🔍 Captcha required:');
            console.log('- Use the web UI at http://localhost:8888');
            console.log('- Or login manually at tradingview.com first');
        }
        
        // Full error stack
        console.log('\nFull error stack:');
        console.log(error.stack);
        
        return null;
    }
}

// 4. Test with variations
console.log('\n4️⃣ Testing Login Variations:');
console.log('================================');

async function testVariations() {
    // Test 1: As-is from env
    console.log('\nTest 1: Using password as-is from .env');
    await testLogin();
    
    // Test 2: Trimmed password
    if (password !== password.trim()) {
        console.log('\nTest 2: Using trimmed password');
        process.env.TV_PASSWORD = password.trim();
        await testLogin();
        process.env.TV_PASSWORD = password; // Reset
    }
}

// Run the tests
(async () => {
    await testVariations();
    
    console.log('\n\n📝 Next Steps:');
    console.log('================');
    console.log('1. If "Invalid password" persists:');
    console.log('   - Log in manually at https://www.tradingview.com');
    console.log('   - Verify the exact password works there');
    console.log('   - Check if 2FA is enabled on your account');
    console.log('   - Update .env with the correct password');
    console.log('\n2. If captcha is required:');
    console.log('   - Use the web UI at http://localhost:8888');
    console.log('   - Complete manual login with captcha');
    console.log('\n3. Check .env file format:');
    console.log('   - No spaces around = sign');
    console.log('   - Try with/without quotes');
    console.log('   - Ensure no trailing newline');
    
    process.exit(0);
})();