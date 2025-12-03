/**
 * Test reading password from .env
 */

require('dotenv').config();

console.log('🔍 Testing password reading from .env\n');

const password = process.env.TV_PASSWORD;

console.log('Password details:');
console.log(`- Length: ${password ? password.length : 0} characters`);
console.log(`- First 3 chars: ${password ? password.substring(0, 3) : 'N/A'}...`);
console.log(`- Last 3 chars: ...${password ? password.substring(password.length - 3) : 'N/A'}`);
console.log(`- Has spaces: ${password ? password.includes(' ') : 'N/A'}`);
console.log(`- Has quotes: ${password ? (password.includes('"') || password.includes("'")) : 'N/A'}`);
console.log(`- Has dollar sign: ${password ? password.includes('$') : 'N/A'}`);
console.log(`- Has backslash: ${password ? password.includes('\\') : 'N/A'}`);

// Check for common issues
if (password) {
    if (password !== password.trim()) {
        console.log('\n⚠️  WARNING: Password has leading/trailing whitespace!');
    }
    
    // Check for characters that might need escaping
    const specialChars = ['$', '\\', '"', "'", '`', '\n', '\r'];
    const foundSpecial = specialChars.filter(char => password.includes(char));
    if (foundSpecial.length > 0) {
        console.log(`\n⚠️  WARNING: Password contains special characters: ${foundSpecial.join(', ')}`);
        console.log('These might need escaping in .env file');
    }
}

console.log('\n📝 If password has special characters, try:');
console.log('1. Wrap in double quotes: TV_PASSWORD="your$password"');
console.log('2. Escape with backslash: TV_PASSWORD=your\\$password');
console.log('3. Use single quotes: TV_PASSWORD=\'your$password\'');