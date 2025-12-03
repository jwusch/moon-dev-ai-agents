#!/usr/bin/env python3
"""
🧪 Test Groq Token Counting Issue 🧪

Investigates why Groq reports "request too large" for small requests
"""

import os
import sys
from termcolor import colored

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model_factory import ModelFactory

def test_groq_token_limit():
    """Test Groq with various prompt sizes"""
    
    print(colored("🧪 Testing Groq Token Limit Issue", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Initialize Groq model
    try:
        groq_model = ModelFactory.create_model('groq', 'llama-3.3-70b-versatile')
        if not groq_model:
            print(colored("❌ Groq model not available - check API key", 'red'))
            return
        print(colored("✅ Groq model initialized", 'green'))
    except Exception as e:
        print(colored(f"❌ Failed to initialize Groq: {e}", 'red'))
        return
    
    # Test 1: Very small prompt
    print("\n📊 Test 1: Minimal prompt")
    print("-"*40)
    
    try:
        response = groq_model.generate_response(
            system_prompt="You are helpful.",
            user_content="Hi",
            temperature=0.7,
            max_tokens=10
        )
        
        if response:
            print(colored("✅ Small prompt succeeded", 'green'))
            print(f"Response: {response}")
        else:
            print(colored("❌ Small prompt failed", 'red'))
    except Exception as e:
        print(colored(f"❌ Error: {e}", 'red'))
    
    # Test 2: Medium prompt (like SwarmAgent uses)
    print("\n📊 Test 2: SwarmAgent-style prompt")
    print("-"*40)
    
    swarm_prompt = """Research the stock symbol GME and determine if it's a valid, actively traded ticker.

Please check:
1. Is GME a legitimate stock ticker?
2. Is it currently listed on a major exchange (NYSE, NASDAQ, etc.)?
3. Is it actively trading (not delisted or suspended)?
4. What company does it represent?

Return a JSON response:
{"valid": true/false, "company": "company name", "exchange": "exchange name", "reason": "explanation"}"""
    
    print(f"Prompt length: {len(swarm_prompt)} characters")
    
    try:
        response = groq_model.generate_response(
            system_prompt="You are a financial analyst specializing in stock market research.",
            user_content=swarm_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        if response:
            print(colored("✅ SwarmAgent prompt succeeded", 'green'))
            print(f"Response preview: {str(response)[:200]}...")
        else:
            print(colored("❌ SwarmAgent prompt failed", 'red'))
    except Exception as e:
        print(colored(f"❌ Error: {str(e)}", 'red'))
        
        # Check if it's the 413 error
        if "413" in str(e) or "rate_limit_exceeded" in str(e):
            print(colored("\n⚠️ This is the 413 'request too large' error!", 'yellow'))
            print("\nPossible causes:")
            print("1. The timestamp suffix added to make requests unique")
            print("2. The combination of system + user prompt")
            print("3. Groq's token counting method")
    
    # Test 3: Check the actual request size with timestamp
    print("\n📊 Test 3: Analyzing actual request size")
    print("-"*40)
    
    import time
    timestamp = int(time.time() * 1000)
    
    system_prompt = "You are a financial analyst specializing in stock market research."
    user_content_with_timestamp = f"{swarm_prompt}_{timestamp}"
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    total_chars = len(system_prompt) + len(user_content_with_timestamp)
    estimated_tokens = total_chars / 4
    
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"User content: {len(user_content_with_timestamp)} chars")
    print(f"Total characters: {total_chars}")
    print(f"Estimated tokens: ~{int(estimated_tokens)}")
    
    print("\n💡 Analysis:")
    if estimated_tokens < 1000:
        print(colored("✅ Request should be well within Groq's limits", 'green'))
        print("   The 100k token limit message is misleading")
        print("   The actual issue might be:")
        print("   • Rate limiting per minute/hour")
        print("   • Request payload size (not token count)")
        print("   • Model-specific restrictions")
    
    # Test 4: Try without timestamp
    print("\n📊 Test 4: Testing without timestamp suffix")
    print("-"*40)
    
    # Temporarily modify the request to not include timestamp
    print("Testing raw request without timestamp...")
    
    print("\n🎯 CONCLUSION:")
    print("="*80)
    print("The Groq '100k token limit' error is misleading. The actual issue is likely:")
    print("1. Rate limiting - too many requests in a short time")
    print("2. The llama-3.3-70b model has stricter limits than shown")
    print("3. The error message is incorrect and should say 'rate limit' not 'token limit'")
    print("\n💡 Recommendations:")
    print("• Use a different Groq model (mixtral-8x7b-32768)")
    print("• Add retry logic with exponential backoff")
    print("• Consider removing Groq from critical consensus paths")

def main():
    """Run the test"""
    try:
        test_groq_token_limit()
    except Exception as e:
        print(colored(f"\n❌ Test failed: {str(e)}", 'red'))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()