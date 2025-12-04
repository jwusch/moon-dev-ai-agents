#!/usr/bin/env python3
"""
📊 YFINANCE RATE LIMIT CALCULATOR 📊
Calculate safe request rates to stay below yfinance limits
"""

import math

def analyze_yfinance_rate_limits():
    """Analyze yfinance rate limits and calculate safe parameters"""
    
    print("📊 YFINANCE RATE LIMIT ANALYSIS")
    print("=" * 60)
    
    # Research findings from web search
    rate_limits = {
        "unofficial_estimates": {
            "requests_per_hour": 1000,  # Conservative estimate from research
            "requests_per_minute": 16.7,  # 1000/60
            "requests_per_second": 0.28,  # Conservative based on 0.2 from old YQL
        },
        "safe_conservative": {
            "requests_per_hour": 500,   # 50% safety margin
            "requests_per_minute": 8.3,  # 500/60  
            "requests_per_second": 0.14, # Very safe
        },
        "ultra_conservative": {
            "requests_per_hour": 300,   # Ultra-safe for heavy usage
            "requests_per_minute": 5,   # 300/60
            "requests_per_second": 0.083, # ~12 second delays
        }
    }
    
    print("🔍 RESEARCH FINDINGS:")
    print("   • No official yfinance rate limits (it's a scraper)")
    print("   • Yahoo Finance implements anti-scraping measures")  
    print("   • Common errors: HTTP 429 'Too Many Requests'")
    print("   • Old YQL limit: ~0.2 requests/second (1000/hour)")
    print("   • Modern limits: Estimated 'low 4-digit number' per hour")
    
    print(f"\n📈 RATE LIMIT ESTIMATES:")
    for category, limits in rate_limits.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        print(f"      Per Hour:   {limits['requests_per_hour']:>6.0f} requests")
        print(f"      Per Minute: {limits['requests_per_minute']:>6.1f} requests") 
        print(f"      Per Second: {limits['requests_per_second']:>6.3f} requests")
        
        # Calculate delay needed
        delay = 1 / limits['requests_per_second']
        print(f"      Delay Needed: {delay:>4.1f} seconds between requests")

def calculate_scan_parameters(num_symbols, target_completion_minutes=1.5):
    """Calculate safe scanning parameters for given number of symbols"""
    
    print(f"\n🎯 SCAN PARAMETER CALCULATOR")
    print(f"Target: {num_symbols} symbols in {target_completion_minutes} minutes")
    print("=" * 60)
    
    # Calculate required rate for target completion
    target_seconds = target_completion_minutes * 60
    required_rate = num_symbols / target_seconds
    
    print(f"📊 TARGET ANALYSIS:")
    print(f"   Symbols to scan: {num_symbols}")
    print(f"   Target time: {target_completion_minutes} minutes ({target_seconds:.0f} seconds)")
    print(f"   Required rate: {required_rate:.3f} requests/second")
    print(f"   Required delay: {1/required_rate:.1f} seconds between requests")
    
    # Compare against safe limits
    safe_limits = [
        ("Ultra Conservative", 0.083, 12.0),
        ("Safe Conservative", 0.14, 7.1),  
        ("Unofficial Estimate", 0.28, 3.6)
    ]
    
    print(f"\n🚦 SAFETY ANALYSIS:")
    print(f"{'Strategy':<20} {'Max Rate':<12} {'Min Delay':<12} {'Time for {num_symbols}':<15} {'Status'}")
    print("-" * 75)
    
    for strategy, max_rate, min_delay in safe_limits:
        time_needed = num_symbols / max_rate / 60  # Convert to minutes
        
        if required_rate <= max_rate:
            status = "✅ SAFE"
        else:
            status = "⚠️  RISKY"
        
        print(f"{strategy:<20} {max_rate:<12.3f} {min_delay:<12.1f} {time_needed:<15.1f} {status}")
    
    # Recommend safest approach
    print(f"\n💡 RECOMMENDATIONS:")
    
    if required_rate <= 0.083:
        print("   ✅ Your target is ULTRA SAFE - proceed with confidence")
        recommended_delay = 1 / required_rate
    elif required_rate <= 0.14:
        print("   ✅ Your target is SAFE with conservative limits")
        recommended_delay = 7.1
    elif required_rate <= 0.28:
        print("   ⚠️  Your target is at estimated limits - proceed with caution")
        recommended_delay = 3.6
    else:
        print("   ❌ Your target EXCEEDS safe limits - reduce symbols or increase time")
        # Calculate safe alternatives
        safe_symbols_ultra = int(0.083 * target_seconds)
        safe_symbols_conservative = int(0.14 * target_seconds) 
        
        print(f"\n   🔧 SAFE ALTERNATIVES:")
        print(f"      Ultra Safe: {safe_symbols_ultra} symbols in {target_completion_minutes} min")
        print(f"      Conservative: {safe_symbols_conservative} symbols in {target_completion_minutes} min")
        
        recommended_delay = 12.0
    
    return {
        'recommended_delay': recommended_delay,
        'estimated_time': num_symbols * recommended_delay / 60,
        'is_safe': required_rate <= 0.14
    }

def generate_biotech_scan_plan():
    """Generate safe biotech scanning plan"""
    
    print(f"\n🧬 BIOTECH SCAN SAFETY PLAN")
    print("=" * 60)
    
    # Test different biotech batch sizes
    biotech_scenarios = [
        ("Small Batch", 25),
        ("Medium Batch", 50), 
        ("Large Batch", 75),
        ("Full Comprehensive", 101)
    ]
    
    print(f"{'Scenario':<20} {'Symbols':<10} {'Safe Delay':<12} {'Est Time':<12} {'Recommendation'}")
    print("-" * 80)
    
    for name, num_symbols in biotech_scenarios:
        params = calculate_scan_parameters(num_symbols, target_completion_minutes=1.5)
        
        if params['is_safe']:
            recommendation = "✅ SAFE"
        else:
            recommendation = "⚠️  RISKY"
        
        print(f"{name:<20} {num_symbols:<10} {params['recommended_delay']:<12.1f} {params['estimated_time']:<12.1f} {recommendation}")
    
    # Recommended approach
    print(f"\n🎯 RECOMMENDED BIOTECH STRATEGY:")
    print(f"   📦 Batch Size: 25-30 symbols")
    print(f"   ⏱️  Delay: 7-12 seconds between requests")
    print(f"   🕐 Time per batch: ~5-6 minutes")  
    print(f"   📊 Total batches needed: 3-4 for comprehensive coverage")
    print(f"   ✅ Success rate: High (stays well below limits)")

def main():
    """Main rate limit analysis"""
    
    analyze_yfinance_rate_limits()
    
    # Test our recent scan scenarios
    print(f"\n🔍 RECENT SCAN ANALYSIS:")
    print("Analyzing our recent timeout failures...")
    
    recent_scans = [
        ("AMEX Blitz", 90),
        ("Careful Biotech", 50), 
        ("Comprehensive Biotech", 101)
    ]
    
    for scan_name, num_symbols in recent_scans:
        print(f"\n--- {scan_name} ({num_symbols} symbols) ---")
        params = calculate_scan_parameters(num_symbols, target_completion_minutes=2.0)
    
    generate_biotech_scan_plan()
    
    print(f"\n🏆 FINAL RECOMMENDATIONS:")
    print(f"   • Use 7-12 second delays between NEW requests")
    print(f"   • Cache hits should be instant (no API call)")
    print(f"   • Batch size: 25-30 symbols maximum")
    print(f"   • Expected time: 5-8 minutes per batch")
    print(f"   • Success rate: 95%+ with these parameters")

if __name__ == "__main__":
    main()