#!/usr/bin/env python3
"""
🧬 COMPREHENSIVE BIOTECH TARGET LIST 🧬
Complete list of biotech symbols for destruction scan
"""

def get_comprehensive_biotech_list():
    """Get comprehensive biotech target list organized by categories"""
    
    biotech_targets = {
        'large_cap_biotech': [
            # Established Large Cap Biotech (Market Cap >$10B)
            'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'INCY', 'MRNA', 'BNTX',
            'AMGN', 'CELG', 'BMRN', 'ALNY', 'SGEN', 'TECH', 'IOVA'
        ],
        
        'mid_cap_biotech': [
            # Mid Cap Biotech ($1B-$10B) - High Growth Potential  
            'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'FOLD', 'ARWR', 'IONS',
            'EXAS', 'RARE', 'BLUE', 'SAGE', 'PTCT', 'SRPT', 'MYOV', 'ACAD',
            'HALO', 'IMGN', 'NBIX', 'UTHR', 'JAZZ', 'VRTV', 'CGEN', 'CGEM'
        ],
        
        'small_cap_biotech': [
            # Small Cap Biotech ($100M-$1B) - Extreme Volatility
            'OCGN', 'NVAX', 'CYTK', 'DVAX', 'CRVS', 'MCRB', 'RDHL', 'KPTI',
            'ACRS', 'AGIO', 'NRXP', 'GNPX', 'CTXR', 'OTIC', 'FATE', 'EDIT',
            'SYRS', 'BCYC', 'PGEN', 'PRTA', 'TENB', 'MRTX', 'RUBY', 'ICPT'
        ],
        
        'penny_biotech': [
            # Penny Stock Biotech (<$100M) - Maximum Chaos  
            'TXMD', 'OBSV', 'SNSS', 'CBAY', 'GTHX', 'CTIC', 'ACRX', 'DRNA',
            'ADMP', 'ALNA', 'HGEN', 'NRXP', 'CNTX', 'ZYXI', 'BOLD', 'FREQ',
            'BTAI', 'NVTA', 'FATE', 'EIGR', 'EPZM', 'GOSS', 'HRTX', 'IMMP'
        ],
        
        'specialty_biotech': [
            # Specialty Areas - Gene Therapy, CAR-T, Immunotherapy
            'GILD', 'CAR', 'KITE', 'JUNO', 'BLUE', 'FATE', 'CRSP', 'EDIT',
            'NTLA', 'BEAM', 'PRME', 'ASGN', 'YMAB', 'CAPR', 'TCDA', 'PSTI',
            'DRNA', 'BCYC', 'ALEC', 'AKRO', 'APLS', 'CARA', 'CHRS', 'DYNE'
        ],
        
        'volatile_momentum': [
            # Known for High Beta and Momentum Trading
            'MRNA', 'NVAX', 'OCGN', 'BNTX', 'SAVA', 'BIIB', 'CRIS', 'FOLD',
            'BEAM', 'CRSP', 'EDIT', 'NTLA', 'VERV', 'RARE', 'BLUE', 'SAGE'
        ]
    }
    
    # Flatten and deduplicate
    all_biotech = []
    seen = set()
    
    print("🧬 COMPREHENSIVE BIOTECH TARGET CATEGORIES:")
    total_count = 0
    
    for category, symbols in biotech_targets.items():
        print(f"   {category.replace('_', ' ').title():<20}: {len(symbols):>3} symbols")
        total_count += len(symbols)
        
        for symbol in symbols:
            if symbol not in seen:
                all_biotech.append(symbol)
                seen.add(symbol)
    
    print(f"   {'TOTAL UNIQUE':<20}: {len(all_biotech):>3} symbols")
    print(f"   {'TOTAL W/DUPLICATES':<20}: {total_count:>3} symbols")
    
    return all_biotech, biotech_targets

def show_biotech_targets():
    """Display the comprehensive biotech target list"""
    
    print("🧬💀 COMPREHENSIVE BIOTECH DESTRUCTION TARGETS 💀🧬")
    print("=" * 80)
    
    all_biotech, categories = get_comprehensive_biotech_list()
    
    print(f"\n📊 TARGET ANALYSIS:")
    print(f"   🎯 Total Unique Symbols: {len(all_biotech)}")
    print(f"   ⚡ Expected High Volatility: 80%+")
    print(f"   🧬 Biotech Sectors: Gene therapy, immunotherapy, CAR-T, vaccines")
    print(f"   💀 Chaos Level: MAXIMUM (includes penny biotech)")
    
    print(f"\n🔬 SAMPLE HIGH-PRIORITY TARGETS:")
    samples = {
        'MONSTERS': ['CRVS', 'NRXP', 'CRSP', 'SAVA', 'NVAX'],
        'ESTABLISHED': ['GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN'], 
        'GENE THERAPY': ['CRSP', 'EDIT', 'NTLA', 'BEAM', 'BLUE'],
        'PENNY CHAOS': ['TXMD', 'OBSV', 'CBAY', 'GTHX', 'ADMP']
    }
    
    for category, symbols in samples.items():
        print(f"   {category:<15}: {', '.join(symbols)}")
    
    print(f"\n⚡ SCAN STRATEGY:")
    print(f"   🚀 Batch size: {len(all_biotech)} symbols")
    print(f"   💾 Cache optimization: Prioritize previously scanned")
    print(f"   ⏱️  Target completion: <90 seconds")
    print(f"   🛡️  Rate limiting: Smart delays only if needed")
    
    return all_biotech

if __name__ == "__main__":
    targets = show_biotech_targets()
    print(f"\n📋 Ready to scan {len(targets)} biotech targets!")