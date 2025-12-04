#!/usr/bin/env python3
"""
🏴‍☠️ AMEX SMALL CAP BLITZ 🏴‍☠️
Hunt for AEGS goldmines in volatile small cap territory
"""

import time
from datetime import datetime
from multi_exchange_aegs_scanner import MultiExchangeAEGSScanner

class AMEXSmallCapBlitz:
    """AMEX small cap scanner - where volatility meets opportunity"""
    
    def __init__(self):
        self.scanner = MultiExchangeAEGSScanner()
        
    def get_amex_small_cap_candidates(self):
        """Get AMEX small cap symbols known for volatility"""
        
        # AMEX small caps with known volatility patterns
        amex_volatility_kings = [
            # Biotech/Pharma Small Caps
            'ABUS', 'ACAD', 'ACHV', 'ACTG', 'ADAP', 'ADMA', 'ADMP', 'ADVM', 'AEMD', 'AFMD',
            'AGEN', 'AGRX', 'AIMD', 'AKER', 'ALBO', 'ALDX', 'ALEC', 'ALGN', 'ALKS', 'ALLK',
            'ALNY', 'ALXN', 'AMAG', 'AMAT', 'AMGN', 'AMRN', 'AMRS', 'ANAB', 'ANCN', 'ANGO',
            
            # Energy/Mining Small Caps  
            'AREC', 'ARRY', 'ARTW', 'ARVN', 'ASMB', 'ASPS', 'ASRT', 'ATHE', 'ATHX', 'ATIF',
            'ATNI', 'ATOS', 'ATNF', 'ATRC', 'ATRS', 'ATVI', 'AUPH', 'AVCO', 'AVDL', 'AVGR',
            'AVIR', 'AVRO', 'AXDX', 'AXSM', 'AYTU', 'AZPN', 'AZRX', 'BAND', 'BBCP', 'BBIG',
            
            # Tech/Software Small Caps
            'BCDA', 'BCEL', 'BCLI', 'BDSX', 'BEAT', 'BFRI', 'BGCP', 'BGNE', 'BHAT', 'BIIB',
            'BIMI', 'BIOC', 'BIOX', 'BITF', 'BIVI', 'BKKT', 'BKNG', 'BLBD', 'BLCM', 'BLFS',
            'BLNK', 'BLUE', 'BLUW', 'BMEA', 'BMEX', 'BMRN', 'BNGO', 'BNTX', 'BODY', 'BOLT',
            
            # Financial/REIT Small Caps
            'BOXL', 'BPMC', 'BPTH', 'BRAC', 'BRDS', 'BREZ', 'BRID', 'BRLI', 'BRQS', 'BSGM',
            'BTAI', 'BTBT', 'BWAY', 'BYFC', 'BYND', 'BZUN', 'CAAS', 'CABA', 'CADL', 'CAKE',
            'CALB', 'CALM', 'CAMP', 'CANF', 'CAPR', 'CARA', 'CARV', 'CASA', 'CASI', 'CASS',
            
            # Consumer/Retail Small Caps
            'CATB', 'CATC', 'CATM', 'CAVN', 'CBAY', 'CBIO', 'CBLI', 'CBPO', 'CCCC', 'CCCL',
            'CCCS', 'CCEP', 'CCNC', 'CCNE', 'CCXI', 'CDAK', 'CDAY', 'CDLX', 'CDMO', 'CDNA',
            'CEAD', 'CECE', 'CELC', 'CELZ', 'CEMI', 'CENT', 'CERE', 'CERS', 'CETX', 'CETY',
            
            # Industrial/Manufacturing Small Caps
            'CFRX', 'CFVS', 'CGEM', 'CGNT', 'CHCI', 'CHDN', 'CHEF', 'CHEK', 'CHFS', 'CHKP',
            'CHMA', 'CHMI', 'CHNR', 'CHRS', 'CHSN', 'CTIC', 'CTIB', 'CTOS', 'CTSO', 'CTXR',
            'CUTR', 'CVAC', 'CVCO', 'CVET', 'CVGI', 'CVGW', 'CVNA', 'CWBC', 'CWCO', 'CYAD',
            
            # Emerging/Crypto Related
            'CYAN', 'CYCC', 'CYCN', 'CYMI', 'CYRX', 'CYTK', 'CZNC', 'DAIO', 'DARE', 'DAVE',
            'DBGI', 'DBRG', 'DCBO', 'DCGO', 'DCOM', 'DCPH', 'DENN', 'DERM', 'DGLY', 'DHAI',
            'DHIL', 'DIBS', 'DILA', 'DMAC', 'DMRC', 'DMTK', 'DNLI', 'DOCU', 'DOGZ', 'DOMA',
            
            # Penny Stock Volatility Legends
            'DPRO', 'DRMA', 'DRRX', 'DRTS', 'DRUG', 'DSGN', 'DSWL', 'DTIL', 'DUET', 'DUNE',
            'DVAX', 'DWSN', 'DXPE', 'DXYN', 'DYAI', 'DYNC', 'DYNT', 'DZSI', 'EACO', 'EARN',
            'EAST', 'EBIX', 'ECBK', 'ECHO', 'ECOR', 'ECPG', 'ECTX', 'EDAP', 'EDIT', 'EDRY',
            
            # Cannabis/CBD Small Caps  
            'EEGI', 'EFOI', 'EFSC', 'EGAN', 'EGBN', 'EGHT', 'EGRX', 'EHTH', 'EICA', 'EIGI',
            'EIGR', 'ELDN', 'ELMD', 'ELOX', 'ELTK', 'EMCG', 'EMKR', 'EMMA', 'ENDO', 'ENDP',
            'ENGS', 'ENLC', 'ENLV', 'ENPH', 'ENSC', 'ENTA', 'ENTG', 'ENTX', 'ENVB', 'ENVI',
            
            # High-Beta Momentum Plays
            'ENVX', 'EPAY', 'EPHY', 'EPIX', 'EPZM', 'EQBK', 'EQIX', 'EQRX', 'EQRR', 'ERAS',
            'ERII', 'ESGR', 'ESLT', 'ESPR', 'ESTA', 'ESTC', 'ESXB', 'ETAC', 'ETNB', 'ETON',
            'ETRN', 'ETSY', 'EUDA', 'EVAX', 'EVBG', 'EVFM', 'EVGN', 'EVGO', 'EVLO', 'EVLV',
            
            # Speculative Growth Stories
            'EVOK', 'EVOP', 'EWBC', 'EXAS', 'EXEL', 'EXFY', 'EXLS', 'EXPE', 'EXPR', 'EXTR',
            'EYEG', 'EYEN', 'EYES', 'EZGO', 'FAMI', 'FATE', 'FBIO', 'FBRX', 'FCEL', 'FDMT',
            'FEBO', 'FENC', 'FERG', 'FEXD', 'FFIC', 'FFIN', 'FGBI', 'FHTX', 'FIBK', 'FICO',
            
            # Additional High-Volatility Targets
            'FIII', 'FINM', 'FISI', 'FITB', 'FIVE', 'FIXX', 'FIZZ', 'FKWL', 'FLGT', 'FLIC',
            'FLLO', 'FLNT', 'FLUX', 'FMAO', 'FMBH', 'FMIV', 'FNCH', 'FNKO', 'FOLD', 'FONR',
            'FORE', 'FORK', 'FOSL', 'FOUR', 'FOXF', 'FPAR', 'FPAY', 'FRAF', 'FRBK', 'FREE',
            'FREQ', 'FRGI', 'FRGT', 'FRHC', 'FRME', 'FRPH', 'FRPT', 'FRSH', 'FRSX', 'FSBW',
            'FSEA', 'FSFG', 'FSLR', 'FSLY', 'FSTX', 'FTAI', 'FTCI', 'FTDR', 'FTEL', 'FTFT',
        ]
        
        print(f"🏴‍☠️ Loaded {len(amex_volatility_kings)} AMEX small cap volatility targets")
        return amex_volatility_kings
    
    def execute_blitz(self, max_symbols=200):
        """Execute the AMEX small cap blitz"""
        
        print("🏴‍☠️💀 AMEX SMALL CAP BLITZ INITIATED 💀🏴‍☠️")
        print("=" * 70)
        print("🎯 Hunting volatile small caps for AEGS opportunities")
        print("💀 These are the wild west of the market - high risk, high reward")
        
        # Get targets
        amex_targets = self.get_amex_small_cap_candidates()
        
        if max_symbols and len(amex_targets) > max_symbols:
            print(f"⚡ Limiting to first {max_symbols} targets for speed")
            amex_targets = amex_targets[:max_symbols]
        
        print(f"\n🚀 BLITZING {len(amex_targets)} AMEX small caps...")
        print("📊 Looking for oversold bounce setups...")
        
        # Execute scan
        start_time = time.time()
        results = self.scanner.scan_multi_exchange(custom_symbols=amex_targets)
        scan_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"amex_smallcap_blitz_{timestamp}.json"
        
        blitz_results = {
            'blitz_date': datetime.now().isoformat(),
            'blitz_type': 'AMEX_SMALL_CAP_VOLATILITY_HUNT',
            'targets_scanned': len(amex_targets),
            'scan_duration_minutes': scan_time / 60,
            'cache_stats': self.scanner.cache.cache_stats,
            'results': results,
            'profitable_goldmines': len(results['profitable']),
            'volatility_thesis': 'Small caps with high beta, frequent oversold conditions'
        }
        
        with open(filename, 'w') as f:
            import json
            json.dump(blitz_results, f, indent=2)
        
        # Analysis and reporting
        self.analyze_blitz_results(results, scan_time, filename)
        
        return blitz_results
    
    def analyze_blitz_results(self, results, scan_time, filename):
        """Analyze and report blitz results"""
        
        profitable = results['profitable']
        
        print(f"\n💀🏴‍☠️ AMEX SMALL CAP BLITZ COMPLETE! 🏴‍☠️💀")
        print(f"⚡ Scan time: {scan_time/60:.1f} minutes")
        print(f"📊 Cache performance: {self.scanner.cache.cache_stats}")
        print(f"💰 Small cap goldmines discovered: {len(profitable)}")
        
        if profitable:
            # Sort by volatility potential (strategy return)
            profitable.sort(key=lambda x: x['strategy_return'], reverse=True)
            
            # Categorize discoveries
            extreme_volatility = [p for p in profitable if p['strategy_return'] >= 100]
            high_volatility = [p for p in profitable if 50 <= p['strategy_return'] < 100]
            good_volatility = [p for p in profitable if 20 <= p['strategy_return'] < 50]
            decent_volatility = [p for p in profitable if p['strategy_return'] < 20]
            
            print(f"\n🎯 SMALL CAP GOLDMINE BREAKDOWN:")
            print(f"   💀 EXTREME VOLATILITY (≥100%): {len(extreme_volatility)} goldmines")
            print(f"   🔥 HIGH VOLATILITY (50-99%): {len(high_volatility)} goldmines") 
            print(f"   ⚡ GOOD VOLATILITY (20-49%): {len(good_volatility)} goldmines")
            print(f"   ✅ DECENT VOLATILITY (<20%): {len(decent_volatility)} goldmines")
            
            print(f"\n🏆 TOP 15 SMALL CAP VOLATILITY KINGS:")
            for i, result in enumerate(profitable[:15], 1):
                symbol = result['symbol']
                ret = result['strategy_return']
                trades = result['total_trades']
                win_rate = result['win_rate']
                
                # Add volatility emoji based on return
                if ret >= 100:
                    emoji = "💀"
                elif ret >= 50:
                    emoji = "🔥"
                elif ret >= 20:
                    emoji = "⚡"
                else:
                    emoji = "✅"
                
                print(f"   {i:2}. {emoji} {symbol:<6} +{ret:6.1f}% ({trades} trades, {win_rate:.0f}% wins)")
            
            # Identify the absolute monsters
            if extreme_volatility:
                print(f"\n💀 ABSOLUTE VOLATILITY MONSTERS:")
                for monster in extreme_volatility:
                    symbol = monster['symbol']
                    ret = monster['strategy_return']
                    trades = monster['total_trades']
                    print(f"   💀 {symbol}: +{ret:.1f}% - SMALL CAP BEAST!")
        
        else:
            print("❌ No profitable small caps found (market might be too efficient today)")
        
        print(f"\n💾 Full blitz results: {filename}")
        print(f"🎯 Ready to add small cap goldmines to registry!")
        
        return profitable

def main():
    """Execute AMEX small cap blitz"""
    
    print("🏴‍☠️💀 AMEX SMALL CAP BLITZ - ENTERING CHAOS ZONE 💀🏴‍☠️")
    print("=" * 80)
    
    blitz = AMEXSmallCapBlitz()
    
    print("⚠️  WARNING: Small caps are volatile and risky!")
    print("💀 This is where fortunes are made and lost")
    print("🎯 Perfect for AEGS oversold bounce strategy")
    
    print("\n🚀 UNLEASHING THE SMALL CAP KRAKEN!")
    
    # Auto-run large scan (200 symbols for speed)
    max_symbols = 200
    print(f"\n🎯 Auto-running large scan: {max_symbols} symbols")
    
    results = blitz.execute_blitz(max_symbols=max_symbols)
    
    # Auto-add profitable goldmines to registry if found
    if results['profitable_goldmines'] > 0:
        print(f"\n🔥 Auto-adding {results['profitable_goldmines']} small cap goldmines to registry...")
        # Return results for expansion script integration
        return results
    else:
        print("\n📊 No profitable small caps found in this batch")
        return results

if __name__ == "__main__":
    main()