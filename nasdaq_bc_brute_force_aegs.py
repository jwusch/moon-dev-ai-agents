#!/usr/bin/env python3
"""
🔥💎 NASDAQ 'B' & 'C' SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥
Complete brute force backtest on all NASDAQ B and C symbols
Finding profitable AEGS candidates across the alphabet
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from termcolor import colored
import warnings
import time
import concurrent.futures
from typing import List, Dict, Optional
warnings.filterwarnings('ignore')

class NASDAQBCBruteForceAEGS:
    """Brute force AEGS backtest for NASDAQ B and C symbols"""
    
    def __init__(self):
        self.nasdaq_bc_symbols = []
        self.brute_force_results = []
        
    def get_all_nasdaq_bc_symbols(self) -> List[str]:
        """Get comprehensive list of NASDAQ B and C symbols"""
        
        print(colored("🔥 BRUTE FORCE MODE: All NASDAQ 'B' & 'C' symbols", 'red', attrs=['bold']))
        print("=" * 60)
        
        # Comprehensive NASDAQ B and C symbol list
        nasdaq_bc_symbols = [
            # === B SYMBOLS ===
            # Major B symbols
            'BBBY', 'BBIO', 'BBSI', 'BBVA', 'BBWI', 'BCAB', 'BCAT', 'BCBP', 'BCDA',
            'BCEL', 'BCLI', 'BCOV', 'BCPC', 'BCRX', 'BCS', 'BCYC', 'BDSX', 'BDTX',
            'BEAM', 'BEAT', 'BECN', 'BEDU', 'BEKE', 'BELFA', 'BELFB', 'BENE', 'BENF',
            'BFAM', 'BFIN', 'BFRI', 'BGCP', 'BGFV', 'BGLI', 'BGNE', 'BGRN', 'BHAC',
            'BHAT', 'BHC', 'BHEL', 'BHTG', 'BIAF', 'BIDU', 'BIGC', 'BILI', 'BINJ',
            'BIOC', 'BIOL', 'BIOR', 'BIOX', 'BIRD', 'BITE', 'BJRI', 'BKCC', 'BKEP',
            'BKKT', 'BKNG', 'BKSC', 'BKSY', 'BKTI', 'BKYI', 'BL', 'BLBD', 'BLBX',
            'BLCM', 'BLDP', 'BLDR', 'BLI', 'BLIN', 'BLKB', 'BLMN', 'BLND', 'BLNK',
            'BLPH', 'BLRX', 'BLTE', 'BLUE', 'BLZE', 'BMBL', 'BMEA', 'BMEZ', 'BMRA',
            'BMRN', 'BNAI', 'BNED', 'BNGO', 'BNTX', 'BNZI', 'BOCN', 'BODY', 'BOLT',
            'BOOM', 'BOOT', 'BORR', 'BOSC', 'BOTJ', 'BOWX', 'BPMC', 'BPOP', 'BPRN',
            'BPTH', 'BPTS', 'BPYP', 'BRAC', 'BRBS', 'BRBR', 'BRCC', 'BRDS', 'BREA',
            'BREZ', 'BRID', 'BRKL', 'BRKR', 'BRLT', 'BRMK', 'BRNS', 'BROG', 'BROS',
            'BRPM', 'BRQS', 'BRY', 'BSAC', 'BSBR', 'BSFC', 'BSGM', 'BSIG', 'BSKY',
            'BSM', 'BSQR', 'BSRR', 'BSVN', 'BSY', 'BTAI', 'BTBD', 'BTBT', 'BTCS',
            'BTCM', 'BTDR', 'BTNB', 'BTWN', 'BTTX', 'BTTR', 'BTWN', 'BUKS', 'BURL',
            'BUSE', 'BWB', 'BWAY', 'BWFG', 'BWIN', 'BWMX', 'BXMT', 'BYFC', 'BYND',
            'BYSI', 'BZ', 'BZFD', 'BZUN',
            
            # More B symbols
            'BABY', 'BACK', 'BANC', 'BANF', 'BANR', 'BATRA', 'BATRK', 'BBAI', 'BBCP',
            'BBDC', 'BBH', 'BBIG', 'BBLN', 'BBLG', 'BBRX', 'BBUC', 'BCAL', 'BCAX',
            'BCEL', 'BCH', 'BCLI', 'BCML', 'BCOW', 'BCSF', 'BCYC', 'BDGE', 'BDRY',
            'BDSI', 'BDSX', 'BDTX', 'BDVS', 'BEAM', 'BEAR', 'BEAT', 'BECN', 'BEDU',
            'BEEM', 'BEKE', 'BENF', 'BERG', 'BEST', 'BFAC', 'BFAM', 'BFIN', 'BFRI',
            'BFST', 'BGCP', 'BGLC', 'BGNE', 'BHAT', 'BHFAL', 'BHFAP', 'BIDU', 'BIEN',
            'BIGC', 'BIGZ', 'BILI', 'BINJ', 'BIOC', 'BIOF', 'BIOL', 'BIOR', 'BIOT',
            'BIOX', 'BIRD', 'BITE', 'BITF', 'BJDX', 'BJRI', 'BKCC', 'BKEP', 'BKKT',
            'BKNG', 'BKSC', 'BKSY', 'BKTI', 'BKYI', 'BLBD', 'BLBX', 'BLCM', 'BLCN',
            'BLDP', 'BLDR', 'BLEE', 'BLI', 'BLIN', 'BLKB', 'BLMN', 'BLND', 'BLNK',
            'BLPH', 'BLRX', 'BLTE', 'BLUE', 'BMBL', 'BMEA', 'BMEZ', 'BMRA', 'BMRN',
            'BNAI', 'BNED', 'BNGO', 'BNOX', 'BNTX', 'BNZI', 'BOCN', 'BODY', 'BOLT',
            'BOOM', 'BOOT', 'BORR', 'BOSC', 'BOTJ', 'BOWX', 'BPMC', 'BPOP', 'BPTH',
            'BPTS', 'BPYP', 'BRAC', 'BRBS', 'BRBR', 'BRCC', 'BRDS', 'BREA', 'BREZ',
            'BRID', 'BRKL', 'BRKR', 'BRLT', 'BRMK', 'BRNS', 'BROG', 'BROS', 'BRPM',
            'BRQS', 'BRY', 'BSAC', 'BSBR', 'BSFC', 'BSGM', 'BSIG', 'BSKY', 'BSM',
            'BSQR', 'BSRR', 'BSVN', 'BSY', 'BTAI', 'BTBD', 'BTBT', 'BTCS', 'BTCM',
            'BTDR', 'BTNB', 'BTWN', 'BTTX', 'BTTR', 'BUKS', 'BURL', 'BUSE', 'BWB',
            'BWAY', 'BWFG', 'BWIN', 'BWMX', 'BXMT', 'BYFC', 'BYND', 'BYSI', 'BZ',
            'BZFD', 'BZUN',
            
            # === C SYMBOLS ===
            # Major C symbols  
            'CAAS', 'CACC', 'CACI', 'CADL', 'CAE', 'CAFD', 'CAKE', 'CALM', 'CAMP',
            'CAMT', 'CAN', 'CAPR', 'CAR', 'CARA', 'CARE', 'CARG', 'CARV', 'CASA',
            'CASH', 'CASI', 'CASS', 'CASY', 'CAT', 'CATB', 'CATC', 'CATY', 'CAWN',
            'CBAT', 'CBAY', 'CBD', 'CBFV', 'CBIO', 'CBLI', 'CBNK', 'CBOE', 'CBPO',
            'CBRE', 'CBRL', 'CBSH', 'CBTX', 'CCAP', 'CCB', 'CCBG', 'CCCC', 'CCCS',
            'CCEL', 'CCG', 'CCIX', 'CCJ', 'CCLP', 'CCMP', 'CCNE', 'CCO', 'CCOI',
            'CCRD', 'CCSI', 'CCU', 'CCV', 'CCXI', 'CD', 'CDAK', 'CDAQ', 'CDAY',
            'CDE', 'CDIO', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDRO', 'CDT', 'CDTX',
            'CDW', 'CDXC', 'CDXS', 'CDZI', 'CEAD', 'CECO', 'CEE', 'CEI', 'CEIX',
            'CELC', 'CELH', 'CELU', 'CELZ', 'CEM', 'CENT', 'CENTA', 'CENX', 'CEPU',
            'CEQP', 'CERN', 'CERS', 'CET', 'CETX', 'CEVA', 'CF', 'CFB', 'CFBK',
            'CFFI', 'CFFN', 'CFG', 'CFIV', 'CGEN', 'CGC', 'CGEM', 'CGNX', 'CGON',
            'CGTX', 'CHCI', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHGG',
            'CHI', 'CHKP', 'CHMA', 'CHMI', 'CHN', 'CHPT', 'CHRD', 'CHRS', 'CHRW',
            'CHSCP', 'CHT', 'CHTR', 'CHW', 'CHWY', 'CHX', 'CHY', 'CI', 'CIA',
            'CIB', 'CIDM', 'CIEN', 'CIFR', 'CIG', 'CIIG', 'CIM', 'CINC', 'CINF',
            'CING', 'CINT', 'CIO', 'CION', 'CISO', 'CITE', 'CITM', 'CIVB', 'CIXX',
            'CIZN', 'CJJD', 'CKH', 'CKPT', 'CL', 'CLAR', 'CLBK', 'CLBT', 'CLDI',
            'CLDR', 'CLDT', 'CLDX', 'CLF', 'CLFD', 'CLGN', 'CLH', 'CLIR', 'CLLS',
            'CLM', 'CLMB', 'CLMT', 'CLNE', 'CLNN', 'CLOE', 'CLOV', 'CLPS', 'CLPT',
            'CLR', 'CLRB', 'CLS', 'CLSD', 'CLSK', 'CLSN', 'CLVR', 'CLVS', 'CLW',
            'CLWT', 'CLXT', 'CLYM', 'CM', 'CMA', 'CMAX', 'CMBM', 'CMC', 'CMCA',
            'CMCL', 'CMCO', 'CMCSA', 'CMCT', 'CMD', 'CME', 'CMLS', 'CMPI', 'CMPO',
            'CMPR', 'CMPS', 'CMRA', 'CMRX', 'CMS', 'CMTG', 'CMTL', 'CNA', 'CNBKA',
            'CNC', 'CNDT', 'CNF', 'CNFR', 'CNHI', 'CNK', 'CNM', 'CNMD', 'CNNE',
            'CNO', 'CNOB', 'CNP', 'CNQ', 'CNS', 'CNSL', 'CNTB', 'CNTG', 'CNTM',
            'CNTY', 'CNX', 'CNXC', 'CNXN', 'CO', 'COCO', 'CODE', 'CODI', 'COFS',
            'COGT', 'COHR', 'COHU', 'COIN', 'COKE', 'COLB', 'COLD', 'COLE', 'COLL',
            'COLM', 'COMM', 'COMT', 'CONL', 'CONN', 'COO', 'COOK', 'COOL', 'COPR',
            'CORE', 'CORN', 'CORR', 'CORS', 'CORTX', 'COST', 'COUP', 'COV', 'COYA',
            'COZI', 'CPA', 'CPAC', 'CPAH', 'CPB', 'CPBI', 'CPHI', 'CPHC', 'CPIX',
            'CPK', 'CPNG', 'CPOP', 'CPRI', 'CPRT', 'CPS', 'CPSH', 'CPSS', 'CPT',
            'CPTK', 'CPTN', 'CPUH', 'CPZ', 'CQP', 'CR', 'CRAI', 'CRBP', 'CRC',
            'CRCT', 'CRDF', 'CRDL', 'CRDO', 'CREG', 'CRESY', 'CREX', 'CRF', 'CRGO',
            'CRGY', 'CRH', 'CRIS', 'CRK', 'CRL', 'CRM', 'CRMD', 'CRMT', 'CRNC',
            'CRNT', 'CRNX', 'CRON', 'CROP', 'CROS', 'CRP', 'CRSP', 'CRSR', 'CRT',
            'CRTO', 'CRUS', 'CRVL', 'CRVS', 'CRWD', 'CRWS', 'CRXT', 'CRZN', 'CS',
            'CSA', 'CSAN', 'CSB', 'CSBR', 'CSCO', 'CSGP', 'CSGS', 'CSIQ', 'CSL',
            'CSLM', 'CSLT', 'CSQ', 'CSR', 'CSSE', 'CSTL', 'CSTM', 'CSV', 'CSWC',
            'CSWI', 'CSX', 'CTAS', 'CTBB', 'CTBI', 'CTCX', 'CTG', 'CTHR', 'CTIC',
            'CTIB', 'CTLP', 'CTMX', 'CTO', 'CTOS', 'CTR', 'CTRA', 'CTRN', 'CTS',
            'CTSH', 'CTSO', 'CTT', 'CTV', 'CTVA', 'CTXR', 'CTXS', 'CUB', 'CUBE',
            'CUBI', 'CULP', 'CUMN', 'CUTR', 'CUZ', 'CVA', 'CVAC', 'CVBF', 'CVC',
            'CVCO', 'CVE', 'CVEO', 'CVGI', 'CVGW', 'CVI', 'CVKD', 'CVLT', 'CVLG',
            'CVM', 'CVNA', 'CVR', 'CVS', 'CVU', 'CVV', 'CVX', 'CW', 'CWBC', 'CWBR',
            'CWCO', 'CWD', 'CWEB', 'CWH', 'CWK', 'CWST', 'CWT', 'CX', 'CXAI', 'CXE',
            'CXDO', 'CXM', 'CXP', 'CXT', 'CXW', 'CYAD', 'CYAN', 'CYBE', 'CYBR',
            'CYCC', 'CYCL', 'CYD', 'CYH', 'CYTH', 'CYTK', 'CYTO', 'CYTR', 'CZFS',
            'CZR', 'CZWI',
            
            # Additional C symbols
            'CAAP', 'CABA', 'CADC', 'CADL', 'CAFD', 'CAKE', 'CALM', 'CAMP', 'CAMT',
            'CANF', 'CANG', 'CAPL', 'CAPR', 'CAPX', 'CARB', 'CARD', 'CARE', 'CARG',
            'CARR', 'CART', 'CARV', 'CAS', 'CASA', 'CASE', 'CASH', 'CASI', 'CASS',
            'CASY', 'CATB', 'CATC', 'CATS', 'CATX', 'CAVM', 'CBAH', 'CBAT', 'CBAY',
            'CBFV', 'CBIO', 'CBLI', 'CBNK', 'CBOE', 'CBPO', 'CBRE', 'CBRL', 'CBSH',
            'CBTX', 'CCAP', 'CCBG', 'CCCC', 'CCCS', 'CCEL', 'CCEP', 'CCG', 'CCIX',
            'CCJ', 'CCLP', 'CCMP', 'CCNE', 'CCO', 'CCOI', 'CCRD', 'CCSI', 'CCU',
            'CCV', 'CCXI', 'CD', 'CDAK', 'CDAQ', 'CDAY', 'CDE', 'CDIO', 'CDLX',
            'CDMO', 'CDNA', 'CDNS', 'CDRO', 'CDT', 'CDTX', 'CDW', 'CDXC', 'CDXS',
            'CDZI', 'CEAD', 'CECO', 'CEE', 'CEI', 'CEIX', 'CELC', 'CELH', 'CELU',
            'CELZ', 'CEM', 'CENT', 'CENTA', 'CENX', 'CEPU', 'CEQP', 'CERN', 'CERS',
            'CET', 'CETX', 'CEVA', 'CF', 'CFB', 'CFBK', 'CFFI', 'CFFN', 'CFG',
            'CFIV', 'CGEN', 'CGC', 'CGEM', 'CGNX', 'CGON', 'CGTX', 'CHCI', 'CHCO',
            'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHGG', 'CHI', 'CHKP', 'CHMA',
            'CHMI', 'CHN', 'CHPT', 'CHRD', 'CHRS', 'CHRW', 'CHSCP', 'CHT', 'CHTR',
            'CHW', 'CHWY', 'CHX', 'CHY', 'CI', 'CIA', 'CIB', 'CIDM', 'CIEN', 'CIFR',
            'CIG', 'CIIG', 'CIM', 'CINC', 'CINF', 'CING', 'CINT', 'CIO', 'CION',
            'CISO', 'CITE', 'CITM', 'CIVB', 'CIXX', 'CIZN', 'CJJD', 'CKH', 'CKPT',
            'CL', 'CLAR', 'CLBK', 'CLBT', 'CLDI', 'CLDR', 'CLDT', 'CLDX', 'CLF',
            'CLFD', 'CLGN', 'CLH', 'CLIR', 'CLLS', 'CLM', 'CLMB', 'CLMT', 'CLNE',
            'CLNN', 'CLOE', 'CLOV', 'CLPS', 'CLPT', 'CLR', 'CLRB', 'CLS', 'CLSD',
            'CLSK', 'CLSN', 'CLVR', 'CLVS', 'CLW', 'CLWT', 'CLXT', 'CLYM', 'CM',
            'CMA', 'CMAX', 'CMBM', 'CMC', 'CMCA', 'CMCL', 'CMCO', 'CMCSA', 'CMCT',
            'CMD', 'CME', 'CMLS', 'CMPI', 'CMPO', 'CMPR', 'CMPS', 'CMRA', 'CMRX',
            'CMS', 'CMTG', 'CMTL', 'CNA', 'CNBKA', 'CNC', 'CNDT', 'CNF', 'CNFR',
            'CNHI', 'CNK', 'CNM', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNQ',
            'CNS', 'CNSL', 'CNTB', 'CNTG', 'CNTM', 'CNTY', 'CNX', 'CNXC', 'CNXN',
            'CO', 'COCO', 'CODE', 'CODI', 'COFS', 'COGT', 'COHR', 'COHU', 'COIN',
            'COKE', 'COLB', 'COLD', 'COLE', 'COLL', 'COLM', 'COMM', 'COMT', 'CONL',
            'CONN', 'COO', 'COOK', 'COOL', 'COPR', 'CORE', 'CORN', 'CORR', 'CORS',
            'COST', 'COUP', 'COV', 'COYA', 'COZI', 'CPA', 'CPAC', 'CPAH', 'CPB',
            'CPBI', 'CPHI', 'CPHC', 'CPIX', 'CPK', 'CPNG', 'CPOP', 'CPRI', 'CPRT',
            'CPS', 'CPSH', 'CPSS', 'CPT', 'CPTK', 'CPTN', 'CPUH', 'CPZ', 'CQP',
            'CR', 'CRAI', 'CRBP', 'CRC', 'CRCT', 'CRDF', 'CRDL', 'CRDO', 'CREG',
            'CRESY', 'CREX', 'CRF', 'CRGO', 'CRGY', 'CRH', 'CRIS', 'CRK', 'CRL',
            'CRM', 'CRMD', 'CRMT', 'CRNC', 'CRNT', 'CRNX', 'CRON', 'CROP', 'CROS',
            'CRP', 'CRSP', 'CRSR', 'CRT', 'CRTO', 'CRUS', 'CRVL', 'CRVS', 'CRWD',
            'CRWS', 'CRXT', 'CRZN', 'CS', 'CSA', 'CSAN', 'CSB', 'CSBR', 'CSCO',
            'CSGP', 'CSGS', 'CSIQ', 'CSL', 'CSLM', 'CSLT', 'CSQ', 'CSR', 'CSSE',
            'CSTL', 'CSTM', 'CSV', 'CSWC', 'CSWI', 'CSX', 'CTAS', 'CTBB', 'CTBI',
            'CTCX', 'CTG', 'CTHR', 'CTIC', 'CTIB', 'CTLP', 'CTMX', 'CTO', 'CTOS',
            'CTR', 'CTRA', 'CTRN', 'CTS', 'CTSH', 'CTSO', 'CTT', 'CTV', 'CTVA',
            'CTXR', 'CTXS', 'CUB', 'CUBE', 'CUBI', 'CULP', 'CUMN', 'CUTR', 'CUZ',
            'CVA', 'CVAC', 'CVBF', 'CVC', 'CVCO', 'CVE', 'CVEO', 'CVGI', 'CVGW',
            'CVI', 'CVKD', 'CVLT', 'CVLG', 'CVM', 'CVNA', 'CVR', 'CVS', 'CVU',
            'CVV', 'CVX', 'CW', 'CWBC', 'CWBR', 'CWCO', 'CWD', 'CWEB', 'CWH',
            'CWK', 'CWST', 'CWT', 'CX', 'CXAI', 'CXE', 'CXDO', 'CXM', 'CXP',
            'CXT', 'CXW', 'CYAD', 'CYAN', 'CYBE', 'CYBR', 'CYCC', 'CYCL', 'CYD',
            'CYH', 'CYTH', 'CYTK', 'CYTO', 'CYTR', 'CZFS', 'CZR', 'CZWI'
        ]
        
        # Remove duplicates, filter valid symbols, and sort
        nasdaq_bc_symbols = sorted(list(set([s for s in nasdaq_bc_symbols if len(s) <= 5])))
        
        print(f"🔥 BRUTE FORCE TARGET: {len(nasdaq_bc_symbols)} B & C symbols")
        print("📊 NO FILTERING - Testing everything that moves!")
        
        # Show sample
        print("\nSample symbols:")
        for i in range(0, min(30, len(nasdaq_bc_symbols)), 6):
            sample = nasdaq_bc_symbols[i:i+6]
            print("   " + ", ".join(sample))
        
        if len(nasdaq_bc_symbols) > 30:
            print(f"   ... and {len(nasdaq_bc_symbols) - 30} more symbols")
        
        self.nasdaq_bc_symbols = nasdaq_bc_symbols
        return nasdaq_bc_symbols
    
    def brute_force_backtest_all(self, max_workers=20):
        """Brute force backtest ALL B and C symbols"""
        
        symbols = self.nasdaq_bc_symbols
        
        print(colored(f"\n🔥💎 BRUTE FORCE AEGS BACKTEST - ALL {len(symbols)} B & C SYMBOLS 💎🔥", 'red', attrs=['bold']))
        print("=" * 80)
        print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO MERCY")
        print("=" * 80)
        
        results = []
        
        # Use ThreadPoolExecutor for maximum parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._brute_force_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            completed = 0
            total = len(symbols)
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        
                        # Progress with key metrics
                        trades = result['total_trades']
                        win_rate = result['win_rate']
                        strategy_return = result['strategy_return']
                        excess_return = result['excess_return']
                        
                        # Enhanced status indicators
                        if strategy_return > 50:
                            status = "🚀"
                        elif strategy_return > 20:
                            status = "🔥"
                        elif strategy_return > 0:
                            status = "✅"
                        elif excess_return > 20:
                            status = "💎"  # Good relative performance
                        else:
                            status = "❌"
                        
                        print(f"[{completed:3d}/{total}] {status} {symbol:6s}: "
                              f"{trades:2d} trades, {win_rate:4.0f}% win, "
                              f"Strategy: {strategy_return:+6.1f}%, "
                              f"Excess: {excess_return:+6.1f}%")
                    else:
                        print(f"[{completed:3d}/{total}] 💀 {symbol:6s}: NO DATA")
                        
                except Exception as e:
                    print(f"[{completed:3d}/{total}] 💥 {symbol:6s}: ERROR")
        
        self.brute_force_results = results
        print(f"\n🎯 BRUTE FORCE COMPLETE: {len(results)}/{total} symbols backtested successfully")
        return results
    
    def _brute_force_single_symbol(self, symbol):
        """Brute force backtest single symbol"""
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')
            
            if len(df) < 50:
                return None
            
            # Simple price filter - exclude penny stocks
            current_price = df['Close'].iloc[-1]
            if current_price < 1.0:
                return None
            
            # Calculate AEGS indicators
            df = self._calculate_aegs_indicators(df)
            df = self._apply_aegs_strategy(df)
            
            # Run backtest
            return self._run_aegs_backtest(df, symbol)
            
        except Exception as e:
            return None
    
    def _calculate_aegs_indicators(self, df):
        """Calculate AEGS technical indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Daily change
        df['Daily_Change'] = df['Close'].pct_change()
        
        return df
    
    def _apply_aegs_strategy(self, df):
        """Apply AEGS strategy signals"""
        
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            signal_strength = 0
            
            # RSI oversold
            if pd.notna(row['RSI']):
                if row['RSI'] < 30:
                    signal_strength += 35
                elif row['RSI'] < 35:
                    signal_strength += 20
            
            # Bollinger Band position
            if pd.notna(row['BB_Position']):
                if row['BB_Position'] < 0:
                    signal_strength += 35
                elif row['BB_Position'] < 0.2:
                    signal_strength += 20
            
            # Volume surge with price drop
            if pd.notna(row['Volume_Ratio']) and pd.notna(row['Daily_Change']):
                if row['Volume_Ratio'] > 2.0 and row['Daily_Change'] < -0.05:
                    signal_strength += 30
                elif row['Volume_Ratio'] > 1.5:
                    signal_strength += 10
            
            df.iloc[i, df.columns.get_loc('Signal_Strength')] = signal_strength
            
            # Signal threshold
            if signal_strength >= 70:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
        
        return df
    
    def _run_aegs_backtest(self, df, symbol):
        """Run AEGS backtest"""
        
        df['Position'] = 0
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = 0
        
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        
        for i in range(len(df)):
            current_date = df.index[i]
            
            if df.iloc[i]['Signal'] == 1 and position == 0:
                # Enter position
                position = 1
                entry_price = df.iloc[i]['Close']
                entry_date = current_date
                df.iloc[i, df.columns.get_loc('Position')] = 1
                
            elif position == 1:
                df.iloc[i, df.columns.get_loc('Position')] = 1
                
                # Exit conditions
                current_price = df.iloc[i]['Close']
                returns = (current_price - entry_price) / entry_price
                days_held = (current_date - entry_date).days
                
                exit_position = False
                exit_reason = ""
                
                # AEGS exit rules
                if returns >= 0.3:  # 30% profit target
                    exit_position = True
                    exit_reason = "Profit Target 30%"
                elif returns <= -0.2:  # 20% stop loss
                    exit_position = True
                    exit_reason = "Stop Loss 20%"
                elif days_held >= 30 and returns > 0:
                    exit_position = True
                    exit_reason = "Time Exit (Profitable)"
                elif days_held >= 60:
                    exit_position = True
                    exit_reason = "Force Exit"
                
                if exit_position:
                    position = 0
                    
                    trades.append({
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'exit_date': current_date.strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': returns * 100,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    df.iloc[i, df.columns.get_loc('Position')] = 0
        
        # Calculate strategy returns
        for i in range(1, len(df)):
            if df.iloc[i]['Position'] == 1:
                df.iloc[i, df.columns.get_loc('Strategy_Returns')] = df.iloc[i]['Returns']
        
        # Performance metrics
        total_return = (1 + df['Strategy_Returns']).cumprod().iloc[-1] - 1
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        winning_trades = [t for t in trades if t['return_pct'] > 0]
        losing_trades = [t for t in trades if t['return_pct'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Current price for context
        current_price = df['Close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'strategy_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': (total_return - buy_hold_return) * 100,
            'trades': trades[-3:] if trades else []
        }
    
    def analyze_brute_force_results(self):
        """Analyze brute force results"""
        
        if not self.brute_force_results:
            print("❌ No brute force results to analyze")
            return
        
        print(colored(f"\n🔥💎 NASDAQ B & C BRUTE FORCE AEGS RESULTS 💎🔥", 'red', attrs=['bold']))
        print("=" * 80)
        
        # Sort by strategy return (actual profitability)
        sorted_by_strategy = sorted(self.brute_force_results, key=lambda x: x['strategy_return'], reverse=True)
        
        # Also sort by excess return for comparison
        sorted_by_excess = sorted(self.brute_force_results, key=lambda x: x['excess_return'], reverse=True)
        
        # Top profitable performers (strategy return > 0)
        profitable_stocks = [r for r in sorted_by_strategy if r['strategy_return'] > 0]
        
        print(colored(f"\n🚀 TOP 20 ACTUALLY PROFITABLE STOCKS (Strategy Return > 0%):", 'green', attrs=['bold']))
        print("=" * 110)
        print(f"{'#':<3} {'Symbol':<8} {'Price':<10} {'Trades':<7} {'Win%':<6} {'Strategy%':<11} {'Excess%':<9} {'Avg Win':<9}")
        print("=" * 110)
        
        for i, result in enumerate(profitable_stocks[:20], 1):
            symbol = result['symbol']
            price = result['current_price']
            trades = result['total_trades']
            win_rate = result['win_rate']
            strategy_return = result['strategy_return']
            excess_return = result['excess_return']
            avg_win = result['avg_win']
            
            # Color coding for profitability
            if strategy_return > 50:
                color = 'red'
                attrs = ['bold']
            elif strategy_return > 20:
                color = 'yellow'
                attrs = ['bold']
            elif strategy_return > 10:
                color = 'green'
                attrs = ['bold']
            else:
                color = 'green'
                attrs = []
            
            print(colored(f"{i:<3} {symbol:<8} ${price:<9.2f} {trades:<7} {win_rate:<6.0f} "
                         f"{strategy_return:<11.1f} {excess_return:<9.1f} {avg_win:<9.1f}", color, attrs=attrs))
        
        # Summary statistics
        total_analyzed = len(self.brute_force_results)
        with_trades = [r for r in self.brute_force_results if r['total_trades'] > 0]
        positive_excess = [r for r in self.brute_force_results if r['excess_return'] > 0]
        actually_profitable = [r for r in self.brute_force_results if r['strategy_return'] > 0]
        goldmines = [r for r in self.brute_force_results if r['strategy_return'] > 50]
        solid_performers = [r for r in self.brute_force_results if r['strategy_return'] > 20 and r['total_trades'] >= 3]
        
        if with_trades:
            avg_win_rate = np.mean([r['win_rate'] for r in with_trades])
            avg_strategy_return = np.mean([r['strategy_return'] for r in self.brute_force_results])
            avg_excess = np.mean([r['excess_return'] for r in self.brute_force_results])
            
            print(colored(f"\n📊 B & C SYMBOLS SUMMARY STATISTICS:", 'cyan', attrs=['bold']))
            print("=" * 50)
            print(f"   Total Symbols Tested: {total_analyzed}")
            print(f"   Symbols with Trades: {len(with_trades)}")
            print(f"   Average Win Rate: {avg_win_rate:.1f}%")
            print(f"   Average Strategy Return: {avg_strategy_return:.1f}%")
            print(f"   Average Excess Return: {avg_excess:.1f}%")
            print(f"   Positive Excess Returns: {len(positive_excess)}/{total_analyzed} ({len(positive_excess)/total_analyzed*100:.0f}%)")
            print(colored(f"   🚀 ACTUALLY PROFITABLE: {len(actually_profitable)}/{total_analyzed} ({len(actually_profitable)/total_analyzed*100:.0f}%)", 'green', attrs=['bold']))
            print(colored(f"   🔥 GOLDMINES (>50% return): {len(goldmines)}", 'red', attrs=['bold']))
            print(colored(f"   💎 SOLID PERFORMERS (>20% return): {len(solid_performers)}", 'yellow', attrs=['bold']))
        
        # Category breakdown
        self._analyze_by_categories(actually_profitable)
        
        # Save results
        self._save_bc_results(sorted_by_strategy, actually_profitable)
        
        return sorted_by_strategy
    
    def _analyze_by_categories(self, profitable_stocks):
        """Analyze profitable stocks by categories/sectors"""
        
        print(colored(f"\n📈 PROFITABLE STOCKS BY CATEGORY:", 'magenta', attrs=['bold']))
        print("=" * 60)
        
        # Simple category classification
        biotech = [s for s in profitable_stocks if any(x in s['symbol'] for x in ['BIO', 'CELL', 'CRIS', 'CRSP', 'BEAM'])]
        crypto = [s for s in profitable_stocks if any(x in s['symbol'] for x in ['BTC', 'COIN', 'CRYPTO', 'BITF', 'CLSK'])]
        tech = [s for s in profitable_stocks if any(x in s['symbol'] for x in ['CRWD', 'CRM', 'CSCO', 'CIEN', 'CDW'])]
        healthcare = [s for s in profitable_stocks if any(x in s['symbol'] for x in ['CHWY', 'CVS', 'CI', 'CNC'])]
        
        categories = [
            ("🧬 Biotech", biotech),
            ("💰 Crypto/Mining", crypto),
            ("💻 Technology", tech),
            ("🏥 Healthcare", healthcare)
        ]
        
        for category_name, stocks in categories:
            if stocks:
                avg_return = np.mean([s['strategy_return'] for s in stocks])
                print(f"{category_name}: {len(stocks)} stocks, avg return: {avg_return:.1f}%")
                top_3 = sorted(stocks, key=lambda x: x['strategy_return'], reverse=True)[:3]
                for stock in top_3:
                    print(f"   {stock['symbol']}: {stock['strategy_return']:+.1f}%")
    
    def _save_bc_results(self, sorted_results, profitable_stocks):
        """Save B & C results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nasdaq_bc_brute_force_aegs_results_{timestamp}.json'
        
        # Top performers and profitable stocks
        top_performers = sorted_results[:30]
        
        export_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'BRUTE FORCE - B & C symbols',
            'criteria': 'All NASDAQ B & C symbols, $1+ price filter only',
            'total_analyzed': len(self.brute_force_results),
            'top_performers': top_performers,
            'profitable_stocks': profitable_stocks,
            'goldmines': [r for r in sorted_results if r['strategy_return'] > 50],
            'solid_performers': [r for r in sorted_results if r['strategy_return'] > 20 and r['total_trades'] >= 3],
            'summary': {
                'total_analyzed': len(self.brute_force_results),
                'profitable_count': len(profitable_stocks),
                'avg_strategy_return': np.mean([r['strategy_return'] for r in self.brute_force_results]),
                'avg_excess_return': np.mean([r['excess_return'] for r in self.brute_force_results]),
                'goldmine_count': len([r for r in sorted_results if r['strategy_return'] > 50]),
                'solid_performer_count': len([r for r in sorted_results if r['strategy_return'] > 20 and r['total_trades'] >= 3])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n💾 B & C results saved to: {filename}")
        print(colored(f"\n🎯 PROFITABLE CANDIDATES FOR AEGS WATCHLIST:", 'green', attrs=['bold']))
        print("=" * 60)
        
        for i, stock in enumerate(profitable_stocks[:10], 1):
            symbol = stock['symbol']
            strategy_return = stock['strategy_return']
            win_rate = stock['win_rate']
            trades = stock['total_trades']
            price = stock['current_price']
            
            print(f"{i:2d}. {symbol:6s} (${price:6.2f}): {strategy_return:+6.1f}% strategy return, "
                  f"{win_rate:4.0f}% win rate, {trades} trades")

def main():
    """Run brute force AEGS backtest on NASDAQ B & C symbols"""
    
    print(colored("🔥💎 NASDAQ 'B' & 'C' SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥", 'red', attrs=['bold']))
    print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO PRESELECTION")
    print("Complete alphabet expansion - B through C symbols")
    print("=" * 70)
    
    backtester = NASDAQBCBruteForceAEGS()
    
    # Step 1: Get ALL NASDAQ B & C symbols
    symbols = backtester.get_all_nasdaq_bc_symbols()
    
    if not symbols:
        print("❌ No symbols found")
        return
    
    # Step 2: Brute force backtest everything
    print(f"\n🔥 INITIATING BRUTE FORCE ATTACK ON {len(symbols)} B & C SYMBOLS...")
    results = backtester.brute_force_backtest_all(max_workers=25)
    
    if not results:
        print("❌ No successful backtests")
        return
    
    # Step 3: Analyze results focusing on actual profitability
    backtester.analyze_brute_force_results()
    
    print(colored(f"\n🎯 NASDAQ B & C BRUTE FORCE MISSION COMPLETE!", 'green', attrs=['bold']))
    print(f"Found profitable AEGS candidates across B and C symbols!")

if __name__ == "__main__":
    main()