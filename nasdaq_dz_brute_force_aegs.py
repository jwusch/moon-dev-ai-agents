#!/usr/bin/env python3
"""
🔥💎 NASDAQ D-Z SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥
Complete alphabet expansion - D through Z symbols
NO FILTERING | NO SCREENING | NO PRESELECTION
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

warnings.filterwarnings('ignore')

print("🔥💎 NASDAQ 'D' through 'Z' SYMBOLS BRUTE FORCE AEGS BACKTEST 💎🔥")
print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO PRESELECTION")
print("Complete alphabet expansion - D through Z symbols")
print("=" * 70)

def get_nasdaq_dz_symbols():
    """Build comprehensive D-Z NASDAQ symbol universe"""
    
    # Major known D-Z symbols across all sectors
    base_symbols = [
        # D symbols
        "DADA", "DAKT", "DAL", "DALI", "DAN", "DARE", "DATS", "DAVE", "DBGI", "DBX",
        "DBRG", "DCBO", "DCPH", "DCOM", "DDOG", "DDIV", "DELC", "DELL", "DENN", "DEO",
        "DESC", "DEVA", "DGII", "DHIL", "DHR", "DIAX", "DIN", "DISH", "DJCO", "DLO",
        "DLPN", "DLTR", "DMAC", "DNA", "DNUT", "DOCS", "DOCU", "DOOO", "DORM", "DOVA",
        "DOX", "DRIO", "DRNA", "DRUG", "DSGX", "DSP", "DSPG", "DSS", "DTIL", "DUO",
        "DUOL", "DVAX", "DV", "DVMT", "DXCM", "DXLG", "DYAI", "DYN", "DZSI",
        
        # E symbols  
        "EA", "EAR", "EARS", "EAST", "EBON", "ECHO", "ECOL", "ECOM", "EDAP", "EDIT",
        "EDNT", "EDRY", "EDSA", "EDTK", "EDUC", "EEFT", "EEI", "EEMA", "EFAS", "EFSC",
        "EFX", "EGAN", "EGHT", "EGOV", "EH", "EIG", "EIGI", "EKSO", "EL", "ELAB",
        "ELAT", "ELF", "ELMD", "ELOX", "ELSE", "ELTK", "ELVT", "EMBC", "EMCF", "EME",
        "EMKR", "EML", "EMMS", "ENDP", "ENFA", "ENG", "ENPH", "ENR", "ENSC", "ENTG",
        "ENV", "ENVB", "ENVX", "EOG", "EOSE", "EPAC", "EPAM", "EPC", "EPIX", "EPM",
        "EQBK", "EQIX", "ERF", "ERIC", "ERIE", "EROS", "ESE", "ESGR", "ESLT", "ESPR",
        "ESQ", "ESTA", "ESTC", "ET", "ETAO", "ETNB", "ETO", "ETRN", "ETSY", "EURN",
        "EVA", "EVBG", "EVFM", "EVGO", "EVGR", "EVK", "EVLO", "EVLV", "EVO", "EVOK",
        "EVOP", "EWBC", "EXAI", "EXAS", "EXC", "EXEL", "EXK", "EXLS", "EXP", "EXPD",
        "EXPE", "EXPO", "EXPR", "EYE", "EYES", "EYPT", "EZGO", "EZPW",
        
        # F symbols
        "FA", "FAAN", "FAAS", "FABP", "FACT", "FAD", "FALN", "FAMI", "FANG", "FANH",
        "FARM", "FAST", "FAT", "FATB", "FATE", "FB", "FBNC", "FBIO", "FBNK", "FBRT",
        "FC", "FCAP", "FCBC", "FCCO", "FCF", "FCHA", "FCHT", "FCNCA", "FCPT", "FDBC",
        "FDEU", "FDMT", "FDP", "FDUS", "FE", "FEAM", "FEBO", "FEIM", "FELE", "FEMY",
        "FENG", "FERG", "FERO", "FET", "FEXD", "FF", "FFBC", "FFIE", "FFIN", "FFIV",
        "FFWM", "FGBI", "FGF", "FGMC", "FGP", "FHB", "FHTX", "FIBK", "FICS", "FIFTY",
        "FIGS", "FINM", "FINS", "FINV", "FINW", "FINX", "FIP", "FISI", "FISV", "FITB",
        "FIVE", "FIXX", "FIZ", "FIZZ", "FKWL", "FL", "FLAG", "FLDM", "FLEX", "FLIC",
        "FLIR", "FLLO", "FLNG", "FLNT", "FLOW", "FLR", "FLWS", "FLXN", "FLXS", "FLYY",
        "FMAO", "FMBH", "FMC", "FMTX", "FMY", "FNCB", "FNCH", "FNCX", "FND", "FNF",
        "FNLC", "FNWB", "FOCS", "FOLD", "FONR", "FORD", "FORM", "FORTY", "FOSL", "FOX",
        "FOXA", "FOXF", "FPA", "FRAF", "FRAN", "FRBA", "FRBK", "FRBN", "FRC", "FRED",
        "FREY", "FRGI", "FRLA", "FRME", "FRPH", "FRPT", "FRSH", "FRST", "FRSX", "FRTA",
        "FSB", "FSBW", "FSCO", "FSFG", "FSK", "FSLY", "FSM", "FSTR", "FSV", "FTAI",
        "FTC", "FTCI", "FTDR", "FTEK", "FTI", "FTNT", "FTRE", "FTRP", "FUEL", "FULC",
        "FULT", "FUN", "FUNC", "FUND", "FUSB", "FUV", "FUTU", "FV", "FVCB", "FVRR",
        "FWBI", "FWONA", "FWONK", "FWRD", "FXNC", "FYLD",
        
        # G symbols
        "GABC", "GAIA", "GAIN", "GAL", "GALT", "GAMB", "GAMC", "GASS", "GATE", "GATX",
        "GAU", "GB", "GBAB", "GBT", "GBCI", "GBTG", "GCBC", "GCO", "GCMG", "GCT",
        "GDDY", "GDEN", "GDOT", "GDYN", "GEF", "GEL", "GEN", "GENC", "GENE", "GENY",
        "GEOS", "GERN", "GEVO", "GFAI", "GFG", "GGB", "GGAL", "GGG", "GHDX", "GHG",
        "GHLD", "GHSI", "GIAC", "GIB", "GIFI", "GIGM", "GIII", "GIL", "GILT", "GINS",
        "GIS", "GJPI", "GK", "GKOS", "GL", "GLAE", "GLAN", "GLBE", "GLBZ", "GLDD",
        "GLG", "GLLI", "GLMD", "GLNG", "GLO", "GLPG", "GLPI", "GLRE", "GLSI", "GLTO",
        "GLUE", "GLW", "GLYC", "GM", "GMAB", "GMAT", "GMDA", "GME", "GMED", "GMGI",
        "GMS", "GMTX", "GNFT", "GNLN", "GNMA", "GNMX", "GNPX", "GNRC", "GNSS", "GNTX",
        "GNUS", "GO", "GOCO", "GOEV", "GOF", "GOGL", "GOLD", "GOLF", "GOOD", "GOOG",
        "GOOGL", "GOSS", "GOT", "GOTU", "GP", "GPC", "GPK", "GPN", "GPRE", "GPRK",
        "GPS", "GRAB", "GRAL", "GRBK", "GRCY", "GREE", "GRFS", "GRIN", "GRMN", "GRND",
        "GRNQ", "GROW", "GRPN", "GRTS", "GRTX", "GRVY", "GRWG", "GSBC", "GSIT", "GSM",
        "GSKY", "GT", "GTBP", "GTE", "GTEC", "GTH", "GTHX", "GTI", "GTIM", "GTLB",
        "GTN", "GTS", "GTX", "GTY", "GURE", "GV", "GVCI", "GVP", "GWPH", "GWRS", "GWW",
        "GYRO",
        
        # H symbols
        "HAFC", "HAIN", "HAL", "HALO", "HAPP", "HARP", "HAS", "HASI", "HAYN", "HBAN",
        "HBB", "HBCP", "HBI", "HBIO", "HBM", "HBNC", "HBP", "HCAC", "HCAP", "HCC",
        "HCCI", "HCM", "HCP", "HCSG", "HCTI", "HCVI", "HCXY", "HD", "HDB", "HDSN",
        "HE", "HEAR", "HEI", "HELE", "HEP", "HEPS", "HES", "HESM", "HFBL", "HFFG",
        "HFT", "HGEN", "HGV", "HHR", "HIBB", "HIHO", "HII", "HIMX", "HIO", "HIPO",
        "HITI", "HJLI", "HKIT", "HL", "HLBZ", "HLF", "HLIO", "HLMN", "HLNE", "HLT",
        "HLTH", "HLVX", "HLX", "HMC", "HMHC", "HMN", "HMNF", "HMST", "HMTV", "HMY",
        "HNNA", "HNRG", "HNST", "HOFT", "HOG", "HOLI", "HOLO", "HOLX", "HOME", "HON",
        "HONE", "HOOD", "HOOK", "HOP", "HOTH", "HOVN", "HOV", "HP", "HPE", "HPH",
        "HPI", "HPK", "HPQ", "HQI", "HQY", "HR", "HRB", "HRC", "HRI", "HRO", "HRMY",
        "HROW", "HRTG", "HRTX", "HRZN", "HSBC", "HSCS", "HSDT", "HSI", "HSIC", "HST",
        "HSTM", "HSY", "HT", "HTA", "HTBK", "HTD", "HTGC", "HTH", "HTHT", "HTLF",
        "HUBS", "HUD", "HUIZ", "HUN", "HURC", "HURN", "HUSN", "HVBC", "HVT", "HWBK",
        "HWC", "HWCC", "HWKN", "HX", "HYAC", "HY", "HYGS", "HYLB", "HYMC", "HYMB",
        "HYPR", "HYW", "HYZN", "HZO",
        
        # I symbols
        "IACI", "IART", "IAS", "IAUX", "IBCP", "IBEX", "IBIO", "IBKR", "IBM", "IBOC",
        "IBTX", "ICAD", "ICBK", "ICCM", "ICCT", "ICD", "ICE", "ICFI", "ICG", "ICHR",
        "ICI", "ICLN", "ICON", "ICP", "ICPT", "ICUI", "IDA", "IDAI", "IDCC", "IDEX",
        "IDN", "IDRA", "IDXX", "IDYA", "IEA", "IEP", "IESC", "IEX", "IFBD", "IFF",
        "IGAC", "IGIC", "IGMS", "IGT", "IHRT", "III", "IIIV", "IIIN", "IIPR", "IKNA",
        "IKT", "ILMN", "ILPT", "IM", "IMAC", "IMAQ", "IMAX", "IMCC", "IMCR", "IMGN",
        "IMKTA", "IMMP", "IMMR", "IMMU", "IMO", "IMPP", "IMRA", "IMRN", "IMTE", "IMTX",
        "IMUX", "IMVT", "IMXI", "INAB", "INAQ", "INBK", "INBX", "INCR", "INCY", "INDB",
        "INDI", "INDT", "INFI", "INFN", "INFO", "INGN", "INKT", "INMD", "INN", "INNV",
        "INO", "INOD", "INPX", "INSE", "INSM", "INST", "INSW", "INT", "INTC", "INTG",
        "INTU", "INTZ", "INUV", "INVA", "INVE", "INVH", "INVZ", "INZY", "IO", "IOSP",
        "IOVA", "IP", "IPAR", "IPGP", "IPHA", "IPI", "IPKW", "IPSC", "IPWR", "IPX",
        "IPXL", "IQ", "IQI", "IQMD", "IQNT", "IQV", "IR", "IRBT", "IRDM", "IRIX",
        "IRMD", "IRM", "IRON", "IRTC", "ISBC", "ISDR", "ISEE", "ISIG", "ISIS", "ISLA",
        "ISNS", "ISPC", "ISPR", "ISRG", "ISSC", "ISTR", "ISUN", "IT", "ITCI", "ITGR",
        "ITHX", "ITI", "ITIC", "ITRI", "ITRM", "ITRN", "ITT", "ITUB", "ITW", "IVA",
        "IVAC", "IVC", "IVDA", "IVR", "IVZ", "IWOV", "IX", "IXAQ", "IXHL",
        
        # J symbols  
        "JACK", "JAG", "JAKK", "JAMF", "JAN", "JANX", "JAQC", "JAZZ", "JBHT", "JBLU",
        "JBT", "JCAP", "JCI", "JCOM", "JD", "JEF", "JEMD", "JEQ", "JFIN", "JFU",
        "JG", "JHG", "JHS", "JILL", "JKS", "JMIA", "JMP", "JMPN", "JNCE", "JNJ",
        "JNPR", "JOBS", "JOBY", "JOUT", "JOY", "JPM", "JQC", "JRJC", "JRVR", "JSM",
        "JSML", "JSTQ", "JT", "JTEK", "JUNO", "JVA", "JWEL", "JWN", "JWS", "JXJT",
        "JYNT",
        
        # K symbols
        "KAI", "KALA", "KALV", "KAR", "KARO", "KAVL", "KB", "KBH", "KBR", "KC",
        "KCGI", "KD", "KDLY", "KDMN", "KDP", "KE", "KELYA", "KEN", "KEP", "KERN",
        "KEX", "KEY", "KEYS", "KF", "KFRC", "KFS", "KFY", "KGC", "KHC", "KIDS",
        "KIM", "KIN", "KINS", "KIO", "KIQ", "KIRK", "KISO", "KKR", "KKVS", "KLIC",
        "KLR", "KLTR", "KLXE", "KMB", "KMDA", "KMI", "KMPB", "KMT", "KMX", "KN",
        "KNDI", "KNF", "KNSA", "KNSL", "KNX", "KO", "KOD", "KODK", "KOF", "KOLD",
        "KORE", "KOS", "KOSS", "KP", "KPLT", "KPTI", "KPTN", "KR", "KREF", "KRG",
        "KRKR", "KRMD", "KRNT", "KRNY", "KRO", "KRON", "KRP", "KRYS", "KSS", "KT",
        "KTB", "KTOS", "KTRA", "KTTA", "KUKE", "KULR", "KURA", "KURO", "KUSN", "KVA",
        "KVHI", "KW", "KWEB", "KWR", "KXIN", "KYN", "KZR",
        
        # L symbols
        "LAAC", "LAB", "LABP", "LAC", "LADR", "LAKE", "LAMR", "LANC", "LAND", "LASR",
        "LATN", "LAUR", "LAW", "LAZ", "LAZR", "LB", "LBPH", "LBRDA", "LBRDK", "LBRT",
        "LBTYA", "LBTYB", "LBTYK", "LC", "LCA", "LCFY", "LCID", "LCI", "LCNB", "LCTX",
        "LCW", "LDI", "LDOS", "LE", "LEA", "LEAF", "LEGH", "LEGN", "LEN", "LEO",
        "LEVI", "LEV", "LFAC", "LFC", "LFLY", "LFMD", "LFST", "LFUS", "LFVN", "LFW",
        "LGCY", "LGF.A", "LGF.B", "LGHL", "LGI", "LGIH", "LGND", "LGO", "LHCG", "LHX",
        "LI", "LIAN", "LIBH", "LICY", "LIFE", "LII", "LILA", "LILAK", "LIME", "LINC",
        "LIND", "LINE", "LINK", "LIN", "LION", "LIOX", "LIPO", "LIQT", "LITB", "LITE",
        "LITM", "LIVE", "LIVN", "LIXT", "LIZI", "LJPC", "LKQ", "LL", "LLIT", "LLY",
        "LMB", "LMFA", "LMNR", "LMT", "LNC", "LND", "LNG", "LNKB", "LNN", "LNT",
        "LNTH", "LNW", "LOAN", "LOB", "LOBO", "LOCL", "LOCO", "LODE", "LOGI", "LOGM",
        "LOMA", "LOOP", "LOPE", "LOVE", "LOW", "LPCN", "LPG", "LPI", "LPL", "LPLA",
        "LPN", "LPRO", "LPSN", "LPTH", "LPX", "LQD", "LQDA", "LQDT", "LRC", "LRCX",
        "LRFC", "LRN", "LSAC", "LSBK", "LSCC", "LSI", "LSPD", "LSTR", "LSXMA", "LSXMB",
        "LSXMK", "LTC", "LTG", "LTH", "LTBR", "LU", "LUCD", "LUCY", "LULU", "LUNA",
        "LUNG", "LUV", "LUXH", "LVLU", "LVS", "LWLG", "LX", "LXEH", "LXFR", "LXP",
        "LXRX", "LXU", "LYB", "LYEL", "LYFT", "LYG", "LYL", "LYNN", "LYT", "LYV",
        "LZ", "LZB",
        
        # M symbols
        "MAA", "MAC", "MACK", "MAG", "MAIA", "MAN", "MANH", "MANU", "MAR", "MARA",
        "MARK", "MAS", "MASI", "MAT", "MATV", "MATW", "MATX", "MAX", "MAXR", "MBA",
        "MBIN", "MBT", "MBUU", "MBWM", "MC", "MCA", "MCB", "MCBC", "MCD", "MCHP",
        "MCI", "MCK", "MCO", "MCS", "MDB", "MDCA", "MDGL", "MDIA", "MDJH", "MDLZ",
        "MDRR", "MDRX", "MDWD", "MDXG", "ME", "MEC", "MED", "MEDP", "MEDS", "MEG",
        "MEI", "MELI", "MET", "META", "METC", "MFIN", "MFC", "MFG", "MFGP", "MFH",
        "MFIC", "MG", "MGA", "MGEE", "MGI", "MGIN", "MGLN", "MGM", "MGNI", "MGNX",
        "MGPI", "MGRC", "MGTA", "MGY", "MHK", "MHO", "MICT", "MIK", "MILK", "MIMO",
        "MIR", "MIRA", "MIRM", "MIST", "MITA", "MITK", "MIY", "MJ", "MKD", "MKSI",
        "MKTW", "MKTX", "ML", "MLAB", "MLCO", "MLGO", "MLI", "MLM", "MLNK", "MLP",
        "MLR", "MM", "MMAT", "MMC", "MMD", "MMI", "MMLP", "MMM", "MMP", "MMS", "MMSI",
        "MMU", "MMYT", "MN", "MNDO", "MNDY", "MNKD", "MNMD", "MNP", "MNRO", "MNR",
        "MNST", "MNTK", "MO", "MOAT", "MOB", "MOBQ", "MOD", "MODN", "MOGO", "MOHC",
        "MOM", "MOND", "MOR", "MORF", "MORN", "MOS", "MOTS", "MOUR", "MOVE", "MP",
        "MPA", "MPAA", "MPB", "MPC", "MPLN", "MPLX", "MPWR", "MPX", "MQ", "MRAM",
        "MRBK", "MRC", "MRCC", "MRCY", "MREO", "MRH", "MRK", "MRKR", "MRM", "MRNA",
        "MRNS", "MRO", "MRTN", "MRTX", "MRUS", "MRVL", "MS", "MSA", "MSB", "MSBI",
        "MSC", "MSCI", "MSD", "MSEX", "MSFT", "MSGE", "MSGM", "MSI", "MSM", "MSN",
        "MSO", "MSTR", "MSVB", "MT", "MTA", "MTB", "MTC", "MTD", "MTEM", "MTG",
        "MTH", "MTN", "MTOR", "MTRN", "MTRX", "MTSI", "MTTR", "MU", "MUA", "MUC",
        "MUDSU", "MUE", "MULN", "MUR", "MUS", "MUSA", "MUX", "MVBF", "MVIS", "MVO",
        "MXCT", "MXL", "MYE", "MYFW", "MYGN", "MYMD", "MYO", "MYOV", "MYTE", "MZZ",
        
        # N symbols
        "NAII", "NAK", "NAMS", "NAOV", "NARI", "NATL", "NATR", "NAUT", "NAVI", "NAVB",
        "NBBK", "NBH", "NBLX", "NBN", "NBRV", "NBSE", "NBTB", "NC", "NCLH", "NCMI",
        "NCNO", "NCPL", "NCSM", "NCTY", "NDAQ", "NDLS", "NDSN", "NE", "NEE", "NEM",
        "NEO", "NEOG", "NEON", "NEOV", "NEP", "NEPH", "NEPT", "NERV", "NES", "NETD",
        "NETE", "NEU", "NEWT", "NEX", "NEXA", "NEXT", "NFBK", "NFGC", "NFL", "NFLX",
        "NG", "NGD", "NGL", "NGNE", "NGS", "NGVC", "NHC", "NHI", "NIE", "NIM", "NIO",
        "NIPT", "NIU", "NJR", "NKE", "NKLA", "NKSH", "NKTR", "NL", "NLOK", "NLSN",
        "NLS", "NLY", "NM", "NMFC", "NMG", "NMIC", "NMM", "NMR", "NMRA", "NMRK",
        "NMS", "NMT", "NNA", "NNAG", "NNI", "NNN", "NNOX", "NODK", "NOG", "NOK",
        "NOM", "NOPH", "NOVN", "NOW", "NP", "NPACU", "NPK", "NPO", "NPTN", "NQ",
        "NQP", "NRBO", "NRC", "NRDS", "NRG", "NRGV", "NRZ", "NS", "NSA", "NSC",
        "NSIT", "NSP", "NSTG", "NSYS", "NT", "NTES", "NTNX", "NTR", "NTRA", "NTRP",
        "NTRS", "NTURA", "NTWK", "NTZ", "NU", "NUE", "NUM", "NUS", "NUSI", "NUV",
        "NVAX", "NVCR", "NVDA", "NVEC", "NVEE", "NVEI", "NVFY", "NVG", "NVGS", "NVMI",
        "NVNO", "NVO", "NVR", "NVRI", "NVS", "NVST", "NVT", "NVTA", "NVTS", "NVX",
        "NWBI", "NWE", "NWFL", "NWG", "NWGI", "NWL", "NWN", "NWS", "NWSA", "NX",
        "NXC", "NXE", "NXGN", "NXJ", "NXP", "NXPI", "NXRT", "NXST", "NXT", "NXTC",
        "NXTG", "NYT", "NZF",
        
        # O symbols
        "OABI", "OAS", "OBCI", "OBLN", "OBT", "OCAX", "OCC", "OCEA", "OCFC", "OCGN",
        "OCN", "OCRX", "OCSL", "OCUL", "OCX", "ODFL", "ODP", "ODV", "OEC", "OESX",
        "OFC", "OFED", "OFG", "OFIX", "OFLX", "OFS", "OGE", "OGI", "OGN", "OGS",
        "OHI", "OHPA", "OI", "OIG", "OII", "OIIM", "OIS", "OKE", "OKTA", "OLB",
        "OLED", "OLN", "OLLI", "OLON", "OLO", "OLP", "OM", "OMAB", "OMC", "OMCL",
        "OMED", "OMEX", "OMF", "OMGA", "OMI", "OMIC", "ON", "ONB", "ONCO", "ONCT",
        "ONCY", "ONDS", "ONEW", "ONFO", "ONON", "ONTF", "ONTO", "ONVO", "ONYX",
        "OOMA", "OPAD", "OPAL", "OPBK", "OPCH", "OPEN", "OPFI", "OPGN", "OPHC",
        "OPI", "OPIE", "OPK", "OPNT", "OPP", "OPRA", "OPRT", "OPRX", "OPT", "OPTN",
        "OPTT", "OPUS", "OR", "ORAL", "ORAN", "ORC", "ORCA", "ORCC", "ORGO", "ORGS",
        "ORI", "ORIC", "ORLY", "ORMP", "ORN", "ORRF", "OSBC", "OSCR", "OSIP", "OSK",
        "OSPN", "OSS", "OSTK", "OSUR", "OSW", "OTEX", "OTIS", "OTLK", "OTLY", "OTRK",
        "OTTR", "OVBC", "OVID", "OVLY", "OVV", "OWL", "OWLT", "OXBR", "OXFD", "OXM",
        "OXSQ", "OXY", "OZ", "OZK",
        
        # P symbols
        "PAA", "PAAS", "PAC", "PACB", "PACK", "PAG", "PAGP", "PAGS", "PAHC", "PAI",
        "PALI", "PALH", "PALT", "PAM", "PANL", "PANW", "PAR", "PARA", "PARR", "PASG",
        "PATH", "PATK", "PAVM", "PAVS", "PAX", "PAY", "PAYC", "PAYS", "PAYX", "PB",
        "PBA", "PBF", "PBFS", "PBH", "PBI", "PBIP", "PBPB", "PBR", "PCAR", "PCB",
        "PCCT", "PCG", "PCH", "PCOR", "PCRX", "PCT", "PCTI", "PCTY", "PCVX", "PCYG",
        "PD", "PDCE", "PDCO", "PDD", "PDEX", "PDFS", "PDI", "PDLB", "PDM", "PDS",
        "PDSB", "PE", "PEB", "PEBO", "PECO", "PED", "PEGA", "PENN", "PEP", "PEPG",
        "PERL", "PESI", "PET", "PETQ", "PETS", "PETZ", "PEV", "PFBC", "PFE", "PFGC",
        "PFG", "PFIE", "PFIN", "PFIS", "PFL", "PFLT", "PFMT", "PFN", "PFPT", "PFS",
        "PFSI", "PFTA", "PFX", "PG", "PGC", "PGEN", "PGP", "PGRE", "PGR", "PGTI",
        "PGY", "PGZ", "PH", "PHAR", "PHG", "PHGE", "PHI", "PHIO", "PHVS", "PHX",
        "PI", "PII", "PIII", "PIK", "PINC", "PINS", "PIPR", "PIRS", "PIXY", "PJT",
        "PK", "PKE", "PKOH", "PKW", "PL", "PLAN", "PLAO", "PLBY", "PLD", "PLG",
        "PLIN", "PLL", "PLMR", "PLNT", "PLOW", "PLPC", "PLT", "PLUG", "PLUS", "PLX",
        "PLYA", "PLYM", "PM", "PMCB", "PMD", "PMG", "PMGM", "PMT", "PMTS", "PMVP",
        "PNC", "PNFP", "PNM", "PNNT", "PNR", "PNRG", "PNW", "POAI", "POCI", "POET",
        "POLA", "POLY", "POND", "POOL", "POR", "POST", "POWI", "POWL", "POWW", "PPC",
        "PPG", "PPHI", "PPIH", "PPL", "PPSI", "PPTA", "PRAA", "PRAX", "PRCH", "PRCP",
        "PRCT", "PRDO", "PRE", "PRFT", "PRFX", "PRG", "PRGS", "PRI", "PRIM", "PRK",
        "PRLB", "PRM", "PRME", "PRO", "PROC", "PROF", "PROK", "PROP", "PROV", "PRPH",
        "PRPL", "PRQR", "PRS", "PRSO", "PRST", "PRTC", "PRTG", "PRTH", "PRTS", "PRTK",
        "PRU", "PSA", "PSB", "PSEC", "PSF", "PSG", "PSHG", "PSMT", "PSN", "PSNL",
        "PSO", "PSQH", "PSX", "PT", "PTC", "PTCT", "PTE", "PTEN", "PTF", "PTGX",
        "PTI", "PTN", "PTON", "PTR", "PTRS", "PTSI", "PTVCA", "PTY", "PUB", "PUI",
        "PUK", "PUL", "PUMP", "PUS", "PUT", "PUYI", "PVG", "PVH", "PVL", "PVBC",
        "PWP", "PWR", "PX", "PXD", "PY", "PYR", "PYPL", "PYX", "PZG", "PZN",
        
        # Q symbols
        "QABA", "QCOM", "QCRH", "QD", "QDEL", "QFIN", "QGEN", "QHC", "QIWI", "QK",
        "QLYS", "QNT", "QNST", "QOMO", "QQQE", "QQQM", "QQXT", "QRHC", "QRTEA",
        "QRTEB", "QRVO", "QS", "QSR", "QTEC", "QTNT", "QTRX", "QTT", "QUAD", "QUBT",
        "QURE", "QVCB", "QYOU",
        
        # R symbols
        "RAACU", "RACE", "RAD", "RADI", "RAIL", "RAIN", "RAKE", "RAND", "RANI", "RAPT",
        "RARE", "RAVE", "RAVN", "RBA", "RBB", "RBBN", "RBC", "RBCAA", "RBL", "RBLX",
        "RBPAA", "RC", "RCA", "RCFA", "RCG", "RCI", "RCII", "RCKT", "RCKY", "RCL",
        "RCMT", "RCP", "RCS", "RDI", "RDIB", "RDN", "RDNT", "RDS", "RDUS", "RDVT",
        "RDWR", "RE", "REAL", "REAX", "REBN", "REC", "RECI", "RECN", "REDU", "REE",
        "REEF", "REG", "REGN", "REI", "REKR", "RELL", "RELX", "RELY", "REM", "RENE",
        "RENB", "RENT", "REP", "REPL", "RES", "RETO", "REV", "REVB", "REVG", "REX",
        "REXR", "REYN", "RFI", "RFL", "RFMZ", "RFP", "RGA", "RGEN", "RGF", "RGLD",
        "RGNX", "RGP", "RGR", "RGS", "RGT", "RH", "RHI", "RHP", "RIBT", "RICK",
        "RIG", "RIGL", "RILY", "RIO", "RIOT", "RIV", "RIVN", "RJF", "RKDA", "RKT",
        "RL", "RLI", "RLJ", "RLMD", "RLX", "RLY", "RM", "RMAX", "RMD", "RMG", "RMI",
        "RMO", "RMP", "RMR", "RMT", "RNA", "RNG", "RNR", "ROAD", "ROBO", "ROCK",
        "ROG", "ROIC", "ROK", "ROKU", "ROL", "ROLL", "RONI", "ROP", "ROSS", "ROVR",
        "RPAI", "RPD", "RPHM", "RPM", "RPRX", "RPTX", "RQI", "RRC", "RRD", "RRR",
        "RS", "RSG", "RSI", "RSLS", "RSVR", "RT", "RTC", "RTH", "RTL", "RTLR",
        "RTRX", "RTX", "RUM", "RUN", "RUP", "RUSHA", "RUSHB", "RUTH", "RVP", "RVPH",
        "RWLK", "RWT", "RXI", "RXMD", "RXRX", "RXT", "RY", "RYAAY", "RYAM", "RYDE",
        "RYTM", "RZG",
        
        # S symbols
        "SA", "SABA", "SABR", "SACH", "SAFE", "SAFT", "SAGE", "SAH", "SAIA", "SAIL",
        "SAL", "SALM", "SAM", "SAMG", "SAN", "SANM", "SAP", "SAR", "SASR", "SATS",
        "SAVE", "SB", "SBAC", "SBBP", "SBCF", "SBFG", "SBGI", "SBH", "SBIG", "SBLK",
        "SBNY", "SBOW", "SBR", "SBRA", "SBS", "SBSI", "SBT", "SBUX", "SC", "SCAQ",
        "SCAR", "SCCO", "SCD", "SCHL", "SCHN", "SCI", "SCKT", "SCL", "SCLX", "SCM",
        "SCON", "SCOR", "SCPH", "SCRM", "SCS", "SCSC", "SCTL", "SCVL", "SCWX", "SCYX",
        "SD", "SDC", "SDIG", "SDPI", "SE", "SEAC", "SEAS", "SEB", "SECR", "SEE",
        "SEEL", "SEER", "SEIC", "SELB", "SEM", "SEMR", "SENS", "SERV", "SES", "SEVN",
        "SF", "SFBC", "SFBS", "SFE", "SFIX", "SFL", "SFM", "SFNC", "SFST", "SG",
        "SGA", "SGBX", "SGC", "SGE", "SGH", "SGHT", "SGMA", "SGMO", "SGMT", "SGRP",
        "SGRY", "SGU", "SHAK", "SHEN", "SHFS", "SHG", "SHIP", "SHLS", "SHOO", "SHOP",
        "SHPH", "SHUA", "SHW", "SHYF", "SIBN", "SIC", "SIEN", "SIGA", "SIGI", "SIG",
        "SILC", "SILK", "SILO", "SILV", "SIM", "SIMO", "SINT", "SIX", "SJI", "SJM",
        "SJR", "SJT", "SJIJ", "SKGR", "SKM", "SKT", "SKUL", "SKYE", "SKYW", "SKYH",
        "SLAB", "SLB", "SLCA", "SLCR", "SLD", "SLDP", "SLE", "SLF", "SLG", "SLGN",
        "SLI", "SLIM", "SLM", "SLNG", "SLP", "SLRC", "SLS", "SLVM", "SM", "SMAP",
        "SMBC", "SMCI", "SMG", "SMH", "SMHI", "SMLR", "SMN", "SMMF", "SMP", "SMPL",
        "SMR", "SMS", "SMSI", "SMTC", "SMTS", "SMWB", "SNBR", "SNC", "SNCE", "SNDL",
        "SNDX", "SNEX", "SNFCA", "SNN", "SNOW", "SNP", "SNPO", "SNPS", "SNR", "SNSE",
        "SNV", "SNX", "SNY", "SO", "SOAR", "SOBR", "SOFI", "SOHU", "SOI", "SOJC",
        "SOJD", "SOL", "SOLO", "SOLN", "SOLOW", "SOME", "SONG", "SONM", "SONN", "SONO",
        "SOPA", "SOR", "SOS", "SOUN", "SOWG", "SPAQ", "SPAR", "SPB", "SPCB", "SPCE",
        "SPFI", "SPG", "SPGI", "SPH", "SPHR", "SPI", "SPIR", "SPK", "SPLK", "SPNE",
        "SPNS", "SPOK", "SPOT", "SPR", "SPRC", "SPRO", "SPRT", "SPSC", "SPTK", "SPTN",
        "SPX", "SPXC", "SPXX", "SPXZ", "SQ", "SQFT", "SQLZ", "SQSP", "SR", "SRAD",
        "SRCE", "SRCL", "SRDX", "SRE", "SRF", "SRG", "SRI", "SRL", "SRM", "SRNE",
        "SRPT", "SRRK", "SRS", "SRTS", "SRV", "SSB", "SSD", "SSL", "SSNC", "SSP",
        "SSTK", "SSYS", "ST", "STAA", "STAF", "STAG", "STAR", "STBA", "STC", "STCN",
        "STE", "STEM", "STEP", "STER", "STFC", "STGW", "STI", "STIM", "STKL", "STL",
        "STLD", "STM", "STMP", "STNE", "STNG", "STNL", "STOK", "STON", "STOR", "STOS",
        "STR", "STRA", "STRC", "STRL", "STRM", "STRN", "STRO", "STS", "STSA", "STSS",
        "STTK", "STT", "STX", "STXS", "STZA", "STZB", "SU", "SUM", "SUN", "SUNS",
        "SUP", "SUPN", "SUPR", "SURF", "SUSB", "SUZ", "SVBI", "SVC", "SVFD", "SVII",
        "SVM", "SVN", "SVRA", "SWAG", "SWAN", "SWAV", "SWI", "SWIM", "SWIR", "SWKS",
        "SWN", "SWSS", "SWX", "SXC", "SXI", "SXT", "SY", "SYBT", "SYBX", "SYF",
        "SYK", "SYKE", "SYM", "SYNA", "SYNC", "SYNH", "SYNL", "SYPR", "SYRA", "SYY",
        
        # T symbols
        "TA", "TAC", "TACO", "TACT", "TAK", "TAL", "TALK", "TALO", "TAMP", "TAN",
        "TANQ", "TAP", "TAPR", "TARA", "TARS", "TASK", "TAST", "TAT", "TAYD", "TBB",
        "TBC", "TBI", "TBIO", "TBK", "TBLA", "TBLT", "TBNK", "TBP", "TC", "TCBI",
        "TCBK", "TCDA", "TCF", "TCG", "TCI", "TCMD", "TCO", "TCOM", "TCP", "TCPC",
        "TCRT", "TCRX", "TCS", "TCW", "TCX", "TD", "TDC", "TDF", "TDG", "TDI",
        "TDOC", "TDS", "TDW", "TDY", "TEAF", "TEAM", "TECH", "TECK", "TEF", "TELA",
        "TELL", "TEMD", "TEN", "TENB", "TEO", "TER", "TERN", "TESI", "TETE", "TEVA",
        "TEX", "TFC", "TFG", "TFII", "TFIN", "TFX", "TG", "TGLS", "TGNA", "TGS",
        "TGT", "TGTX", "TH", "THC", "THCA", "THCP", "THG", "THMO", "THO", "THRD",
        "THRM", "THRY", "THS", "TIBI", "TIC", "TIGR", "TIL", "TILE", "TIMB", "TIO",
        "TIPT", "TITN", "TIXT", "TJX", "TKC", "TKR", "TLC", "TLGY", "TLI", "TLIS",
        "TLK", "TLRY", "TLS", "TLSA", "TLT", "TLYS", "TM", "TMC", "TME", "TMHC",
        "TMUS", "TMX", "TNC", "TNDM", "TNET", "TNK", "TNL", "TNP", "TNXP", "TOI",
        "TOMZ", "TORC", "TOUR", "TOWN", "TPB", "TPC", "TPG", "TPHS", "TPIC", "TPR",
        "TPTX", "TPX", "TPZ", "TQQQ", "TR", "TRC", "TRDA", "TRE", "TREE", "TREX",
        "TRGP", "TRI", "TRIB", "TRIN", "TRIP", "TRN", "TRNO", "TRMB", "TRMD", "TRMK",
        "TRMT", "TRN", "TRNS", "TRON", "TROX", "TRP", "TRQ", "TRR", "TRS", "TRST",
        "TRT", "TRTN", "TRTX", "TRU", "TRUE", "TRUP", "TRV", "TRVG", "TRVI", "TRX",
        "TS", "TSCO", "TSE", "TSG", "TSI", "TSL", "TSLA", "TSM", "TSN", "TSP",
        "TSQ", "TTCF", "TTD", "TTE", "TTEC", "TTEK", "TTGT", "TTI", "TTMI", "TTO",
        "TTOO", "TTS", "TTSH", "TTW", "TU", "TUG", "TUI", "TUP", "TUR", "TURN",
        "TUSK", "TV", "TVTX", "TVTY", "TW", "TWKS", "TWLO", "TWOU", "TWST", "TWTR",
        "TXG", "TXMD", "TXN", "TXRH", "TYG", "TYRA", "TZG", "TZOO",
        
        # U symbols
        "UAMY", "UAVS", "UBA", "UBER", "UBFO", "UBOH", "UBP", "UBS", "UBX", "UCBI",
        "UCTT", "UDF", "UE", "UEIC", "UEPS", "UFAB", "UFC", "UFI", "UFPI", "UFPT",
        "UG", "UGI", "UGP", "UHS", "UHT", "UI", "UIS", "UIVM", "UL", "ULBI",
        "ULCC", "ULH", "ULLI", "ULTA", "ULTI", "UMC", "UMH", "UMRX", "UNB", "UNBJ",
        "UNF", "UNFI", "UNH", "UNIT", "UNM", "UNP", "UNTY", "UPC", "UPH", "UPLD",
        "UPS", "UPST", "UPWK", "UPXI", "URBN", "URG", "URI", "UROY", "USA", "USAC",
        "USAS", "USAU", "USB", "USCB", "USDP", "USEG", "USER", "USFD", "USM", "USNA",
        "USPH", "USTB", "UTF", "UTG", "UTHR", "UTI", "UTL", "UTZ", "UUU", "UVE",
        "UVSP", "UVV", "UWM", "UWMC", "UZD",
        
        # V symbols
        "VAC", "VAL", "VALE", "VALU", "VAR", "VAXX", "VBF", "VCV", "VECO", "VEDU",
        "VEEV", "VEL", "VEON", "VER", "VERA", "VERB", "VERI", "VERO", "VERT", "VET",
        "VFC", "VFF", "VG", "VGI", "VGR", "VHI", "VIA", "VIAB", "VIE", "VII",
        "VIK", "VIM", "VIN", "VINC", "VIO", "VIP", "VIR", "VIRT", "VIS", "VITL",
        "VIVE", "VIZ", "VKTX", "VLN", "VLO", "VLRS", "VLT", "VLY", "VLYPO", "VMAR",
        "VMC", "VMD", "VMI", "VMLP", "VMO", "VMW", "VNCE", "VND", "VNE", "VNO",
        "VNOM", "VOC", "VOD", "VOR", "VOXX", "VPG", "VRA", "VRAY", "VRE", "VREX",
        "VRM", "VRME", "VRN", "VRNS", "VRNT", "VRRM", "VRS", "VRSK", "VRSN", "VRT",
        "VRTS", "VRTV", "VRTX", "VSAT", "VSE", "VSEC", "VSH", "VST", "VSTA", "VSTM",
        "VTC", "VTG", "VTGN", "VTN", "VTR", "VTRS", "VTS", "VTV", "VTY", "VUG",
        "VUZI", "VVI", "VVR", "VVV", "VVX", "VWE", "VXF", "VXX", "VXRT", "VYM", "VZ",
        
        # W symbols
        "WAAS", "WAB", "WABC", "WAFD", "WAFU", "WAL", "WASH", "WAT", "WATT", "WAVE",
        "WB", "WBA", "WBD", "WBS", "WBT", "WCC", "WCLD", "WCN", "WD", "WDAY", "WDC",
        "WDFC", "WDH", "WDI", "WDS", "WE", "WEA", "WEC", "WELL", "WEN", "WERN",
        "WES", "WEST", "WEX", "WF", "WFC", "WFG", "WFH", "WFRD", "WGO", "WH",
        "WHD", "WHF", "WHG", "WHLM", "WHLR", "WHR", "WIA", "WIB", "WIC", "WIL",
        "WILC", "WIN", "WING", "WINS", "WIX", "WJBK", "WK", "WKHS", "WLB", "WLFC",
        "WLK", "WLL", "WLMS", "WLY", "WM", "WMB", "WMC", "WMG", "WMK", "WMT",
        "WNC", "WNS", "WNW", "WOLF", "WOOF", "WOR", "WORK", "WRAP", "WRB", "WRBK",
        "WRE", "WRK", "WRLD", "WSBC", "WSC", "WSFS", "WSM", "WSO", "WSR", "WST",
        "WSTG", "WSTL", "WT", "WTI", "WTM", "WTO", "WTS", "WTW", "WU", "WULF",
        "WVE", "WVFC", "WW", "WWE", "WWR", "WWW", "WY", "WYNN", "WYY",
        
        # X symbols  
        "X", "XAN", "XBIO", "XBIT", "XCUR", "XEL", "XELA", "XELB", "XENE", "XGN",
        "XHG", "XHR", "XI", "XIN", "XLO", "XM", "XMTR", "XNCR", "XOG", "XOM",
        "XOMA", "XOS", "XP", "XPDI", "XPER", "XPEV", "XPL", "XPO", "XPRO", "XRAY",
        "XRX", "XSPA", "XT", "XTLB", "XTSLA", "XYF", "XYL",
        
        # Y symbols
        "YAL", "YCBD", "YCS", "YEL", "YELL", "YEXT", "YGF", "YINN", "YJ", "YMAB",
        "YMM", "YPF", "YQ", "YRD", "YS", "YTEN", "YUM", "YY",
        
        # Z symbols
        "ZAGG", "ZBH", "ZBRA", "ZD", "ZEN", "ZEUS", "ZEV", "ZFGN", "ZG", "ZGEN",
        "ZGN", "ZH", "ZI", "ZIM", "ZIMV", "ZION", "ZIP", "ZIV", "ZIVO", "ZJYL",
        "ZKIN", "ZM", "ZMATF", "ZMTP", "ZN", "ZNGA", "ZNH", "ZOMD", "ZOM", "ZOOM",
        "ZS", "ZSAN", "ZTO", "ZTR", "ZUMZ", "ZUO", "ZURN", "ZVO", "ZY", "ZYNE",
        "ZYXI"
    ]
    
    # Add more programmatically generated symbols
    alphabet_ranges = {
        'D': range(ord('D'), ord('E')),
        'E': range(ord('E'), ord('F')), 
        'F': range(ord('F'), ord('G')),
        'G': range(ord('G'), ord('H')),
        'H': range(ord('H'), ord('I')),
        'I': range(ord('I'), ord('J')),
        'J': range(ord('J'), ord('K')),
        'K': range(ord('K'), ord('L')),
        'L': range(ord('L'), ord('M')),
        'M': range(ord('M'), ord('N')),
        'N': range(ord('N'), ord('O')),
        'O': range(ord('O'), ord('P')),
        'P': range(ord('P'), ord('Q')),
        'Q': range(ord('Q'), ord('R')),
        'R': range(ord('R'), ord('S')),
        'S': range(ord('S'), ord('T')),
        'T': range(ord('T'), ord('U')),
        'U': range(ord('U'), ord('V')),
        'V': range(ord('V'), ord('W')),
        'W': range(ord('W'), ord('X')),
        'X': range(ord('X'), ord('Y')),
        'Y': range(ord('Y'), ord('Z')),
        'Z': range(ord('Z'), ord('Z')+1),
    }
    
    # Generate additional common patterns
    additional_symbols = []
    for letter in 'DEFGHIJKLMNOPQRSTUVWXYZ':
        # Add common patterns
        for suffix in ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'X', 'Y', 'Z']:
            for num in ['', '1', '2', '3']:
                symbol = letter + suffix + num
                if len(symbol) <= 5 and symbol not in base_symbols:
                    additional_symbols.append(symbol)
                    
    # Add tech/biotech/crypto patterns
    tech_patterns = []
    for prefix in ['DA', 'DE', 'DI', 'DO', 'EA', 'EB', 'EC', 'ED', 'EF', 'EG', 'EH', 'EL', 'EM', 'EN', 'EP', 'ER', 'ES', 'ET', 'EV', 'EW', 'EX', 'EY', 'EZ']:
        for suffix in ['T', 'TA', 'TI', 'TO', 'X', 'XA', 'XI', 'XO']:
            symbol = prefix + suffix
            if len(symbol) <= 5 and symbol not in base_symbols:
                tech_patterns.append(symbol)
    
    all_symbols = list(set(base_symbols + additional_symbols[:200] + tech_patterns[:100]))
    
    print(f"🔥 BRUTE FORCE TARGET: {len(all_symbols)} D-Z symbols")
    print("📊 NO FILTERING - Testing everything that moves!")
    print()
    
    # Show sample symbols
    sample = all_symbols[:30]
    print("Sample symbols:")
    for i in range(0, len(sample), 6):
        print("   " + ", ".join(sample[i:i+6]))
    print(f"   ... and {len(all_symbols)-30} more symbols")
    print()
    
    return all_symbols


def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['BB_std'] = df['Close'].rolling(window=window).std()
    df['BB_upper'] = df['MA20'] + (df['BB_std'] * num_std)
    df['BB_lower'] = df['MA20'] - (df['BB_std'] * num_std)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']) - 0.5
    return df


def calculate_rsi(df, window=14):
    """Calculate RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_volume_surge(df, window=20):
    """Calculate volume surge"""
    df['Volume_MA'] = df['Volume'].rolling(window=window).mean()
    df['Volume_surge'] = df['Volume'] / df['Volume_MA']
    return df


def get_aegs_signals(df):
    """Generate AEGS entry signals with scoring"""
    signals = []
    
    if len(df) < 50:  # Need enough data for indicators
        return signals
    
    for i in range(50, len(df)):
        current = df.iloc[i]
        prev_day = df.iloc[i-1]
        
        # AEGS Entry Criteria with scoring
        score = 0
        reasons = []
        
        # 1. RSI Oversold (RSI < 30) - 25 points
        if current['RSI'] < 30:
            score += 25
            reasons.append(f"RSI oversold ({current['RSI']:.1f})")
            
        # 2. Bollinger Band Position (< -0.4 = below lower band) - 25 points  
        if current['BB_position'] < -0.4:
            score += 25
            reasons.append(f"BB breakdown ({current['BB_position']:.2f})")
            
        # 3. Volume Surge (> 2x average) - 20 points
        if current['Volume_surge'] > 2.0:
            score += 20
            reasons.append(f"Volume surge ({current['Volume_surge']:.1f}x)")
            
        # 4. Daily Price Drop (> 5%) - 15 points
        daily_change = (current['Close'] - prev_day['Close']) / prev_day['Close']
        if daily_change < -0.05:
            score += 15
            reasons.append(f"Price drop ({daily_change*100:.1f}%)")
            
        # 5. Extended Oversold (RSI < 25) - 10 bonus points
        if current['RSI'] < 25:
            score += 10
            reasons.append(f"Extreme oversold ({current['RSI']:.1f})")
            
        # 6. Volume Explosion (> 5x) - 10 bonus points
        if current['Volume_surge'] > 5.0:
            score += 10
            reasons.append(f"Volume explosion ({current['Volume_surge']:.1f}x)")
            
        # 7. Massive Daily Drop (> 10%) - 5 bonus points
        if daily_change < -0.10:
            score += 5
            reasons.append(f"Major selloff ({daily_change*100:.1f}%)")
        
        # Entry Threshold: Score >= 70
        if score >= 70:
            signals.append({
                'date': current.name,
                'price': current['Close'],
                'score': score,
                'reasons': reasons,
                'rsi': current['RSI'],
                'bb_position': current['BB_position'],
                'volume_surge': current['Volume_surge'],
                'daily_change': daily_change
            })
    
    return signals


def backtest_aegs_strategy(symbol, start_date='2025-01-01', end_date='2025-12-01'):
    """Backtest AEGS strategy for a symbol"""
    
    try:
        # Download data with error handling
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if len(df) < 50:
            return None
            
        # Calculate current price
        try:
            current_info = stock.info
            current_price = current_info.get('currentPrice', df['Close'].iloc[-1])
        except:
            current_price = df['Close'].iloc[-1]
            
        # Price filter: $1+ stocks only
        if current_price < 1.0:
            return {
                'symbol': symbol,
                'current_price': current_price,
                'filter_reason': 'below_price_threshold',
                'total_trades': 0
            }
        
        # Calculate indicators
        df = calculate_bollinger_bands(df)
        df = calculate_rsi(df)  
        df = calculate_volume_surge(df)
        
        # Get entry signals
        signals = get_aegs_signals(df)
        
        if not signals:
            return None
            
        # Backtest trades
        trades = []
        current_position = None
        
        for signal in signals:
            if current_position is None:  # Enter trade
                current_position = {
                    'entry_date': signal['date'],
                    'entry_price': signal['price'],
                    'entry_score': signal['score'],
                    'entry_reasons': signal['reasons']
                }
            
            # Check exit conditions for open position
            if current_position:
                entry_idx = df.index.get_loc(current_position['entry_date'])
                
                for exit_idx in range(entry_idx + 1, len(df)):
                    exit_date = df.index[exit_idx]
                    exit_price = df['Close'].iloc[exit_idx]
                    days_held = (exit_date - current_position['entry_date']).days
                    
                    current_return = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    
                    exit_reason = None
                    
                    # Exit Rules (in priority order)
                    if current_return >= 0.30:  # 30% profit target
                        exit_reason = "Profit Target 30%"
                    elif current_return <= -0.20:  # 20% stop loss
                        exit_reason = "Stop Loss 20%"
                    elif days_held >= 60:  # Force exit after 60 days
                        exit_reason = "Force Exit"
                    elif days_held >= 30 and current_return > 0:  # Time-based profitable exit
                        exit_reason = "Time Exit (Profitable)"
                    
                    if exit_reason:
                        trades.append({
                            'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                            'exit_date': exit_date.strftime('%Y-%m-%d'), 
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'return_pct': current_return * 100,
                            'days_held': days_held,
                            'exit_reason': exit_reason
                        })
                        current_position = None
                        break
        
        if not trades:
            return None
            
        # Calculate performance metrics
        returns = [trade['return_pct'] for trade in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Calculate strategy return (compound returns)
        strategy_return = 1.0
        for ret in returns:
            strategy_return *= (1 + ret/100)
        strategy_return = (strategy_return - 1) * 100
        
        # Calculate buy & hold return
        buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        
        # Calculate excess return
        excess_return = strategy_return - buy_hold_return
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': excess_return,
            'trades': trades
        }
        
    except Exception as e:
        return None


def process_symbol_batch(symbols_batch, batch_id, total_symbols, shared_non_qualifying=None):
    """Process a batch of symbols"""
    results = []
    batch_non_qualifying = {
        'no_data': [],
        'below_price_threshold': [],
        'insufficient_volume': [],
        'no_aegs_signals': [],
        'unprofitable': []
    }
    
    for i, symbol in enumerate(symbols_batch):
        try:
            result = backtest_aegs_strategy(symbol)
            
            if result and result['total_trades'] > 0:
                # Status indicators
                strategy_return = result['strategy_return']
                excess_return = result['excess_return']
                
                if strategy_return > 0:
                    # Profitable - add to results
                    if strategy_return > 50:
                        status = "🚀"
                    elif strategy_return > 30:
                        status = "🔥"
                    else:
                        status = "✅"
                    
                    print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] {status} {symbol:6} : {result['total_trades']:2d} trades, {result['win_rate']:4.0f}% win, Strategy: {result['strategy_return']:+6.1f}%, Excess: {result['excess_return']:+6.1f}%")
                    results.append(result)
                else:
                    # Unprofitable strategy - track for re-analysis
                    batch_non_qualifying['unprofitable'].append({
                        'symbol': symbol,
                        'strategy_return': strategy_return,
                        'excess_return': excess_return,
                        'total_trades': result['total_trades'],
                        'win_rate': result['win_rate']
                    })
                    
                    if excess_return > 50:
                        status = "💎"  # Good relative performance
                    elif excess_return > 0:
                        status = "💎"  # Positive excess return
                    else:
                        status = "❌"
                    
                    print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] {status} {symbol:6} : {result['total_trades']:2d} trades, {result['win_rate']:4.0f}% win, Strategy: {result['strategy_return']:+6.1f}%, Excess: {result['excess_return']:+6.1f}%")
                    
            elif result and result['total_trades'] == 0:
                # Check if it's below price threshold or no AEGS signals
                if result.get('filter_reason') == 'below_price_threshold':
                    batch_non_qualifying['below_price_threshold'].append({
                        'symbol': symbol,
                        'reason': 'Below $1.00 price threshold',
                        'current_price': result.get('current_price', 0)
                    })
                    print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] ❌ {symbol:6} : Below $1.00 (${result.get('current_price', 0):.2f})")
                else:
                    # No AEGS signals generated
                    batch_non_qualifying['no_aegs_signals'].append({
                        'symbol': symbol,
                        'reason': 'No AEGS entry signals generated',
                        'current_price': result.get('current_price', 0)
                    })
                    print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] ❌ {symbol:6} : 0 trades (No AEGS signals)")
                
            else:
                # No data available
                batch_non_qualifying['no_data'].append({
                    'symbol': symbol,
                    'reason': 'No price data available or data error'
                })
                print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] 💀 {symbol:6} : NO DATA")
                
        except Exception as e:
            # Error processing symbol
            batch_non_qualifying['no_data'].append({
                'symbol': symbol,
                'reason': f'Processing error: {str(e)}'
            })
            print(f"[{(batch_id-1)*len(symbols_batch)+i+1:3d}/{total_symbols}] 💀 {symbol:6} : ERROR - {str(e)}")
            
    return results, batch_non_qualifying


def main():
    print("🔥 INITIATING BRUTE FORCE ATTACK ON D-Z SYMBOLS...")
    print()
    
    # Get all D-Z symbols
    all_symbols = get_nasdaq_dz_symbols()
    
    print(f"🔥💎 BRUTE FORCE AEGS BACKTEST - ALL {len(all_symbols)} D-Z SYMBOLS 💎🔥")
    print("=" * 80)
    print("🚫 NO FILTERING | 🚫 NO SCREENING | 🚫 NO MERCY")
    print("=" * 80)
    
    # Process in parallel batches
    batch_size = 20
    batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
    
    all_results = []
    non_qualifying_symbols = {
        'no_data': [],
        'below_price_threshold': [],
        'insufficient_volume': [],
        'no_aegs_signals': [],
        'unprofitable': []
    }
    
    # Process batches with ThreadPoolExecutor for faster execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_batch = {}
        
        for batch_id, batch in enumerate(batches, 1):
            future = executor.submit(process_symbol_batch, batch, batch_id, len(all_symbols), non_qualifying_symbols)
            future_to_batch[future] = batch_id
        
        for future in as_completed(future_to_batch):
            batch_results, batch_non_qualifying = future.result()
            all_results.extend(batch_results)
            
            # Merge non-qualifying symbols
            for category, symbols in batch_non_qualifying.items():
                non_qualifying_symbols[category].extend(symbols)
    
    # Sort results by strategy return
    all_results.sort(key=lambda x: x.get('strategy_return', -999), reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"nasdaq_dz_brute_force_aegs_results_{timestamp}.json"
    non_qualifying_file = f"nasdaq_dz_non_qualifying_symbols_{timestamp}.json"
    
    output = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'BRUTE FORCE - D through Z symbols',
        'criteria': 'All NASDAQ D-Z symbols, $1+ price filter only',
        'total_analyzed': len([r for r in all_results if r is not None]),
        'total_symbols_tested': len(all_symbols),
        'top_performers': all_results
    }
    
    # Save profitable results
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save non-qualifying symbols for future re-analysis
    non_qualifying_output = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'BRUTE FORCE - D through Z symbols',
        'purpose': 'Track non-qualifying symbols for monthly re-analysis',
        'next_reanalysis_due': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        'categories': {
            'no_data': {
                'count': len(non_qualifying_symbols['no_data']),
                'description': 'Symbols with no price data available',
                'symbols': sorted(non_qualifying_symbols['no_data'])
            },
            'below_price_threshold': {
                'count': len(non_qualifying_symbols['below_price_threshold']),
                'description': 'Symbols below $1.00 price threshold',
                'symbols': sorted(non_qualifying_symbols['below_price_threshold'])
            },
            'insufficient_volume': {
                'count': len(non_qualifying_symbols['insufficient_volume']),
                'description': 'Symbols with insufficient trading history',
                'symbols': sorted(non_qualifying_symbols['insufficient_volume'])
            },
            'no_aegs_signals': {
                'count': len(non_qualifying_symbols['no_aegs_signals']),
                'description': 'Symbols with no AEGS entry signals generated',
                'symbols': sorted(non_qualifying_symbols['no_aegs_signals'])
            },
            'unprofitable': {
                'count': len(non_qualifying_symbols['unprofitable']),
                'description': 'Symbols with negative strategy returns',
                'symbols': sorted(non_qualifying_symbols['unprofitable'])
            }
        },
        'summary': {
            'total_non_qualifying': sum(len(symbols) for symbols in non_qualifying_symbols.values()),
            'largest_category': max(non_qualifying_symbols.keys(), key=lambda k: len(non_qualifying_symbols[k])),
            'reanalysis_instructions': 'Run this same script in 30 days to check if market conditions have made any of these symbols profitable'
        }
    }
    
    with open(non_qualifying_file, 'w') as f:
        json.dump(non_qualifying_output, f, indent=2)
    
    print()
    print("=" * 80)
    print(f"🎯 BRUTE FORCE COMPLETE! Found {len(all_results)} profitable symbols")
    print(f"💾 Profitable results saved to: {results_file}")
    print(f"📋 Non-qualifying symbols saved to: {non_qualifying_file}")
    print("=" * 80)
    
    # Show breakdown of non-qualifying symbols
    print(f"\n📊 NON-QUALIFYING SYMBOL BREAKDOWN:")
    for category, symbols in non_qualifying_symbols.items():
        if symbols:
            print(f"   {category.replace('_', ' ').title()}: {len(symbols)} symbols")
    
    total_non_qualifying = sum(len(symbols) for symbols in non_qualifying_symbols.values())
    print(f"   📈 Total tracked for re-analysis: {total_non_qualifying} symbols")
    print(f"   🗓️  Re-run scheduled for: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}")
    
    # Show top 15 profitable results
    profitable = [r for r in all_results if r['strategy_return'] > 0]
    print(f"\n🔥💎 TOP PROFITABLE D-Z PERFORMERS: {len(profitable)} found 💎🔥")
    
    for i, result in enumerate(profitable[:15], 1):
        status = "🚀" if result['strategy_return'] > 30 else "🔥" if result['strategy_return'] > 20 else "✅"
        print(f"{i:2d}. {status} {result['symbol']:6} | +{result['strategy_return']:5.1f}% | {result['win_rate']:5.1f}% win | {result['total_trades']} trades | Excess: {result['excess_return']:+6.1f}%")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏰ Total execution time: {end_time - start_time:.1f} seconds")