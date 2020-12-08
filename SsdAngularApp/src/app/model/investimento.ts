/**
 * @example
 * {
 *   "horizon"       : 24,
 *   "S&P_500_INDEX" : 0.2,
 *   "FTSE_MIB_INDEX": 0.15,
 *   "GOLD_SPOT_$_OZ": 0.1,
 *   "MSCI_EM"       : 0.1,
 *   "MSCI_EURO"     : 0.2,
 *   "All_Bonds_TR"  : 0.15,
 *   "U.S._Treasury" : 0.1
 * }
 */
export interface Investimento {
  horizon: number;
  'S&P_500_INDEX': number;
  'FTSE_MIB_INDEX': number;
  'GOLD_SPOT_$_OZ': number;
  'MSCI_EM': number;
  'MSCI_EURO': number;
  'All_Bonds_TR': number;
  'U.S._Treasury': number;
}

export function createInvestimento(
  horizon: number,
  SP_500_INDEX: number,
  FTSE_MIB_INDEX: number,
  GOLD_SPOT_$_OZ: number,
  MSCI_EM: number,
  MSCI_EURO: number,
  All_Bonds_TR: number, // tslint:disable-line
  US_Treasury: number, // tslint:disable-line
): Investimento {
  return {
    horizon,
    'S&P_500_INDEX': SP_500_INDEX,
    FTSE_MIB_INDEX,
    GOLD_SPOT_$_OZ,
    MSCI_EM,
    MSCI_EURO,
    All_Bonds_TR,
    'U.S._Treasury': US_Treasury,
  };
}
