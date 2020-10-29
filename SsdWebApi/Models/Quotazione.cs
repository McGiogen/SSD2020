using System;

namespace SsdWebApi
{
  public class Quotazione
  {
    public int id { get; set; }
    public int Data { get; set; }
    public string SP_500 { get; set; }
    public string FTSE_MIB { get; set; }
    public string GOLD_SPOT { get; set; }
    public string MSCI_EM { get; set; }
    public string MSCI_EURO { get; set; }
    public string All_Bonds { get; set; }
    public string US_Treasury { get; set; }
  }
}
