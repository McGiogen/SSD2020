using System.Text.Json.Serialization;

namespace SsdWebApi.Models
{
  public class PortafoglioResult
  {
    public Portfolio portfolio { get; set; }
    public string[] text { get; set; }
    public string[][] img { get; set; }
  }

  /**
    Esempio JSON:
    {
      "horizon"       : 24,
      "S&P_500_INDEX" : 0.2,
      "FTSE_MIB_INDEX": 0.15,
      "GOLD_SPOT_$_OZ": 0.1,
      "MSCI_EM"       : 0.1,
      "MSCI_EURO"     : 0.2,
      "All_Bonds_TR"  : 0.15,
      "U.S._Treasury" : 0.1
    }
  */
  public class Portfolio
  {
    public int horizon { get; set; }
    [JsonPropertyName("S&P_500_INDEX")]
    public double SP_500 { get; set; }
    [JsonPropertyName("FTSE_MIB_INDEX")]
    public double FTSE_MIB { get; set; }
    [JsonPropertyName("GOLD_SPOT_$_OZ")]
    public double GOLD_SPOT { get; set; }
    [JsonPropertyName("MSCI_EM")]
    public double MSCI_EM { get; set; }
    [JsonPropertyName("MSCI_EURO")]
    public double MSCI_EURO { get; set; }
    [JsonPropertyName("All_Bonds_TR")]
    public double All_Bonds { get; set; }
    [JsonPropertyName("U.S._Treasury")]
    public double US_Treasury { get; set; }
  }
}
