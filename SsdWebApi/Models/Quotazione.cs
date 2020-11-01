using System.Text.Json.Serialization ;

namespace SsdWebApi
{
  public class Quotazione
  {
    public int id { get; set; }
    public string Data { get; set; }
    [JsonPropertyName("SP_500")]
    public double SP_500 { get; set; }
    [JsonPropertyName("FTSE_MIB")]
    public double FTSE_MIB { get; set; }
    [JsonPropertyName("GOLD_SPOT")]
    public double GOLD_SPOT { get; set; }
    [JsonPropertyName("MSCI_EM")]
    public double MSCI_EM { get; set; }
    [JsonPropertyName("MSCI_EURO")]
    public double MSCI_EURO { get; set; }
    [JsonPropertyName("All_Bonds")]
    public double All_Bonds { get; set; }
    [JsonPropertyName("US_Treasury")]
    public double US_Treasury { get; set; }
  }
}
