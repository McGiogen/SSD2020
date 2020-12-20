namespace SsdWebApi.Models
{
  public class ForecastResult
  {
    public string text { get; set; }
    public string[] img { get; set; }
    public double[] pctChanges { get; set; }
    public double revenue { get; set; }
    public double revenuePerc { get; set; }
    public double mape { get; set; }
    public double valueAtRisk { get; set; }
  }
}
