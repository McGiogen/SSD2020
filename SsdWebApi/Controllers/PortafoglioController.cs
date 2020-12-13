using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using SsdWebApi.Models;
using SsdWebApi.Services;

// Documentazione https://docs.microsoft.com/it-it/aspnet/core/data/ef-mvc/crud?view=aspnetcore-3.1

namespace SsdWebApi.Controllers
{
  [ApiController]
  [Route("api/[controller]")]
  public class PortafoglioController : ControllerBase
  {
    private readonly QuotazioneContext _context;

    private readonly Persistence persistence;

    public PortafoglioController(QuotazioneContext context)
    {
      _context = context;
      persistence = new Persistence(context);
    }

    [HttpGet]
    public async Task<ActionResult<PortafoglioResult>> GetAll([FromQuery(Name = "type")] string type)
    {
      string[] indices = new string[] { "SP_500", "FTSE_MIB", "GOLD_SPOT", "MSCI_EM", "MSCI_EURO", "All_Bonds", "US_Treasury" };

      // Forecast
      Forecast forecast = new Forecast();
      double[] indexesRevenue = new double[indices.Length];
      double[] indexesRisk = new double[indices.Length];
      List<string> text = new List<string>();
      List<string[]> img = new List<string[]>();
      for (int i = 0; i < indices.Length; i++) {
        ForecastResult forecastResult = forecast.forecastIndex(indices[i], type);
        indexesRevenue[i] = forecastResult.revenue;
        indexesRisk[i] = forecastResult.risk;
        text.Add(forecastResult.text);
        img.Add(forecastResult.img);
      }

      // Ottimizzazione portafoglio
      double[] portfolioValues = PSOHandler.start(0, indexesRevenue, indexesRisk).globalBest;

      // Output data
      Portfolio portfolio = new Portfolio();
      portfolio.horizon = 12;
      portfolio.SP_500 = portfolioValues[0];
      portfolio.FTSE_MIB = portfolioValues[1];
      portfolio.GOLD_SPOT = portfolioValues[2];
      portfolio.MSCI_EM = portfolioValues[3];
      portfolio.MSCI_EURO = portfolioValues[4];
      portfolio.All_Bonds = portfolioValues[5];
      portfolio.US_Treasury = portfolioValues[6];

      PortafoglioResult res = new PortafoglioResult();
      res.portfolio = portfolio;
      res.text = text.ToArray();
      res.img = img.ToArray();
      return res;
    }
  }
}
