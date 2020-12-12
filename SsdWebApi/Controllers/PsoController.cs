using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using SsdWebApi.Models;
using SsdWebApi.Services;

// Documentazione https://docs.microsoft.com/it-it/aspnet/core/data/ef-mvc/crud?view=aspnetcore-3.1

namespace SsdWebApi.Controllers
{
  [ApiController]
  [Route("api/[controller]")]
  public class PsoController : ControllerBase
  {
    private readonly QuotazioneContext _context;

    private readonly Persistence persistence;

    public PsoController(QuotazioneContext context)
    {
      _context = context;
      persistence = new Persistence(context);
    }

    // [HttpGet] public async Task<ActionResult<List<Quotazione>>> GetAll() => await _context.indici.ToListAsync();

    [HttpGet("{id}")]
    public async Task<ActionResult<double[]>> Get(int id)
    {
      return PSOHandler.start(id).globalBest;

      // if (id > 8) return NotFound();
      // string[] indices = new string[] { "id", "Data", "SP_500", "FTSE_MIB", "GOLD_SPOT", "MSCI_EM", "MSCI_EURO", "All_Bonds", "US_Treasury" };
      // string index = indices[id];

      // // Forecast
      // Forecast forecast = new Forecast();
      // return forecast.forecastIndex(index, type);
    }
  }
}
