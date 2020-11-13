using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using SsdWebApi.Models;

namespace SsdWebApi.Controllers
{
  public class Persistence
  {
    private QuotazioneContext context;

    public Persistence(QuotazioneContext context)
    {
      this.context = context;
    }

    public async Task<List<string>> readIndex(string attribute)
    {
      List<string> serie = new List<string>();
      serie.Add(attribute);
      using (var command = context.Database.GetDbConnection().CreateCommand())
      {
        command.CommandText = $"SELECT {attribute} FROM indici";
        await context.Database.OpenConnectionAsync();
        using (var reader = await command.ExecuteReaderAsync())
        {
          while (await reader.ReadAsync())
          {
            serie.Add(reader[attribute].ToString());
          }
        }
      }

      return serie;
    }

    public void writeCsv(string attribute, List<string> serie)
    {
      using (StreamWriter writer = new StreamWriter(attribute + ".csv", false))
      {
        foreach (string row in serie)
        {
          writer.WriteLine(row);
        }
      }
    }
  }
}
