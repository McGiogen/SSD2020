using Microsoft.EntityFrameworkCore;

namespace SsdWebApi.Models
{
  public class QuotazioneContext : DbContext
  {
    public QuotazioneContext(DbContextOptions<QuotazioneContext> options) : base(options) { }
    public DbSet<Quotazione> indici { get; set; }
  }
}
