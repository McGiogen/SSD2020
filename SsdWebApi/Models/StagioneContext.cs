using Microsoft.EntityFrameworkCore;

namespace SsdWebApi.Models
{
  public class StagioneContext : DbContext
  {
    public StagioneContext(DbContextOptions<StagioneContext> options) : base(options) { }
    public DbSet<Stagione> cronistoria { get; set; }
  }
}