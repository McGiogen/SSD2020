using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SsdWebApi.Models;

// Documentazione https://docs.microsoft.com/it-it/aspnet/core/data/ef-mvc/crud?view=aspnetcore-3.1

namespace SsdWebApi.Controllers
{
  [ApiController]
  [Route("api/Stagione")]
  public class StagioneController : ControllerBase
  {
    private readonly StagioneContext _context;

    public StagioneController(StagioneContext context)
    {
      _context = context;
    }

    [HttpGet] public ActionResult<List<Stagione>> GetAll() => _context.cronistoria.ToList();

    [HttpGet("{id}")]
    public ActionResult<Stagione> Get(int id)
    {
      return _context.cronistoria.Find(id);
    }

    [HttpPost]
    public async Task<ActionResult<Stagione>> Create(Stagione entity)
    {
      _context.cronistoria.Add(entity);
      await _context.SaveChangesAsync();
      return entity;
    }

    [HttpPut]
    public async Task<ActionResult<Stagione>> Update(/*int id,*/ Stagione entity)
    {
      _context.cronistoria.Update(entity);
      await _context.SaveChangesAsync();
      return entity;
    }

    [HttpDelete("{id}")]
    public async Task<ActionResult<Stagione>> Delete(int id)
    {
      var entity = await _context.cronistoria.FindAsync(id);
      if (entity == null) return NotFound();

      _context.cronistoria.Remove(entity);
      await _context.SaveChangesAsync();
      return entity;
    }
  }
}
