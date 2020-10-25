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
    public async Task<ActionResult<Stagione>> Get(int id)
    {
      var entity = await _context.cronistoria.FindAsync(id);
      if (entity == null) return NotFound();
      return entity;
    }

    [HttpPost]
    // [Route("[action]")] per ottenere il seguente route: /api/Stagione/<nome-funzione>
    public async Task<ActionResult<Stagione>> Create(Stagione entity)
    {
      try
      {
        _context.cronistoria.Add(entity);
        await _context.SaveChangesAsync();
        return entity;
      }
      catch (Exception ex)
      {
        Console.WriteLine("[ERROR] " + ex.Message);
        return StatusCode(500, ex);
      }
    }

    [HttpPut("{id}")]
    public async Task<ActionResult<Stagione>> Update(int id, Stagione entity)
    {
      if (id != entity.id) return BadRequest();

      // _context.cronistoria.Update(entity);
      // await _context.SaveChangesAsync();
      // return entity;

      _context.Entry(entity).State = EntityState.Modified;

      try
      {
        await _context.SaveChangesAsync();
      }
      catch (Exception ex)
      {
        if (!_context.cronistoria.Any(s => s.id == id))
        {
          return NotFound();
        }
        else
        {
          Console.WriteLine("[ERROR] " + ex.Message);
          return StatusCode(500, ex);
        }
      }

      return Ok();
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
