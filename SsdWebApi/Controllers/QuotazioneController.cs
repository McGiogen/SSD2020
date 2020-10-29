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
  [Route("api/[controller]")]
  public class QuotazioneController : ControllerBase
  {
    private readonly QuotazioneContext _context;

    public QuotazioneController(QuotazioneContext context)
    {
      _context = context;
    }

    [HttpGet] public async Task<ActionResult<List<Quotazione>>> GetAll() => await _context.indici.ToListAsync();

    [HttpGet("{id}")]
    public async Task<ActionResult<Quotazione>> Get(int id)
    {
      var entity = await _context.indici.FindAsync(id);
      if (entity == null) return NotFound();
      return entity;
    }

    [HttpPost]
    // [Route("[action]")] per ottenere il seguente route: /api/Quotazione/<nome-funzione>
    public async Task<ActionResult<Quotazione>> Create(Quotazione entity)
    {
      try
      {
        _context.indici.Add(entity);
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
    public async Task<ActionResult<Quotazione>> Update(int id, Quotazione entity)
    {
      if (id != entity.id) return BadRequest();

      _context.Entry(entity).State = EntityState.Modified;

      try
      {
        await _context.SaveChangesAsync();
      }
      catch (Exception ex)
      {
        if (!_context.indici.Any(s => s.id == id))
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
    public async Task<ActionResult<Quotazione>> Delete(int id)
    {
      var entity = await _context.indici.FindAsync(id);
      if (entity == null) return NotFound();

      _context.indici.Remove(entity);
      await _context.SaveChangesAsync();
      return entity;
    }
  }
}
