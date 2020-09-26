using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SsdWebApi;
using SsdWebApi.Models;

namespace SsdWebApi.Controllers
{
  [Route("api/[controller]")]
  [ApiController]
  public class GenStagioneController : ControllerBase
  {
    private readonly StagioneContext _context;

    public GenStagioneController(StagioneContext context)
    {
      _context = context;
    }

    // GET: api/GenStagione
    [HttpGet]
    public async Task<ActionResult<IEnumerable<Stagione>>> Getcronistoria()
    {
      return await _context.cronistoria.ToListAsync();
    }

    // GET: api/GenStagione/5
    [HttpGet("{id}")]
    public async Task<ActionResult<Stagione>> GetStagione(int id)
    {
      var stagione = await _context.cronistoria.FindAsync(id);

      if (stagione == null)
      {
        return NotFound();
      }

      return stagione;
    }

    // PUT: api/GenStagione/5
    // To protect from overposting attacks, enable the specific properties you want to bind to, for
    // more details, see https://go.microsoft.com/fwlink/?linkid=2123754.
    [HttpPut("{id}")]
    public async Task<IActionResult> PutStagione(int id, Stagione stagione)
    {
      if (id != stagione.id)
      {
        return BadRequest();
      }

      _context.Entry(stagione).State = EntityState.Modified;

      try
      {
        await _context.SaveChangesAsync();
      }
      catch (DbUpdateConcurrencyException)
      {
        if (!StagioneExists(id))
        {
          return NotFound();
        }
        else
        {
          throw;
        }
      }

      return NoContent();
    }

    // POST: api/GenStagione
    // To protect from overposting attacks, enable the specific properties you want to bind to, for
    // more details, see https://go.microsoft.com/fwlink/?linkid=2123754.
    [HttpPost]
    public async Task<ActionResult<Stagione>> PostStagione(Stagione stagione)
    {
      _context.cronistoria.Add(stagione);
      await _context.SaveChangesAsync();

      return CreatedAtAction("GetStagione", new { id = stagione.id }, stagione);
    }

    // DELETE: api/GenStagione/5
    [HttpDelete("{id}")]
    public async Task<ActionResult<Stagione>> DeleteStagione(int id)
    {
      var stagione = await _context.cronistoria.FindAsync(id);
      if (stagione == null)
      {
        return NotFound();
      }

      _context.cronistoria.Remove(stagione);
      await _context.SaveChangesAsync();

      return stagione;
    }

    private bool StagioneExists(int id)
    {
      return _context.cronistoria.Any(e => e.id == id);
    }
  }
}
