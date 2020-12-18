using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using System.IO;
using SsdWebApi.Models;

namespace SsdWebApi
{
  public class Startup
  {
    public Startup(IConfiguration configuration)
    {
      Configuration = configuration;
    }

    public IConfiguration Configuration { get; }

    // This method gets called by the runtime. Use this method to add services to the container.
    public void ConfigureServices(IServiceCollection services)
    {
      // https://dzone.com/articles/cors-in-net-core-net-core-security-part-vi
      services.AddCors(options =>
      {
        options.AddPolicy("AllowAnyOrigin", builder => builder.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod());
      });
      services.AddDbContext<QuotazioneContext>(options => options.UseSqlite("Data Source=finindices.sqlite"));
      services.AddControllers();
    }

    // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
      if (env.IsDevelopment())
      {
        app.UseDeveloperExceptionPage();
      }
      app.Use(async (context, next) => {
        await next();

        if (context.Response.StatusCode == 404 &&
              !Path.HasExtension(context.Request.Path.Value) &&
              !context.Request.Path.Value.StartsWith("/api/"))
        {
          context.Request.Path = "/index.html";

          await next();
        }
      });

      // Per servire il client
      app.UseDefaultFiles();
      app.UseStaticFiles(new StaticFileOptions
      {
          FileProvider = new PhysicalFileProvider(
              Path.Combine(env.ContentRootPath, "views")),
          RequestPath = ""
      });

      app.UseHttpsRedirection();

      app.UseRouting();

      app.UseCors("AllowAnyOrigin");


      app.UseAuthorization();
      app.UseEndpoints(endpoints =>
      {
        endpoints.MapControllers();
      });
    }
  }
}
