using System;
using System.Drawing;
using SsdWebApi.Services;

namespace SsdWebApi.Models
{
  public class Forecast
  {
    public Forecast() {

    }

    public string forecastSARIMAIndex(string index) {
      string res = "\"text\": \"";
      string interpreter = @"python3";
      string environment = "";
      int timeout = 5000;
      PythonRunner runner = new PythonRunner(interpreter, environment, timeout);
      Bitmap bmp = null;

      try {
        string command = $"Services/forecast.py {index}.csv";
        string list = runner.runDosCommands(command);

        if (string.IsNullOrWhiteSpace(list)) {
          Console.WriteLine("Error in the script call");
          return res;
        }

        string[] lines = list.Split(new[] {Environment.NewLine }, System.StringSplitOptions.None);
        string strBitmap="";

        foreach (string s in lines) {
          if (s.StartsWith("MAPE")) {
            Console.WriteLine(s);
            res += s;
          }
          if (s.StartsWith("b'")) {
            strBitmap = s.Trim();
            break;
          }
          if (s.StartsWith("Actual")) {
            double fcast = Convert.ToDouble(s.Substring(s.LastIndexOf(" ")));
            Console.WriteLine(fcast);
          }
        }

        strBitmap = strBitmap.Substring(strBitmap.IndexOf("b'"));
        res += $"\", \"img\": \"{strBitmap}\"";
        try {
          bmp = runner.FromPythonBase64String(strBitmap);
        } catch (Exception e) {
          throw new System.Exception("An error occurred while trying to create an image from Python script output.", e);
        }
      } catch (Exception e) {
        Console.WriteLine(e.ToString());
      }

      return res;
    }
  }
}
