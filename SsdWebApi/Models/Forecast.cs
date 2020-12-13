using System;
using SsdWebApi.Services;
using System.Collections.Generic;

namespace SsdWebApi.Models
{
  public class Forecast
  {
    public Forecast() {

    }

    public ForecastResult forecastIndex(string index, string type) {
      ForecastResult res = new ForecastResult();
      string interpreter = @"/usr/bin/python3";
      string environment = "";
      int timeout = 5000;
      PythonRunner runner = new PythonRunner(interpreter, environment, timeout);

      try {
        string command = $"/home/gioele/workspace/uni/ssd/esame/SsdWebApi/Services/forecast.py {index}.csv {type}";
        string list = runner.runDosCommands(command);

        if (string.IsNullOrWhiteSpace(list)) {
          Console.WriteLine("Error in the script call: " + command);
          return res;
        }

        string[] lines = list.Split(new[] {Environment.NewLine }, System.StringSplitOptions.None);
        List<string> strBitmapArray= new List<string>();

        foreach (string s in lines) {
          if (s.StartsWith("LOG ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
          }
          if (s.StartsWith("b'")) {
            strBitmapArray.Add(s.Trim().Substring(2, s.Length - 3));
            Console.WriteLine("Image found");
          }
          if (s.StartsWith("REVENUE ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.revenue = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ")));
          }
          if (s.StartsWith("RISK ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.risk = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ")));
          }
        }
        res.img = strBitmapArray.ToArray();
      } catch (Exception e) {
        Console.WriteLine(e.ToString());
      }

      return res;
    }
  }
}
