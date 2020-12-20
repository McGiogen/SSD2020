using System;
using SsdWebApi.Services;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

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
        string command = $"/home/gioele/workspace/uni/ssd/esame/SsdWebApi/Models/forecast.py {index}.csv {type}";
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
            strBitmapArray.Add(s.Trim().Substring(s.IndexOf("'") + 1, s.Length - (s.Length - s.LastIndexOf("'")) - s.IndexOf("'") - 1));
            Console.WriteLine("Image found");
          }
          if (s.StartsWith("REVENUE ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.revenue = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ") + 1), CultureInfo.InvariantCulture);
          }
          if (s.StartsWith("REVENUE_PERC ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.revenuePerc = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ") + 1), CultureInfo.InvariantCulture);
          }
          if (s.StartsWith("PCT_CHANGES ")) {
            Console.WriteLine(s);
            int startIndex = s.Trim().IndexOf(" [")+2;
            int endIndex = s.Trim().Length-1;
            res.pctChanges = s[startIndex..endIndex].Split(",").Select(v => Convert.ToDouble(v, CultureInfo.InvariantCulture)).ToArray();
          }
          if (s.StartsWith("MAPE ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.mape = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ") + 1), CultureInfo.InvariantCulture);
          }
          if (s.StartsWith("VAR ")) {
            Console.WriteLine(s);
            res.text += s + Environment.NewLine;
            res.valueAtRisk = Convert.ToDouble(s.Trim().Substring(s.Trim().IndexOf(" ") + 1), CultureInfo.InvariantCulture);
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
