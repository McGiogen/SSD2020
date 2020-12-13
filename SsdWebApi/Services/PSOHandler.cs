using System;
using System.Linq;
using System.Collections.Generic;

namespace SsdWebApi.Services
{
  public class PSOHandler
  {
    public static PSO start(int idTest, double[] indexesRevenue = null, double[] indexesRisk = null) {
      // double res = double.MinValue;
      int dimensions = 0;
      double vMin = 0, vMax = 0;
      Func<double[], double> calculateFitness = null;
      Action<Particle> adeguateLimits = null;
      int iters = 1000;

      int numNeighbours = 5;
      int numParticels = 50;

      if (idTest == 1) {
        dimensions = 20;
        vMin = -100;
        vMax = 100;
        calculateFitness = PSOHandler.paraboloid;
        adeguateLimits = PSOHandler.adeguateLimitsPerDimension(vMin, vMax);
      } else if (idTest == 2) {
        dimensions = 30;
        vMin = -2048;
        vMax = 2048;
        iters = 2000;
        calculateFitness = PSOHandler.rosenbrock;
        adeguateLimits = PSOHandler.adeguateLimitsPerDimension(vMin, vMax);
      } else {
        if (indexesRevenue == null || indexesRisk == null) throw new Exception("indexesRevenue and indexesRisk cannot be null");
        dimensions = 7;
        vMin = 5;
        vMax = (100 - vMin*dimensions)/dimensions * 2 + vMin;
        iters = 2000;
        calculateFitness = PSOHandler.indexesFitness(indexesRevenue, indexesRisk);
        adeguateLimits = PSOHandler.adeguateLimitsPerc(5);
      }

      PSO pso = new PSO(0.25, 1.5, 2.0, vMin, vMax, calculateFitness, adeguateLimits);
      pso.calculate(numParticels, dimensions, iters, numNeighbours);

      return pso;
    }

    static public double paraboloid(double[] xvec)
    {
      double sum = 0;
      int i;
      for (i = 0; i < xvec.Length; i++)
        sum += Math.Pow(xvec[i], 2);
      return -sum;
    }

    static public double rosenbrock(double[] xvec)
    {
      double sum = 0;
      int i, dim = xvec.Length;
      for (i = 0; i < dim - 1; i++)
        sum += 100 * Math.Pow((xvec[i + 1] - Math.Pow(xvec[i], 2)), 2) + Math.Pow((1 - xvec[i]), 2);
      return -sum;
    }

    static public Func<double[], double> indexesFitness(double[] revenue, double[] risk)
    {
      return (double[] xvec) => {
        double sum = 0;
        int i, dim = xvec.Length;
        for (i = 0; i < dim - 1; i++)
          sum += revenue[i] - risk[i];
        return sum;
      };
    }

    static public Action<Particle> adeguateLimitsPerDimension(double minValue, double maxValue) {
      return (Particle particle) => {
        foreach (int d in Enumerable.Range(0, particle.value.Length))
        {
          // position entro i limiti
          if (particle.value[d] < minValue)
          {
            particle.value[d] = minValue;
            particle.velocity[d] *= -1;
          }
          else if (particle.value[d] > maxValue)
          {
            particle.value[d] = maxValue;
            particle.velocity[d] *= -1;
          }
        }
      };
    }

    static public Action<Particle> adeguateLimitsPerc(double minPerc) {
      return (Particle particle) => {
        int dimensions = particle.value.Length;

        // Rimuove il valore minimo - minPerc - e calcola il totale
        double[] values = particle.value.Select(v => v - minPerc > 0 ? v - minPerc : 0).ToArray();
        double tot = values.Sum();

        // Verifica il raggiungimento del 100% ed eventualmente adegua
        double desiredTot = 100 - minPerc*dimensions;
        if (tot != desiredTot)
        {
          foreach (int d in Enumerable.Range(0, values.Length))
          {
            values[d] = tot == 0 ? 0 : values[d] / tot * desiredTot;
            particle.velocity[d] *= -1;
          }
        }

        // Aggiunge il limite minimo - minPerc
        Array.Copy(particle.value, values.Select(v => v + 5).ToArray(), dimensions);
      };
    }
  }
}
