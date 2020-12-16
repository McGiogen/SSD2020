using System;
using System.Linq;
using System.Collections.Generic;

namespace SsdWebApi.Services
{
  public class PSOHandler
  {
    public static PSO start(int id, double[] indexesRevenue = null, double[] indexesRisk = null) {
      // double res = double.MinValue;
      int dimensions = 0;
      double valueMin = 0, valueMax = 0, velocityMin = 0, velocityMax = 0;
      Func<double[], double> calculateFitness = null;
      Action<Particle> adeguateLimits = null;
      int iters = 1000;

      int numNeighbours = 5;
      int numParticels = 50;

      double c0 = 0.25, c1 = 1.5, c2 = 2.0;

      if (id == 1) {
        dimensions = 20;
        valueMin = velocityMin = -100;
        valueMax = velocityMax = 100;
        calculateFitness = PSOHandler.paraboloid;
        adeguateLimits = PSOHandler.adeguateLimitsPerDimension(valueMin, valueMax);
      } else if (id == 2) {
        dimensions = 30;
        valueMin = velocityMin = -2048;
        valueMax = velocityMax = 2048;
        iters = 2000;
        calculateFitness = PSOHandler.rosenbrock;
        adeguateLimits = PSOHandler.adeguateLimitsPerDimension(valueMin, valueMax);
      } else {
        if (indexesRevenue == null || indexesRisk == null) throw new Exception("indexesRevenue and indexesRisk cannot be null");
        dimensions = 7;
        valueMin = 5;
        valueMax = (100 - valueMin*dimensions) * 2/dimensions + valueMin;
        velocityMin = 0;
        velocityMax = 5;
        iters = 2000;
        calculateFitness = PSOHandler.indexesFitness(indexesRevenue, indexesRisk);
        adeguateLimits = PSOHandler.adeguateLimitsPerc(5);
      }

      PSO pso = new PSO(c0, c1, c2, valueMin, valueMax, velocityMin, velocityMax, calculateFitness, adeguateLimits);
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
        // Con la divisione si comparano meglio guadagni e rischi alti con guadagni e rischi bassi
        // Con la potenza si ottiene sempre un valore del rischio positivo (per non modificare il segno di revenue)
        // e si da minor valore alle ricompense ad alto rischio (revenue aumenta in maniera lineare mentre risk in maniera esponenziale)
        for (i = 0; i < dim - 1; i++)
          sum += revenue[i]*xvec[i] / Math.Pow(risk[i]*xvec[i], 2);
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
        Array.Copy(values.Select(v => v + 5).ToArray(), particle.value, dimensions);
      };
    }
  }
}
