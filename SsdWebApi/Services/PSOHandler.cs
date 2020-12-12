using System;

namespace SsdWebApi.Services
{

  public class PSOHandler
  {
    public static PSO start(int idTest) {
      // double res = double.MinValue;
      int dimensions = 0;
      double xmin = 0, xmax = 0;
      Func<double[], double> calculateFitness = null;

      if (idTest == 1) {
        dimensions = 20;
        xmin = -100;
        xmax = 100;
        calculateFitness = PSOHandler.paraboloid;
      } else if (idTest == 2) {
        dimensions = 30;
        xmin = -2048;
        xmax = 2048;
        calculateFitness = PSOHandler.rosenbrock;
      }

      int iters = 1000;
      int numNeighbours = 5;
      int numParticels = 50;

      PSO pso = new PSO(0.25, 1.5, 2.0, xmin, xmax, calculateFitness);
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

  }
}
