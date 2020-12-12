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
        calculateFitness = PSO.paraboloid;
      } else if (idTest == 2) {
        dimensions = 30;
        xmin = -2048;
        xmax = 2048;
        calculateFitness = PSO.rosenbrock;
      }

      int iters = 1000;
      int numNeighbours = 5;
      int numParticels = 50;

      PSO pso = new PSO(0.25, 1.5, 2.0, xmin, xmax, calculateFitness);
      pso.calculate(numParticels, dimensions, iters, numNeighbours);

      return pso;
    }

  }
}
