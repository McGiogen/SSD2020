using System;
using System.Linq;

namespace SsdWebApi.Services
{

  public class PSO
  {
    public double c0, c1, c2;
    public double minValue, maxValue;
    public double fitGlobalBest;
    public double[] globalBest;
    public Func<double[], double> calculateFitness;

    public PSO(double c0, double c1, double c2, double minValue, double maxValue, Func<double[], double> calculateFitness)
    {
      this.c0 = c0;
      this.c1 = c1;
      this.c2 = c2;
      this.minValue = minValue;
      this.maxValue = maxValue;
      this.fitGlobalBest = double.MinValue;
      this.calculateFitness = calculateFitness;
    }

    public double calculate(int numParticels, int dimensions, int iters, int numNeighbours)
    {
      System.Random rnd = new System.Random(666);
      globalBest = new double[dimensions];

      // Init
      Particle[] particles = Enumerable.Range(0, numParticels).Select(i => new Particle()).ToArray();
      foreach (Particle particle in particles)
      {
        // position and velocity
        var dimensionsRange = Enumerable.Range(0, dimensions);
        particle.value = dimensionsRange.Select(i => rnd.NextDouble() * (maxValue - minValue) - minValue).ToArray();
        particle.velocity = dimensionsRange.Select(i => (rnd.NextDouble() - rnd.NextDouble()) * 0.5 * (maxValue - minValue) - minValue).ToArray();
        particle.personalBest = dimensionsRange.Select(i => particle.value[i]).ToArray();
        particle.localBest = dimensionsRange.Select(i => particle.value[i]).ToArray();

        // fit
        particle.fit = calculateFitness(particle.value);
        particle.fitPersonalBest = particle.fit;

        // neighbours
        particle.neighbours = new int[numNeighbours];
        foreach (int i in Enumerable.Range(0, particle.neighbours.Length)) {
          int id;
          do
          {
            id = rnd.Next(numParticels);
          } while (Array.IndexOf(particle.neighbours, id) == -1);
          particle.neighbours[i] = id;
        }
      }

      foreach (int iter in Enumerable.Range(0, iters))
      {
        Console.WriteLine($"Ciclo {iter}, best result {fitGlobalBest}");

        // Update all particles
        foreach (Particle particle in particles)
        {
          foreach (int d in Enumerable.Range(0, dimensions))
          {
            {
              // Velocity
              particle.velocity[d] = c0 * particle.velocity[d]
                + c1 * rnd.NextDouble() * (particle.personalBest[d] - particle.value[d])
                + c2 * rnd.NextDouble() * (particle.localBest[d] - particle.value[d]);

              // Position
              particle.value[d] += particle.velocity[d];

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

              // Fitness
              particle.fit = calculateFitness(particle.value);

              // Personal best
              if (particle.fit > particle.fitPersonalBest)
              {
                particle.fitPersonalBest = particle.fit;
                Array.Copy(particle.value, particle.personalBest, particle.value.Length);
              }

              // Local best
              particle.fitLocalBest = particle.fitPersonalBest;
              foreach (int neighbourId in particle.neighbours)
              {
                Particle neighbour = particles[neighbourId];
                if (neighbour.fit > particle.fitLocalBest)
                {
                  particle.fitLocalBest = neighbour.fit;
                }
              }

              // Global best
              if (particle.fit > fitGlobalBest)
              {
                fitGlobalBest = particle.fit;
                Array.Copy(particle.value, globalBest, particle.value.Length);
              }
            }
          }
        }
      }
      return fitGlobalBest;
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
