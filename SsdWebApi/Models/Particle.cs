namespace SsdWebApi.Services
{
  public class Particle
  {
    public double[] value;
    public double[] personalBest;
    public double[] localBest;
    public double[] velocity;
    public double fit;
    public double fitPersonalBest;
    public double fitLocalBest;
    public int[] neighbours;
  }
}
