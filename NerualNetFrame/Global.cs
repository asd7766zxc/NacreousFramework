using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetFrame
{
    public static class Global
    {
        public static Random rand = new Random();
        public static double learningRate = 0;
        public static double Scalar { get { return rand.NextDouble() >0.5 ? -1 : 1; } set { } }
        public static double Clip = 0.7;
        public static void Normalize(ref List<double> values)
        {
            double mean = values.Sum() / values.Count;
            double variance = 0;
            foreach(var i in values)
            {
                variance += (i - mean) * (i - mean);
            }
            double standardDeviation = Math.Sqrt(variance / values.Count);
            for(int i = 0; i < values.Count; i++)
            {
                values[i] = (values[i] - mean) / standardDeviation;
            }
        }
        public static double ScalingNormalize(ref List<double> values)
        {
            double variance = 0;
            foreach (var i in values)
            {
                variance += i*i;
            }
            double length = Math.Sqrt(variance);
            for (int i = 0; i < values.Count; i++)
            {
                values[i] = (values[i]) / length;
            }
            return length;
        }
    }
}
