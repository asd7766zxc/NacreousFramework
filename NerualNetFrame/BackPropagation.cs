using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NerualNetFrame
{
    [Serializable]
    public class BackPropagation
    {
        public List<Layer>? layers;
        public List<double> _testData;
   
        const double d = 1E-10;
        public BackPropagation()
        {

        }
 

        //微分過一次神經元 必定會有一個 activefunction

        public void Prop(int cindex)
        {
            // dC/d
            var outputlayer = layers.LastOrDefault();
            //最後一層 C =sigma[ (t(i)-p(i))^2 ] ; dC/dr(i) =[ 2r(i)-2t(i) ] 
            double expsum = 0;//for softmax
            double avgsum = 0;
            double max = 0;
            List<double> Z_Table = new List<double>();
            for (int i = 0; i < outputlayer._neuron.Count; i++)
            {
                var neuron = outputlayer._neuron[i];
                Z_Table.Add(neuron.z);
            }
           double length = Global.ScalingNormalize(ref Z_Table);
            avgsum = max;
           
            for (int i = 0; i < Z_Table.Count; i++)
            {
                var safevalue = Math.Pow(Math.E, Z_Table[i] - avgsum);
                if (safevalue < 1E-50)
                    safevalue = 0;
                expsum += safevalue;
            }
            if (expsum == 0)
                Console.WriteLine(); 
            for (int i = 0; i < outputlayer._neuron.Count; i++)
            {
                var neuron = outputlayer._neuron[i];
                neuron.output = Math.Pow(Math.E, Z_Table[i] - avgsum) / expsum;
                if (double.IsNaN(neuron.output))
                    Console.WriteLine("error");
                //calculat the softmax 簡單來說 進行歸一 
            }
            for (int i = 0; i < outputlayer._neuron.Count; i++)
            {
                //dC/dr 
                var neuron = outputlayer._neuron[i];
                var dCR = (neuron.output - _testData[i]);
              // double sz = neuron.x;
                 /*sigmoid*/
               //neuron.dCdu[cindex] = sz*(1-sz)*dCR;
                //softmax 需要偏微
                //因 數據在送入計算前 進行過 scaling  
                //因此在微分需要把scaling 乘回來
               var dcu =  ((Math.Pow(Math.E, Z_Table[i] - avgsum) * expsum) / (Math.Pow(expsum, 2))) * dCR;

                neuron.dCdu[cindex] = length * dcu;
            }
            //Partial Neuron 
            for (int r = layers.Count - 2; r >= 0; r--)
            {
                Parallel.For(0, layers[r]._neuron.Count, (i) =>
                {
                    double partialSum = 0;
                    for (int a = 0; a < layers[r + 1]._neuron.Count; a++)
                    {
                        //Chain Rule 
                        // u[r+1][a] = sigmoid(u[r][i].z) => s(z)(1-s(z))
                        // sigmoid  var sz =ActiveFunctions.Sigmoid( layers[r + 1]._neuron[a]._weights[i]*layers[r]._neuron[i].x + layers[r + 1]._neuron[a]._bias[i]);
                        // sigmoid partialSum += layers[r + 1]._neuron[a]._weights[i] * sz * (1 - sz) * layers[r + 1]._neuron[a].dCdu[cindex];
                        double factor = 0.01;
                        if (layers[r]._neuron[a].z > 0)
                            factor = 1;
                        var dp = layers[r + 1]._neuron[a]._weights[i] * factor * layers[r + 1]._neuron[a].dCdu[cindex];

                            partialSum += dp;
                        // z = a*w + b 
                        //Partial Weights 
                        // sigmoid layers[r + 1]._neuron[a].d_weights[cindex][i] = layers[r]._neuron[i].x * sz * (1 - sz) * layers[r + 1]._neuron[a].dCdu[cindex];
                 
                        var dw = layers[r]._neuron[i].x * factor * layers[r + 1]._neuron[a].dCdu[cindex];

                            layers[r + 1]._neuron[a].d_weights[cindex][i] = dw;
                        //Partial Bias 
                        //bias is constant => db = 1 
                        //sigmoid layers[r + 1]._neuron[a].d_bias[cindex][i] =1 * sz * (1 - sz) * layers[r + 1]._neuron[a].dCdu[cindex];
                        var db = 1 * factor * layers[r + 1]._neuron[a].dCdu[cindex];

                        layers[r + 1]._neuron[a].d_bias[cindex] = db;
                    }
                    layers[r]._neuron[i].dCdu[cindex] = partialSum;
                });

            }

        }
        public int Result()
        {
            var outputlayer = layers.LastOrDefault();
            double expsum = 0;//for softmax
            List<double> Z_Table = new List<double>();
            for (int i = 0; i < outputlayer._neuron.Count; i++)
            {
                var neuron = outputlayer._neuron[i];
                Z_Table.Add(neuron.z);
            }
            Global.ScalingNormalize(ref Z_Table);
            for (int i = 0; i < Z_Table.Count; i++)
            {
                var safevalue = Math.Pow(Math.E, Z_Table[i] );
                if (safevalue < 1E-50)
                    safevalue = 0;
                expsum += safevalue;
            }
            if (expsum == 0)
                Console.WriteLine();
            for (int i = 0; i < outputlayer._neuron.Count; i++)
            {
                var neuron = outputlayer._neuron[i];
                neuron.output = Math.Pow(Math.E, Z_Table[i] ) / expsum;
                if (double.IsNaN(neuron.output))
                    Console.WriteLine("error");
                //calculat the softmax 簡單來說 進行歸一 
            }
            List<double> a = new List<double>();
            foreach (var i in layers.LastOrDefault()._neuron)
            {
                a.Add(i.output);
            }
            return a.IndexOf(a.Max());
        }
        public double Cost()
        {           
            double cost = 0;
            for (int i = 0; i < _testData.Count; i++)
            {
                cost += Math.Pow((layers.LastOrDefault()._neuron[i].output- _testData[i]), 2);
            }
            return cost;
        }

    }
}
