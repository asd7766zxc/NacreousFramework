using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetFrame
{
    public static class ActiveFunctions
    {
        public static double Active(ActiveTypes type,double value)
        {
            switch(type)
            {
                case ActiveTypes.None:
                    return value;
                case ActiveTypes.Sigmoid:
                    return Sigmoid(value);
                case ActiveTypes.ReLU:
                    return ReLU(value);
                default: 
                    return value;
            }

        }
        public static double Sigmoid(double x)
        {
            double r = 1 / (1 + Math.Pow(Math.E, -1 * x));
            return r;
        }
        public static double ReLU(double x)
        {
            if (x < 0)
                return 0.01*x;
            else
                return x;
        }
    }
    public enum ActiveTypes
    {
        None,
        Sigmoid,
        ReLU,
        SoftMax,
    }
}
