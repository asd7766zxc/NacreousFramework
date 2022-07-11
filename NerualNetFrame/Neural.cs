using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetFrame
{
    public class Neural
    {   
        
        public List<double> _weights = new List<double>();
        public List<double> d_weights = new List<double>();
        public List<double> _bias = new List<double>();
        public List<double> d_bias = new List<double>();
        //此的 權重為 此層與上層神經的連接權重
        public double _value = 0;
        public double _u { get { return ActiveFunctions.Sigmoid(_value); } set { _u = value; } }
        // _value 為原始值 _u是要傳到下一層的
    }
}
