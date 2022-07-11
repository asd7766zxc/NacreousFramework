using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetFrame
{
    [Serializable]
    public class Neuron
    {
        public bool isOutputLayer = false;
        public List<double> _weights = new List<double>();
        public List<List<double>> d_weights = new List<List<double>>();
        public List<double> weightsVelocity = new List<double>();
        public double _bias;
        public List<double> d_bias = new List<double>();
        public ActiveTypes _activeTypes = ActiveTypes.ReLU;
        //此的 權重為 此層與上層神經的連接權重
        public double z = 0;
        //public double x { get { return ActiveFunctions.Sigmoid(z); } set { x = value; } }
        public double _x = 0 ;
        public double x { get {  _x  = isOutputLayer ? output:ActiveFunctions.Active(_activeTypes, z); return _x; } set { _x = value; } }
        public double output;
        // _value 為原始值 _u是要傳到下一層的
        //current neuron tial derivative to Cost Function 
        public List<double> dCdu = new List<double>();

    }
}
