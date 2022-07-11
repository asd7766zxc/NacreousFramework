using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetFrame
{
    [Serializable]
    public class Layer
    {
        public List<Neuron> _neuron = new List<Neuron>();
        Layer? _lastlayers;
        ActiveTypes _actypes;//feature 
        public Layer(Layer _lastLayer, ActiveTypes ac_type)
        {
            _lastlayers = _lastLayer;
            _actypes = ac_type;
        }
        public Layer(Layer _lastLayer)
        {
            _lastlayers = _lastLayer;
        }
        public Layer()
        {

        }
        public void ResizeNeuron(int count)
        {
            for (int i = 0; i < count; i++)
            {
                _neuron.Add(new Neuron());
            }
        }
        public void init(int batchSize, bool isOuput = false)
        {
            ActiveTypes active = ActiveTypes.ReLU ;
            if (_lastlayers == null)
            {
                _lastlayers = new Layer();
                _lastlayers._neuron = new List<Neuron>();
                _lastlayers._neuron.Add(new Neuron());
                active = ActiveTypes.None;
            }
            foreach (Neuron current_neural in _neuron)
            {
                current_neural._activeTypes = active;
                current_neural.isOutputLayer = isOuput;
                current_neural.z = current_neural._bias;
                for (int i = 0; i < _lastlayers._neuron.Count; i++)
                {
                    current_neural._weights.Add(1 * Global.rand.NextDouble() * Global.Scalar);
                    current_neural.weightsVelocity.Add(0);
                    current_neural._bias = 0;
                    current_neural.z += current_neural._weights[i] * _lastlayers._neuron[i].x;
                }
                    Global.ScalingNormalize(ref current_neural._weights);
                for (int r = 0; r < batchSize; r++)
                {
                    current_neural.d_weights.Add(new List<double>());
                    current_neural.d_bias.Add(0);
                    current_neural.dCdu.Add(0);
                    for (int i = 0; i < _lastlayers._neuron.Count; i++)
                    {
                        current_neural.d_weights[r].Add(0);
                        current_neural.d_bias[r] = 0;
                    }
                }
            }

        }
        public void ForwardPass()
        {
            if (_lastlayers != null)
                Parallel.For(0, _neuron.Count, a =>
                {
                    _neuron[a].z = _neuron[a]._bias;
                    for (int i = 0; i < _lastlayers._neuron.Count; i++)
                    {
                        _neuron[a].z += _neuron[a]._weights[i] * _lastlayers._neuron[i].x ;
                    }
                });
        }
        public void UpdateParameter(double beta)
        {
            Parallel.For(0, _neuron.Count, a =>
            {
                for (int i = 0; i < _neuron[a].d_weights[0].Count; i++) //每次權重數量一樣 所以取第一次就好
                {
                    double dw = 0;
                    for (int u = 0; u < _neuron[a].d_weights.Count; u++)
                    {
                        dw += _neuron[a].d_weights[u][i]; //把此次 batch內所有的斜率加起來平均 找出這次batch的整體下降量
                        _neuron[a].d_weights[u][i] = 0;//清空 留給下次batch 
                    }
                    dw = dw / (double)_neuron[a].d_weights.Count;
                    //梯度剪切
                    //momentum
                    _neuron[a].weightsVelocity[i] = beta * _neuron[a].weightsVelocity[i] + 1 * Global.learningRate * dw;
                    _neuron[a]._weights[i] = _neuron[a]._weights[i] - _neuron[a].weightsVelocity[i];
                    //_neuron[a]._weights[i] = _neuron[a]._weights[i] + (-1)* Global.learningRate * dw;
                }
                Global.ScalingNormalize(ref _neuron[a]._weights);
                double db = 0;
                for (int u = 0; u < _neuron[a].d_bias.Count; u++)
                {

                    //梯度剪切
                    db += _neuron[a].d_bias[u];
                    _neuron[a].d_bias[u] = 0;
                }
                db = db / (double)_neuron[a].d_bias.Count;
                _neuron[a]._bias += -1 * Global.learningRate * db;
            });
        }
    }
}
