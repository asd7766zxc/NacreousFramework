using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Diagnostics;
using System.Collections.Concurrent;

namespace NerualNetFrame
{
    [Serializable]
    public class NetworkManager
    {
        public List<string> ModelInfo = new List<string>();
        List<byte[]> TrainingSet = new List<byte[]>();
        List<List<byte[]>> TrainingSetBatch = new List<List<byte[]>>();
        List<List<double>> TrainingLabelSet = new List<List<double>>();

        List<List<List<double>>> TrainingLabelBatch = new List<List<List<double>>>();
        List<byte> LabelSet = new List<byte>();
        public double LearningRate = 0.3;
        BackPropagation BP;
        const int InputSize = 784;
        const int OutputSize = 10;
        public string ModelName = "";
        public void Initialize(int hiddenLayerCount, List<int> neuronLsit)
        {
            //DNN
            int totalParameterCount = 0;
            Global.learningRate = LearningRate;
            BP = new BackPropagation();
            BP.layers = new List<Layer>();
            //Input Layer Layer(1) 
            Layer InputLayer = new Layer();
            InputLayer.ResizeNeuron(InputSize);//img input size
            InputLayer.init(BatchSize);
            BP.layers.Add(InputLayer);
            int lastCount = 0;
            for (int i = 0; i < hiddenLayerCount; i++)
            {
                if (i == 0)
                {
                    int hiddenLayerNeuron = neuronLsit[i];
                    lastCount = hiddenLayerNeuron;
                    totalParameterCount += InputSize * hiddenLayerNeuron * 2;
                    Layer HiddenLayer1 = new Layer(InputLayer);
                    HiddenLayer1.ResizeNeuron(hiddenLayerNeuron);
                    HiddenLayer1.init(BatchSize);
                    BP.layers.Add(HiddenLayer1);

                }
                else
                {
                    int hiddenLayerNeuron = neuronLsit[i];
                    totalParameterCount += lastCount * hiddenLayerNeuron * 2;
                    lastCount = hiddenLayerNeuron;
                    Layer HiddenLayer1 = new Layer(BP.layers.LastOrDefault());
                    HiddenLayer1.ResizeNeuron(hiddenLayerNeuron);
                    HiddenLayer1.init(BatchSize);
                    BP.layers.Add(HiddenLayer1);
                }
            }
            Layer OutputLayer = new Layer(BP.layers.LastOrDefault());
            OutputLayer.ResizeNeuron(OutputSize);
            OutputLayer.init(BatchSize, true);
            BP.layers.Add(OutputLayer);
            totalParameterCount += lastCount * OutputSize * 2;
            #region info
            Console.WriteLine();
            Console.WriteLine("     -- Model Info --");
            Console.WriteLine("     Hidden Layers : " + hiddenLayerCount);
            Console.WriteLine("     Hidden Layer Neurons : " + hiddenLayerCount);
            Console.WriteLine("     Learning Rate : " + LearningRate);
            Console.WriteLine("     Training Count : " + TrainingCount);
            Console.WriteLine("     Testing Count : " + TestingCount);
            Console.WriteLine("     Loop Times(Epoch) : " + TrainingLoopTimes);
            Console.WriteLine("     Batch Size : " + BatchSize);
            Console.WriteLine("     Batch Count : " + TrainingSetBatch.Count);
            Console.WriteLine("     Total Parameters : " + totalParameterCount);
            Console.WriteLine();
            ModelInfo.Add("Model Name : " + ModelName);
            ModelInfo.Add("Hidden Layers : " + hiddenLayerCount);
            ModelInfo.Add("Hidden Layer Neurons : " + hiddenLayerCount);
            ModelInfo.Add("Learning Rate : " + LearningRate);
            ModelInfo.Add("Training Count : " + TrainingCount);
            ModelInfo.Add("Testing Count : " + TestingCount);
            ModelInfo.Add("Loop Times(Epoch) : " + TrainingLoopTimes);
            ModelInfo.Add("Batch Size : " + BatchSize);
            ModelInfo.Add("Batch Count : " + TrainingSetBatch.Count);
            ModelInfo.Add("Total Parameters : " + totalParameterCount);
            #endregion
        }
        public int TrainingCount = 150;
        public int TrainingLoopTimes = 1;
        public int TrainingIndex = 0;
        public int BatchSize = 10;
        public int TrainingNeuronIndex = 0;
        int d = 0;
        public void TrainWithBatch()
        {
            Console.WriteLine(DateTime.Now + " Batch Training Start");
            for (int z = 0; z < TrainingLoopTimes; z++)
            {

                d = 0;
                // Console.WriteLine("Epoch " + z + "/" + TrainingLoopTimes);
                for (int a = 0; a < TrainingCount; a++)
                {
                    Stopwatch sw1 = new Stopwatch();

                    for (int b = 0; b < BatchSize; b++)
                    {
                        TrainingIndex = a * BatchSize + b;
                        TrainActionPerBatch(a, b, z, TrainingIndex);
                    }
                    #region Test
                    //sw1.Restart();
                    //sw1.Start();
                    //sw1.Stop();
                    //Console.WriteLine(" Bacth Training Time: " + sw1.ElapsedMilliseconds + " ms");

                    //Stopwatch sw  = Stopwatch.StartNew();
                    /*sw.Start();
                    Task.WaitAll(tasks.ToArray());
                    sw.Stop();
                    Console.WriteLine(" Wait All time" + sw.ElapsedMilliseconds + " ms");*/
                    #endregion
                 //   Console.WriteLine(DateTime.Now + " SGD ({0}/{1})", a, TrainingCount);
                    for (int i = 1; i < BP.layers.Count; i++)
                    {
                        BP.layers[i].UpdateParameter(1);

                    }
                }
                TrainingIndex = 0;
            }
        }
        public void TrainActionPerBatch(int a, int b, int z, int trainingindex)
        {
            List<double> currentTrainData = new List<double>();
            for (int i = 0; i < 784; i++)
            {
                currentTrainData.Add((double)TrainingSetBatch[a][b][i]);
            }
            Global.Normalize(ref currentTrainData);
            for (int i = 0; i < 784; i++)
            {
                //[0] => input layer
                //acessor 
                BP.layers[0]._neuron[i].z = currentTrainData[i];
            }
            BP._testData = TrainingLabelBatch[a][b];
            //第0層是輸入層 所以不用碰
            for (int i = 1; i < BP.layers.Count; i++)
            {
                BP.layers[i].ForwardPass();
            }
            //第幾梯次做prop
            BP.Prop(b);
            if (TrainingIndex > d)
            {

                var str = (100 * ((double)TrainingIndex / (TrainingCount * BatchSize))).ToString();
                string str1 = "";
                if (str.Length > 5)
                    str1 = str.Substring(0, 5);
                else
                    str1 = str;
                Console.WriteLine("Epoch " + z + "/" + TrainingLoopTimes);
                Console.WriteLine(DateTime.Now + " [+] " + TrainingIndex + " Data Trained (" + str1 + "% ) ");
                double c = BP.Cost();
                Console.WriteLine("    Loss R: " + c + "  Rate: " + ActiveFunctions.Sigmoid(c));
                d += 1000;
            }
            #region Test
             for (int i = 1; i < BP.layers.Count; i++)
            {
                BP.layers[i].ForwardPass();
            }
            /* foreach (var u in BP.layers.LastOrDefault()._neuron)
             {
                 if (u.output < 1E-5)
                     Console.Write(0 + ".0 ");
                 else if (u.x.ToString().Length > 3)
                     Console.Write(u.output.ToString().Substring(0, 3) + " ");
                 else
                     Console.Write(u.output.ToString() + " ");
             }
             Console.WriteLine();*/
            #endregion
        }

        public void Train()
        {
            Console.WriteLine(DateTime.Now + " Training Start");
            int d = 100;
            for (int a = 0; a < TrainingCount; a++)
            {
                for (int z = 0; z < TrainingLoopTimes; z++)
                {
                    TrainingIndex = a;
                    for (int i = 0; i < 784; i++)
                    {
                        //[0] => input layer
                        BP.layers[0]._neuron[i].z = TrainingSet[a][i];
                    }
                    BP._testData = TrainingLabelSet[a];
                    for (int i = 1; i < BP.layers.Count; i++)
                    {
                        BP.layers[i].ForwardPass();
                    }
                    foreach (var i in BP.layers.LastOrDefault()._neuron)
                    {
                        var u = i.x.ToString();
                        //Console.Write(" " + (u).Remove(5,u.Length-5-1));
                    }
                    //Console.WriteLine();
                    // BP.Prop();
                    for (int i = 1; i < 4; i++)
                    {
                        BP.layers[i].UpdateParameter(0.09);
                    }
                    if (a > d)
                    {
                        Console.WriteLine(DateTime.Now + " [+] " + d + " Data Trained (" + 100 * ((double)a / TrainingCount) + "% )");
                        d += 100;
                    }
                }
            }

        }
        public void SettingUpTrainingData()
        {
            string path = Directory.GetCurrentDirectory();
            var imgs = File.ReadAllBytes(path + @"\DataSet\train\train-images.idx3-ubyte");
            var labels = File.ReadAllBytes(path + @"\DataSet\train\train-labels.idx1-ubyte");
            Console.WriteLine("     -- Images --");
            List<string> info = new List<string>()
            {
                "magic number",
                "number of images",
                "number of rows",
                "number of columns",
            };
            //前面都info
            // img data 在 0016 開始 offset 16 
            List<byte> infa = new List<byte>();
            for (int i = 0; i < 16; i++)
            {
                infa.Add(imgs[i]);
            }
            var imgset = imgs.ToList();
            imgset.RemoveRange(0, 16);
            for (int i = 0; i < 4; i++)
            {
                var a = infa.Chunk(4).ToList()[i];
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(a);
                int u = BitConverter.ToInt32(a, 0);
                Console.WriteLine("    " + info[i] + " " + u);
            }
            Console.WriteLine();
            Console.WriteLine("     -- Label --");
            List<string> info1 = new List<string>()
            {
                "magic number",
                "number of items",
            };
            List<byte> infa1 = new List<byte>();
            // label data 在 0008 開始 offset 8
            for (int i = 0; i < 8; i++)
            {
                infa1.Add(labels[i]);
            }
            var labelset = labels.ToList();
            labelset.RemoveRange(0, 8);
            for (int i = 0; i < 2; i++)
            {
                var a = infa1.Chunk(4).ToList()[i];
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(a);
                int u = BitConverter.ToInt32(a, 0);
                Console.WriteLine("    " + info1[i] + " " + u);
            }

            TrainingSet = imgset.Chunk(784).ToList();

            for (int i = 0; i < labelset.Count; i++)
            {
                List<double> CU = new List<double>(10);
                for (int j = 0; j < 10; j++)
                    CU.Add(0);
                CU[labelset[i]] = 1;
                TrainingLabelSet.Add(CU);
                LabelSet.Add(labelset[i]);
            }
            //SaveToimg(BImgSet[0]);

            foreach (var i in TrainingSet.ToList().Chunk(BatchSize))
            {
                TrainingSetBatch.Add(i.ToList());
            }

            foreach (var i in TrainingLabelSet.ToList().Chunk(BatchSize))
            {
                TrainingLabelBatch.Add(i.ToList());
            }

        }
        public int TestingCount = 500;
        public void SaveToimg(byte[] img)
        {
            /*MemoryStream ms = new MemoryStream();
            ms.Write(img, 0, img.Length);
            Bitmap bmp = new Bitmap(28, 28);
            var s = img.Chunk(28).ToList();
            for (int i = 0; i < 28; i++)
            {
                for (int u = 0; u < 28; u++)
                    bmp.SetPixel(i, u, Color.FromArgb(s[i][u]));

            }
            bmp.Save("Write.Bmp", System.Drawing.Imaging.ImageFormat.Bmp);*/
        }
        public void Test()
        {
            Console.WriteLine(DateTime.Now + " Start Testing");
            int offset = TrainingCount * BatchSize; //註: 此的TrainingCount 是指 多少batch
            int correct = 0;
            for (int a = 0 + offset; a < TestingCount + offset; a++)
            {
                for (int i = 0; i < 784; i++)
                {
                    BP.layers[0]._neuron[i].z = TrainingSet[a][i];
                }
                BP._testData = TrainingLabelSet[a];
                for (int i = 1; i < BP.layers.Count; i++)
                {
                    BP.layers[i].ForwardPass();
                }
                if (BP.Result() == LabelSet[a])
                    correct++;
            }
            Console.WriteLine();
            // Console.WriteLine("Loss Rate : " + BP.Cost());
            Console.WriteLine(DateTime.Now + " AC ratio:" + ((double)correct / TestingCount));
        }
    }
}
