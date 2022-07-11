// See https://aka.ms/new-console-template for more information
using NerualNetFrame;
Console.WriteLine("ReadModel:");
string path = Directory.GetCurrentDirectory();
Console.WriteLine("-1 : new model");
int i = 0; 
List<string> modellist = new List<string>();
foreach (var r in Directory.GetFiles(path + @"\Model\"))
{
    string h = r.Replace(path+@"\Model\", "");
    modellist.Add(h);
    Console.WriteLine(i+" "+h);
    i++;
}
int u =int.Parse(Console.ReadLine());
NetworkManager model;
if (u == -1)
{
    Console.Write("Model Name: ");
    string modelname = "";
    modelname = Console.ReadLine();
    Console.Write("Loop Times ((Epoch)): ");
    int loopTimes = 0;
    loopTimes = int.Parse(Console.ReadLine());

    Console.Write("Training Count (Times Count): ");
    int trainCount = 0;
    trainCount = int.Parse(Console.ReadLine());

    Console.Write("Batch Size: ");
    int batchCo = 0;
    batchCo = int.Parse(Console.ReadLine());

    Console.Write("Testing Count (Data Count): ");
    int testCount = 0;
    testCount = int.Parse(Console.ReadLine());

    Console.Write("Hidden Layer Count: ");
    int Hdlayer = 0;
    Hdlayer = int.Parse(Console.ReadLine());

    List<int> NeuronList = new List<int>();
    for(int n = 0; n < Hdlayer; n++)
    {
        Console.Write("Hidden Layer [{0}] Neuron: ",n);
       int  neuron = int.Parse(Console.ReadLine());
        NeuronList.Add(neuron);
    }

    Console.Write("Learning Rate: ");
    double lr = 0;
    lr = double.Parse(Console.ReadLine());


    Console.WriteLine(DateTime.Now + " Start Building Model "+ modelname);
    model = new NetworkManager();
    model.ModelName = modelname;
    model.TrainingLoopTimes = loopTimes;
    model.TrainingCount = trainCount;
    model.TestingCount = testCount;
    model.LearningRate = lr;
    model.BatchSize = batchCo;
    model.SettingUpTrainingData();
    model.Initialize(Hdlayer, NeuronList);
    model.TrainWithBatch();
    Console.WriteLine(DateTime.Now+" Training Completed");
    NerualNetFrame.Tools.ModelSaver.SaveModel(model, modelname+".mdl");
    Console.WriteLine(DateTime.Now+" Model : " +modelname+".mdl" +" Saved");
    model.Test();
}
else
{
    model = NerualNetFrame.Tools.ModelSaver.ReadModel<NetworkManager>(modellist[u]);
    foreach(var bv in model.ModelInfo)
    {
        Console.WriteLine("     "+bv);
    }
    model.Test();
}

//manager1.Test();
//NerualNetFrame.Tools.ModelSaver.SaveModel(manager, "model1.mdl");
Console.WriteLine("Press Any Key To Leave....");
Console.Read();


