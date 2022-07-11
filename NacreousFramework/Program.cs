// See https://aka.ms/new-console-template for more information
using NacreousFramework.CKIP;
using NacreousFramework.Helpers;
using Newtonsoft.Json;
using System.Text;

//Console.WriteLine("Hello, World!");
//MLTest mt = new MLTest();
//mt.Run();

CKIPClient cc = new CKIPClient();
var r = cc.sendText("耶");

Console.WriteLine(r);
