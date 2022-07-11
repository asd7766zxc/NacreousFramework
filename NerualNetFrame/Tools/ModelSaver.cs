using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization.Formatters.Binary;

namespace NerualNetFrame.Tools
{
    public static class ModelSaver
    {
        public static void SaveModel<T>(T Model,string name)
        {
            Console.WriteLine(DateTime.Now+" Start Saving...");
            try
            {
                string path = Directory.GetCurrentDirectory();
                WriteToBinaryFile<T>(path + @"\Model\" + name, Model);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Saving Failed : " + ex.Message);
            }
            
        }
        public static T ReadModel<T>(string name)
        {
            string path = Directory.GetCurrentDirectory();
           return ReadFromBinaryFile<T>(path + @"\Model\" + name);

        }
        public static void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
        {
            using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
            {
                var binaryFormatter = new BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }
        public static T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new BinaryFormatter();
                return (T)binaryFormatter.Deserialize(stream);
            }
        }
    }
}
