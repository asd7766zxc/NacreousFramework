using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace NacreousFramework.CKIP
{
    public class CKIPClient
    {
        HttpClient? client;

        public bool Initialize()
        {
            return true;
        }
        public string? sendText(string text)
        {
            client = new HttpClient();
            HttpRequestMessage request = new HttpRequestMessage();
            request.Method = HttpMethod.Post;
            CKIPConfig config = new CKIPConfig();
            config.sentence_list = "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。\n美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。\n土地公有政策?？還是土地婆有政策。.\n… 你確定嗎… 不要再騙了……\n最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.\n科長說:1,坪數對人數為1:3。2,可以再增加。\n";
            config.use_recommend = false;
            config.recommend_dictionary = "仁今 1\n緯來體育台 1\n";
            config.use_coerce = false;
            config.coerce_dictionary = "土地公 1\n土地婆 1\n公有 2\n";
            config.use_delimeter = false;
            config.segment_delimiter_set = "";
            config.run_pos = true;
            config.run_ner = true;
            Console.WriteLine(config.ToString());
            request.RequestUri = new Uri("https://ckip.iis.sinica.edu.tw/service/ckiptagger/run_ckiptagger");
            request.Content = new StringContent(config.ToString(), Encoding.UTF8,"application/json");
            var res = client.Send(request);
            Result? r = JsonConvert.DeserializeObject<Result>(new StreamReader(res.Content.ReadAsStream(), Encoding.UTF8).ReadToEnd());
            return r == null ?  "NULL" : r.result;
        }
        public void SequenceString(ref string result, string[] input)
        {
            foreach (var item in input)
            {
                result += item + "\n";
            }
        }
    }
    public class Result
    {
        public string? result { get; set; }
    }
    public class CKIPConfig
    {
        public string? sentence_list { get; set; }
        public bool use_recommend { get; set; }
        public string? recommend_dictionary { get; set; }
        public bool use_coerce { get; set; }
        public string? coerce_dictionary { get; set; }
        public bool use_delimeter { get; set; } 
        public string? segment_delimiter_set    { get; set; }
        public bool run_pos { get; set; }
        public bool run_ner { get; set; }
        public override string ToString()
        => JsonConvert.SerializeObject(this);
       
    }
}
