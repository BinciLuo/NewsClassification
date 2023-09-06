import datetime
import requests
import time
import json
import os



request_classes=["society","law","ent","tech","life","economy_zixun","edu"]

for request_class in request_classes:
    titles_hans=[]
    for i in range(10):
        try:
            ret=requests.get(url='https://news.cctv.com/2019/07/gaiban/cmsdatainterface/page/'+request_class+'_'+str(i+1)+'.jsonp?cb=society')
            ret_json_str=ret.text.encode('ISO-8859-1').decode('utf-8')[len(request_class)+1:-1]
            ret_json=json.loads(ret_json_str)["data"]["list"]
            for each in ret_json:
                titles_hans.append(each["title"])
        except:
            pass

    try:
        with open('/root/python/intel-internet-experiment/data/'+request_class+".txt","r+",encoding="utf-8") as f:
            current_datas_t=f.readlines()
            current_datas=[line.replace('\n','') for line in current_datas_t]
            for line in titles_hans:
                if line not in current_datas:
                    print(f"    [{request_class}] New append : {line}")
                    f.write(line+"\n")
    except:
        pass
        #with open('/root/python/intel-internet-experiment/data/'+request_class+".txt","w",encoding="utf-8") as f:
        #    for line in titles_hans:
        #        f.write(line+"\n")
localtime = time.asctime( time.localtime(time.time()) )            
print("finished in ",localtime)
