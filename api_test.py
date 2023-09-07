import requests,json
url="http://localhost:27778/predict/"

samples=['专访英国前首相布莱尔夫人14岁理想成为女首相',
         '陈奕迅4月在沪再开唱 自称有“上海情结”(图)',
         '演艺圈“爱”仍在继续 网友纪念视频引爆点击率',
         '私募基金产品首季回报率两极相差3.35倍']
for text in samples:
    data={'text': text}
    response=requests.post(url, data=json.dumps(data))
    print({
        'text':text,
        'response_status' :response.status_code,
        'response_text' :response.text
    })