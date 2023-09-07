import torch
from preprocess import label_to_id,id_to_label
from preprocess import Vocabulary
import mdtex2html
import gradio as gr

# 加载模型
"""

"""
vocab = Vocabulary()
vocab.load_vocabulary()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_path='./res_model/model_epoch8.pth'
model=torch.load(load_path,map_location=device)
model.eval()

id2label=id_to_label()

def text_to_id(text, max_len):
    text_ids = vocab.text_to_ids(text)
    lenx = len(text_ids)
    if lenx < max_len: text_ids += [0 for _ in range(max_len - lenx)]
    else: text_ids = text_ids[0:max_len]
    return torch.tensor([text_ids])


# 定义推理函数
"""

"""
def infer(infer_text):
    inputs=text_to_id(infer_text,20)
    with torch.no_grad():
        inputs=inputs.to(device)
        preds = model(inputs)
        all_preds=[preds]
        all_preds = torch.cat(all_preds, dim=0)
        label_idx=all_preds.argmax(dim=1)
        infered_label = id2label[int(label_idx)]
        print(infer_text,infered_label,flush=True)
        return infered_label


# API
"""

""" 

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn

app = FastAPI()

# 输入数据模型
class NewsInput(BaseModel):
    text: str

# 输出数据模型
class NewsOutput(BaseModel):
    news_type: str

@app.post("/predict/", response_model=NewsOutput)
async def predict_news_type(news_input: NewsInput):
    predicted_news_type = infer(news_input.text)
    
    return {"news_type": predicted_news_type}

uvicorn.run(app, host="0.0.0.0", port=27778)
