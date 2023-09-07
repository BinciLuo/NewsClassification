import torch
import gradio as gr
from predict import init,predict
from model import BiLSTMClassifier

# 加载模型
init()

# 定义推理函数
infer=predict

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
