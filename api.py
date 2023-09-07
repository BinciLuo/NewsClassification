import torch
import gradio as gr

# 加载模型
"""

"""


# 定义推理函数
"""

"""
def infer(infer_text):
    infered_label=""
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
