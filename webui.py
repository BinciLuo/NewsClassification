import torch
import mdtex2html
import gradio as gr
from predict import init,predict
from model import BiLSTMClassifier

# 加载模型
"""

"""
init()

# 定义推理函数
"""

"""
infer=predict

# GRADIO
"""

"""
def chat(inputs, chatbot):
    infer_text=inputs
    chatbot.append((input, ""))
    infered_label = infer(infer_text)
    chatbot[-1] = (infer_text,infered_label)
    print(infer_text,infered_label,flush=True)       

    yield chatbot, None

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], []

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">新闻标题文本分类</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="输入新闻标题...", lines=10,container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")

    history = gr.State([])

    submitBtn.click(chat, [user_input, chatbot], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0',server_port=27777,debug=True)