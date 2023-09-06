import torch
from preprocess import label_to_id,id_to_label
from preprocess import Vocabulary
import mdtex2html
import gradio as gr

# 构建词汇表
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

def predict(inputs, chatbot):
    infer_text=inputs
    chatbot.append((input, ""))
    inputs=text_to_id(inputs,20)
    with torch.no_grad():
        inputs=inputs.to(device)
        preds = model(inputs)
        all_preds=[preds]
        all_preds = torch.cat(all_preds, dim=0)
        print(inputs,all_preds)
        label_idx=all_preds.argmax(dim=1)
        chatbot[-1] = (infer_text,id2label[int(label_idx)])
        print((infer_text,id2label[int(label_idx)]))       

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
                user_input = gr.Textbox(show_label=False, placeholder="输入新闻标题...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0',server_port=27777)