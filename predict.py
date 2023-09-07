from model import BiLSTMClassifier
from preprocess import *
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init():
    global model, idx2label, label2idx, wvm, embedding_matrix
    model = torch.load('pth/epoch_4.pth',map_location=device)
    model.eval()
    idx2label = pickle_load('idx2label')
    label2idx = pickle_load('label2idx')
    wvm = pickle_load('wvm')
    embedding_matrix = pickle_load('embedding_matrix')

def predict(text):
    with torch.no_grad():
        test_outputs = model(torch.LongTensor([text_to_id(text, wvm, 30)]).to(device))
        _, predicted = torch.max(test_outputs, 1)
        predicted = predicted.to('cpu').numpy()
        print(predicted)
        predicted = idx2label[predicted[0]]
    return predicted



