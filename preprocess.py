# %%
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# %%
def load_data(data_dir):
    data = []
    labels = []

    # 遍历data目录下的所有txt文件
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if category_path.endswith(".txt"):
            with open(category_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                lines = [line.split('\t')[-1].strip() for line in lines]
                data += lines
                labels += [category[:-4]] * len(lines)

    return data, labels

# %%
def split_train_test_data(data, labels, test_size=0.2, random_state=42):
    # 划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    return train_data, test_data, train_labels, test_labels

# %%
def pickle_dump(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# %%
def create_vector(train_data, train_labels, vector_size):
    sentences = [list(line) for line in train_data]
    word2vec_model = gensim.models.Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, sg=0)
    word2vec_model.wv['<UNK>'] = np.zeros((vector_size))
    embedding_matrix = np.zeros((len(word2vec_model.wv.index_to_key), word2vec_model.vector_size))
    for i, word in enumerate(word2vec_model.wv.index_to_key):
        embedding_matrix[i] = word2vec_model.wv[word]
    return word2vec_model

# %%
def text_to_vector(text, word2vec_model, max_length):
    words = list(text)  # 将文本拆分为单词
    vector = []
    for word in words:
        if word in word2vec_model.wv:
            vector.append(word2vec_model.wv[word])  # 如果单词在Word2Vec模型中存在，将其词向量添加到序列中
        else:
            vector.append(word2vec_model.wv['<UNK>'])  # 否则，使用<UNK>标记的向量代替未登录词
    # 填充或截断向量以适应固定长度
    if len(vector) < max_length:
        vector.extend([word2vec_model.wv['<UNK>']] * (max_length - len(vector)))  # 填充向量
    else:
        vector = vector[:max_length]  # 截断向量以达到最大长度
    return vector


# %%
def text_to_id(text, word2vec_model, max_length):
    words = list(text)  # 将文本拆分为单词
    vector = []
    for word in words:
        if word in word2vec_model.wv:
            vector.append(word2vec_model.wv.key_to_index[word])  # 如果单词在Word2Vec模型中存在，将其词向量添加到序列中
        else:
            vector.append(word2vec_model.wv.key_to_index['<UNK>'])  # 否则，使用<UNK>标记的向量代替未登录词
    # 填充或截断向量以适应固定长度
    if len(vector) < max_length:
        vector.extend([word2vec_model.wv.key_to_index['<UNK>']] * (max_length - len(vector)))  # 填充向量
    else:
        vector = vector[:max_length]  # 截断向量以达到最大长度
    return vector


