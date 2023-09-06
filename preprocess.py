import os
import json
import jieba
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle

# 从THUCNews中提取新闻标题，保存到THUCNews-Title文件夹中
def get_title():
    for class_name in os.listdir('./THUCNews'):
        content = []
        if class_name[0] == '.':continue # 忽略系统文件
        for news in os.listdir(os.path.join('./THUCNews', class_name)):
            news_path = os.path.join(os.path.join('./THUCNews', class_name),news)
            with open(news_path, 'r') as f:
                content.append(f.readline())
        with open('./THUCNews-Title/' + class_name + '.txt', 'w') as f:
            for i in content:
                f.write(class_name + '\t' + i)


# 读取数据集，读取到`data.txt`中，格式为`label + '\t' + content`
def read_dataset() -> None:
    dataset_path = "THUCNews-Title"
    data_txt = open("./data.txt", "w")

    for label in os.listdir(dataset_path):
        data_path = os.path.join(dataset_path, label)
        
        with open(data_path, 'r') as f:
            for line in f.readlines():
                data_txt.write(line)
    
    data_txt.close()

# 数据集划分
def dataset_partition() -> None:
    with open('./data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            # 处理空行
            if len(parts) == 2:
                data.append(parts)

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    with open('./train.txt', 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line[0] + '\t' + line[1] + '\n')
    with open('./test.txt', 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line[0] + '\t' + line[1] + '\n')
    with open('./val.txt', 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(line[0] + '\t' + line[1] + '\n')
 
# 将label转化为id
def label_to_id() -> dict:
    res = []
    for label in os.listdir("./THUCNews-Title"):
        res.append(label.split('.')[0])
    res.sort()
    return dict(zip(res, range(len(res))))

def id_to_label() -> dict:
    res = []
    for label in os.listdir("./THUCNews-Title"):
        res.append(label.split('.')[0])
    res.sort()
    return dict(zip(range(len(res)),res))
    
# 构建词汇表类
class Vocabulary:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_freq = defaultdict(int)
        self.num_words = 0
        self.len_of_text = []

    def add_text(self, text):
        words = jieba.lcut(text)
        for word in words:
            self.word_freq[word] += 1
        self.len_of_text.append(len(words))

    def build_vocab(self):
        sorted_words = sorted(self.word_freq.items(), key=lambda x: (-x[1], x[0]))
        sorted_words = [word for word, freq in sorted_words]
        
        # 添加特殊标记
        self.add_word("<PAD>")
        self.add_word("<UNK>")

        for word in sorted_words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.num_words
            self.id_to_word[self.num_words] = word
            self.num_words += 1

    def text_to_ids(self, text):
        words = jieba.lcut(text)
        ids = []
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id["<UNK>"])
        return ids
    
    def save_vocabulary(self):
        save_path = './vocabulary/'
        with open(save_path + 'word_to_id.pickle', 'wb') as file:
            pickle.dump(self.word_to_id, file)
        with open(save_path + 'id_to_word.pickle', 'wb') as file:
            pickle.dump(self.id_to_word, file)
        with open(save_path + 'word_freq.pickle', 'wb') as file:
            pickle.dump(self.word_freq, file)
        with open(save_path + 'num_words.pickle', 'wb') as file:
            pickle.dump(self.num_words, file)
        with open(save_path + 'len_of_text.pickle', 'wb') as file:
            pickle.dump(self.len_of_text, file)

    def load_vocabulary(self):
        with open('vocabulary/word_to_id.pickle', 'rb') as file:
            self.word_to_id = pickle.load(file)
        with open('vocabulary/id_to_word.pickle', 'rb') as file:
            self.id_to_word = pickle.load(file)
        with open('vocabulary/word_freq.pickle', 'rb') as file:
            self.word_freq = pickle.load(file)
        with open('vocabulary/num_words.pickle', 'rb') as file:
            self.num_words = pickle.load(file)
        with open('vocabulary/len_of_text.pickle', 'rb') as file:
            self.len_of_text = pickle.load(file)

if __name__ == "__main__":
    # get_title()
    read_dataset()
    dataset_partition()
    print(label_to_id())