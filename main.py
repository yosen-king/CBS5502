# 1、处理数据
import os
import copy
from collections import defaultdict
import jieba


data_dict = defaultdict(list)
single_data_dict = defaultdict(list)
path = 'data/train'
root_dirs = os.listdir(path)


for dir_ in root_dirs:
    fir_dir = os.path.join(path, dir_)
    for root, sec_dir, files in os.walk(fir_dir):
        for file in files:
            file_abs_path = os.path.join(root, file)
            try:
                data = open(file_abs_path, 'r', encoding='gbk')
                for line in data:
                    line = line.replace('$LOTOzf$', '')
                    if len(line) < 80:
                        single_data_dict[dir_].append(line)
                        continue
                    fir_sent = line[:len(line) // 2]
                    sec_sent = line[len(line) // 2:]
                    data_dict[dir_].append(copy.deepcopy([fir_sent, sec_sent]))
            except:
                print(file_abs_path)
                continue          

import random

    
def cut_words(text):
    return list(jieba.cut(text))
    
    
classify_dict = ['文学', '体育', '女性', '校园']
writer = open('data/semantic_disambiguation_lstm.txt', 'w')
temp_dict = {}

for k, v in data_dict.items():
    temp = ['文学', '体育', '女性', '校园']
    cur_classify = classify_dict.index(k)
    temp.pop(cur_classify)
    other_data = [d for i, data in data_dict.items() if i != k for da in data for d in da]
    for sents in v:
        fir_sent = sents[0]
        sec_sent = sents[1]
        single_data_dict[k].extend([fir_sent, sec_sent])
        random_index = random.randint(0, len(single_data_dict[k])-1)
        
        random_index2 = random.randint(0,len(other_data)-1)
        neg_fir_sent = single_data_dict[k].pop(random_index)
        neg_sec_sent = other_data[random_index2]
        
        fir_sent = cut_words(fir_sent)
        sec_sent = cut_words(sec_sent)
        neg_fir_sent = cut_words(neg_fir_sent)
        neg_sec_sent = cut_words(neg_sec_sent)
        
        writer_str = f'{k}\t{fir_sent}\t{sec_sent}\t{1}\n'
        writer.write(writer_str)
        writer_str = f'{k}\t{neg_fir_sent}\t{neg_sec_sent}\t{0}\n'
        writer.write(writer_str)

writer.close()   
    
## 2、处理词向量,作词向量字典
import numpy as np
import torch 

vec_path = 'word_vector/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
vector_dict = {}
with open(vec_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split()
        vector_dict[line[0]] = np.array(line[1:])  

import random


data_set = open('data/semantic_disambiguation_lstm.txt', 'r').readlines()
random.shuffle(data_set)

data_array = []
for d in data_set:
    try:
        temp_data = d.split('\t')
        a = eval(temp_data[1])
        b = eval(temp_data[2])
        lable = int(temp_data[-1].strip('\n'))
        vec_a = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in a if vector_dict.get(word) is not None], axis=1)
        vec_b = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in b if vector_dict.get(word) is not None], axis=1)
    except:
        continue
    data_array.append([vec_a, vec_b, lable])

test_set = data_array[:len(data_array) // 7]
train_set = data_array[len(data_array) // 7:]

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123) #保证每次运行初始化的随机数相同

# vocab_size = 5000   #词表大小
embedding_size = 300   #词向量维度
num_classes = 2    #二分类
# sentence_max_len = 64  #单个句子的长度
hidden_size = 256

num_layers = 1  #一层lstm
num_directions = 2  #双向lstm
lr = 1e-3
batch_size = 1  
epochs = 6

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#Bi-LSTM模型
class RNNModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(RNNModel, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x):
        #lstm的输入维度为 [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        
        #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)
        
        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        #out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        
        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        # [batch, 1, hidden_size]  * [batch, seq_len, hidden_size]
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len],权重归一化
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x
     
        
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas in test_loader:
        inputs_0 = torch.from_numpy(datas[0]).float()
        inputs_1 = torch.from_numpy(datas[1]).float()
        if datas[2] == 1:
            labels = torch.LongTensor([0, 1]).unsqueeze(0).float()
        else:
            labels = torch.LongTensor([1, 0]).unsqueeze(0).float()
        inputs = torch.cat([inputs_0, inputs_1], dim=1).permute(1, 0).unsqueeze(0) # len x 300
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        preds = model(inputs)
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * inputs.size(0)
        
        #获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / len(test_loader)
    test_acc = corrects / len(test_loader)
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc


def train(model, train_loader,test_loader, optimizer, loss_func, epochs, model_save_path):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    train_loss = 0
    step = 0
    total_train_loss = []
    total_eval_loss = []
    mean_loss = []
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        
        for datas in train_loader:
            inputs_0 = torch.from_numpy(datas[0]).float()
            inputs_1 = torch.from_numpy(datas[1]).float()
            if datas[2] == 1:
                labels = torch.LongTensor([0, 1]).unsqueeze(0).float()
            else:
                labels = torch.LongTensor([1, 0]).unsqueeze(0).float()
            inputs = torch.cat([inputs_0, inputs_1], dim=1).permute(1, 0).unsqueeze(0) # len x 300
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs)
            loss = loss_func(preds, labels)
            step += 1
            train_loss += loss.mean().item()
            if step % 10 == 0:
                mean_loss.append(train_loss / step)
                print(f'Epoch {epoch}/ {step} train_loss  {train_loss / step}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item() * inputs.size(0)
            total_train_loss.append(loss.item())            
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader)
        train_acc = corrects / len(train_loader)
        
        if(epoch % 1 == 0):
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            test_acc = test(model, test_loader, loss_func)
            if(best_val_acc < test_acc):
                torch.save(model.state_dict(), model_save_path)
                print(f'保存模型成功! 路径为: ’{model_save_path}‘')
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    
    return model, total_train_loss, mean_loss


def predict(model, origin, entity_descs:list):
    """
    origin: 原始文本
    entity_descs:原始文本中具有歧义的实体词的多个异项的相关描述放到entity_descs中。
    比如：origin：苹果12马上就要发布了。
    歧义实体为”苹果“
    entity_descs：
    ['《苹果》是由李玉执导，范冰冰、佟大为、梁家辉、金燕玲领衔主演的黑色幽默剧情电影。于2007年5月18日在中国大陆上映。影片讲述了两个贫富家庭之间离奇诡异的矛盾冲突和感情错位的戏剧性故事。'
, 'iPhone 12是美国苹果公司研发的iPhone手机，采用了直面边框设计，支持5G，搭载A14 Bionic芯片，双镜头后置摄像头系统。支持北斗导航 [1]  ，有黑色、白色、红色、绿色、蓝色五种配色。'
]
    entity_descs中的句子分别和origin做相似度得分，得分高的为正确意项
    """
    scores = []
    for entity_desc in entity_descs:
        origin_vector = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in origin if vector_dict.get(word) is not None], axis=1)
        origin_vector = torch.from_numpy(origin_vector).float()
        entity_desc_vector = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in entity_desc if vector_dict.get(word) is not None], axis=1)
        entity_desc_vector = torch.from_numpy(entity_desc_vector).float()
        inputs = torch.cat([origin_vector, entity_desc_vector], dim=1).permute(1, 0).unsqueeze(0) # bsz x len x 300
        inputs = inputs.to(device)
        preds = model(inputs)
        score = preds.view(-1).tolist()[1]
#         print(preds)
        scores.append(score)
    scores = torch.tensor(scores)
    index = scores.argmax()
    return entity_descs[index]

model = RNNModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
model, total_train_loss, mean_loss = train(model, train_set, test_set, optimizer, loss_func, epochs, model_save_path='model/word_sense_disambiguation.bin')

# origin = '苹果12手机马上就要发布了。'
entity_descs = ['《苹果》是由李玉执导，范冰冰、佟大为、梁家辉、金燕玲领衔主演的黑色幽默剧情电影。于2007年5月18日在中国大陆上映。影片讲述了两个贫富家庭之间离奇诡异的矛盾冲突和感情错位的戏剧性故事。'
, 'iPhone 是美国苹果公司研发的iPhone手机，采用了直面边框设计，支持5G，搭载A14 Bionic芯片，双镜头后置摄像头系统。支持北斗导航 [1]  ，有黑色、白色、红色、绿色、蓝色五种配色。',
'苹果，又称柰或林檎，是苹果 树（学名：Malus domestica）的果实。苹果树是蔷薇科苹果亚科苹果属植物，为落叶乔木，果实富含矿物质和维生素，是人们最常食用的水果之一。'
]

origin = '我们家苹果树上的的苹果果实快要熟了，你要不要来几斤？'
# entity_descs = ['《苹果》是由李玉执导，范冰冰、佟大为、梁家辉、金燕玲领衔主演的黑色幽默剧情电影。于2007年5月18日在中国大陆上映。影片讲述了两个贫富家庭之间离奇诡异的矛盾冲突和感情错位的戏剧性故事。'
# , 'iPhone 是美国苹果公司研发的iPhone手机，采用了直面边框设计，支持5G，搭载A14 Bionic芯片，双镜头后置摄像头系统。支持北斗导航 [1]  ，有黑色、白色、红色、绿色、蓝色五种配色。',
#  '苹果，又称柰或林檎，是苹果 树（学名：Malus domestica）的果实。苹果树是蔷薇科苹果亚科苹果属植物，为落叶乔木，果实富含矿物质和维生素，是人们最常食用的水果之一。'                    
# ]
right_text = predict(model, origin, entity_descs)
print(right_text)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
import torch

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for datas in test_loader:
            inputs_0 = torch.from_numpy(datas[0]).float()
            inputs_1 = torch.from_numpy(datas[1]).float()
            label = datas[2]
            
            inputs = torch.cat([inputs_0, inputs_1], dim=1).permute(1, 0).unsqueeze(0)  # len x 300
            inputs = inputs.to(device)
            
            preds = model(inputs)
            pred_label = torch.argmax(preds, dim=1).item()
            
            all_preds.append(pred_label)
            all_labels.append(label)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"模型评测结果：")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return acc, precision, recall, f1

def evaluate_traditional_method(test_loader, vector_dict):
    """ 使用词向量 + 余弦相似度进行分类 """
    all_preds = []
    all_labels = []
    
    for datas in test_loader:
        a = datas[0]
        b = datas[1]
        label = datas[2]
        
        # 计算余弦相似度
        try:
            vec_a = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in a if vector_dict.get(word) is not None], axis=1)
            vec_b = np.concatenate([vector_dict.get(word).reshape(300, -1).astype(float) for word in b if vector_dict.get(word) is not None], axis=1)
            similarity = 1 - cosine(vec_a.mean(axis=1), vec_b.mean(axis=1))  # 计算两个句子的平均词向量
            pred_label = 1 if similarity > 0.5 else 0  # 设定阈值
        except:
            pred_label = 0
        
        all_preds.append(pred_label)
        all_labels.append(label)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"传统方法（词向量 + 余弦相似度）评测结果：")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return acc, precision, recall, f1

# 运行评测
print("=== 评测基于 Bi-LSTM + Attention 的模型 ===")
model_acc, model_prec, model_rec, model_f1 = evaluate_model(model, test_set)

print("\n=== 评测基于 词向量 + 余弦相似度 的传统方法 ===")
trad_acc, trad_prec, trad_rec, trad_f1 = evaluate_traditional_method(test_set, vector_dict)

# 对比结果
print("\n=== 对比评测结果 ===")
print(f"{'方法':<30}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-score':<10}")
print(f"{'Bi-LSTM + Attention':<30}{model_acc:.4f}{model_prec:.4f}{model_rec:.4f}{model_f1:.4f}")
print(f"{'词向量 + 余弦相似度':<30}{trad_acc:.4f}{trad_prec:.4f}{trad_rec:.4f}{trad_f1:.4f}")