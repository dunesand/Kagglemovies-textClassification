import pandas as pd
from numpy import *

def loadDataSet():
    fTrain=open('F:\MachineLearning\DATA\Movie_kaggle\_train.tsv')
    dataTrain=pd.read_csv(fTrain,sep='\t')
    fTest=open('F:\MachineLearning\DATA\Movie_kaggle\_test.tsv')
    dataTest=pd.read_csv(fTest,sep='\t')
    return dataTrain,dataTest

def clean_text(x):
    #去掉无意义的符号
    punts=[',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    #将test_sentence中的punts替换成空格
    x=str(x)
    for X in x:
        if X in punts:
            x=x.replace(X,' ')
    return x

#建立词表
def builtVocab(data):
    Vocab=set([])
    texts=[]
    for i in range(len(data)):
        dataLower=data['Phrase'].loc[i].lower()  #转换为小写
        data_clean=clean_text(dataLower)
        text=data_clean.split()#每行文本words
        Vocab=Vocab|set(text)
        texts.append(text)
    return texts,Vocab

dataTrain,dataTest=loadDataSet()
trainText,trainVocab=builtVocab(dataTrain)
testText,testVocab=builtVocab(dataTest)
vocab=set(trainVocab|testVocab)
Vocab=list(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0]='<unk>'


#准备好进入模型的数据 (例如将单词转换成整数索引,并将其封装在变量中)    
import torch
import torch.nn as nn
def preprocess_imdb(texts):
    #max_1=500    # 将每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        if len(x) > 30:
            return x[:30]  
        else:
            return x+[0] * (30 - len(x))
                                                                                                                                                                              
    textIds=[]
    for text in texts:
        #print(type(text))
        #textId=[word_to_idx[w] for w in text]
        text=tuple(text)
        textId=[]
        for w in text:
            wordId=word_to_idx[w]
            textId.append(wordId)
        a=pad(textId)
        textIds.append(a)
    #return textIds
    feature=torch.tensor(textIds)     
    #labels=torch.tensor(data['Sentiment'] )
    return feature
        
trainFeature=preprocess_imdb(dataTrain)
trainLabel=torch.tensor(dataTrain['Sentiment'] )
testFeature=preprocess_imdb(testText)

#加载glove
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# 已有的glove词向量
glove_file = datapath('glove.6B.300d.txt')
#print(glove_file)
# 指定转化为word2vec格式后文件的位置
tmp_file = get_tmpfile("text_word2vec.txt")
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
print(tmp_file)

import gensim
#加载glove
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/winner/AppData/Local/Temp/text_word2vec.txt', 
                                                          binary=False, encoding='utf-8')

vocab_size = len(vocab) + 1
embed_size = 300
weight = torch.zeros(vocab_size, embed_size)

for i in range(1,len(vocab)):
    if idx_to_word[i] in wvmodel.index2word:
        weight[i]=torch.from_numpy(wvmodel.get_vector(idx_to_word[i]))
    else:weight[i]=torch.randn(embed_size)
print(weight)    #预训练的初始词向量

#创建数据迭代器
import torch.utils.data as Data
batch_size = 64
train_set = Data.TensorDataset(trainFeature,trainLabel)
#test_set = Data.TensorDataset(testFeature, vocab)  #可能要从训练集中抽出一部分做测试集
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
#test_iter = Data.DataLoader(test_set, batch_size)

#由于PyTorch没有自带全局的最大池化层，所以类似5.8节我们可以通过普通的池化来实现全局池化。
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
         # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])

class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 5)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 300, 50, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

#加载预训练的词向量
net.embedding.weight.data.copy_(weight)
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

def train(train_iter, net, loss, optimizer, device, num_epochs):
    import time
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        #test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec' 
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

        #训练并评价模型
lr, num_epochs = 0.01, 5  #这里暂时只运行5个epoch
device='cpu'
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, net, loss, optimizer, device, num_epochs)
