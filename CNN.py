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

#由于PyTorch没有自带全局的最大池化层，所以通过普通的池化来实现全局池化。
class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab)+1, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 5)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = 2*embed_size, 
                                        out_channels = c, 
                                        kernel_size = k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs), 
            self.constant_embedding(inputs)), dim=2) # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs
    
#创建一个TextCNN实例。它有3个卷积层，它们的核宽分别为3、4和5，输出通道数均为100。
embed_size, kernel_sizes, nums_channels = 300, [3, 4, 5], [100, 100, 100]
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)

#加载预训练的50维GloVe词向量，并分别初始化嵌入层embedding和constant_embedding，前者参与训练，而后者权重固定。
net.embedding.weight.data.copy_(weight)
net.constant_embedding.weight.data.copy_(weight)
net.constant_embedding.weight.requires_grad = False

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
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

#训练模型
lr, num_epochs = 0.001, 1951
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
device='cpu'
train(train_iter, net, loss, optimizer,device, num_epochs)
