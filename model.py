import pandas as pd
from numpy import *

#读取数据
f_train=open('...\_train.tsv')
dataTrain=pd.read_csv(f_train,sep='\t')
f_test=open('F:\MachineLearning\DATA\Movie_kaggle\_test.tsv')
data_test=pd.read_csv(f_test,sep='\t')

#处理数据
#筛选出只有一个单词的数据
filDataSet=[]
def filt(dataTrain):
    for i in range(len(dataTrain)):
        phrWord=dataTrain['Phrase'][i].split()
        if len(phrWord)==1:
            filDataSet.append(dataTrain.loc[i])
        else:
            continue
    return pd.DataFrame(filDataSet)
bagDataSet=filt(dataTrain)

#建立词表
def createVocablist(dataSet):
    vocablist=set(dataSet)
    return list(vocablist)
wordSet=list(bagDataSet['Phrase'])  #所有的单词
wordSetLower=[word.lower() for word in wordSet]  #将字符串转换成小写
vocabList=createVocablist(wordSetLower)   #得到词表

#处理训练数据
trainDocSet=list(dataTrain['Phrase'])
trainDocSetLower=[doc.lower() for doc in trainDocSet]  #全部小写
#切分文本
trainDocSplit=[]
for document in trainDocSetLower:
    trainDocSplit.append(document.split())

#构建词向量，得到基于词袋模型特征
def setofWordVector(vocabList,wordSetLower):
    returnVec=[0]*len(vocabList)
    for word in wordSetLower:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#得到特征向量
trainMat=[]
for document in trainDocSplit:
    trainMat.append(setofWordVector(vocabList,document))

#定义softmax函数
def softmax(weights,docMat): #如果X是n维1列文档矩阵，即词向量，W则是c行n列的权重矩阵，c是类别，这里是5，W需初始化
    sentiments=[]
    for i in range(len(docMat)):
        sentiment=[]
        for j in range(len(weights)):
            pSoft=(exp(dot(weights[j],docMat[i])))/sum(exp(dot(weights,docMat[i])))
        sentiment.append(pSoft)
        sentiments.append(max(sentiment))    
    return sentiments

#mini-batch梯度下降法
def`miniBatchGradDescent():
    for i in range(578):  #分成578个mini-batch，每个batch中270条训练数据
        m,n=shape(trainMat[i:i+270]) #每次只迭代270条训练数据
        labels=list(dataTrain['Sentiment'][i:i+270])
        weights=ones((5,270)) #初始化权重
        alpha=0.01
        maxCycles=500
        for j in range(maxCycles):
            y=softmax(weights,trainMat[i:i+270])
            error=[labels-y for labels,y in zip(labels,y)]
            weights=weights+alpha*trainMat[i,i+270].transpose()*error  #trainMat.transpose()在这里是梯度下降函数的参数
        return weights
    return weights

