import pandas as pd
from numpy import *

#加载数据
def loadDataSet():
    fTrain=open('F:\MachineLearning\DATA\Movie_kaggle\_train.tsv')
    dataTrain=pd.read_csv(fTrain,sep='\t')
    fTest=open('F:\MachineLearning\DATA\Movie_kaggle\_test.tsv')
    dataTest=pd.read_csv(fTest,sep='\t')
    return dataTrain,dataTest

#处理数据
#筛选首字母大写，最后字符是“.”的训练数据，即一个完整不重复的句子
def filt(data):
    filData=[]
    for i in range(len(data)):
        if data['Phrase'][i][0].isupper () is True and data['Phrase'][i][-1]=='.':
            filData.append(data.iloc[i])
        else:
            continue
    return pd.DataFrame(filData)

#对筛选出来的训练数据切分文本
def split(preFilData):
    phrase=[word.lower() for word in preFilData['Phrase']]  #将文本全部转换成小写
    itemSplit=[]
    for i in range(len(preFilData)):
        itemSplit.append(phrase[i].split(',') )    #逗号前后不能组词，先用逗号分句
        
    elements=[]
    for i in range(len(itemSplit)):
        for j in range(len(itemSplit[i])):
            elements.append(itemSplit[i][j].split(' ')) #用空格将文本中切分成每个字符串是一个单词的形式
    
    vocabFilt=[]
    for i in range(len(elements)):
        if len(elements[i])<=1:  
            continue            #去掉只有一个单词的向量
        else:
            elementsFilt=[]
            for j in range(len(elements[i])):
                if elements[i][j]=='' or elements[i][j]==',' or elements[i][j]=='.': #去掉无意义的符号和空白符
                    continue
                else:
                    elementsFilt.append(elements[i][j])
        vocabFilt.append(elementsFilt)
    return vocabFilt

def nGramVocab(vocablist,n):
    nGramVocab=[]
    for i in range(len(vocablist)):
        for j in range(len(vocablist[i])):
            nGramVocabItem=' '.join(vocablist[i][j:j+n])  #单词之间以空白符相连
            if nGramVocabItem.count(' ')==n-1:  #只留下有n个单词的元素
                nGramVocab.append(nGramVocabItem)
            else:
                continue
    return list(set(nGramVocab))    #用set()去掉重复元素，得到n元特征词汇表


#处理训练数据
def nGramDoc(data,n):
    trainDoc=list(data['Phrase'])
    trainDocLower=[doc.lower() for doc in trainDoc]  #全部小写
    #切分训练文本
    docItemSplit=[]
    for i in range(len(data)):
        docItemSplit.append(trainDocLower[i].split(' ') )

    docFilt=[]
    for i in range(len(docItemSplit)):
        docElementsFilt=[]
        for j in range(len(docItemSplit[i])):
            if docItemSplit[i][j]=='' or docItemSplit[i][j]==',' or docItemSplit[i][j]=='.': #去掉无意义的符号和空白符
                continue
            else:
                docElementsFilt.append(docItemSplit[i][j])
        docFilt.append(docElementsFilt)
    
    docVocablist=[]
    for i in range(len(docFilt)):
        docItemlist=[]
        for j in range(len(docFilt[i])):
            docItemlist.append(' '.join(docFilt[i][j:j+n]))
        docVocablist.appen`d(docItemlist)
    return docVocablist


dataTrain,dataTest=loadDataSet()
filData=filt(dataTrain)
vocabFilt=split(filData)
biGramVocab=nGramVocab(vocabFilt,2)  #二元特征词表
#triGramVocab=nGramVocab(vocabFilt,3)#三元特征词表

biGramDoc=nGramDoc(dataTrain,2)
#triGramDoc=nGramDoc(dataTrain,3)

#构建词向量
def setofWordVector(Vocablist,trainDocSet):
    Vector=[0]*len(Vocablist)
    for word in trainDocSet:
        if word in Vocablist:
            Vector[Vocablist.index(word)]+=1
        else:
            continue
    return Vector

#训练文本数据转换为矩阵
def matrix(Vocablist,trainDocSet):   
    trainMat=[]
    for postinDoc in trainDocSet:
        trainMat.append(setofWordVector(Vocablist,postinDoc))
    return trainMat

biGramMat=matrix(biGramVocab,biGramDoc)
#triGramMat=matrix(triGramVocab,triGramDoc)

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

#训练权重
weights=gradientDescent(biGramMat,dataTrain['Sentiment'])

