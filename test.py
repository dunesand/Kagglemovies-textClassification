#定义预测函数
def predict_sentiment(net, vocab, testFeature):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    testFeature = torch.tensor(testFeature, device=device)
    label = [torch.argmax(net(sentence.view((1, -1))), dim=1) for sentence in testFeature]
    return  label

labels=predict_sentiment(net, vocab, testFeature)

labels=[label.item() for label in labels]
import pandas as pd
data={'phraseId':dataTest['PhraseId'],'sentiment':labels}
test=pd.DataFrame(data)
test.to_csv('.../test.csv',encoding='gbk')
