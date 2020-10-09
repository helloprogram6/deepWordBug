import pandas as pd
import numpy as np

path = "D:\\yanjiu\\datasets"
#'../textdata_small/imdb_csv/test.csv'
def trainToCsv():
    imdb_df = pd.read_csv(path + "\imdb_master.csv", encoding='latin-1')
    imdb_df.review = imdb_df.apply(lambda x: x.review.lower(), axis=1)  # review 小写

    traindata = imdb_df.copy()
    traindata = traindata[(traindata.type == 'train') & (traindata.label != 'unsup')]
    traindata.label = np.where(traindata.label == "neg", 1, 2)  # 标签转为 数字

    traindata[['label','review']] = traindata[['review','label']] # 交换类别和内容的顺序
    traindata = traindata.loc[:, ["review", "label"]]
    traindata.to_csv(path + "\\imdb_csv\\train.csv", encoding='utf-8',index=False, header=False)

def testToCsv():
    imdb_df = pd.read_csv(path + "\imdb_master.csv", encoding='latin-1')
    imdb_df.review = imdb_df.apply(lambda x: x.review.lower(), axis=1)  # review 小写

    testdata = imdb_df.copy()
    testdata = testdata[(testdata.type == 'test') & (testdata.label != 'unsup')]
    testdata.label = np.where(testdata.label == "neg", 1, 2)  # 标签转为 数字
    testdata[['label', 'review']] = testdata[['review', 'label']] # 交换类别和内容的顺序
    testdata = testdata.loc[:, ["review", "label"]]
    testdata.to_csv(path + "\\imdb_csv\\xin\\test.csv", encoding='utf-8',index=False, header=False)


def toSmallTrainCsv():
    smallTrainData = pd.read_csv(path + "\\imdb_csv\\xin\\train.csv",nrows=5000)
    smallTrainData.to_csv(path + "\\imdb_csv\\train.csv", encoding='utf-8',index=False, header=False)

def toSmallTestCsv():
    smallTestData = pd.read_csv(path + "\\imdb_csv\\xin\\test.csv", nrows=5000)
    smallTestData.to_csv(path + "\\imdb_csv\\test.csv", encoding='utf-8',index=False, header=False)

if __name__ == "__main__":
    testToCsv()
    #toSmallTrainCsv()
    #oSmallTestCsv()
    #with open(path + "\\imdb_csv\\dd.xls") as f:
    #    print(f.encoding)