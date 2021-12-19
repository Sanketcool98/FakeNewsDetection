import pandas as pd
import re
import string

LEN = 2

# Fetching the raw true news data from the csv file
print("Fetching the dataset for true news")
TrueNews = pd.read_csv('C:/Users/Owner/Desktop/Project/Datasets/True.csv')

# Fetching the raw fake news data from the csv file
print("Fetching the dataset for the fake news")
FakeNews = pd.read_csv('C:/Users/Owner/Desktop/Project/Datasets/Fake.csv')

TrueNews["label"] = 1
FakeNews["label"] = 0
MergeDF = pd.concat([TrueNews, FakeNews], axis=0)
UsedDF = MergeDF.drop(["title", "subject", "date"], axis=1)
UsedDF = UsedDF.sample(frac=1)
UsedDF.reset_index(inplace=True)
UsedDF.drop(["index"], axis=1, inplace=True)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # text = re.sub('[0-9]','', text)
    return text


UsedDF["text"] = UsedDF["text"].apply(wordopt)
UsedDF.to_csv("C:/Users/Owner/Desktop/Project/Datasets/Data.csv")
