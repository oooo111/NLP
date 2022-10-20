import nltk
nltk.download()

# 获取emma全文
from nltk.corpus import gutenberg
import pandas as pd
emma_text=gutenberg.raw("austen-emma.txt")
dic={"text":emma_text}
data=pd.DataFrame(dic,index=[0])

## 文本预处理
import re
import string
#删除url 统一资源定位符
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', str(text))

##### 去除html标签
def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', str(text))

#### 删除标点符号
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)  #string.punctuation 返回所有标点符号集合  maketrans 为字典或映射表
    return text.translate(table)  # 该translate()方法返回一个字符串，其中某些指定字符被替换为字典或映射表中描述的字符。

## 去除停用词
from nltk.corpus import stopwords, wordnet
from wordcloud import WordCloud, STOPWORDS
nltk.download('stopwords')
stop = set(stopwords.words('english'))
clean_text=[]
for word in data["text"].values:
    if word not in stop:
        clean_text.append(word)
    else:
        continue

data["clean_text"]=clean_text
data["clean_text"]=data["clean_text"].apply(lambda x : remove_URL(x))
data["clean_text"]=data["clean_text"].apply(lambda x : remove_html(x))
data["clean_text"]=data["clean_text"].apply(lambda x : remove_punct(x))


## 转化为词向量
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None, 
                             preprocessor = None,
                             stop_words = None,  
                             max_features = 5000)
train_data_features = vectorizer.fit_transform(data["clean_text"])


train_data_features = train_data_features.toarray()


vocab = vectorizer.get_feature_names()

print(train_data_features)
print(vocab)