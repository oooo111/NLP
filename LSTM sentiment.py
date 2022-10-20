import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')
import numpy as np # linear algebra
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# load data
train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')
train.shape, test.shape


# plot the sentiment in train
train['Sentiment'].value_counts().plot.bar()



################################## 文本清洗函数
def data_preprocessing(df):
    reviews = []
    for raw in tqdm(df['Phrase']):
        # remove html tag
        text = BeautifulSoup(raw, 'lxml').get_text()
        # remove non-letters
        letters_only = re.sub('[^a-zA-Z]', ' ', text)
        # split(lowercase)
        words = word_tokenize(letters_only.lower())
        # get stoplist words
        stops = set(stopwords.words('english'))
        # remove stopwords / get non-stopwords list
        non_stopwords = [word for word in words if not word in stops]
        # lemmatize word to its lemma
        lemma_words = [lemmatizer.lemmatize(word) for word in non_stopwords]    
        reviews.append(lemma_words)
    return reviews


# data cleaning for train and test
%time train_sentences = data_preprocessing(train)
%time test_sentences = data_preprocessing(test)
len(train_sentences), len(test_sentences)



from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


# extract label columns and to_categorical
target = train.Sentiment.values
y_target = to_categorical(target)
num_classes = y_target.shape[1]

# train set => split to train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_sentences, y_target, test_size=0.2, stratify=y_target)


# keras tokenzier initialization
unique_words = set()  #稀有词列表  
len_max = 0
for sent in tqdm(X_train):
    unique_words.update(sent) # 将未出先过的元素添加到集合当中  
    if len_max < len(sent):
        len_max = len(sent)
len(list(unique_words)), len_max



# transfer to keras tokenizer
tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train) #将文本中的每个文本转换为整数序列编码。
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)

X_train = sequence.pad_sequences(X_train, maxlen=len_max) #将序列进行填充 扩充矩阵 max_len 表示最大列数
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)

X_train.shape, X_val.shape, X_test.shape



from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam


# build model
model = Sequential()
model.add(Embedding(len(list(unique_words)), 300, input_length=len_max))  #输入维度 输出维度
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])
model.summary()



%%time

# fit
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=256, verbose=1, callbacks=callback)


# submit 预测
y_pred = model.predict_classes(X_test)
submission = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
submission.Sentiment = y_pred
submission.to_csv('submission.csv', index=False)