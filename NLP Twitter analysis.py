# Most basic stuff for EDA.

from this import d
import pandas as pd
from zmq import THREAD_AFFINITY_CPU_REMOVE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Core packages for text processing.

import string
import re

# Libraries for text preprocessing.

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Loading some sklearn packaces for modelling.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import f1_score, accuracy_score

# Some packages for word clouds and NER.

from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image
import spacy
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
import en_core_web_sm

# Core packages for general use throughout the notebook.

import random
import warnings
import time
import datetime

# For customizing our plots.

from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Loading pytorch packages.

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

# Setting some options for general use.

stop = set(stopwords.words('english'))
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')

#Setting seeds for consistent results.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Loading the train and test data for visualization & exploration.
trainv = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
testv = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

idx=len(trainv)
dataset=pd.concat([trainv,testv])
dataset=dataset.reset_index(drop=True)

################################################ #################Clean text-data

# Some basic helper functions to clean text by removing urls, emojis, html tags and punctuations.

#??????url ?????????????????????
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

##??????????????????
def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

##### ??????html??????
def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

#### ??????????????????
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)  #string.punctuation ??????????????????????????????  maketrans ?????????????????????
    return text.translate(table)  # ???translate()?????????????????????????????????????????????????????????????????????????????????????????????????????????

# Applying helper functions
dataset['text_clean'] = dataset['text'].apply(lambda x: remove_URL(x))
dataset['text_clean'] = dataset['text_clean'].apply(lambda x: remove_emoji(x))
dataset['text_clean'] = dataset['text_clean'].apply(lambda x: remove_html(x))
dataset['text_clean'] = dataset['text_clean'].apply(lambda x: remove_punct(x))


# Tokenizing the tweet base texts.
dataset['tokenized'] = dataset['text_clean'].apply(word_tokenize)  # word_tokenize ??????
dataset.head()

# ?????????????????????
dataset['lower'] = dataset['tokenized'].apply(lambda x: [word.lower() for word in x])

#dataset["lower"]=dataset[""].apply(lambda x: [word.lower() for word in x])


#???????????????
dataset["stopwords_removed"]=dataset["lower"].apply(lambda x:[word for word in x if word not in stop])

# ??????????????????. 
dataset['pos_tags'] = dataset['stopwords_removed'].apply(nltk.tag.pos_tag) 

# Converting part of speeches to wordnet format. ?????????????????? wordnet_pos
def get_wordnet_pos(tag):
    if tag.startswith('J'): #????????????????????? ??????????????????
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV                     
    else:
        return wordnet.NOUN

dataset['wordnet_pos'] = dataset['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

# Applying word lemmatizer. ????????????
wnl = WordNetLemmatizer()
dataset['lemmatized'] = dataset['wordnet_pos'].apply(
    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

dataset['lemmatized'] = dataset['lemmatized'].apply(
    lambda x: [word for word in x if word not in stop])

#############???????????????????????????
dataset['lemma_str'] = [' '.join(map(str, l)) for l in dataset['lemmatized']]


trainv=dataset.iloc[:idx,:]
testv=dataset.iloc[idx:,:]
trainv["target"]=trainv["target"].astype(int)


############# ????????????????????? (???????????????)
lis = [
    trainv[trainv['target'] == 0]['lemma_str'],
    trainv[trainv['target'] == 1]['lemma_str']
]


# Displaying most common words.
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()

for i, j in zip(lis, axes):

    new = i.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]

    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:30]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    sns.barplot(x=y, y=x, palette='plasma', ax=j)
axes[0].set_title('Non Disaster Tweets')

axes[1].set_title('Disaster Tweets')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Word')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Word')

fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')
plt.tight_layout()


################################################### ngram
def ngrams(n, title):
    """A Function to plot most common ngrams"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes = axes.flatten()
    for i, j in zip(lis, axes):

        new = i.str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i]

        def _get_top_ngram(corpus, n=None):
            #getting top ngrams
            vec = CountVectorizer(ngram_range=(n, n),
                                  max_df=0.9,
                                  stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx])
                          for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:15]

        top_n_bigrams = _get_top_ngram(i, n)[:15]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y, y=x, palette='plasma', ax=j)
        
        axes[0].set_title('Non Disaster Tweets')
        axes[1].set_title('Disaster Tweets')
        axes[0].set_xlabel('Count')
        axes[0].set_ylabel('Words')
        axes[1].set_xlabel('Count')
        axes[1].set_ylabel('Words')
        fig.suptitle(title, fontsize=24, va='baseline')
        plt.tight_layout()
        
ngrams(2, 'Most Common Bigrams')

################################ WorldCloud grapth

def plot_wordcloud(text, title, title_size):
    """ A function for creating wordcloud images """
    words = text
    allwords = []
    for wordlist in words:
        allwords += wordlist
    mostcommon = FreqDist(allwords).most_common(140)
    wordcloud = WordCloud(
        width=400,
        height=300,
        background_color='black',
        stopwords=set(STOPWORDS),
        max_words=150,
        scale=3,
        contour_width=0.1,
        contour_color='grey',
    ).generate(str(mostcommon))    

    def grey_color_func(word,
                        font_size,
                        position,
                        orientation,
                        random_state=None,
                        **kwargs):
        # A definition for creating grey color shades.
        return 'hsl(0, 0%%, %d%%)' % random.randint(60, 100)

    fig = plt.figure(figsize=(12, 12), facecolor='white')
    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=42),
               interpolation='bilinear')
    plt.axis('off')
    plt.title(title,
              fontdict={
                  'size': title_size,
                  'verticalalignment': 'bottom'
              })
    plt.tight_layout(pad=0)
    plt.show()

######### ???????????????
plot_wordcloud(trainv[trainv['target'] == 0]['lemmatized'],
               'Most Common Words in Non-Disaster Tweets',
               title_size=30)

plot_wordcloud(trainv[trainv['target'] == 1]['lemmatized'],
               'Most Common Words in Disaster Tweets',
               title_size=30)



###########################  ???????????? BERT model


########## ??????GPU??????
if torch.cuda.is_available():    
    device = torch.device('cuda')    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')
    
  
train=trainv
test=testv

# Tokenizing the combined text data using bert tokenizer.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', lower=True)

# max_len ??????????????????
max_len = 0
for text in combined:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.  
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Update the maximum sentence length.  
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

############################## Model ??????
class MovieReviewsDataset(Dataset):
    def __init__(self, df, max_len, test_only=False):
        self.max_len = max_len
        self.test_only = test_only
        self.text = df['??????'].tolist()
        if not self.test_only:
            self.sentiments = df['????????????'].values
            
        self.encode = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True
        )
        
    def __getitem__(self, i):
        input_ids = torch.tensor(self.encode['input_ids'][i])
        attention_mask = torch.tensor(self.encode['attention_mask'][i])
        
        if self.test_only:
            return (input_ids, attention_mask)
        else:
            sentiments = self.sentiments[i]
            return (input_ids, attention_mask, sentiments)
    
    def __len__(self):
        return len(self.text)
    
###################### ??????????????????torch??????
max_len =55
train_dataset = MovieReviewsDataset(train, max_len)
test_dataset = MovieReviewsDataset(test, max_len, test_only=True)
lengths = [int(round(len(train_dataset) * 0.8)), int(round(len(train_dataset) * 0.2))]
train_dataset, valid_dataset = random_split(train_dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(valid_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        bert_base_config = AutoConfig.from_pretrained('bert-base-uncased')
        self.bert_base = AutoModel.from_pretrained('bert-base-uncased', config=bert_base_config)
        self.classifier = nn.Linear(bert_base_config.hidden_size, 2)  #????????????

    def forward(self, input_ids, attention_mask):
        bert_base_output = self.bert_base(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = bert_base_output[1] # [batch_size, hidden] 
        out = self.classifier(pooler_output)
        return out
    
################################################### ????????????
epochs=10
total_steps = len(train_dataloader) * epochs
model = Model()
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criteron = nn.CrossEntropyLoss()  

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,num_training_steps=total_steps)  

#scheduler = get_cosine_schedule_with_warmup(
        #optimizer, num_warmup_steps=100, num_training_steps=total_steps)

################################################################################## ????????????
total_loss = []
total_val_acc = []
for epoch in range(10):  # epoch ????????????
    model.train()
    epoch_loss = []
    for input_ids, attention_mask, target in tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)            
        target = target.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(input_ids, attention_mask)
        
        loss = criteron(y_pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss.append(loss.item())

    input_ids = input_ids.to(torch.device('cpu'))
    attention_mask = attention_mask.to(torch.device('cpu'))            
    target = target.to(torch.device('cpu'))

    val_accs = []
    model.eval()
    for input_ids, attention_mask, target in tqdm(val_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)        
        y_pred = model(input_ids, attention_mask)
        _, y_pred = torch.max(y_pred, -1)
        acc = torch.mean((torch.tensor(y_pred.cpu() == target.cpu(), dtype=torch.float)))
        val_accs.append(acc.cpu())

    el = sum(epoch_loss)/len(epoch_loss)
    total_loss.append(el)
    acc = np.array(val_accs).mean()
    total_val_acc.append(acc)
    print("Epoch:", epoch+1, "-- loss:", el, "-- acc:", acc)
    
####################### ??????

model.eval()
predictions = []
for text, attention_mask in tqdm(test_dataloader):
    text = text.to(device)
    attention_mask = attention_mask.to(device)
    preds = model(text, attention_mask)
    _, preds = torch.max(preds, -1)
    for pred in preds: predictions.append(pred.item())
print(len(predictions))

submission = pd.DataFrame()
submission['PhraseId'] = test_df['PhraseId']
submission['Sentiment'] = predictions
submission.to_csv("submission.csv", index=False)
print("Sumbssion is ready!")










