import random
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW

################################################################设定随机种子
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

################################读取文件
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_submission = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')


train_df = pd.read_csv('train.tsv', sep='\t')
print(train_df.shape)
print(train_df.info())
train_df.head()

test_df = pd.read_csv('test.tsv', sep='\t')  
print(test_df.shape)
print(test_df.info())
test_df.head()

################################Bert预训练模型 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', lower=True)

class MovieReviewsDataset(Dataset):
    def __init__(self, df, max_len, test_only=False):
        self.max_len = max_len
        self.test_only = test_only
        self.text = df['Phrase'].tolist()
        if not self.test_only:
            self.sentiments = df['Sentiment'].values
            
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
    
max_len = 64  ###文本最大长度
train_dataset = MovieReviewsDataset(train_df, max_len)
test_dataset = MovieReviewsDataset(test_df, max_len, test_only=True)

lengths = [int(round(len(train_dataset) * 0.8)), int(round(len(train_dataset) * 0.2))]
train_dataset, valid_dataset = random_split(train_dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(valid_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

############################################################ Model架构
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        bert_base_config = AutoConfig.from_pretrained('bert-base-uncased')
        self.bert_base = AutoModel.from_pretrained('bert-base-uncased', config=bert_base_config)
        self.classifier = nn.Linear(bert_base_config.hidden_size, 5) #分类类别

    def forward(self, input_ids, attention_mask):
        bert_base_output = self.bert_base(input_ids=input_ids, attention_mask=attention_mask)
        # get last hidden state
        # bert_base_last_hidden_state = bert_base_output[0]
        # or
        # roberta_base_last_hidden_state = roberta_base_output.hidden_states[-1]

        # pooler_output – Last layer hidden-state of the first token of the sequence 
        # (classification token) further processed by a Linear layer and a Tanh activation function
        pooler_output = bert_base_output[1] # [batch_size, hidden] 
        out = self.classifier(pooler_output)
        return out
    
model = Model()
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criteron = nn.CrossEntropyLoss()

####################################训练模型
total_loss = []
total_val_acc = []
for epoch in range(1):
    model.train()
    epoch_loss = []
    for input_ids, attention_mask, target in tqdm(train_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)            
        target = target.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(input_ids, attention_mask)
        aux_logits=False
        loss = criteron(y_pred, target)
        loss.backward()
        optimizer.step()
        
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
    
    
    
################################预测   
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




