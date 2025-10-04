# Binary Classification with Neural Networks
## Aim:


## Algorithm:

## Program:
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline
df = pd.read_csv('income.csv')

print(len(df))
df.head()

df['label'].value_counts()
df.columns

cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']
print(f'cat_cols has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col has {len(y_col)} column')

for col in cat_cols:
  df[col] = df[col].astype('category')
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

for col in cat_cols:
  df[col] = df[col].astype('category')

for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].cat.add_categories('Missing').fillna('Missing')

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats[:5]
cats = torch.tensor(cats, dtype=torch.int64)
cats

for col in cont_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]
conts = torch.tensor(conts, dtype=torch.float32)
conts

df.dropna(subset=y_col, inplace=True)
y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
b = len(df)
t = 5000
cat_train = cats[:b-t]
con_train = conts[:b-t]
y_train = y[:b-t]
cat_test = cats[b-t:b-t+t]
con_test = conts[b-t:b-t+t]
y_test = y[b-t:b-t+t]
torch.manual_seed(33)

class TabularModel(nn.Module):
  def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
    super().__init__()
    self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
    self.emb_drop = nn.Dropout(p)
    self.bn_cont = nn.BatchNorm1d(n_cont)
    layerlist = []
    n_emb = sum((nf for ni,nf in emb_szs))
    n_in = n_emb + n_cont
    for i in layers:
      layerlist.append(nn.Linear(n_in,i))
      layerlist.append(nn.ReLU(inplace=True))
      layerlist.append(nn.BatchNorm1d(i))
      layerlist.append(nn.Dropout(p))
      n_in = i
    layerlist.append(nn.Linear(layers[-1],out_sz))
    self.layers = nn.Sequential(*layerlist)
  def forward(self, x_cat, x_cont):
    embeddings = []
    for i,e in enumerate(self.embeds):
      embeddings.append(e(x_cat[:,i]))
    x = torch.cat(embeddings, 1)
    x = self.emb_drop(x)
    x_cont = self.bn_cont(x_cont)
    x = torch.cat([x, x_cont], 1)
    x = self.layers(x)
    return x

model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
  i+=1
  y_pred = model(cat_train, con_train)
  loss = criterion(y_pred, y_train)
  losses.append(loss)
  if i%25 == 1:
    print(f'epoch: {i:3} loss: {loss.item():10.8f}')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
print(f'epoch: {i:3} loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

plt.plot([loss.item() for loss in losses])
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.show()

with torch.no_grad():
  y_val = model(cat_test, con_test)
  loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')

correct = 0
for i in range(len(y_test)):
  if y_val[i].argmax().item() == y_test[i].item():
    correct += 1
accuracy = correct / len(y_test) * 100
print(f'{correct} out of {len(y_test)} = {accuracy:.2f}% correct')
```

## Result:
