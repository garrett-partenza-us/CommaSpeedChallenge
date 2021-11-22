#!/usr/bin/env python

import os
import cv2
from models import SFF
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tqdm import tqdm
import statistics
from utils import flow_to_image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print("using gpu...")
else:
    print("no gpu detected...")

    

print("generating optical flow images...")
#create dataset of resized flatten optical flow images
features = []
for filename in tqdm(sorted(os.listdir('speedchallenge/data/ofs/'))):
    #load of numpy array
    of = np.load('speedchallenge/data/ofs/'+filename)
    #convert to torch
    of = torch.from_numpy(of)
    #permutate
    of = of[0].permute(1,2,0).numpy()
    #color
    of = flow_to_image(of)
    #resize
    of = cv2.resize(of, dsize=(120, 180), interpolation=cv2.INTER_CUBIC)
    #flatten
    of = of.flatten()
    #convert to float tensor
    of = torch.tensor(of)
    features.append(of)



print("getting labels...")
labels = None
#read labels from text file
with open("speedchallenge/data/train.txt") as f:
    #only get necessary subset of labels and map to floats
    labels = map(float, f.read().splitlines()[:len(features)])
    #convert to float tensors
    labels = list(torch.tensor(x) for x in labels)



print("splitting train, val, test...")
#split train=0.8, val=0.1, test=0.1
x_train, x_holdout, y_train, y_holdout = train_test_split(features,labels,test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_holdout,y_holdout,test_size=0.5)



print("creating nueral network...")
#create network
sff = SFF(n_feature=64800)     
optimizer = torch.optim.Adam(sff.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
print("creating dataloaders...")
#create torch dataset
dataset_train = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size=16,shuffle=True)
dataset_val = torch.utils.data.DataLoader(list(zip(x_val,y_val)),batch_size=16,shuffle=True)
dataset_test = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size=16,shuffle=True)



print("beggining trainig...")
train_loss = []
val_loss = []

for epoch in range(10):
    print("epoch: ", epoch)
    #train
    batch_loss = []
    sff.train()
    for x, y in dataset_train:
        predictions = sff(x.float())
        loss = loss_func(predictions, y)     
        optimizer.zero_grad()  
        loss.backward()         
        optimizer.step()
        batch_loss.append(loss.item())
        print("loss: ", loss.item())
    train_loss.append(statistics.mean(batch_loss))
    
    #eval
    sff.eval()
    batch_loss = []
    for x, y in dataset_val:
        predictions = sff(x.float())
        loss = loss_func(predictions, y)
        batch_loss.append(loss.item())
    val_loss.append(statistics.mean(batch_loss))
    print("val loss: ", statistics.mean(batch_loss))

plt.plot(train_loss, color='blue', label='train loss')
plt.plot(val_loss, color='green', label='val loss')
plt.savefig('loss.jpg')

print("beggining testing...")



batch_loss = []
for x, y in tqdm(dataset_test):
    predictions = sff(x.float())
    loss = loss_func(predictions, y)
    batch_loss.append(loss.item())
print("test acc: ", statitics.mean(batch_loss))

