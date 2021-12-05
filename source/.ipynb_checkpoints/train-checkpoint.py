#!/usr/bin/env python

import os
import cv2
import csv
import statistics
from models import SFF
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tqdm import tqdm
import statistics
from utils import flow_to_image
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 

print("using gpu: ", torch.cuda.is_available())

images = pd.read_pickle('../data/images.pkl')

print("getting labels...")
labels = None
#read labels from text file
with open("../speedchallenge/data/train.txt") as f:
    #only get necessary subset of labels and map to floats
    labels = (list(map(float, f.read().splitlines()))[1:])
    labels = list(torch.tensor(x) for x in labels)
    
print("splitting train, val, test...")
#split train=0.8, val=0.1, test=0.1
x_train, x_holdout, y_train, y_holdout = train_test_split(images,labels,test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_holdout,y_holdout,test_size=0.5)
print("creating nueral network...")
#create network
sff = SFF(n_feature=64800)     
optimizer = torch.optim.Adam(sff.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
print("creating dataloaders...")
#create torch dataset
dataset_train = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size=32,shuffle=True)
dataset_val = torch.utils.data.DataLoader(list(zip(x_val,y_val)),batch_size=32,shuffle=True)
dataset_test = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size=32,shuffle=True)

print("beggining training...")
train_loss = []
val_loss = []
batch_losses = []

for epoch in range(1):
    print("*"*25)
    print("epoch: ", epoch)
    print("*"*25)
    #train
    count = 1
    for x, y in dataset_train:
        sff.train()
        predictions = sff(x.float())
        train_loss = loss_func(predictions, y)     
        optimizer.zero_grad()  
        train_loss.backward(retain_graph=True)         
        optimizer.step()
        if count%10==0:
            #eval
            sff.eval()
            temp = []
            for x, y in dataset_val:
                predictions = sff(x.float())
                val_loss = loss_func(predictions, y)
                temp.append(val_loss.item())
            batch_losses.append([epoch,train_loss.item(),statistics.mean(temp)])
            print("val loss: ", statistics.mean(temp))
        else:
            batch_losses.append([epoch,train_loss.item(),-1])
            print("train loss: ", train_loss.item())
        count+=1

plt.plot(train_loss, color='blue', label='train loss')
plt.plot(val_loss, color='green', label='val loss')
plt.savefig('../charts/sff_loss.jpg')

print("beggining testing...")

fields = ['epoch', 'train', 'val'] 
with open('../data/loss.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(batch_losses)

batch_loss = []
for x, y in tqdm(dataset_test):
    predictions = sff(x.float())
    loss = loss_func(predictions, y)
    batch_loss.append(loss.item())
print("test acc: ", statitics.mean(batch_loss))

torch.save(sff.state_dict(), '../models/sff.pt')
