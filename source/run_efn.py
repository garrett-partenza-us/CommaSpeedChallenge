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
import math
from functools import reduce
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print("using gpu...")
else:
    print("no gpu detected...")

print("downloading efficientnet-b0...")
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')


print("generating optical flow images...")
#create dataset of resized flatten optical flow images
features = []
no_testcases = 10000
for filename in tqdm(sorted(os.listdir('speedchallenge/data/ofs/'))[:no_testcases]):
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
    #convert to tensor
    of = torch.tensor(of)
    #extract features using efn-b0
    of = model.extract_features(of.reshape(1,3,180,120).float())
    #flatten
    of = of.flatten()
    #append to feature list
    features.append(of)



print("getting labels...")
labels = None
#read labels from text file
with open("speedchallenge/data/train.txt") as f:
    #only get necessary subset of labels and map to floats
    labels = map(float, f.read().splitlines()[1:no_testcases+1]) #raft omits first image
    #convert to float tensors
    labels = list(torch.tensor(x) for x in labels)



print("splitting train, val, test...")
#split train=0.8, val=0.1, test=0.1
x_train, x_holdout, y_train, y_holdout = train_test_split(features,labels,test_size=0.2,shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_holdout,y_holdout,test_size=0.5,shuffle=True)



print("creating nueral network...")
#create network
sff = SFF(n_feature=len(features[0]))     
optimizer = torch.optim.Adam(sff.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
print("creating dataloaders...")
#create torch dataset
dataset_train = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size=32,shuffle=True,drop_last=True)
dataset_val = torch.utils.data.DataLoader(list(zip(x_val,y_val)),batch_size=32,shuffle=True,drop_last=True)
dataset_test = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size=32,shuffle=True,drop_last=True)



print("beggining trainig...")
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

print("beggining testing...")
temp = []
for x, y in tqdm(dataset_test):
    predictions = sff(x.float())
    loss = loss_func(predictions, y)
    temp.append(loss.item())
batch_losses.append(["TEST", "TEST", statistics.mean(temp)])
import csv
fields = ['epoch', 'train', 'val'] 
with open('loss.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(batch_losses)
