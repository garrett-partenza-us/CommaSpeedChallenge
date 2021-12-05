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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model type", default='sff')
parser.add_argument("-t", "--testcases", help="number of testcases", default=20399, type=int)
args = parser.parse_args()

def get_images():
    return pd.read_pickle('../data/images.pkl')

def get_labels():
    labels = None
    with open("../speedchallenge/data/train.txt") as f:
        labels = (list(map(float, f.read().splitlines()))[1:])
    return list(torch.tensor(x) for x in labels)

def build_model():
    model = SFF(n_feature=64800)     
    return model
    
def build_datasets(images, labels):
    x_train, x_holdout, y_train, y_holdout = train_test_split(images,labels,test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_holdout,y_holdout,test_size=0.5)
    train = torch.utils.data.DataLoader(list(zip(x_train,y_train)),batch_size=32,shuffle=True)
    val = torch.utils.data.DataLoader(list(zip(x_val,y_val)),batch_size=32,shuffle=True)
    test = torch.utils.data.DataLoader(list(zip(x_test,y_test)),batch_size=32,shuffle=True)
    return train, val, test

def train(dataset_train, dataset_val, dataset_test, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.MSELoss()
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
            model.train()
            predictions = model(x.float())
            train_loss = loss_func(predictions, y)     
            optimizer.zero_grad()  
            train_loss.backward(retain_graph=True)         
            optimizer.step()
            if count%10==0:
                #eval
                model.eval()
                temp = []
                for x, y in dataset_val:
                    predictions = model(x.float())
                    val_loss = loss_func(predictions, y)
                    temp.append(val_loss.item())
                batch_losses.append([epoch,train_loss.item(),statistics.mean(temp)])
                print("val loss: ", statistics.mean(temp))
            else:
                batch_losses.append([epoch,train_loss.item(),-1])
                print("train loss: ", train_loss.item())
            count+=1
            
    torch.save(model.state_dict(), '../models/'+args.model+'.pt')
    
    fields = ['epoch', 'train', 'val'] 
    with open('../data/loss.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(batch_losses)

    test(model, dataset_test)

def test(model, dataset_test):
    batch_loss = []
    for x, y in tqdm(dataset_test):
        predictions = model(x.float())
        loss = loss_func(predictions, y)
        batch_loss.append(loss.item())
    print("test loss: ", statitics.mean(batch_loss))


if __name__ == '__main__':
    print("using gpu: ", torch.cuda.is_available())
    warnings.filterwarnings("ignore")
    images = get_images()[:args.testcases]
    labels = get_labels()[:args.testcases]
    model = build_model()
    dataset_train, dataset_val, dataset_test = build_datasets(images, labels)
    train(dataset_train, dataset_val, dataset_test, model)