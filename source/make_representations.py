#!/usr/bin/env python

import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import flow_to_image
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print("using gpu...")
else:
    print("no gpu detected...")

print("downloading efficientnet-b0...")
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

print("reading pickle...")
images = pd.read_pickle('images.pkl')

print("generating representation features...")
reps = {}
for filename, image in tqdm(images.items()):
    rep = model.extract_features(image.reshape(1,3,180,120).float()).flatten()
    reps[filename] = reps
    
s = pd.Series(reps, name='Features')
s.index.name = 'Filename'
s.reset_index()
s.to_pickle('representations.pkl')
