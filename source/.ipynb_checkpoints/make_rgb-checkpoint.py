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


    
print("generating RGB images...")
images = {}
for filename in tqdm(sorted(os.listdir('../speedchallenge/data/ofs/'))):
    #load of numpy array
    of = np.load('../speedchallenge/data/ofs/'+filename)
    #convert to torch
    of = torch.from_numpy(of)
    #permutate
    of = of[0].permute(1,2,0).numpy()
    #color
    of = flow_to_image(of)
    #resize
    #of = cv2.resize(of, dsize=(120, 180), interpolation=cv2.INTER_CUBIC)
    #flatten
    of = of.flatten()
    #convert to float tensor
    of = torch.tensor(of)
    #add row
    images[filename]=of
    
s = pd.Series(images, name='Image')
s.index.name = 'Filename'
s.reset_index()
s.to_pickle('../data/images.pkl')
