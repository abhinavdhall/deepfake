import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.io import wavfile
import python_speech_features

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class deepfake_3d(data.Dataset):
    def __init__(self,out_dir,
                 mode='train',
                 transform=None):
        self.mode = mode
        self.transform = transform
        self.out_dir = out_dir

        # splits
        if mode == 'train':
            split = os.path.join(self.out_dir,'train_split.csv')
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'test'):
            split = os.path.join(self.out_dir,'test_split.csv')
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get label list
        self.label_dict_encode = {}
        self.label_dict_decode = {}
        self.label_dict_encode['fake'] = 0
        self.label_dict_decode['0'] = 'fake'
        self.label_dict_encode['real'] = 1
        self.label_dict_decode['1'] = 'real'

        self.video_info = video_info

    def __getitem__(self, index):
        try:
            vpath, audiopath, label = self.video_info.iloc[index]
            seq = [pil_loader(os.path.join(vpath,img)) for img in os.listdir(vpath)]

            t_seq = self.transform(seq) # apply same transform
            
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
            t_seq = t_seq.view(1, 30, C, H, W).transpose(1,2)

            sample_rate, audio = wavfile.read(audiopath)
            
            mfcc = zip(*python_speech_features.mfcc(audio,sample_rate,nfft=2048))
            mfcc = np.stack([np.array(i) for i in mfcc])

            cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
            cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

            vid = self.encode_label(label)
            
        except:
            return None
        return t_seq, cct, torch.LongTensor([vid]), audiopath

    def __len__(self):
        return len(self.video_info)

    def encode_label(self, label_name):
        return self.label_dict_encode[label_name]

    def decode_label(self, label_code):
        return self.label_dict_decode[label_code]