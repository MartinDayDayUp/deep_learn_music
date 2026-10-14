# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:59:21 2026

@author: huawei
"""


from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch


class DoaDataSet(Dataset):
    def __init__(self,dataName,labelName):

        datadf = pd.read_csv(dataName,header=None)
        labeldf = pd.read_csv(labelName,header=None)
        self.Rxx = datadf.values.astype(np.float32)
        self.labels = labeldf.values.astype(np.float32)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):

        currRxx = torch.tensor(self.Rxx[idx])
        currlabel = torch.tensor(self.labels[idx])

        features = currRxx.view(2,64)
        features = features.view(2,8,8)
        label = currlabel
        return features,label


if __name__ == '__main__':
    doaset = DoaDataSet('./dataCsv.csv','./labelCsv.csv')
    tmpF,tmpL = doaset[1]
    