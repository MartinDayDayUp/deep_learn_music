# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:21:23 2026

@author: huawei
"""

import torch
import torch.nn as nn
from torchinfo import summary


class DOAModel(nn.Module):
    def __init__(self,angleGird,inputChannel):
        super(DOAModel,self).__init__()

        self.model = nn.Sequential(
                nn.Conv2d(inputChannel,256,3,padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(256,256,2,padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256,256,2,padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256,256,2,padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Flatten(),
                nn.Linear(256*8*8,4096),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(4096,2048),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(2048,1024),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(1024,angleGird)
   

              


            )



    def forward(self,x):

        logits = self.model(x)
        return logits




if __name__ == '__main__':
    doaModel = DOAModel(141,2,) 
    print(doaModel)
    aaa= 1
    print(f'ssss {aaa}')
    
    train_d = torch.randn(1,2,8,8)
    tmp = doaModel(train_d)
    print(tmp.size())
    
    summary(doaModel,input_size=(1,2,8,8))