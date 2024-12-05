# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:56:57 2024

@author: ABaraka
"""


import torch
import torch.nn as nn


#%%**************************************** IFE Module

class Attention(nn.Module):
    def __init__(self, input_size,output_size,r=1):
        super(Attention, self).__init__()
        self.r=r
        self.attention=nn.Sequential( 
            nn.Linear(input_size,output_size,bias=False),       
            nn.Softmax(dim=1))
    def forward(self,x):
        crnt_attn=self.attention(x)
        for param in self.attention.parameters():
            w=param
        z=torch.matmul(crnt_attn,torch.exp(w*self.r))
        attention_weights=torch.mean(z,dim=0)          
        return attention_weights


class IterativeAttention(nn.Module):
  def __init__(self, input_size,output_size,r=1):
    super(IterativeAttention, self).__init__()
    self.ite_attn = nn.ModuleList(
        [Attention(input_size,output_size,r) for i in range(input_size)]
        )
  def forward(self, x):  
      com_attn= torch.zeros(x.size(1)).to(x.device).type(torch.float)
      com_attn=com_attn.unsqueeze(-1)
      ite_result=[] 
      
      for i, l in enumerate(self.ite_attn):
          mask = torch.ones_like(x[0, :])
          mask[i] = 0
          x_masked = x * mask    
          ite_result.append(self.ite_attn[i](x_masked).unsqueeze(-1))
          
      com_attn=torch.concat((ite_result),dim=1)
      com_mean=torch.mean(com_attn,dim=1)
      com_mean=torch.softmax(com_mean,dim=0)
      return  com_mean   

class IFEM(nn.Module):
    def __init__(self, x, hidden_size,output_size,r=1):
        super(IFEM, self).__init__()       
        self.feature_importance=0
        hidden_size=hidden_size 
        self.bn =nn.BatchNorm1d(x.size(1))
        self.ite_attn=IterativeAttention(x.size(1),output_size,r)
        self.fc1 = nn.Linear(x.size(1), hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x=self.bn(x)  
        ite_attn=self.ite_attn(x)
        self.feature_importance=ite_attn
        x=x*ite_attn
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        #x=torch.sigmoid(x)
        return x
