import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from module import Module

'''
optimizers mainly focus on the calculation to update trainable parameters in the network in each step
detailed computation equation is available in report with clear formular

'''


class SGD():
    def __init__(self, model,lr=1e-3, momentum=0.):
        self.lr=lr
        self.layers=model.layers
        self.momentum=momentum
        
    def step(self):
        for temp_layer in self.layers:
            temp_param=temp_layer.param()
            if len(temp_param)>0:
                _ , temp_dw, _ , temp_db = temp_param
                temp_layer.w=temp_layer.w-self.lr*temp_dw
                temp_layer.b=temp_layer.b-self.lr*temp_db
                
    def zero_grad(self):
        for temp_layer in self.layers:
            try:
                temp_layer.zero_grad()
            except:
                continue
                
                
                
                
class Adam():
    def __init__(self, model,lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8 ):
        self.lr=lr
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.eps=eps
        self.layers=model.layers
        self.time=0
        
        self.num_grad=0
        
        self.m=[]
        self.v=[]
        for temp_layer in self.layers:
            temp_param=temp_layer.param()
            if (len(temp_param))>0:
                
                self.m.append(Tensor(temp_param[1].size()).zero_())
                self.m.append(Tensor(temp_param[3].size()).zero_())
                self.v.append(Tensor(temp_param[1].size()).zero_())
                self.v.append(Tensor(temp_param[3].size()).zero_())
                self.num_grad+=2

                               
    def step(self):
        i=0
        for temp_layer in self.layers:
            temp_param=temp_layer.param()
            if len(temp_param)>0:
                _ , temp_dw, _ , temp_db = temp_param
                # update weight
                self.m[i]=self.beta_1*self.m[i]+(1-self.beta_1)*temp_dw
                self.v[i]=self.beta_2*self.v[i]+(1-self.beta_2)*(temp_dw.pow(2))
                temp_m_hat=self.m[i]/(1-self.beta_1**(self.time+1))
                temp_v_hat=self.v[i]/(1-self.beta_2**(self.time+1))
                temp_layer.w-=self.lr*temp_m_hat/(temp_v_hat.pow(0.5)+self.eps)
                
                # update bias
                self.m[i+1]=self.beta_1*self.m[i+1]+(1-self.beta_1)*temp_db
                self.v[i+1]=self.beta_2*self.v[i+1]+(1-self.beta_2)*(temp_db.pow(2))
                temp_m_hat=self.m[i+1]/(1-self.beta_1**(self.time+1))
                temp_v_hat=self.v[i+1]/(1-self.beta_2**(self.time+1))
                temp_layer.b-=self.lr*temp_m_hat/(temp_v_hat.pow(0.5)+self.eps)
                
                i+=2
                
        self.time+=1
                
                
    def zero_grad(self):
        for temp_layer in self.layers:
            try:
                temp_layer.zero_grad()
            except:
                continue