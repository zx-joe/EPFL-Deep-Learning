import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt

'''
only forward and backward gradient calculation is needed in loss function
detailed computation equation is available in report with clear formular


'''

class LossMSE():
    def __init__(self):
        self.pred=None
        self.label=None
        
    def forward(self, pred, label):
        self.pred=pred.clone()
        self.label=label.clone()
        all_loss=(self.label - self.pred).pow(2)
        mse=all_loss.mean()
        return mse
    
    def backward(self, pred, label):
        self.pred=pred
        self.label=label
        grad=2*(self.label - self.pred)*(-1)
        return grad
        
    def param(self):
        return []
        
        
class LossMAE():
    def __init__(self):
        self.pred=None
        self.label=None
        
    def forward(self, pred, label):
        self.pred=pred.clone()
        self.label=label.clone()
        all_loss=(self.label - self.pred).abs()
        mae=all_loss.mean()
        return mae
    
    def backward(self, pred, label):
        self.pred=pred
        self.label=label
        grad=pred
        grad[grad>=self.pred]=1.
        grad[grad<self.pred]=-1.
        return grad
        
    def param(self):
        return []
        
        
class CrossEntropy():
    def __init__(self):
        self.pred=None
        self.label=None
        
    def forward(self, pred, label, eps=1e-2):
        self.pred=pred
        self.label=label
        self.eps=eps
        
        prob_label=Tensor(label.size()).zero_()
        _,label_class=torch.max(label,0)
        prob_label[label_class]=1.
        
        pred.add_(eps)
        entropys=-(pred.log())*prob_label
        self.output=entropys.sum(0)
        return self.output
    
    
    def backward(self, pred, label, eps=1e-2):
        self.pred=pred
        self.label=label
        self.eps=eps
        #prob_pred=pred.softmax(0)
        
        prob_label=FloatTensor(label.size()).zero_()
        _,label_class=torch.max(label,0)
        prob_label[label_class]=1.
        
        #grad=-prob_label/pred
      
        grad=FloatTensor(pred.size()).zero_()

        grad[0]=-prob_label[0]/(pred[0]+eps)+prob_label[1]/(1-pred[0]+eps)
        grad[1]=-prob_label[1]/(pred[1]+eps)+prob_label[0]/(1-pred[1]+eps)

 
        return grad
    
        
    def param(self):
        return []