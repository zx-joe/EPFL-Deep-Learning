import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        self.p=p
        self.prob_matrix=None
        
    def forward(self,x):
        self.prob_matrix=Tensor(x.size())
        # build bernoulli matrix according to probability, only activate pass through the matrix
        self.prob_matrix=self.prob_matrix.bernoulli_(1.-self.p).div_(1.-self.p)
        return x * self.prob_matrix
    
    def backward(self,gradwrtoutput):
        # use the same matrix in forward pass
        return gradwrtoutput*self.prob_matrix.t()
    
    def param(self):
        return []
              