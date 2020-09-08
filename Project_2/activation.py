import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from module import Module

'''
only forward and backward gradient calculation is needed in activation module
detailed computation equation is available in report with clear formular
'''


class ReLU(Module):
    def __init__(self):
        self.input=None
        
    def forward(self, x):
        # relu(x)= x (where input >=0) or 0 (where input <0)
        self.input=x.clone()
        temp_coef= (x>=0).float()
        output=x*temp_coef
        return output
    
    def backward(self, gradwrtoutput):
        # gradient = 1 (where input >=0) or 0 (where input < 0) 
        temp_coef= (self.input>=0).float()
        temp_grad=gradwrtoutput*(temp_coef)
        return temp_grad
        
        
    def param(self):
        return []
        
        
class Tanh(Module):
    def __init__(self):
        self.input=None
        
    def forward(self, x):
        self.input=x.clone()
        output= (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
        return output
    
    def backward(self, gradwrtoutput):
        temp_grad= 4./(  ( self.input.exp() + (-self.input).exp() ).pow(2)  )
        temp_grad=temp_grad*gradwrtoutput
        return temp_grad
    
    def param(self):
        return []
                 
                 
class Sigmoid(Module):
    def __init__(self):
        self.input=None
     
    def forward(self, x):
        self.input=x.clone()
        output=1./(1+(-self.input).exp())
        return output
    
    def backward(self,gradwrtoutput ):
        temp_grad=(-self.input).exp()/( (1+(-self.input).exp()).pow(2) )
        temp_grad=temp_grad*gradwrtoutput
        return temp_grad
    
    def param(self):
        return []
        
        
class SeLU(Module):
    def __init__(self):
        self.input=None
        
    def forward(self, x):
        self.input=x.clone()
        output=x.clone()
        output[output<0]=output[output<0].exp()-1.
        return output
    
    def backward(self, gradwrtoutput):
        grad=self.input.clone()
        grad[grad>=0]=1.
        grad[grad<0]=grad[grad<0].exp()
        return grad * gradwrtoutput
        
        
    def param(self):
        return []
        
class Softmax(Module):
    def __init__(self):
        self.input=None
        
    def forward(self, x):
        self.input=x.clone()
        
        prob_pred=x.exp()
        prob_pred_sum=prob_pred.sum(0)
        self.output=prob_pred/prob_pred_sum
        
        return self.output
    
    def backward(self, gradwrtoutput):
        
        prob_pred=self.input.exp()
        prob_pred_sum=prob_pred.sum(0)
        grad=(prob_pred_sum-prob_pred)/(prob_pred_sum.pow(2))
        
        return grad * gradwrtoutput
        
        
    def param(self):
        return []