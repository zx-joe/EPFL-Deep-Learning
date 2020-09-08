import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from module import Module

class Linear (Module) :
    
    # one fully-connected layer
    
    
    def __init__(self, in_dim, out_dim, eps=1., method='xavier'):
        self.in_dim=in_dim
        self.out_dim=out_dim
        
        # define weight, bias and their gradient
        self.w=FloatTensor(out_dim, in_dim)
        self.dw=FloatTensor(out_dim, in_dim)
        self.b=FloatTensor(out_dim)
        self.db=FloatTensor(out_dim)
        
        # initialization: defaulted as Xavier 
        if method=='zero':
            self.w=self.w.fill_(0)
            self.b=self.w.fill_(0)
        elif method=='normal':
            self.w=self.w.normal_(mean=0,std=eps)
            self.w=self.b.normal_(mean=0,std=eps)
        else:
            temp_std=1./math.sqrt((self.in_dim + self.out_dim)/2)
            self.w=self.w.normal_(mean=0,std=temp_std)
            self.b=self.b.normal_(mean=0,std=temp_std)
            
        # zero gradient intialization
        self.dw=self.dw.zero_()
        self.db=self.db.zero_()


    def forward( self ,x ):
        
        # y = w * x + b
        
        self.input=x.clone()
        self.output=self.w.matmul(self.input)+self.b
        #self.output=self.w @ self.input + self.b

        return self.output


        
    def backward( self , gradwrtoutput ):
        
        temp_wt=self.w.t()
       
        # dw = dL/dy * x
        temp_dw=gradwrtoutput.view(-1,1).mm(self.input.view(1,-1))
        self.dw.add_(temp_dw)
        
        # db = dL/dy
        temp_db=gradwrtoutput.clone()
        self.db.add_(temp_db)

        
        # dx = w.T * dL/dy
        temp_dx=temp_wt.matmul(gradwrtoutput)
        

        return temp_dx
        

        
    def param(self ) :      
        return [ self.w, self.dw  , self.b, self.db]
    
    
    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()