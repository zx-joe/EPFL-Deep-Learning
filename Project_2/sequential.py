import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt
from module import Module


class Sequential(Module):
    def __init__(self, layers):
        self.layers=[]
        for temp_layer in layers:
            self.layers=self.layers+[temp_layer]
            
    def forward(self, x):
        # input: input data features fed to the whole network
        # output: model prediction sent to loss function
        self.input=x.clone()
        temp_output=self.input.clone()
        for temp_layer in self.layers:
            temp_output=temp_layer.forward(temp_output)
        final_output=temp_output
        return final_output
    
    def backward(self, gradwrtoutput):
        # input: loss gradient wrt network output in forward pass
        # output: loss gradient passed to the first layer sequently
        temp_gradient=gradwrtoutput.clone()
        for temp_layer in self.layers[::-1]:
            temp_gradient=temp_layer.backward(temp_gradient)
        final_gradient=temp_gradient
        return final_gradient
    
    def param(self):
        # all trainable features in the model (weights and bias in Linear module)
        parameters=[temp_layer.param() for temp_layer in self.layers]
        return parameters
           