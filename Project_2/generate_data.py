import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt

def generate_all_datapoints_and_labels(number=1000):
    # generate 2d points 
    points = Tensor(number, 2)
    # uniform in (0,1)
    points=points.uniform_(0, 1)
    
    vector2center=points-Tensor([0.5,0.5])
    dist2center=vector2center.pow(2).sum(1)
    labels=[]
    for i in range(number):
        if dist2center[i]<=1/(2*math.pi):
            labels.append(1.)
        else:
            labels.append(0.)
    labels=Tensor(labels)
    return points, labels
    
def convert_labels(labels):
    new_labels=Tensor(len(labels),2)
    for i in range(len(labels)):
        if labels[i]>0.5:
            new_labels[i]=Tensor([-1.,1.])
        else:
            new_labels[i]=Tensor([1.,-1.])
    return new_labels