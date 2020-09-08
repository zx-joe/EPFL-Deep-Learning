import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

def data_init(train_input,train_target, train_class, num=1000):
    # training data random initialization
    index=[i for i in range(num)]
    random.shuffle(index)
    new_train_input=train_input[index,:,:,:]
    new_train_target=train_target[index]
    new_train_class=train_class[index,:]
    
    return new_train_input,new_train_target, new_train_class

def weights_init(m):
    # initialization of weights in convolution and linear layer
    # conform to a normal distribution
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Linear):
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)