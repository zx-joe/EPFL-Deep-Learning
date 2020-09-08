import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def plot_different_histories(train_loss_list_1,train_accuracy_list_1, test_loss_list_1,test_accuracy_list_1,
                             train_loss_list_2,train_accuracy_list_2, test_loss_list_2,test_accuracy_list_2,
                             label_1='', label_2='', title=''):
    # plot learning curves of 2 set of training history lists
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 7))
    fig.suptitle(title)
    
 
    ax1.set_yscale('log')
    ax1.plot(train_loss_list_1, label = label_1+" training loss",c='r')
    ax1.plot(test_loss_list_1, label = label_1+" validation loss",c='r',linestyle='--')
    ax1.plot(train_loss_list_2, label = label_2+" training loss",c='g')
    ax1.plot(test_loss_list_2, label = label_2+" validation loss",c='g',linestyle='--')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend()
    
    
    ax2.plot(train_accuracy_list_1, label = label_1+" training accuracy",c='r')
    ax2.plot(test_accuracy_list_1, label =label_1+ " validation accuracy",c='r',linestyle='--')
    ax2.plot(train_accuracy_list_2, label = label_2+" training accuracy",c='g')
    ax2.plot(test_accuracy_list_2, label =label_2+ " validation accuracy",c='g',linestyle='--')
    ax2.set_ylabel('classification accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend()
    
    
def plot_4_different_histories(train_loss_list_1,train_accuracy_list_1, test_loss_list_1,test_accuracy_list_1,
                             train_loss_list_2,train_accuracy_list_2, test_loss_list_2,test_accuracy_list_2,
                             train_loss_list_3,train_accuracy_list_3, test_loss_list_3,test_accuracy_list_3,
                             train_loss_list_4,train_accuracy_list_4, test_loss_list_4,test_accuracy_list_4,
                             label_1='', label_2='',label_3='', label_4='', title=''):
    # plot learning curves of 4 set of training history lists
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 7))
    fig.suptitle(title)
    
 
    ax1.set_yscale('log')
    ax1.plot(train_loss_list_1, label = label_1+" training loss",c='r')
    ax1.plot(test_loss_list_1, label = label_1+" validation loss",c='r',linestyle='--')
    ax1.plot(train_loss_list_2, label = label_2+" training loss",c='g')
    ax1.plot(test_loss_list_2, label = label_2+" validation loss",c='g',linestyle='--')
    ax1.plot(train_loss_list_3, label = label_3+" training loss",c='blue')
    ax1.plot(test_loss_list_3, label = label_3+" validation loss",c='blue',linestyle='--')
    ax1.plot(train_loss_list_4, label = label_4+" training loss",c='brown')
    ax1.plot(test_loss_list_4, label = label_4+" validation loss",c='brown',linestyle='--')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend()
    
    
    ax2.plot(train_accuracy_list_1, label = label_1+" training accuracy",c='r')
    ax2.plot(test_accuracy_list_1, label =label_1+ " validation accuracy",c='r',linestyle='--')
    ax2.plot(train_accuracy_list_2, label = label_2+" training accuracy",c='g')
    ax2.plot(test_accuracy_list_2, label =label_2+ " validation accuracy",c='g',linestyle='--')
    ax2.plot(train_accuracy_list_3, label = label_3+" training accuracy",c='blue')
    ax2.plot(test_accuracy_list_3, label =label_3+ " validation accuracy",c='blue',linestyle='--')
    ax2.plot(train_accuracy_list_4, label = label_4+" training accuracy",c='brown')
    ax2.plot(test_accuracy_list_4, label =label_4+ " validation accuracy",c='brown',linestyle='--')
    ax2.set_ylabel('classification accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend()
        
