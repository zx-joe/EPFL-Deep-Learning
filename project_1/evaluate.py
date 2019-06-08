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


def evaluate_accuracy(train_accuracy_list,test_accuracy_list, title):
    # draw learning curves of accuracy on training amd testing set
    # print the accuracy on testing set
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    fig.suptitle(title)
    ax.plot(train_accuracy_list, label = "training accuracy")
    ax.plot(test_accuracy_list, label = "validation accuracy")
    ax.set_ylabel('classification accuracy')
    ax.set_xlabel('epoch')
    ax.legend()
    print("The Best Validation Accuracy is : {:.4f}..".format(max(test_accuracy_list)))

def evaluate_result(train_loss_list,train_accuracy_list, test_loss_list,test_accuracy_list, title):
    # draw learning curves of accuracy and loss of training amd testing set
    # print the accuracy on testing set
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_yscale('log')
    ax1.plot(train_loss_list, label = "training loss")
    ax1.plot(test_loss_list, label = "validation loss")
    ax1.set_ylabel('categorical cross entropy')
    ax1.set_xlabel('epoch')
    ax1.legend()
    ax2.plot(train_accuracy_list, label = "training accuracy")
    ax2.plot(test_accuracy_list, label = "validation accuracy")
    ax2.set_ylabel('classification accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend()
    print("The Best Validation Accuracy is : {:.4f}..".format(max(test_accuracy_list)))
	
def plot_different_histories(train_loss_list_1,train_accuracy_list_1, test_loss_list_1,test_accuracy_list_1,
                             train_loss_list_2,train_accuracy_list_2, test_loss_list_2,test_accuracy_list_2,
                             train_loss_list_3,train_accuracy_list_3, test_loss_list_3,test_accuracy_list_3,
                             train_loss_list_4,train_accuracy_list_4, test_loss_list_4,test_accuracy_list_4,
                             train_loss_list_5=None,train_accuracy_list_5=None, test_loss_list_5=None,test_accuracy_list_5=None,
                             label_1='', label_2='', label_3='', label_4='', label_5='',title=''):
    # plot learning curves of 4 or 5 set of training history lists
    
    # if no loss histories, then just plot the accuracy 
    if train_loss_list_1==None:
        fig, ax = plt.subplots(1,1,figsize=(10, 6))
        fig.suptitle(title)
        ax.plot(train_accuracy_list_1, label = label_1+" training accuracy",c='r')
        ax.plot(test_accuracy_list_1, label =label_1+ " validation accuracy",c='r',linestyle='--')
        ax.plot(train_accuracy_list_2, label = label_2+" training accuracy",c='g')
        ax.plot(test_accuracy_list_2, label =label_2+ " validation accuracy",c='g',linestyle='--')
        ax.plot(train_accuracy_list_3, label = label_3+" training accuracy",c='b')
        ax.plot(test_accuracy_list_3, label =label_3+ " validation accuracy",c='b',linestyle='--')
        ax.plot(train_accuracy_list_4, label = label_4+" training accuracy",c='y')
        ax.plot(test_accuracy_list_4, label =label_4+ " validation accuracy",c='y',linestyle='--')
        ax.set_ylabel('boolean classification accuracy')
        ax.set_xlabel('epoch')
        ax.legend()
        return
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 7))
    fig.suptitle(title)
    ax1.set_yscale('log')         
    
    # if there are only 4 set of histories
    # then plot the learning curves of accuracy and loss of 4 models
    if train_loss_list_5==None:
        ax1.set_yscale('log')
        ax1.plot(train_loss_list_1, label = label_1+" training loss",c='r')
        ax1.plot(test_loss_list_1, label = label_1+" validation loss",c='r',linestyle='--')
        ax1.plot(train_loss_list_2, label = label_2+" training loss",c='g')
        ax1.plot(test_loss_list_2, label = label_2+" validation loss",c='g',linestyle='--')
        ax1.plot(train_loss_list_3, label = label_3+" training loss",c='b')
        ax1.plot(test_loss_list_3, label = label_3+" validation loss",c='b',linestyle='--')
        ax1.plot(train_loss_list_4, label = label_4+" training loss",c='y')
        ax1.plot(test_loss_list_4, label = label_4+" validation loss",c='y',linestyle='--')
        ax1.set_ylabel('categorical cross entropy')
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax2.plot(train_accuracy_list_1, label = label_1+" training accuracy",c='r')
        ax2.plot(test_accuracy_list_1, label =label_1+ " validation accuracy",c='r',linestyle='--')
        ax2.plot(train_accuracy_list_2, label = label_2+" training accuracy",c='g')
        ax2.plot(test_accuracy_list_2, label =label_2+ " validation accuracy",c='g',linestyle='--')
        ax2.plot(train_accuracy_list_3, label = label_3+" training accuracy",c='b')
        ax2.plot(test_accuracy_list_3, label =label_3+ " validation accuracy",c='b',linestyle='--')
        ax2.plot(train_accuracy_list_4, label = label_4+" training accuracy",c='y')
        ax2.plot(test_accuracy_list_4, label =label_4+ " validation accuracy",c='y',linestyle='--')
        ax2.set_ylabel('classification accuracy')
        ax2.set_xlabel('epoch')
        ax2.legend()
        
    # else, then plot the learning curves of accuracy and loss of all 5 models
    else:
        ax1.set_yscale('log')
        ax1.plot(train_loss_list_1, label = label_1+" training loss",c='r')
        ax1.plot(test_loss_list_1, label = label_1+" validation loss",c='r',linestyle='--')
        ax1.plot(train_loss_list_2, label = label_2+" training loss",c='g')
        ax1.plot(test_loss_list_2, label = label_2+" validation loss",c='g',linestyle='--')
        ax1.plot(train_loss_list_3, label = label_3+" training loss",c='b')
        ax1.plot(test_loss_list_3, label = label_3+" validation loss",c='b',linestyle='--')
        ax1.plot(train_loss_list_4, label = label_4+" training loss",c='y')
        ax1.plot(test_loss_list_4, label = label_4+" validation loss",c='y',linestyle='--')
        ax1.plot(train_loss_list_5, label = label_5+" training loss",c='brown')
        ax1.plot(test_loss_list_5, label = label_5+" validation loss",c='brown',linestyle='--')
        ax1.set_ylabel('categorical cross entropy')
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax2.plot(train_accuracy_list_1, label = label_1+" training accuracy",c='r')
        ax2.plot(test_accuracy_list_1, label =label_1+ " validation accuracy",c='r',linestyle='--')
        ax2.plot(train_accuracy_list_2, label = label_2+" training accuracy",c='g')
        ax2.plot(test_accuracy_list_2, label =label_2+ " validation accuracy",c='g',linestyle='--')
        ax2.plot(train_accuracy_list_3, label = label_3+" training accuracy",c='b')
        ax2.plot(test_accuracy_list_3, label =label_3+ " validation accuracy",c='b',linestyle='--')
        ax2.plot(train_accuracy_list_4, label = label_4+" training accuracy",c='y')
        ax2.plot(test_accuracy_list_4, label =label_4+ " validation accuracy",c='y',linestyle='--')
        ax2.plot(train_accuracy_list_5, label = label_5+" training accuracy",c='brown')
        ax2.plot(test_accuracy_list_5, label =label_5+ " validation accuracy",c='brown',linestyle='--')
        ax2.set_ylabel('classification accuracy')
        ax2.set_xlabel('epoch')
        ax2.legend()