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


def train_model(model , 
                train_input, train_target, 
                test_input, test_target,
                optim='SGD',
                epochs=50, batch_size=100, learning_rate=1e-3, momentum=0,
                 if_print=False):
    
    '''
    define training process of basic model (MLP, basic CNN)
    
    input: 
    model: model to train
    train_input, train_target, test_input, test_target: correspond to training and testing set loaded
    if_print: if we want to print the accuracy and loss of data set in each epoch
    other input: hypaerparameters to be tuned, all has a default value
    
    output:
    model: model after training
    train_loss_list,train_accuracy_list: list of loss and accuracy history of training set after each epoch
    test_loss_list,test_accuracy_list: list of loss and accuracy history of testing set after each epoch
    '''
  
    
    #use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # choose optimizer (sgd or adam) with learning rate and momentum by input
    if optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    elif optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # initialize output lists
    train_loss_list=train_accuracy_list=[]
    valid_loss_list=valid_accuracy_list=[]
    test_loss_list=test_accuracy_list=[]
    
    for e in range(epochs):
        # 10 batches per epoch
        for b in range(0, train_input.size(0), batch_size):
            # choose the mini batch each time
            temp_train_input=train_input.narrow(0, b, batch_size)    
            temp_train_target=train_target.narrow(0, b, batch_size)
            train_output = model(temp_train_input)
            # get the loss in each mini batch
            loss = criterion(train_output, temp_train_target)
            # learning by bp through loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
                
        # get the loss of training set
        train_temp_output = model(train_input)
        train_temp_loss = criterion(train_temp_output, train_target)
        # pred on training set
        _, temp_train_pred =torch.max(F.softmax(train_temp_output).data, 1)
        # accuracy of test set
        train_temp_accuracy=((temp_train_pred == train_target).sum().item())/train_input.size(0)
        
        
        # get the loss of testing set
        test_temp_output=model(test_input)
        test_temp_loss = criterion(test_temp_output, test_target)
        # pred on testing set
        _, temp_test_pred =torch.max(F.softmax(test_temp_output).data, 1)
        # accuracy of test set
        test_temp_accuracy=((temp_test_pred == test_target).sum().item())/test_input.size(0)
        
        # add to list each epoch
        train_loss_list=train_loss_list+[train_temp_loss]
        train_accuracy_list=train_accuracy_list+[train_temp_accuracy]
        test_loss_list=test_loss_list+[test_temp_loss]
        test_accuracy_list=test_accuracy_list+[test_temp_accuracy]
        
        # if if_print input is True, print the history per epoch
        if if_print:
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Training Loss: {:.4f}..".format(train_temp_loss),
                  "Training Accuracy: {:.4f}..".format(train_temp_accuracy), 
                  "Validation Loss: {:.4f}..".format(test_temp_loss),
                  "Validation Accuracy: {:.4f}..".format(test_temp_accuracy),  )
                

    return model, train_loss_list,train_accuracy_list, test_loss_list,test_accuracy_list





def train_model_WS(model , 
                train_input, train_target, train_class,
                test_input, test_target, test_class,
                optim='SGD', if_auxiliary_loss=False,auxiliary_loss_ratio=1,
                epochs=50, batch_size=100, learning_rate=1e-2, decay=True,
                 if_print=False):
    
    '''
    
    define training process of model combined with digit information
    
    input: 
    model: model to train
    train_input, train_target, train_class, test_input, test_target, test_class: correspond to training and testing set loaded4
    if_auxiliary_loss: whether we will use auxiliary loss
    
    if_print: if we want to print the accuracy and loss of data set in each epoch
    other input: hypaerparameters to be tuned, all has a default value
    auxiliary_loss_ratio: ratios of auxiliary loss wrt original loss
    decay: whether to give a decay of weights of 10-digit classification through epochs
    output:
    model: model after training
    train_loss_list,train_accuracy_list: list of loss and accuracy history of training set after each epoch
    test_loss_list,test_accuracy_list: list of loss and accuracy history of testing set after each epoch
  
    '''
    #use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # choose optimizer (sgd or adam) with learning rate and momentum by input
    if optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    elif optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters())
    # initialize output lists
    train_loss_list=train_accuracy_list=[]
    valid_loss_list=valid_accuracy_list=[]
    test_loss_list=test_accuracy_list=[]
    for e in range(epochs):
        # 10 batches per epoch
        for b in range(0, train_input.size(0), batch_size):
            # choose the mini batch each time
            temp_train_input=train_input.narrow(0, b, batch_size)    
            temp_train_target=train_target.narrow(0, b, batch_size)
            temp_train_class=train_class.narrow(0, b, batch_size)
            # output of weight sharing model
            # include digit information of 2 digits and binary classification
            x1_output,x2_output,train_output = model(temp_train_input)
            # loss of binary classification
            loss = criterion(train_output, temp_train_target)
            # loss of 2 digit classification, which serves as auxiliary loss
            auxiliary_loss_1 = criterion(x1_output, temp_train_class[:,0])      
            auxiliary_loss_2 = criterion(x2_output, temp_train_class[:,1]) 
            # auxiliary loss with combination of the 2 digit loss
            auxiliary_loss=auxiliary_loss_1+auxiliary_loss_ratio*auxiliary_loss_2
            # if input if_auxiliary_loss is True, we use combinatoin of loss and auxiliary loss
            if if_auxiliary_loss:
                if decay:
                    # a weight decay
                    temp_weight=(epochs-e)/epochs
                else:
                    # no decay
                    temp_weight=1.
                # combined loss
                loss=loss+auxiliary_loss*auxiliary_loss_ratio*temp_weight
                
            # learning by bp through loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
                
        # get the loss of training set
        _,_,train_temp_output = model(train_input)
        train_temp_loss = criterion(train_temp_output, train_target)
        # pred on training set
        _, temp_train_pred =torch.max(F.softmax(train_temp_output).data, 1)
        # accuracy of test set
        train_temp_accuracy=((temp_train_pred == train_target).sum().item())/train_input.size(0)
        
        # get the loss of testing set
        _,_,test_temp_output=model(test_input)
        test_temp_loss = criterion(test_temp_output, test_target)
        # pred on testing set
        _, temp_test_pred =torch.max(F.softmax(test_temp_output).data, 1)
        # accuracy of test set
        test_temp_accuracy=((temp_test_pred == test_target).sum().item())/test_input.size(0)
        
        # add to list each epoch
        train_loss_list=train_loss_list+[train_temp_loss]
        train_accuracy_list=train_accuracy_list+[train_temp_accuracy]
        test_loss_list=test_loss_list+[test_temp_loss]
        test_accuracy_list=test_accuracy_list+[test_temp_accuracy]
        
        # if if_print input is True, print the history per epoch
        if if_print:
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Training Loss: {:.4f}..".format(train_temp_loss),
                  "Training Accuracy: {:.4f}..".format(train_temp_accuracy), 
                  "Validation Loss: {:.4f}..".format(test_temp_loss),
                  "Validation Accuracy: {:.4f}..".format(test_temp_accuracy),  )
                

    return model, train_loss_list,train_accuracy_list, test_loss_list,test_accuracy_list
     
    
def train_by_digit(model , 
                train_input, train_target, train_class,
                test_input, test_target, test_class,
                optim='SGD',
                epochs=50, batch_size=100, learning_rate=1e-3, momentum=0,
                 if_print=False):
    
    '''
    define training process of model trained directly on digit class
    
    input: 
    model: model to train
    train_input, train_target, train_class, test_input, test_target, test_class: correspond to training and testing set loaded4  
    if_print: if we want to print the accuracy and loss of data set in each epoch
    other input: hypaerparameters to be tuned, all has a default value
 
    output:
    model: model after training
    digit_train_loss_list,digit_train_accuracy_list:loss and accuracy hisotry of digit classfication on train set
    train_accuracy_list: accuracy history of binary classfication of target on train set per epcoch
    
    digit_test_loss_list,digit_test_accuracy_list:loss and accuracy hisotry of digit classfication on test set
    test_accuracy_list: accuracy history of binary classfication of target on test set per epcoch
    '''
    
    #use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # choose optimizer (sgd or adam) with learning rate and momentum by input
    if optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    elif optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # initialize output lists
    test_accuracy_list=train_accuracy_list=[]
    digit_train_accuracy_list=digit_test_accuracy_list=[]
    digit_train_loss_list=digit_test_loss_list=[]
    
    for e in range(epochs):
        # 10 batches per epoch
        for b in range(0, train_input.size(0), batch_size):
            # choose the mini batch each time
            temp_train_input=train_input.narrow(0, b, batch_size)    
            temp_train_class=train_class.narrow(0, b, batch_size)
            # seperate each pair into 2 digits
            train_output_1 = model(temp_train_input[:,0,:,:])
            train_output_2 = model(temp_train_input[:,1,:,:])
            # only use the loss of 10-class digit classification
            loss = criterion(train_output_1, temp_train_class[:,0])+criterion(train_output_2, temp_train_class[:,1])
            # learning by bp through loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
        # predict the binary value according to digit classification on train set
        train_temp_output_1 = model(train_input[:,0,:,:])
        train_temp_output_2 = model(train_input[:,1,:,:])
        _, temp_train_pred_1 =torch.max(F.softmax(train_temp_output_1).data, 1)
        _, temp_train_pred_2 =torch.max(F.softmax(train_temp_output_2).data, 1)
        train_temp_output=temp_train_pred_1<=temp_train_pred_2

        
        # accuracy of binary classification on train set
        train_temp_accuracy=((train_temp_output.int() == train_target.int()).sum().item())/train_input.size(0)
        # accuracy of 2 digit classfication on train set
        train_digit_accracy_1=(temp_train_pred_1.int()==train_class[:,0].int()).sum().item()/train_input.size(0)
        train_digit_accracy_2=(temp_train_pred_2.int()==train_class[:,1].int()).sum().item()/train_input.size(0)
        train_digit_accracy=(train_digit_accracy_1+train_digit_accracy_2)/2
        # add to list
        digit_train_accuracy_list=digit_train_accuracy_list+[train_digit_accracy]
        # compute the digit classfication loss on train set
        train_digit_loss = criterion(train_temp_output_1, train_class[:,0])+criterion(train_temp_output_2, train_class[:,1])
        digit_train_loss_list=digit_train_loss_list+[train_digit_loss]
 
        # predict the binary value according to digit classification on test set
        test_temp_output_1 = model(test_input[:,0,:,:])
        test_temp_output_2 = model(test_input[:,1,:,:])
        _, temp_test_pred_1 =torch.max(F.softmax(test_temp_output_1).data, 1)
        _, temp_test_pred_2 =torch.max(F.softmax(test_temp_output_2).data, 1)
        test_pred_output=temp_test_pred_1<=temp_test_pred_2
        
        # compute the digit classfication loss on test set
        test_digit_loss = criterion(test_temp_output_1, test_class[:,0])+criterion(test_temp_output_2, test_class[:,1])
        digit_test_loss_list=digit_test_loss_list+[test_digit_loss]
        

        # accuracy of binary classification on test set
        test_temp_accuracy=((test_pred_output.int() == test_target.int()).sum().item())/test_input.size(0)
        # accuracy of 2 digit classfication on test set
        test_digit_accracy=((temp_test_pred_1.int()==test_class[:,0].int()).sum().item()+(temp_test_pred_2.int()==test_class[:,1].int()).sum().item())/(2*test_input.size(0))
        # add to list
        digit_test_accuracy_list=digit_test_accuracy_list+[test_digit_accracy]
        
        
        train_accuracy_list=train_accuracy_list+[train_temp_accuracy]
        test_accuracy_list=test_accuracy_list+[test_temp_accuracy]
        
        
        # if if_print input is True, print the history per epoch
        if if_print:
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Digit Training Loss: {:.4f}..".format(train_digit_loss),
                  "Digit Training Accuracy: {:.4f}..".format(train_digit_accracy),

                  "Boolea n Training Accuracy: {:.4f}..".format(train_temp_accuracy),
                  
                  "Digit Validation Loss: {:.4f}..".format(test_digit_loss),
                  "Digit Validation Accuracy: {:.4f}..".format(test_digit_accracy),

                  "Boolean Validation Accuracy: {:.4f}..".format(test_temp_accuracy)  )
                

    return model, digit_train_loss_list,digit_train_accuracy_list, train_accuracy_list,digit_test_loss_list,digit_test_accuracy_list,test_accuracy_list
        
        