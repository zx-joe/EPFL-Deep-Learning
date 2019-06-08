import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# loading data
import dlc_practical_prologue as prologue
train_input,train_target,train_class,test_input,test_target,test_class=prologue.generate_pair_sets(1000)

# import our script
import model as model
import evaluate as evaluate
import train as train
import initialization as init



def main():
    # if you wish to train a simple CNN model 
    # just uncomment this and comment the other model
    '''
    temp_train_input,temp_train_target,temp_train_class=init.data_init(train_input,train_target, train_class, num=1000)
    model_cnn=model.ConvNet_2(bn=True,dropout=0.1)
    model_cnn.apply(init.weights_init)
    model_cnn,train_loss_cnn,train_accuracy_cnn, test_loss_cnn,test_accuracy_cnn=\
           train.train_model(model_cnn,\
                       temp_train_input,temp_train_target,\
                       test_input, test_target,\
                       if_print=True, learning_rate=0.01, momentum=0.5)
    evaluate.evaluate_result(train_loss_cnn,train_accuracy_cnn, test_loss_cnn,test_accuracy_cnn, \
               'Learning Curve of Simple CNN')
    '''
    
    
    # if you wish to train a model with weight sharing and auxiliary loss
    # just uncomment this and comment the other model
    '''
    temp_train_input,temp_train_target,temp_train_class=init.data_init(train_input,train_target, train_class, num=1000)
    model_ws_al=model.ConvNet_WS()
    model_ws_al.apply(init.weights_init)
    model_ws_al,train_loss_ws_al,train_accuracy_ws_al, test_loss_ws_al,test_accuracy_ws_al=\
                                  train.train_model_WS(model_ws_al, \
                                                       temp_train_input,temp_train_target,temp_train_class,\
                                                       test_input, test_target, test_class,\
                                                       if_auxiliary_loss=True,epochs=50,if_print=True,auxiliary_loss_ratio=5)
    evaluate.evaluate_result(train_loss_ws_al,train_accuracy_ws_al, test_loss_ws_al,test_accuracy_ws_al,\
                'Learning Curve of Second ConvNet with Weight Sharing and Auxiliart Loss')  '''
    
    
    
    # training data random initialization
    temp_train_input,temp_train_target,temp_train_class=init.data_init(train_input,train_target, train_class, num=1000)
    # import the model
    model_digit=model.CNN_digit()
    # model weight initialization
    model_digit.apply(init.weights_init)
    # get the training history of the model with input hyperparameters
    _,_,_,train_accuracy_from_digit,_,_,test_accuracy_from_digit=train.train_by_digit(model_digit,\
                                                                                  temp_train_input,temp_train_target,temp_train_class, \
                                                                                  test_input, test_target, test_class, \
                                                                                  if_print=True, \
                                                                                  epochs=50,\
                                                                                  optim='sgd',learning_rate=0.01,momentum=0.9)
    # plot the learning curves and print the accuracy on testing set
    evaluate.evaluate_accuracy(train_accuracy_from_digit,test_accuracy_from_digit,\
                           'Accuracy of Boolean Classification from CNN Trained Directly on Digit Class') 

if __name__ == '__main__':
    main()