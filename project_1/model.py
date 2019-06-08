import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')


class MLP(nn.Module):
    '''
    define MLP model
    input: number of layer ; number of hidden neuron in layer
  
    '''
    def __init__(self,nb_layer=1,nb_hidden_neuron=50):
        super(MLP, self).__init__()
        self.fc_begin = nn.Linear(392 ,nb_hidden_neuron)
        self.fc_hidden=nn.Linear(nb_hidden_neuron ,nb_hidden_neuron)
        self.fc_last = nn.Linear(nb_hidden_neuron, 2)
        
        self.fc_one_layer=nn.Linear(392 ,2)
        self.nb_layer=nb_layer

    def forward(self, x):
        x=x.view(x.size(0),-1)
        if self.nb_layer==1:
            x=self.fc_one_layer(x)
        else:
            # build number of linear layers according to input layer number
            x = F.relu(self.fc_begin(x))
            for i in range(self.nb_layer-2):
                x = F.relu(self.fc_hidden(x))
            x = self.fc_last(x)
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)

    
    
class ConvNet_1(nn.Module):
    '''
    define first CNN model
    input parameters can decide the choice of optimizatoin method
  
    input:
    bn: wthether to activate batch normalization layer
    dropout:the probability of dropout in the model
    activation: choice of activation function, default as Relu
 
    '''
    def __init__(self, bn=False, dropout=0, activation='relu' ):
        super(ConvNet_1, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*2*2 ,64)
        self.fc2 = nn.Linear(64, 2)
        self.bn16 = torch.nn.BatchNorm2d(16)
        self.bn16 = torch.nn.BatchNorm2d(32)
        self.bn=bn
        self.activation=activation
        self.dropout=dropout

    def forward(self, x):
        x=self.conv1(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
            
        x=F.max_pool2d(x, kernel_size=2, stride=2)
            
        x=self.conv2(x)
        if self.bn:
            x=self.bn16(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
            
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.dropout(x, p=self.dropout,training=self.training)
        
        
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x = self.fc2(x)
    
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)

    
    

class ConvNet_2(nn.Module):
    '''
    define 2nd CNN model
    input parameters can decide the choice of optimizatoin method
  
    input:
    bn: wthether to activate batch normalization layer
    dropout:the probability of dropout in the model
    activation: choice of activation function, default as Relu
 
    '''
    def __init__(self, bn=False, dropout=0, activation='relu' ):
        super(ConvNet_2, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*6*6 ,64)
        self.fc2 = nn.Linear(64, 2)
        self.bn8 = torch.nn.BatchNorm2d(8)
        self.bn16 = torch.nn.BatchNorm2d(16)
        self.bn32 = torch.nn.BatchNorm2d(32)
        self.bn=bn
        self.activation=activation
        self.dropout=dropout
    def forward(self, x):
        x=self.conv1(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
            
        if self.bn:
            x=self.bn8(x)
            
        x=self.conv2(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
        if self.bn:
            x=self.bn16(x)
        
        x = F.dropout(x, p=self.dropout,training=self.training)
         
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        
        
        
        
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x = self.fc2(x)
    
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
    
    
    
class DeepNet(nn.Module):
    '''
    define Deeper CNN model 
   
    '''
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*6*6 ,64)
        self.fc2 = nn.Linear(64, 2)
        self.bn1=torch.nn.BatchNorm2d(16)
        self.bn2=torch.nn.BatchNorm2d(16)
        self.bn3=torch.nn.BatchNorm2d(16)
        self.bn4=torch.nn.BatchNorm2d(16)
        self.bn5=torch.nn.BatchNorm2d(16)
        self.bn6=torch.nn.BatchNorm2d(16)
        

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        
        
        # after this, build 6 same convolution layers
        x=self.conv3(x)
        x=F.relu(x)
        x=self.bn1(x)
        
        x=self.conv4(x)
        x=F.relu(x)
        x=self.bn2(x)
        
        x=self.conv5(x)
        x=F.relu(x)
        x=self.bn3(x)
        
        x=self.conv6(x)
        x=F.relu(x)
        x=self.bn4(x)
        
        x=self.conv7(x)
        x=F.relu(x)
        x=self.bn5(x)
        
        x=self.conv8(x)
        x=F.relu(x)
        x=self.bn6(x)
        
        
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x = self.fc2(x)
    
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
    
    

    
class DeepNet_Res(nn.Module):
    '''
    define Deeper CNN model with residual block
   
    '''
    def __init__(self):
        super(DeepNet_Res, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*6*6 ,64)
        self.fc2 = nn.Linear(64, 2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1=torch.nn.BatchNorm2d(16)
        self.bn2=torch.nn.BatchNorm2d(16)
        self.bn3=torch.nn.BatchNorm2d(16)
        self.bn4=torch.nn.BatchNorm2d(16)
        self.bn5=torch.nn.BatchNorm2d(16)
        self.bn6=torch.nn.BatchNorm2d(16)
    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        

        # for later residual operation
        x1=x
        # after this, build 6 same convolution layers
        x=self.conv3(x)
        x=F.relu(x)
        x=self.bn1(x)
        
        x=self.conv4(x)
        x=F.relu(x)
        x=self.bn2(x)
        
        x=self.conv5(x)
        x=F.relu(x)
        x=self.bn3(x)
        
        # residual operation
        x=x1+x
        # for later residual operation
        x2=x
        
        x=self.conv6(x)
        x=F.relu(x)
        x=self.bn4(x)
        
        x=self.conv7(x)
        x=F.relu(x)
        x=self.bn5(x)
        
        x=self.conv8(x)
        x=F.relu(x)
        x=self.bn6(x)
        
        #  residual operation
        x=x2+x
       
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)    
        x = self.fc2(x)
    
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)

    
    
    

    
    
class ConvNet_WS(nn.Module):
    '''
    define CNN model combined with weight sharing on digit class
    return:
      temp_x1: prediction of 1st digit in pair
      temp_x2: prediction of 1st digit in pair
      new_x: prediction of binary classification combined with digit information
   
    '''
    def __init__(self ):
        super(ConvNet_WS, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*6*6 ,64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 2)
        #self.fc5  = nn.Linear(64,2)
        self.bn8 = torch.nn.BatchNorm2d(8)
        self.bn16 = torch.nn.BatchNorm2d(16)
        
        
    def subforward(self,temp_x):
        # define the function of training of each digit in a pair
        # return:
        # temp_x: digit information (10) 
        # x: feature map for later binary classification
        temp_x=self.conv1(temp_x)
        temp_x=F.relu(temp_x) 
        
        temp_x=self.conv2(temp_x)
        temp_x=F.relu(temp_x)
        
        temp_x=F.max_pool2d(temp_x, kernel_size=2, stride=2)
        
        temp_x=temp_x.view(temp_x.size(0),-1)
        temp_x=self.fc1(temp_x)
        temp_x=F.relu(temp_x)
        # store current feature map for later binary classification
        x=temp_x   
        # continue for digit classification
        temp_x=self.fc2(temp_x)
        return temp_x, x
        
    def forward(self, x):
        # get 2 digits in each pair
        x1=x[:,0,:,:].view(x.size(0),1,14,14)
        x2=x[:,1,:,:].view(x.size(0),1,14,14)
        
        # get 2 digit classification information and feature maps for later binary classification
        temp_x1,new_x1=self.subforward(x1)
        temp_x2,new_x2=self.subforward(x2)
        # concatenate the feature maps from 2 digit training
        new_x=torch.cat((new_x1.view(-1,64), new_x2.view(-1, 64)), 1)
        # finish the binary classification
        new_x=F.relu(self.fc3(new_x))
        new_x=self.fc4(new_x)
        
        return temp_x1,temp_x2,new_x


class CNN_digit(nn.Module):
    '''
    define same CNN model directly trained on digit class
    input parameters can decide the choice of optimizatoin method
  
    input:
    bn: wthether to activate batch normalization layer
    dropout:the probability of dropout in the model
    activation: choice of activation function, default as Relu
 
    '''
    def __init__(self, bn=False, dropout=0, activation='relu' ):
        super(CNN_digit, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*6*6 ,64)
        self.fc2 = nn.Linear(64, 10)
        self.bn8 = torch.nn.BatchNorm2d(8)
        self.bn16 = torch.nn.BatchNorm2d(16)
        self.bn32 = torch.nn.BatchNorm2d(32)
        self.bn=bn
        self.activation=activation
        self.dropout=dropout
    def forward(self, x):
        x=x.view(x.size(0),1,14,14)
        x=self.conv1(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
            
        if self.bn:
            x=self.bn8(x)
            
        x=self.conv2(x)
        if self.activation=='sigmoid':
            x=F.sigmoid(x)
        else:
            x=F.relu(x)
        if self.bn:
            x=self.bn16(x)
        
        x = F.dropout(x, p=self.dropout,training=self.training)
         
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        
        
        
        
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x = self.fc2(x)
    
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
