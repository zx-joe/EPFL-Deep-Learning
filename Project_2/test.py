import torch
import math
from torch import Tensor, FloatTensor
import matplotlib.pyplot as plt

from generate_data import generate_all_datapoints_and_labels,  convert_labels

from module import Module
from linear import Linear
from sequential import Sequential
from activation import ReLU, Tanh, Sigmoid, SeLU, Softmax
from optimizer import SGD,Adam
from regularization import Dropout
from loss import LossMSE, LossMAE, CrossEntropy

torch.set_grad_enabled(False)

def main():
    # generate data and translate labels
    train_features, train_targets = generate_all_datapoints_and_labels()
    test_features, test_targets = generate_all_datapoints_and_labels()
    train_labels, test_labels = convert_labels(train_targets), convert_labels(test_targets)


    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('Model: Linear + ReLU + Linear +ReLU + Linear + ReLU + Linear + Tanh')
    print('Loss: MSE')
    print('Optimizer: SGD')
    print('*************************************************************************')
    print('Training')
    print('*************************************************************************')
    # build network, loss and optimizer for Model 1
    my_model_design_1=[Linear(2,25), ReLU(), Linear(25,25), Dropout(p=0.5), ReLU(),
                       Linear(25,25), ReLU(),Linear(25,2),Tanh()]
    my_model_1=Sequential(my_model_design_1)
    optimizer_1=SGD(my_model_1,lr=1e-3)
    criterion_1=LossMSE()

    # train Model 1
    batch_size=1
    for epoch in range(50):
        temp_train_loss_sum=0.
        temp_test_loss_sum=0.
        num_train_correct=0
        num_test_correct=0
        
        # trained in batch-fashion: here batch size = 1
        for temp_batch in range(0,len(train_features), batch_size):
            temp_train_features=train_features.narrow(0, temp_batch, batch_size)  
            temp_train_labels=train_labels.narrow(0, temp_batch, batch_size)  
            
            for i in range(batch_size):
                # clean parameter gradient before each batch
                optimizer_1.zero_grad()  
                temp_train_feature=temp_train_features[i]
                temp_train_label=temp_train_labels[i]
                
                # forward pass to compute loss
                temp_train_pred=my_model_1.forward(temp_train_feature)
                temp_train_loss=criterion_1.forward(temp_train_pred,temp_train_label)
                temp_train_loss_sum+=temp_train_loss
                
                _, temp_train_pred_cat=torch.max(temp_train_pred,0)
                _, temp_train_label_cat=torch.max(temp_train_label,0)

                
                if temp_train_pred_cat==temp_train_label_cat:
                    num_train_correct+=1
  
                # calculate gradient according to loss gradient
                temp_train_loss_grad=criterion_1.backward(temp_train_pred,temp_train_label)
                # accumulate parameter gradient in each batch
                my_model_1.backward(temp_train_loss_grad)                       
            
            # update parameters by optimizer
            optimizer_1.step()
            
            
        # evaluate the current model on testing set
        # only forward pass is implemented
        for i_test in range(len(test_features)):
            temp_test_feature=test_features[i_test]
            temp_test_label=test_labels[i_test]

            temp_test_pred=my_model_1.forward(temp_test_feature)
            temp_test_loss=criterion_1.forward(temp_test_pred,temp_test_label)
            temp_test_loss_sum+=temp_test_loss

            
            _, temp_test_pred_cat=torch.max(temp_test_pred,0)
            _, temp_test_label_cat=torch.max(temp_test_label,0)

            if temp_test_pred_cat==temp_test_label_cat:
                num_test_correct+=1
            
            
        temp_train_loss_mean=temp_train_loss_sum/len(train_features)
        temp_test_loss_mean=temp_test_loss_sum/len(test_features)
        
        temp_train_accuracy=num_train_correct/len(train_features)
        temp_test_accuracy=num_test_correct/len(test_features)
        
        print("Epoch: {}/{}..".format(epoch+1, 50),
                      "Training Loss: {:.4f}..".format(temp_train_loss_mean),
                      "Training Accuracy: {:.4f}..".format(temp_train_accuracy), 
                      "Validation/Test Loss: {:.4f}..".format(temp_test_loss_mean),
                      "Validation/Test Accuracy: {:.4f}..".format(temp_test_accuracy),  )
        
        
        
    # # visualize the classification performance of Model 1 on testing set
    test_pred_labels_1=[]
    for i in range(1000): 
        temp_test_feature=test_features[i]
        temp_test_label=test_labels[i]

        temp_test_pred=my_model_1.forward(temp_test_feature)

        _, temp_train_pred_cat=torch.max(temp_test_pred,0)
        if test_targets[i].int() == temp_train_pred_cat.int():
            test_pred_labels_1.append(int(test_targets[i]))
        else:
            test_pred_labels_1.append(2)
            
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.scatter(test_features[:,0], test_features[:,1], c=test_pred_labels_1)
    axes.set_title('Classification Performance of Model 1')
    plt.show()
                      
      
    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('*************************************************************************')
    print('Model: Linear + ReLU + Linear + Dropout+ SeLU + Linear + Dropout + ReLU + Linear + Sigmoid')
    print('Loss: Cross Entropy')
    print('Optimizer: Adam')
    print('*************************************************************************')
    print('Training')
    print('*************************************************************************')
    
    # build network, loss function and optimizer for Model 2
    my_model_design_2=[Linear(2,25), ReLU(), Linear(25,25), Dropout(p=0.5), SeLU(),
                       Linear(25,25),Dropout(p=0.5), ReLU(),Linear(25,2),
                       Sigmoid()]
    my_model_2=Sequential(my_model_design_2)
    optimizer_2=Adam(my_model_2,lr=1e-3)
    criterion_2=CrossEntropy()

    # train Model 2
    batch_size=1
    epoch=0
    while(epoch<25):
        temp_train_loss_sum=0.
        temp_test_loss_sum=0.
        num_train_correct=0
        num_test_correct=0
        
        # trained in batch-fashion: here batch size = 1
        for temp_batch in range(0,len(train_features), batch_size):
            temp_train_features=train_features.narrow(0, temp_batch, batch_size)  
            temp_train_labels=train_labels.narrow(0, temp_batch, batch_size)  
            
            for i in range(batch_size):
                # clean parameter gradient before each batch
                optimizer_2.zero_grad()  
                temp_train_feature=temp_train_features[i]
                temp_train_label=temp_train_labels[i]
                
                # forward pass to compute loss
                temp_train_pred=my_model_2.forward(temp_train_feature)
                temp_train_loss=criterion_2.forward(temp_train_pred,temp_train_label)
                temp_train_loss_sum+=temp_train_loss
                
                _, temp_train_pred_cat=torch.max(temp_train_pred,0)
                _, temp_train_label_cat=torch.max(temp_train_label,0)

                
                if temp_train_pred_cat==temp_train_label_cat:
                    num_train_correct+=1
       
                
                # calculate gradient according to loss gradient
                temp_train_loss_grad=criterion_2.backward(temp_train_pred,temp_train_label)
                '''
                if (not temp_train_loss_grad[0]>=0) and (not temp_train_loss_grad[0]<0):
                    continue
                '''
                # accumulate parameter gradient in each batch
                my_model_2.backward(temp_train_loss_grad)     
                
            # update parameters by optimizer
            optimizer_2.step()
            
        # evaluate the current model on testing set
        # only forward pass is implemented
        for i_test in range(len(test_features)):
            temp_test_feature=test_features[i_test]
            temp_test_label=test_labels[i_test]

            temp_test_pred=my_model_2.forward(temp_test_feature)
            temp_test_loss=criterion_2.forward(temp_test_pred,temp_test_label)
            temp_test_loss_sum+=temp_test_loss

            
            _, temp_test_pred_cat=torch.max(temp_test_pred,0)
            _, temp_test_label_cat=torch.max(temp_test_label,0)

            if temp_test_pred_cat==temp_test_label_cat:
                num_test_correct+=1
            
            
        temp_train_loss_mean=temp_train_loss_sum/len(train_features)
        temp_test_loss_mean=temp_test_loss_sum/len(test_features)
        
        temp_train_accuracy=num_train_correct/len(train_features)
        temp_test_accuracy=num_test_correct/len(test_features)
        
        # in case there is gradient explosion problem, initiliza model again and restart training
        # but the situation seldom happens
        if (not temp_train_loss_grad[0]>=0) and (not temp_train_loss_grad[0]<0):
            epoch=0
            my_model_design_2=[Linear(2,25), ReLU(), Linear(25,25), Dropout(p=0.5), ReLU(),
                       Linear(25,25),Dropout(p=0.5), ReLU(),Linear(25,2),Sigmoid()]
            my_model_2=Sequential(my_model_design_2)
            optimizer_2=Adam(my_model_2,lr=1e-3)
            criterion_2=CrossEntropy()
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')
            print('Restart training because of gradient explosion')
            continue
        
        print("Epoch: {}/{}..".format(epoch+1, 25),
                      "Training Loss: {:.4f}..".format(temp_train_loss_mean),
                      "Training Accuracy: {:.4f}..".format(temp_train_accuracy), 
                      "Validation/Test Loss: {:.4f}..".format(temp_test_loss_mean),
                      "Validation/Test Accuracy: {:.4f}..".format(temp_test_accuracy),  )
        epoch+=1 
        
    # visualize the classification performance of Model 2 on testing set
    test_pred_labels_2=[]
    for i in range(1000): 
        temp_test_feature=test_features[i]
        temp_test_label=test_labels[i]

        temp_test_pred=my_model_2.forward(temp_test_feature)

        _, temp_train_pred_cat=torch.max(temp_test_pred,0)
        if test_targets[i].int() == temp_train_pred_cat.int():
            test_pred_labels_2.append(int(test_targets[i]))
        else:
            test_pred_labels_2.append(2)
            
    fig,axes = plt.subplots(1,1,figsize=(6,6))
    axes.scatter(test_features[:,0], test_features[:,1], c=test_pred_labels_2)
    axes.set_title('Classification Performance of Model 2')
    plt.show()
                
         

if __name__ == '__main__':
    main()