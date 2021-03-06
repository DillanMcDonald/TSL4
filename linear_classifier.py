"""
File: linear_classifier.py
Author(s): Dillan McDonald
Date created: 11-25-18

Description:
A basic Linear Classifier for image processing
"""

import get_data as gd
import loss_func as lf
import reg_func as rf
import numpy as np
import string

#hyperparameters
l=1 #lambda
kfoldnum = 5 #number of folds

class BasicLinearClassifier:

    def __init__(self, loss_func = None,reg_func = None,kfolds = 0):
        self.loss_func = loss_func
        self.reg_func = reg_func
        self.kfolds = kfolds

    def split_train_data(self,imgs,labels,labelstr,kfoldnum): #splits the training data into 5 sets
        i=1
        j=0
        numimgsperblock = (int)(imgs.shape[0]/kfoldnum)
        blockimg = np.zeros((numimgsperblock,imgs.shape[1]))
        blocklbl = np.zeros((numimgsperblock))
        blocklblstr = np.empty((numimgsperblock),str)
        kfoldimg = np.zeros((kfoldnum,numimgsperblock,imgs.shape[1]))
        kfoldlbl = np.zeros((kfoldnum,numimgsperblock))
        kfoldlblstr = np.empty((kfoldnum,numimgsperblock),str)

        while i < kfoldnum+1:
            while j < i*numimgsperblock:
                blockimg[j-((i-1)*numimgsperblock),:] = imgs[j,:]
                blocklbl[j-((i-1)*numimgsperblock)] = labels[j]
                blocklblstr[j-((i-1)*numimgsperblock)] = labelstr[j]
                j += 1

            #print(blockimg.shape)
            kfoldimg[i-1] = blockimg
            kfoldlbl[i-1] = blocklbl
            kfoldlblstr[i-1] = blocklblstr
            i += 1
        return kfoldimg,kfoldlbl,kfoldlblstr

    def validate(self,X,Y,W): #same as train, but is used to optimize your hyperparameters
        i=0
        correct = 0
        while i<X.shape[0]:
            scores = X[i].dot(W)
            prediction_index = np.argmax(scores)
            #print(Y[prediction_index])
            if Y[i] == Y[prediction_index]:
                correct +=1
            i+=1
        percent_correct = correct/X.shape[0]
        return percent_correct

    def train(self,X,Y,W, lr=1e-6): #input the kfold sets here
        i=0
        #print(Y)
        xlen = X.shape[1]
        #W = np.ones((xlen , 100))
        loss,dw = lf.svm_loss(W,X,Y.astype(int),rf.l2_reg(W,l))
        print("Overall Model Loss: ",loss)
        W= np.add(W, lr * dw * -1)
        #print(W.shape)
        return W    #training returns the optimum weights

    def batchtest(self,X,Y,W): #where X is the data the images, Y is the correct Label, and W is the weights
        i=0
        correct = 0
        while i<X.shape[0]:
            scores = X[i].dot(W)
            prediction_index = np.argmax(scores)
            #print(Y[prediction_index])
            if Y[i] == Y[prediction_index]:
                correct +=1
        percent_correct = correct/X.shape[0]
        return percent_correct

#    def singletest(self,X,Y,W):

def main():
    test = BasicLinearClassifier()
    all_train_imgs, all_train_lbls, all_train_lbls_str = gd.get_data(gd.img_dataset_train_path)
    test_imgs, test_lbls, test_lbls_str = gd.get_data(gd.img_dataset_test_path)
    kfoldimg,kfoldlbl,kfoldlblstr = test.split_train_data(all_train_imgs,all_train_lbls,all_train_lbls_str,kfoldnum)
    i=0
    j=0
    #print(all_train_imgs.shape[1])
    W = np.ones((kfoldnum,all_train_imgs.shape[1],100))
    Wf =np.zeros(kfoldnum)
    while i < kfoldnum :
        validate_set_img = kfoldimg[i]
        validate_set_lbl = kfoldlbl[i]
        #validate_set_lbl_str = kfoldlblstr[i]
        while j < kfoldnum:
            if j!=i:
                print("Set ", j)
                W[j] = test.train(kfoldimg[j],kfoldlbl[j],W[j-1])
            else :
                W[j]=0;
            j += 1
        Wf[i] = np.sum(W)/(kfoldnum-1)
        #print("Batch ",i," Performance: ",test.validate(validate_set_img,validate_set_lbl,Wf[i])*100)

        i+=1

if __name__ == '__main__':
    main()