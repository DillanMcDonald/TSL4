"""
File: loss_func.py
Author(s): Dillan McDonald
Date created: 11-25-18

Description:
A collection of loss functions that can be used interchangably with the linear classifier
"""
import numpy as np

def svm_l_i_vectorized(W,X,Y): #inputs are the Weight matrix, and the img data matrix, and the correct labels respectively
    scores = W.dot(X)
    margins = np.maximum(0,scores - scores[Y]+1)
    margins[Y] = 0
    loss_i = np.sum(margins)
    return loss_i;

def svm_loss(W, X, y, reg): #Inputs are matricies
  #loss = 0.0
  #dW = np.zeros(W.shape)
  num_train = X.shape[0]
  scores = X.dot(W)
  yi_scores = scores[np.arange(scores.shape[0]),y]
  #print(W)
  #print(y)
  #print(yi_scores)
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
  margins[np.arange(num_train),y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += 0.5 * reg * np.sum(W * W)

  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  dW /= num_train # Average
  dW += reg*W # Regularize
  #print(dW)
  return loss, dW

#def softmax(W,X):