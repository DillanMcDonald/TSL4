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

def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  # Implement a vectorized version of the structured SVM loss, storing the
  # result in loss.

  scores = X.dot(W)
  yi_scores = scores[np.arange(scores.shape[0]),y] # http://stackoverflow.com/a/23435843/459241
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
  margins[np.arange(num_train),y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += 0.5 * reg * np.sum(W * W)

  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #

  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  # Average
  dW /= num_train

  # Regularize
  dW += reg*W

  return loss, dW

#def softmax(W,X):