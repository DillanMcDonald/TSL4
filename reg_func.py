"""
File: reg_func.py
Author(s): Dillan McDonald
Date created: 11-25-18

Description:
A collection of regression functions that can be used interchangably in the loss function
"""
import numpy as np

def l2_reg(W , l): #needs Weight matrix and lambda
    return -np.sum((W*W))*l

#def l1_reg()
