# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:05 2020

@author: Rob.Cook
"""

import numpy as np

def sigmoid(x):
    # logistic function
    return (1 / (1 + np.exp(-x)))

def dsigmoid(x):
    # first derivative of logistic function
    return sigmoid(x)*sigmoid(-x)