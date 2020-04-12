# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:52:46 2020

@author: Rob.Cook
"""
import numpy as np

class mnist:
    def __init__(self): # defines empty class
        self.tstData = []
        self.tstLabels = []
        self.trnData = []
        self.trnLabels = []
    
    def loadData(self): # loads the mnist database from file
        pass
    
    def __parseLabelFile(self, fileObj):
        pass
    
    def __parseImageFile(self, fileObj):
        pass
    
    def images(self, idx):
        pass