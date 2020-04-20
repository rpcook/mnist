# -*- coding: utf-8 -*-

import numpy as np
from struct import unpack

class database:
    def __init__(self, loadTestOnly=False): # defines empty class
        self.__trainingImages = []
        self.__trainingLabels = []
        self.__testImages = []
        self.__testLabels = []
        self.__loadData(loadTestOnly)
    
    def __loadData(self, loadTestOnly): # loads the mnist database from file
        if loadTestOnly==False:
            with open('train-images.idx3-ubyte', 'rb') as f:
                self.__trainingImages = self.__parseImageFile(f)
            with open('train-labels.idx1-ubyte', 'rb') as f:
                self.__trainingLabels = self.__parseLabelFile(f)
        with open('t10k-images.idx3-ubyte', 'rb') as f:
            self.__testImages = self.__parseImageFile(f)
        with open('t10k-labels.idx1-ubyte', 'rb') as f:
            self.__testLabels = self.__parseLabelFile(f)
    
    def __parseImageFile(self, fileObj):
        magicNumber = unpack('>i', fileObj.read(4))[0]
        nImages = unpack('>i', fileObj.read(4))[0]
        nRows = unpack('>i', fileObj.read(4))[0]
        nColumns = unpack('>i', fileObj.read(4))[0]
        nBytes = nImages * nRows * nColumns
        if magicNumber != 2051:
            raise NameError('Unexpected Magic Number in file')
        if nRows != 28 or nColumns != 28:
            raise NameError('Unexpected number of rows or columns in image file (!=28)')
        return np.array(unpack('{}B'.format(nBytes), fileObj.read(nBytes)), dtype=np.ubyte).reshape(nImages, 28, 28)
    
    def __parseLabelFile(self, fileObj):
        magicNumber = unpack('>i', fileObj.read(4))[0]
        nItems = unpack('>i', fileObj.read(4))[0]
        if magicNumber != 2049:
            raise NameError('Unexpected Magic Number in file')
        return np.array(unpack('{}B'.format(nItems), fileObj.read(nItems)), dtype=np.ubyte)
    
    def getData(self, idx, imgSet): # returns images and labels from the requested indexes of the requested set
        if imgSet == 'training':
            return self.__trainingImages[idx], self.__trainingLabels[idx]
        if imgSet == 'test':
            return self.__testImages[idx], self.__testLabels[idx]
        raise NameError('Unexpected Image Set type (not \'training\' or \'test\')')