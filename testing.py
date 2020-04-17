# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:23:33 2020

@author: Rob.Cook
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import math
import mnist
import nn
from struct import unpack
import numpy as np

# TODO: remove this debug function
from random import random
def randomArray(size, multiplier):
    x=[]
    for i in range(size):
        x.append(1-multiplier*random())
    return x

mnistTestData = mnist.database(loadTestOnly=True)

class testingGUI:
    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network Test Interface')
        
        # drawing canvas area
        pxSize = 15
        self.pixelSize = pxSize
        self.drawingCanvas = tk.Canvas(width=pxSize*28, height=pxSize*28, bg='#000044')
        self.drawingCanvas.grid(row=0, column=0)
        self.drawingCanvas.bind('<B1-Motion>', self.__paintOnCanvas)
        self.drawingCanvas.bind('<Button-1>', self.__paintOnCanvas)
        self.__addDrawingBackground()
        
        self.processDrawingButton = tk.Button(text='Process Digit', command=self.__procDrawing)
        self.processDrawingButton.grid(row=2)
        
        self.clearButton = tk.Button(text='Clear', command=self.__clearDrawingCanvas)
        self.clearButton.grid(row=1)
        
        # pixel canvas area
        self.pixelCanvas = tk.Canvas(width=pxSize*28, height=pxSize*28, bg='#440000')
        self.pixelCanvas.grid(row=0, column=1)
        
        self.testIndex = tk.Spinbox(from_=0, to=9999, command=self.__loadMNIST)
        self.testIndex.grid(row=1,column=1)
        self.testIndex.bind('<Return>', self.__loadMNIST)
        
        self.mnistLabelVar = tk.StringVar()
        self.mnistLabelVar.set('Label: #')
        self.mnistLabel = tk.Label(textvariable=self.mnistLabelVar)
        self.mnistLabel.grid(row=2,column=1)
        
        # neural network area
        # TODO: add output digit
        self.nnProcessButton = tk.Button(text='GO!', command=self.__processNetwork)
        self.nnProcessButton.grid(row=2, column=2)
        
        self.nnCanvas = tk.Canvas(width=pxSize*27, height=pxSize*28)
        self.nnCanvas.grid(row=0,column=2)
        self.nnCanvas.bind('<Button-1>', self.__highlightNode)
        
        self.loadNNbutton = tk.Button(text='Load Neural Network', command=self.__loadNetwork)
        self.loadNNbutton.grid(row=1,column=2)
        
        # TODO: graphics of network activity (pending actually how to evaluate that...)
        
        self.network = nn.network()
        
    def __loadNetwork(self):
        ##### following four lines commented during debug to save mouseclicks
        # file = filedialog.askopenfile(title='Select neural network', filetypes=(("neural network files","*.nn"),))
        # if file is None:
        #     return
        #with open(file.name, 'rb') as f:
        with open('randomNetwork.nn', 'rb') as f:
            nLayers = unpack('B', f.read(1))[0]
            neuronsPerLayer = unpack('{}H'.format(nLayers), f.read(2*nLayers))
            if neuronsPerLayer[0] != 784 or neuronsPerLayer[3] != 10:
                messagebox.showerror('Error', 'Neural network has wrong number\nof input (784) or output (10) nodes')
                return
            self.network = nn.network()
            self.network.setStructure(neuronsPerLayer)
            for i in range(len(neuronsPerLayer)-1):
                weights = np.array(unpack('<{}f'.format(neuronsPerLayer[i]*neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i]*neuronsPerLayer[i+1]))).reshape((neuronsPerLayer[i+1],neuronsPerLayer[i]))
                self.network.setConnectionWeights(i+1, weights)
                biases = np.array(unpack('<{}f'.format(neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i+1])))
                self.network.setNeuronBias(i+1, range(neuronsPerLayer[i+1]), biases)
        self.__drawNetwork()
    
    def __drawNetwork(self):
        self.nnCanvas.delete('all')
        layersToDraw = self.network.getStructure()[1:]
        neuronSpacingX = round((25*self.pixelSize) / (len(layersToDraw) + 1))
        neuronRadius = round(round((28*self.pixelSize) / (max(layersToDraw) + 1)) / 4)
        for a in range(len(layersToDraw)):
            neuronSpacingY = (28*self.pixelSize) / (layersToDraw[a] + 1)
            for b in range(layersToDraw[a]):
                if a==0:
                    lastSpacingY = (28*self.pixelSize) / (self.network.getStructure()[a] + 1)
                    cRange=range(0,28*28,10)
                else:
                    lastSpacingY = (28*self.pixelSize) / (layersToDraw[a-1] + 1)
                    cRange=range(layersToDraw[a-1])
                for c in cRange:
                    self.nnCanvas.create_line(
                        (a+1)*neuronSpacingX,
                        (b+1)*neuronSpacingY,
                        (a+0)*neuronSpacingX,
                        (c+1)*lastSpacingY,
                        fill='#AAAAAA',
                        tags='connection')
        layersToDraw = self.network.getStructure()[1:]
        maxActivation = 0
        for a in range(len(layersToDraw)):
            neuronSpacingY = (28*self.pixelSize) / (layersToDraw[a] + 1)
            for b in range(layersToDraw[a]):
                self.nnCanvas.create_oval(
                    (a+1)*neuronSpacingX-neuronRadius, 
                    (b+1)*neuronSpacingY-neuronRadius, 
                    (a+1)*neuronSpacingX+neuronRadius, 
                    (b+1)*neuronSpacingY+neuronRadius,
                    fill=self.__getNeuronActivationColour(a+1, b),
                    tags=('neuron', 'L{}'.format(a+1), 'N{}'.format(b)))
                if a == len(layersToDraw)-1:
                    self.nnCanvas.create_rectangle(
                        (a+1)*neuronSpacingX+2*neuronRadius-1, 
                        (b+1)*neuronSpacingY-neuronRadius-1, 
                        (a+2)*neuronSpacingX+2*neuronRadius+1, 
                        (b+1)*neuronSpacingY+neuronRadius+1)
                    activation = self.network.getNeuronActivation(a+1, b)
                    rect = self.nnCanvas.create_rectangle(
                        (a+1)*neuronSpacingX+2*neuronRadius, 
                        (b+1)*neuronSpacingY-neuronRadius, 
                        (a+1+activation)*neuronSpacingX+2*neuronRadius+1, 
                        (b+1)*neuronSpacingY+neuronRadius+1,
                        fill='#FF0000',
                        width=0)
                    if activation > maxActivation:
                        maxActivation = activation
                        maxRect = rect
            if maxActivation != 0:
                self.nnCanvas.itemconfig(maxRect, fill='#00FF00')
    
    def __highlightNode(self, event):
        if len(self.nnCanvas.find_all())==0:
            return
        clickedElement = event.widget.find_closest(event.x, event.y)
        clickedTags = self.nnCanvas.itemcget(clickedElement[0], 'tags')
        if len(clickedElement)==1:
            if not ('highlight' in clickedTags and 'current' in clickedTags) or ('highlight' in clickedTags and 'neuron' in clickedTags):
                neuronCoords = self.nnCanvas.coords(clickedElement[0])
                self.nnCanvas.delete('highlight')
            if 'neuron' in clickedTags and 'current' in clickedTags:
                layer = int(clickedTags[clickedTags.find('L')+1:clickedTags.find(' ',clickedTags.find('L'))])
                neuron = int(clickedTags[clickedTags.find('N')+1:clickedTags.find(' ',clickedTags.find('N'))])
                neuronSpacingY = (28*self.pixelSize) / (self.network.getStructure()[layer-1] + 1)
                xDiffBetweenLayers = ((neuronCoords[0]+neuronCoords[2])/2)/layer
                for a in range(self.network.getStructure()[layer-1]):
                    self.nnCanvas.create_line(
                        (neuronCoords[0]+neuronCoords[2])/2,
                        (neuronCoords[1]+neuronCoords[3])/2,
                        (layer-1)*xDiffBetweenLayers,
                        (a+1)*neuronSpacingY,
                        fill=self.__getConnectionWeightColour(layer-1, neuron, a),
                        tags='highlight')
                    if layer > 1:
                        self.nnCanvas.create_oval(
                            neuronCoords[0]-xDiffBetweenLayers,
                            (a+1)*neuronSpacingY-(neuronCoords[2]-neuronCoords[0])/2,
                            neuronCoords[2]-xDiffBetweenLayers,
                            (a+1)*neuronSpacingY+(neuronCoords[2]-neuronCoords[0])/2,
                            fill=self.__getNeuronActivationColour(layer-1, a),
                            tags=('highlight', 'neuron', 'L{}'.format(layer-1), 'N{}'.format(a)))
                self.nnCanvas.create_oval(
                    neuronCoords[0],
                    neuronCoords[1],
                    neuronCoords[2],
                    neuronCoords[3],
                    fill=self.__getNeuronActivationColour(layer, neuron),
                    outline=self.__getNeuronBiasColour(layer, neuron),
                    width=(neuronCoords[2]-neuronCoords[0])/3,
                    tags='highlight')
    
    def __getNeuronBiasColour(self, layer, neuron):
        bias = self.network.getNeuronBias(layer, neuron)
        maxBrightness = 1.0
        brightness = min(int(round(255*bias/maxBrightness)),255)
        hexValue = '%0.2X' % abs(brightness)
        if bias > 0:
            return '#00'+hexValue+'00'
        else:
            return '#'+hexValue+'0000'
    
    def __getConnectionWeightColour(self, layer, currentLayerNeuron, lastLayerNeuron):
        weight = self.network.getConnectionWeights(layer, currentLayerNeuron, lastLayerNeuron)
        maxBrightness = 1.0
        brightness = min(int(round(255*weight/maxBrightness)),255)
        hexValue = '%0.2X' % abs(brightness)
        if weight > 0:
            return '#00'+hexValue+'00'
        else:
            return '#'+hexValue+'0000'
    
    def __getNeuronActivationColour(self, layer, neuron):
        activation = self.network.getNeuronActivation(layer, neuron)
        brightness = int(round(255*activation))
        hexValue = '%0.2X' % brightness
        return '#'+hexValue*3
    
    def __processNetwork(self):
        if self.network.getStructure() == []:
            return
        # TODO: remove next two lines of debug (randomise network contents)
        for i in range(len(self.network.getStructure())):
            self.network.setNeuronActivation(i, range(self.network.getStructure()[i]), randomArray(self.network.getStructure()[i], 1))
        self.__drawNetwork()

    def __loadMNIST(self, *args):
        try:
            testIndex = int(self.testIndex.get())
        except:
            self.testIndex.delete(0, tk.END)
            return
        if testIndex < 0 or testIndex > 9999:
            return
        mnistDigit, mnistLabel = mnistTestData.getData(testIndex, 'test')
        self.mnistLabelVar.set('Label: {}'.format(mnistLabel))
        pxSz = self.pixelSize
        self.pixelCanvas.delete('all')
        for row in range(28):
            for col in range(28):
                pxBrightness = mnistDigit[col][row]
                pxColour = '#%02x%02x%02x' % (int(pxBrightness),int(pxBrightness),int(pxBrightness))
                self.pixelCanvas.create_rectangle(row*pxSz+1,col*pxSz+1,(row+1)*pxSz+2,(col+1)*pxSz+2,fill=pxColour,width=pxSz/10)

    def __procDrawing(self):
        self.pixelCanvas.delete('all')
        if len(self.drawingCanvas.find_all()) == 1:
            return
        self.drawingCanvas.delete('background')
        pxSz = self.pixelSize
        for row in range(28):
            for col in range(28):
                pxX = row*pxSz+pxSz/2
                pxY = col*pxSz+pxSz/2
                closestPointSpecifier = self.drawingCanvas.find_closest(pxX, pxY)
                coords = self.drawingCanvas.coords(closestPointSpecifier[0])
                centreX = (coords[0] + coords[2]) / 2
                centreY = (coords[1] + coords[3]) / 2
                distance = math.sqrt((centreX-pxX)**2+(centreY-pxY)**2)
                distanceN = distance / pxSz
                brightness = (150/(max(max(distanceN,0.5)-0.5,0.1))**2)-50
                brightnessN = max(min(brightness,255),0)
                pxColour = '#%02x%02x%02x' % (int(brightnessN),int(brightnessN),int(brightnessN))
                self.pixelCanvas.create_rectangle(row*pxSz+1,col*pxSz+1,(row+1)*pxSz+2,(col+1)*pxSz+2,fill=pxColour,width=pxSz/10)
        self.__addDrawingBackground()

    def __addDrawingBackground(self):
        pxSize = self.pixelSize
        bg = self.drawingCanvas.create_rectangle(4*pxSize,4*pxSize,24*pxSize,24*pxSize,fill='#000000',tags='background')
        self.drawingCanvas.tag_lower(bg)

    def __clearDrawingCanvas(self):
        self.drawingCanvas.delete('all')
        self.__addDrawingBackground()
        self.__procDrawing()

    def __paintOnCanvas(self, event):
        brushSize = self.pixelSize*1.2
        x1, y1 = (event.x - brushSize), (event.y - brushSize)
        x2, y2 = (event.x + brushSize), (event.y + brushSize)
        self.drawingCanvas.create_oval(x1,y1,x2,y2,fill='#FFFFFF',width=0)

root = tk.Tk()
g = testingGUI(root)
root.mainloop()