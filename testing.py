# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:23:33 2020

@author: Rob.Cook
"""

import tkinter as tk
from tkinter import filedialog
import math
import mnist
import nn
from struct import unpack

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
        # TODO: add canvas, load button, (process button?), output digit
        self.nnProcessButton = tk.Button(text='GO!', command=self.__processNetwork)
        self.nnProcessButton.grid(row=0, column=2)
        
        self.nnCanvas = tk.Canvas(width=pxSize*30, height=pxSize*28)
        self.nnCanvas.grid(row=0,column=3)
        self.nnCanvas.bind('<Button-1>', self.__highlightNode)
        
        self.loadNNbutton = tk.Button(text='Load Neural Network', command=self.__loadNetwork)
        self.loadNNbutton.grid(row=1,column=3)
        
        # TODO: load network structure from file (dialog)
        # TODO: graphics of network structure
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
            return
        self.network.setStructure(neuronsPerLayer)
        self.__drawNetwork()
    
    def __drawNetwork(self):
        self.nnCanvas.delete('all')
        layersToDraw = self.network.getStructure()[1:]
        neuronSpacingX = round((25*self.pixelSize) / (len(layersToDraw) + 1))
        neuronRadius = round(round((28*self.pixelSize) / (max(layersToDraw) + 1)) / 4)
        for a in range(len(layersToDraw)):
            if a==0:
                continue
            neuronSpacingY = (28*self.pixelSize) / (layersToDraw[a] + 1)
            for b in range(layersToDraw[a]):
                lastSpacingY = (28*self.pixelSize) / (layersToDraw[a-1] + 1)
                for c in range(layersToDraw[a-1]):
                    self.nnCanvas.create_line(
                        (a+1)*neuronSpacingX,
                        (b+1)*neuronSpacingY,
                        (a+0)*neuronSpacingX,
                        (c+1)*lastSpacingY,
                        fill='#CCCCCC',
                        tags='connection')
        for a in range(len(layersToDraw)):
            neuronSpacingY = (28*self.pixelSize) / (layersToDraw[a] + 1)
            for b in range(layersToDraw[a]):
                self.nnCanvas.create_oval(
                    (a+1)*neuronSpacingX-neuronRadius, 
                    (b+1)*neuronSpacingY-neuronRadius, 
                    (a+1)*neuronSpacingX+neuronRadius, 
                    (b+1)*neuronSpacingY+neuronRadius,
                    fill='#000000',
                    tags=('neuron', 'L{}'.format(a+1), 'N{}'.format(b)))
    
    def __highlightNode(self, event):
        clickedElement = event.widget.find_closest(event.x, event.y)
        clickedTags = self.nnCanvas.itemcget(clickedElement[0], 'tags')
        if len(clickedElement)==1:
            if not ('highlight' in clickedTags and 'current' in clickedTags):
                self.nnCanvas.delete('highlight')
            if 'neuron' in clickedTags and 'current' in clickedTags:
                neuronCoords = self.nnCanvas.coords(clickedElement[0])
                layer = int(clickedTags[clickedTags.find('L')+1:clickedTags.find(' ',clickedTags.find('L'))])
                neuron = int(clickedTags[clickedTags.find('N')+1:clickedTags.find(' ',clickedTags.find('N'))])
                neuronSpacingY = (28*self.pixelSize) / (self.network.getStructure()[layer-1] + 1)
                for a in range(self.network.getStructure()[layer-1]):
                    self.nnCanvas.create_line(
                        (neuronCoords[0]+neuronCoords[2])/2,
                        (neuronCoords[1]+neuronCoords[3])/2,
                        ((layer-1)/layer)*((neuronCoords[0]+neuronCoords[2])/2),
                        (a+1)*neuronSpacingY,
                        fill='#FF0000',
                        tags='highlight')
                self.nnCanvas.create_oval(
                    neuronCoords[0],
                    neuronCoords[1],
                    neuronCoords[2],
                    neuronCoords[3],
                    fill='#FF0000',
                    tags='highlight')
    
    def __processNetwork(self):
        pass

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