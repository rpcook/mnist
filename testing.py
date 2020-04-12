# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:23:33 2020

@author: Rob.Cook
"""

import tkinter as tk
import math

from random import random

class testingGUI:
    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network Test Interface')
        
        pxSize = 15
        self.pixelSize = pxSize
        self.drawingCanvas = tk.Canvas(master, width=pxSize*28, height=pxSize*28, bg='#000044')
        self.drawingCanvas.grid(row=0, column=0)
        self.drawingCanvas.bind('<B1-Motion>', self.__paintOnCanvas)
        self.__addDrawingBackground()
        
        self.pixelCanvas = tk.Canvas(master, width=pxSize*28, height=pxSize*28, bg='#440000')
        self.pixelCanvas.grid(row=0, column=1)
        
        self.processDrawingButton = tk.Button(text='Process Digit', command=self.__procDrawing)
        self.processDrawingButton.grid(row=2)
        
        self.clearButton = tk.Button(text='Clear', command=self.__clearDrawingCanvas)
        self.clearButton.grid(row=1)

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
                brightness = (25-distance)**3/10
                brightnessN = max(min(brightness,30),0)*(255/30)
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
        brushSize = self.pixelSize*1.3
        x1, y1 = (event.x - brushSize), (event.y - brushSize)
        x2, y2 = (event.x + brushSize), (event.y + brushSize)
        self.drawingCanvas.create_oval(x1,y1,x2,y2,fill='#FFFFFF',width=0)

root = tk.Tk()
g = testingGUI(root)
root.mainloop()