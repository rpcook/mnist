# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:23:33 2020

@author: Rob.Cook
"""

import tkinter as tk

class testingGUI:
    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network Test Interface')
        
        self.label = tk.Label(master, text='sometext')
        self.label.pack()
        
        pxSize = 10
        self.__pixelSize = pxSize
        self.drawingCanvas = tk.Canvas(master, width=pxSize*28, height=pxSize*28, bg='#000044')
        self.drawingCanvas.pack()
        self.drawingCanvas.bind('<B1-Motion>', self.__paintOnCanvas)
        self.drawingCanvas.create_rectangle(4*pxSize,4*pxSize,24*pxSize,24*pxSize,fill='#000000')

    def __paintOnCanvas(self, event):
        brushSize = self.__pixelSize*1.6
        x1, y1 = (event.x - brushSize), (event.y - brushSize)
        x2, y2 = (event.x + brushSize), (event.y + brushSize)
        self.drawingCanvas.create_oval(x1,y1,x2,y2,fill='#FFFFFF',width=0)

root = tk.Tk()
g = testingGUI(root)
root.mainloop()