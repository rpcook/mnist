# -*- coding: utf-8 -*-
import tkinter as tk
import numpy as np

class elements:
    def __init__(self, kwargs):
        if 'rootWindow' in kwargs and 'progressBarWidget' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.progressBarWidget = kwargs['progressBarWidget']
            self.progressBarAvailable = True
        else:
            self.progressBarAvailable = False
            
        if 'rootWindow' in kwargs and 'messageLog' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.messageLog = kwargs['messageLog']
            self.messageLogAvailable = True
        else:
            self.messageLogAvailable = False
            
        if 'rootWindow' in kwargs and 'graphCanvas' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.graphCanvas = kwargs['graphCanvas']
            self.graphAvailable = True
        else:
            self.graphAvailable = False

    def drawGraphs(self, costHistory):
        if self.graphAvailable:
            gc = self.graphCanvas
            gc.delete('all')
            gc.create_text(15,75,angle=90,text='Loss / error')
            boldLine = '#aaaaaa'
            softLine = '#dddddd'
            costLine = '#0000ff'
            if len(costHistory) < 2:
                numMajorGrids = 1
            else:
                numMajorGrids = int(np.ceil(np.log10(10/min(costHistory))))
            
            gc.create_line(30,8,30,145)
            for i in range(numMajorGrids+1):
                gc.create_line(31, 8+i*(136/numMajorGrids), 500, 8+i*(136/numMajorGrids), fill=boldLine)
            for i in range(numMajorGrids):
                for j in range(2,10):
                    yPos = 8+(i+1-np.log10(j))*(136/numMajorGrids)
                    gc.create_line(31, yPos, 500, yPos, fill=softLine)
                    
            if len(costHistory) > 1:
                for i in range(len(costHistory)-1):
                    yPosStart = 8 + 136*((1 - np.log10(costHistory[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(costHistory[i+1])) / numMajorGrids)
                    gc.create_line(31+i*(470/(len(costHistory)-1)), yPosStart, 31+(i+1)*(470/(len(costHistory)-1)), yPosStop, fill=costLine)
            self.rootWindow.update()

    def writeToLog(self, message):
        if self.messageLogAvailable:
            self.messageLog.configure(state='normal')
            if len(message)>7:
                if message[0:5]=='ERROR':
                    self.messageLog.insert(tk.END, message[0:5], 'error')
                    self.messageLog.insert(tk.END, message[5:])
                elif message[0:7]=='WARNING':
                    self.messageLog.insert(tk.END, message[0:7], 'warning')
                    self.messageLog.insert(tk.END, message[7:])
                else:
                    self.messageLog.insert(tk.END, message)
            else:
                self.messageLog.insert(tk.END, message)
            self.messageLog.configure(state='disabled')
            self.messageLog.yview(tk.END)
            self.rootWindow.update()

    def updateProgressBar(self, progress):
        if self.progressBarAvailable:
            self.progressBarWidget['value'] = progress   
            self.rootWindow.update()
