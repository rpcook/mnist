# -*- coding: utf-8 -*-
import tkinter as tk

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
