# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:31:19 2015

@author: Chris
"""

import numpy as np
# from file import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
Stores data for a single performance of a task.
You can pull this data out as a Numpy array.
'''
class Demo:

    def __init__(self,env_height,env_width,filename=None):

        self.t = None
        self.s = None
        self.u = None

        self.env_height = env_height
        self.env_width = env_width

        if not filename is None:
            handle = file(filename,'r')
            (env,time) = self.parse_name(filename)
            self.load(handle)
            self.env = env

    @staticmethod
    def parse_name(filename):
        toks = filename.split('/')[-1].split('.')[0].split('_')
        return (int(toks[1]),toks[2])

    def draw(self):
        plt.plot(self.s[:,0],self.s[:,1])

    '''
    Load demonstration from a file
    '''
    def load(self, handle):

        t = []
        s = []
        u = []

        data = handle.readline()
        while not data is None and len(data)>0:
            data = [float(x) for x in data.split(',')]

            t.append(data[0])
            s.append(data[1:4])
            u.append(data[4:])

            data = handle.readline()

        self.t = np.array(t)#.transpose()
        self.s = np.array(s)#.transpose()
        #self.s[1,:] = self.s[1,:]
        self.u = np.array(u)#.transpose()
