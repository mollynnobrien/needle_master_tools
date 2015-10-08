# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:31:19 2015

@author: Chris
"""

import numpy as np
import needle_master as nm

'''
Stores data for a single performance of a task.
You can pull this data out as a Numpy array.
'''
class Demo:
    
    def __init__(self,filename=None):
        if not filename is None:
            handle = file(filename,'r')
            (env,time) = nm.ParseDemoName(filename)
            self.Load(handle)
            self.env = env

    
    '''
    Load demonstration from a file
    '''
    def Load(self,handle):
        data = handle.readline()
        print data
