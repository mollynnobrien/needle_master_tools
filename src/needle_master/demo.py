# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:31:19 2015

@author: Chris
"""

import numpy as np

'''
Stores data for a single performance of a task.
You can pull this data out as a Numpy array.
'''
class Demo:
    
    def __init__(self,filename=None):
        if not filename is None:
            handle = file(filename,'r')
            self.Load(handle)

    
    '''
    Load demonstration from a file
    '''
    def Load(self,handle):
        pass
