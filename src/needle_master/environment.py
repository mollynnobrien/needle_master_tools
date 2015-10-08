# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:30:52 2015

@author: Chris Paxton
"""

import numpy as np

class Environment:
    
    def __init__(self,filename=None):
        if not filename is None:
            handle = file(filename,'r')
            self.Load(handle)
    
    '''
    Load an environment file.
    '''
    def Load(self,handle):
        pass
    
class Gate:
    
    def __init__(self):
        pass
    
    '''
    Load Gate from file at the current position.
    '''
    def Load(self,handle):
        pass
