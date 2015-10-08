#!/usr/bin/env python

import needle_master as nm
import os

import matplotlib.pyplot as plt

files = os.listdir('.')

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        demo = nm.Demo(file)
for i in range(1,9):
    filename = "environment_%d.txt"%(i)

    # process as an environment
    env = nm.Environment(filename)
    plt.subplot(2,4,i)
    env.Draw()
plt.show()
