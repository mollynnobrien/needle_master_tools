#!/usr/bin/env python

import needle_master as nm
import os

import matplotlib.pyplot as plt

files = os.listdir('.')

envs = [0]*8

for i in range(1,9):
    filename = "environment_%d.txt"%(i)

    # process as an environment
    env = nm.Environment(filename)
    envs[i-1] = env
    plt.subplot(2,4,i)
    env.Draw()

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        (env,t) = nm.ParseDemoName(file)

        # draw
        if env < 9 and env > 0:
            demo = nm.Demo(env_height=envs[env-1].height,env_width=envs[env-1].width,filename=file)
            plt.subplot(2,4,env)
            demo.Draw()

plt.show()
