#!/usr/bin/env python

import needle_master as nm
import os
from pdb import set_trace as woah

import matplotlib

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
files = os.listdir('.')

start_idx = 1
end_idx = 11
envs = [0]*(end_idx-start_idx)
ncols = 5

for i in range(start_idx,end_idx):
    filename = "environment_%d.txt"%(i)

    # process as an environment
    env = nm.Environment(filename)
    envs[i-1] = env
    plt.subplot(2,ncols,i)
    env.Draw()

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        (env,t) = nm.ParseDemoName(file)

        # draw
        if env < end_idx and env >= start_idx:
            demo = nm.Demo(env_height=envs[env-1].height,env_width=envs[env-1].width,filename=file)
            plt.subplot(2,ncols,env)
            demo.Draw()
plt.savefig('test_output.png')
plt.show()
