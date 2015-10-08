#!/usr/bin/env python

import needle_master as nm
import os

files = os.listdir('.')

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        demo = nm.Demo(file)
    elif file[:11] == 'environment':
        # process as an environment
        env = nm.Environment(file)
