import os
import csv
import sys
import numpy as np
from controller import *
from pdb import set_trace as woah
from context import needlemaster as nm

def control(env_path, save_path):
    """
            Lifan 03/04/2019 --- edited from play.py

            use a PID controller to reach a sequence of gates. This runs assuming it has
            access to the true state information

            This function also saves the demonstration so it can be visualized later

            Args:
                env_path: path/to/corresponding/environment/file
                save_path: (OPTIONAL) path/to/save/demo/file. If save_path is empty then images/demonstration won't be saved
    """

    environment        = nm.Environment(env_path, save_path=save_path)
    action_constraints = [10, np.pi/10]           # constraints on allowable motion
    parameters         = [0.1, 0.1]#[0.1,0.0009]             # proportional control parameters --- these have been hand-tuned
    save_images        = (save_path is not None)

    pid = PIDcontroller(params=parameters, bounds=action_constraints)
    """ ..................................... """
    done = False
    environment.render(save_image=save_images)

    while(not done):
        if(environment.next_gate is not None):
            next_gate = environment.gates[environment.next_gate]
        else:
            next_gate = None

        action  = pid.step([environment.needle.x, environment.needle.y, environment.needle.w], next_gate)
        _, _, done   = environment.step(action, 'play')#,save_image=save_images)

    print("________________________")
    environment.score(True)
    print("________________________")
    """ ..................................... """


#-------------------------------------------------------
# main()
args = sys.argv

if len(args) >= 2:
    args.append(None) # If a save path was given, won't access the None, if it wasn't then we will send None as save path
    control(args[1], args[2])

else:
    print("ERROR: command line arguments required")
    print("[Usage] python control.py <path to environment file> <OPTIONAL: save path for demonstration>")
