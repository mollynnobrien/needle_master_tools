import os
import sys
from context import needlemaster as nm
from pdb import set_trace as woah

def playback(env_path, demo_path):
    """
            Molly 11/30/2018

            read in an environment and demonstration and "hallucinate" the
            screen images

            Args:
                env_path: path/to/corresponding/environment/file
                demo_path: path/to/demo/file
    """
    environment = nm.Environment(env_path, save_image = True)
    demo        = nm.Demo(environment.width, environment.height, filename=demo_path)
    actions     = demo.u;
    state       = demo.s;

    """ ..................................... """
    running = True
    environment.render()

    while(running):
        environment.step(actions[environment.t,0:2])
        environment.render()
        running = environment.check_status()

    print("________________________")
    print(" Level " + str(demo.env))
    environment.score(True)
    print("________________________")
    """ ..................................... """


#-------------------------------------------------------
# main()
args = sys.argv
print(len(args))
if(len(args) == 3):
    playback(args[1], args[2])
else:
    print("ERROR: 2 command line arguments required")
    print("[Usage] python play.py <path to environment file> <path to demonstration>")
