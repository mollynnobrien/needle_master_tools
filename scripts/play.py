import os
import sys
from context import needlemaster as nm
from pdb import set_trace as woah

def playback(env_path, demo_path, device):
    """
            Molly 11/30/2018

            read in an environment and demonstration and "hallucinate" the
            screen images

            Args:
                env_path: path/to/corresponding/environment/file
                demo_path: path/to/demo/file
    """
    environment = nm.Environment(env_path)
    demo        = nm.Demo(environment.width, environment.height, filename=demo_path)
    actions     = demo.u;
    state       = demo.s;

    if(device == 'mollys_phone'):
        demo.device_width = 2560
        demo.device_height = 1440
    # convert the coordinates into the default image size
    # demo.convert()
    """ ..................................... """
    running = True
    environment.draw(True)

    while(running):
        running = environment.step(actions[environment.t,0:2])
        environment.draw(True)

    print("________________________")
    print(" Level " + str(demo.env))
    environment.score(True)
    print("________________________")
    """ ..................................... """
    woah()


#-------------------------------------------------------
# main()
args = sys.argv
print(len(args))
if(len(args) == 4):
    playback(args[1], args[2], args[3])
else:
    print("ERROR: 2 command line arguments required")
    print("[Usage] python play.py <path to environment file> <path to demonstration> <demo device>")
