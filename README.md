# NEEDLE MASTER TOOLS

To use this code, download the game Needle Master [from the Google Play store](https://play.google.com/store/apps/details?id=edu.jhu.lcsr.needlemaster). Go into options and enable data collection. Then, plug your Android device into a computer, go to the storage, and copy out the files from the `needle_master_trials` folder.

Installing the requirements for this python module is as simple as calling `python setup.py develop --user`

To run DQN on an example environment, call
`python3 -m rainbow_dqn.main data/environment_14.txt`

To run PPO on an example envrionment, go to lifan2/DDPG_TD3, call
`python -m main_PPO [data/environment_14.txt(choose environment)] [image or state(choose to use state or image)]`


