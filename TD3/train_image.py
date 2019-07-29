import numpy as np
import torch
import random, math
import os, sys, argparse
from os.path import abspath
from os.path import join as pjoin
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import plotly

cur_dir= os.path.dirname(abspath(__file__))
sys.path.append(abspath(pjoin(cur_dir, '..')))
from needlemaster.environment import Environment

'''
Train the CNN to see the position of the needle, before we start
using RL
'''

model = ImageToPos() 

def train():
    

Ts, rewards, Best_avg_reward = [], [], -1e5

def run(args):
    env_data_name = os.path.splitext(
        os.path.basename(args.filename))[0]

    base_filename = '{}_{}_{}'.format(
        args.env_name, args.policy_name, env_data_name)

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
    else:
        args.device = torch.device('cpu')

    ## environment setup
    log_f = open('log_' + base_filename + '.txt', 'w')

    """ setting up environment """
    env = Environment(filename = args.filename, mode=args.mode,
        stack_size = args.stack_size)

    # Initialize policy
    action_dim = 1
    state_dim = len(env.gates) + 9
    from .DDPG_image import ImageToPos

    model = ImageToPos(args.img_stack).to(args.device)

        policy = TD3(state_dim, action_dim, args.stack_size, max_action, args.mode)
    elif args.policy_name == 'ddpg':
        from .DDPG_image import DDPG
        policy = DDPG(state_dim, action_dim, args.stack_size, max_action, args.mode)
    else:
      raise ValueError(
        args.policy_name + ' is not recognized as a valid policy')

    ## load pre-trained policy
    #try:
    #    policy.load(result_path)
    #except:
    #    pass

    replay_buffer = NaivePrioritizedBuffer(int(args.max_size))

    state = env.reset()
    #print('state = ', state) # debug
    total_timesteps = 0
    episode_num = 0
    done = False
    zero_noise = np.zeros((action_dim,))

    policy.actor.eval() # set for batchnorm

    while total_timesteps < args.max_timesteps:

        # Evaluate episode
        if (total_timesteps % args.eval_freq == 0
            and total_timesteps != 0):
            print("Evaluating policy") # debug
            best_reward = evaluate_policy(
                env, args, policy, total_timesteps, test_path, result_path)

        """ exploration rate decay """

        # Check if we should add noise
        percent_greedy = 1. - max(1., total_timesteps / greedy_decay_rate)
        epsilon_greedy = args.epsilon_greedy * percent_greedy
        if random.random() < epsilon_greedy:
            noise_std = ((args.explo_noise - epsilon_final) *
                math.exp(-1. * total_timesteps / decay_rate))
            ep_decay.append(noise_std)
            # log_f.write('epsilon decay:{}\n'.format(noise_std)) # debug
            noise = np.random.normal(0, noise_std, size=action_dim)
        else:
            noise = zero_noise

        # """ using PID controller """
        # state_pid = state[0:3]
        # action = pid.PIDcontroller( state_pid, env.next_gate, env.gates, total_timesteps)
        # print("action based on PID: " + str(action))

        """ action selected based on pure policy """
        action2 = policy.select_action(state)

        # state_pid = state[0:3]
        # guidance = pid.PIDcontroller( state_pid, env.next_gate, env.gates, total_timesteps)
        action = np.clip(action2 + noise, -max_action, max_action)

        #print "action: ", action, "noise: ", noise, "action2: ", action2 # debug

        # Perform action
        new_state, reward, done = env.step(action)

        # Store data in replay buffer
        replay_buffer.add(state, new_state, action, reward, done)

        ## Train over the past episode
        if done:
            print ("Training. episode ", episode_num, "R =", env.total_reward) # debug

            ## training
            str = 'Total:{}, Episode Num:{}, Step:{}, Reward:{}'.format(
              total_timesteps, episode_num, env.t, env.total_reward)

            log_f.write(str + '\n')
            if episode_num % 20 == 0:
                print(str)
                env.render(save_image=True, save_path=save_path)

            policy.actor.train() # Set actor to training mode

            beta = min(1.0, beta_start + total_timesteps *
                (1.0 - beta_start) / beta_frames)
            policy.train(replay_buffer, env.t, beta, args)

            policy.actor.eval() # set for batchnorm

            # Reset environment
            new_state = env.reset()
            done = False
            episode_num += 1

            # print "Training done" # debug

        state = new_state
        total_timesteps += 1

    print("Best Reward: " + best_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', default=False, action='store_true',
        help='Disable CUDA')
    parser.add_argument("--env_name", default="NeedleMaster",
        help='OpenAI gym environment name')
    parser.add_argument("--seed", default=1e6, type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--pid_interval", default=5e3, type=int,
        help='How many time steps purely random policy is run for')
    parser.add_argument("--eval_freq", default=1e3, type=int,
        help='How often (time steps) we evaluate')
    parser.add_argument("--pid_freq", default=1e4, type=int,
        help='How often we get back to pure random action')
    parser.add_argument("--max_timesteps", default=5e6, type=float,
        help='Max time steps to run environment for')
    parser.add_argument("--learning_start", default=0, type=int,
        help='Timesteps before learning')
    parser.add_argument("--save_models", action= "store",
        help='Whether or not models are saved')
    parser.add_argument("--expl_noise", default=0.5, type=float,
        help='Starting std of Gaussian exploration noise')
    parser.add_argument("--epsilon_greedy", default=0.08, type=float,
        help='Starting percentage of choosing random noise')
    parser.add_argument("--batch_size", default=32, type=int,
        help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, type=float,
        help='Discount factor')
    parser.add_argument("--tau", default=0.005, type=float,
        help='Target network update rate')
    parser.add_argument("--policy_noise", default=0.2, type=float,
        help='Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5, type=float,
        help='Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int,
        help='Frequency of delayed policy updates')
    parser.add_argument("--max_size", default=5e4, type=int,
        help='Frequency of delayed policy updates')
    parser.add_argument("--stack_size", default=2, type=int,
        help='How much history to use')
    parser.add_argument("--evaluation_episodes", default=6, type=int)
    parser.add_argument("--profile", default=False, action="store_true",
        help="Profile the program for performance")
    parser.add_argument("--mode", default = 'state',
        help="Choose image or state, options are rgb_array and state")
    parser.add_argument("filename", help='File for environment')
    parser.add_argument("policy_name", default="TD3", type=str)

    args = parser.parse_args()
    if args.profile:
        import cProfile
        cProfile.run('run(args)', sort='cumtime')
    else:
        run(args)
