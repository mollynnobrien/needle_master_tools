#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from datetime import datetime
import random
import torch
from os.path import abspath
from os.path import join as pjoin

from .agent import Agent
from .memory import ReplayMemory
from .test import test

cur_dir = os.path.dirname(abspath(__file__))
sys.path.append(abspath(pjoin(cur_dir, '..')))
from needlemaster.environment_discrete import Environment, PID

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(1e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--channels', type=int, default=1, metavar='C', help='Number of channels per image')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=128, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument("--mode", default = 'state', help="Choose image or state, options are rgb_array and state")
parser.add_argument('filename', help='File for environment')

# for pycharm
# env_name = 'environment_1'
# env_path = 'C:/Users/icage/needle_master_tools-lifan/environments/' + env_name + '.txt'

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

# making dirs
# # get environment name
count = 0
for chr in args.filename:
    if chr == '/':
        start = count
    elif chr == '.':
        end = count
    count += 1
env_name = args.filename[start+1:end]

validation_path = "./" +env_name + "/DQN_validation"
out_path = "./" +env_name +"/DQN_out"
test_path = "./" +env_name +"/DQN_test"
result_path = "./" +env_name +"/DQN_results"
if not os.path.exists(validation_path):
    os.makedirs(validation_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Environment
img_stack = 4

## PID controller
parameter = [0.1, 0.0009]

## for pycharm
# env = Environment(mode = args.mode, stack_size = img_stack, filename = env_path)

# ## for scripts
env = Environment(mode = args.mode, stack_size = img_stack, filename = args.filename)
action_dim = env.action_space()

pid = PID( parameter, env.width, env.height )

# Agent
state = env.reset()
state_dim = state.size()[1]
dqn = Agent(args, state_dim, env)
mem = ReplayMemory(args, args.memory_capacity,  env.reset())
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
# load pre-trained model first (if there is any)
# dqn.load(result_path)

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size,  env.reset())

T, done = 0, True
env.episode_num = 0
episode_timesteps = 0  ## counting timesteps in one episode
print(" ------------ validation ---------- ")
while T < args.evaluation_size:
  if done:
    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (T, env.episode_num, episode_timesteps, env.total_reward))
    env.render(save_image=True, save_path = validation_path)
    state, done = env.reset(), False
    env.episode_num += 1
    episode_timesteps = 0

  next_state, _, done = env.step(random.randint(0, action_dim - 1))
  val_mem.append(state, None, None, done)
  state = next_state
  T += 1
  episode_timesteps += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, test_path, result_path, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

else:
  # Training loop
  dqn.train()
  T, done = 0, True
  episode_timesteps = 0
  while T < args.T_max:
    if done:
        print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (T, env.episode_num, episode_timesteps, env.total_reward))
        # if env.episode_num % 20 == 0:
        env.render(save_image = True, save_path = out_path)
        state, done = env.reset(), False
        env.episode_num += 1
        episode_timesteps = 0

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    state = state.to(args.device)
    action_idx = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action_idx)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action_idx, reward, done)  # Append transition to memory
    T += 1
    episode_timesteps += 1

    if T % args.log_interval == 0:
      log('T = ' + str(T) + ' / ' + str(args.T_max))

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
      # print("update")

      if T % args.replay_frequency == 0:
        # print("replay freq")
        dqn.learn(mem)  # Train with n-step distributional double-Q learning
        print("dqn.learn pass")

      if T % args.evaluation_interval == 0:
        # print("evaluation inter")
        dqn.eval()  # Set DQN (online network) to evaluation mode
        # print("dqn.eval pass")
        avg_reward, avg_Q = test(args, T, dqn, val_mem, test_path, result_path,)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode
        # print("dqn.train pass")

      # Update target network
      if T % args.target_update == 0:
        # print("target update")
        dqn.update_target_net()
        # print("dqn.update pass")

    state = next_state

