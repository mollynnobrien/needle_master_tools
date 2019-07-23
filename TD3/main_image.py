import numpy as np
import torch
import argparse
import os
import random
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import plotly
from .environment import Environment
from .environment import PID
from .utils import NaivePrioritizedBuffer
import math


pi = math.pi
Ts, rewards, Best_avg_reward = [], [], -1e5

# Runs policy for X episodes and returns average reward
def evaluate_policy(args, policy, T, test_path, result_path):
    global Ts, rewards, Best_avg_reward
    Ts.append(T)
    T_rewards = []
    policy.actor.eval()
    done = True
    env.episode_reward = 0
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = policy.select_action(state)
            state, reward, done = env.step(action)
            reward_sum += reward

            if done:
                env.render(save_image=True, save_path=test_path)
                T_rewards.append(reward_sum)
                break
    avg_reward = sum(T_rewards) / len(T_rewards)
    rewards.append(T_rewards)
    _plot_line(Ts, rewards, 'Reward', path = test_path)

    ## same model parameters if improved
    if avg_reward > Best_avg_reward:
        Best_avg_reward = avg_reward
        policy.save(result_path)

    print ("---------------------------------------")
    print ("In %d episodes, avg rewards: %f" % (args.evaluation_episodes, avg_reward))
    print ("---------------------------------------")
    return Best_avg_reward

# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--env_name", default="NeedleMaster")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1e6, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--pid_interval", default=5e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--pid_freq", default=1e4, type=int)  # How often we get back to pure random action
    parser.add_argument("--max_timesteps", default=5e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--learning_start", default=5e2, type=int)  # Timesteps before learning
    parser.add_argument("--save_models", action= "store" )  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_size", default=5e4, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--img_stack", default=4, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--evaluation_episodes", default=6, type=int)  # Frequency of delayed policy updates
    parser.add_argument("filename", help='File for environment')
    parser.add_argument("policy_name", default="TD3")  # Policy name

    args = parser.parse_args()

    # env_name = 'environment_17'
    # env_path = 'C:/Users/icage/needle_master_tools-lifan/environments/' + env_name + '.txt'

    def make_dirs(args):
        count = 0
        for chr in args.filename:
            if chr == '/':
                start = count
            elif chr == '.':
                end = count
            count += 1
        env_name = args.filename[start + 1:end]

        save_path = "./" + env_name + "/" + args.policy_name + "_out"
        test_path = "./" + env_name + "/" + args.policy_name + "_test"
        result_path = "./" + env_name + "/" + args.policy_name + "_results"
        explore_path = "./" + env_name + "/" + args.policy_name + "_explore"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(explore_path):
            os.makedirs(explore_path)
        return save_path, test_path, result_path, explore_path, env_name

    save_path, test_path, result_path, explore_path, env_name = make_dirs(args)
    file_name = "%s_%s_%s" % (env_name,args.policy_name, args.env_name)
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    else:
        args.device = torch.device('cpu')

    ## environment set up
    """ Adding the log file """
    logfile = "%s_%s_%s" % (env_name, args.policy_name, args.env_name)
    log_f = open("log_"+logfile+".txt","w+")

    """ setting up environment """
    action_dim = 1
    ## from pycharm
    # env = Environment("image", args.img_stack, env_path)
    ## from scripts
    env = Environment("image", args.img_stack, args.filename)
    state_dim = len(env.gates) + 9

    """" setting up PID controller """
    action_constrain = [10, np.pi/20]
    # parameter = [0.1,0.0009]
    parameter =  [0.0000001, 0.5]
    pid = PID( parameter, env.width, env.height )

    """ setting up action bound for RL """
    max_action = 1/4 * pi

    """ parameters for epsilon declay """
    epsilon_start = 1
    epsilon_final = 0.01
    decay_rate = 25000
    ep_decay = []

    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000

    """ start straightly """
    evaluations = []

    # Initialize policy
    if args.policy_name == 'TD3':
        from .TD3_image import TD3
        policy = TD3( action_dim, args.img_stack, max_action)
    elif args.policy_name == 'DDPG':
        from .DDPG_image import DDPG
        policy = DDPG(action_dim, args.img_stack, max_action)

    replay_buffer = NaivePrioritizedBuffer(int(args.max_size))

    env.total_timesteps = 0
    timesteps_since_eval = 0
    pid_assist = 0
    done = True

    while env.total_timesteps < args.max_timesteps:

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            Best_reward = evaluate_policy(args, policy, env.total_timesteps, test_path, result_path)
            continue

        ## finish one episode, and train episode_times
        if done:
            log_f.write('~~~~~~~~~~~~~~~~~~~~~~~~ iteration {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'.format(env.episode_num))

            ## load model
            # policy.load(result_path)

            ## training as usual
            # if env.total_timesteps != 0 and env.episode_reward > 500:
            if env.total_timesteps != 0:
                log_f.write('Total:{}, Episode Num:{}, Eposide:{}, Reward:{}\n'.format(env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                log_f.flush()

                if env.total_timesteps > args.learning_start:
                    if env.episode_num % 20 == 0:
                        print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
	                        env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                        env.render( save_image=True, save_path=save_path)
                else:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
		                env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                    env.render(save_image = True, save_path = explore_path)


            ## sampling data before start to train
            if env.total_timesteps > args.learning_start:
                policy.actor.train()
                beta = min(1.0, beta_start + env.total_timesteps * (1.0 - beta_start) / beta_frames)
                policy.train(replay_buffer, episode_timesteps, beta, args)

            # Reset environment
            state = env.reset()
            done = False
            env.episode_num += 1
            env.episode_reward = 0
            episode_timesteps = 0

        """ exploration rate decay """
        args.expl_noise = (epsilon_start - epsilon_final) * math.exp(-1. * env.total_timesteps / decay_rate)
        ep_decay.append(args.expl_noise)
        # log_f.write('epsilon decay:{}\n'.format(args.expl_noise))

        # if env.total_timesteps % 500 == 0 and args.expl_noise > 0:
        #     args.expl_noise *= 0.9

        # """ using PID controller """
        # state_pid = state[0:3]
        # action = pid.PIDcontroller( state_pid, env.next_gate, env.gates, env.total_timesteps)
        # print("action based on PID: " + str(action))

        """ action selected based on pure policy """
        policy.actor.eval()
        action = policy.select_action(state)
        log_f.write('action based on policy:{}\n'.format(action))
        # print("action based on policy:" + str(action))
        if args.expl_noise != 0:
            # state_pid = state[0:3]
            # guidance = pid.PIDcontroller( state_pid, env.next_gate, env.gates, env.total_timesteps)
            noise = np.random.normal(0, args.expl_noise, size=action_dim)
            # print("noise: " + str(noise))
            # action = ((1 - args.expl_noise) * action + args.expl_noise * guidance)
            action = action + noise
            action = np.clip(action, -max_action ,max_action)


        # Perform action
        new_state, reward, done = env.step(action)

        done_bool = 0 if episode_timesteps + 1 == env.max_time else float(done)
        env.episode_reward += reward

        # Store data in replay buffer
        # print("reward in")
        # print(reward)
        replay_buffer.add(state, new_state, action, reward, done_bool)
        # print("state: " + str(state))
        state = new_state

        episode_timesteps += 1
        env.total_timesteps += 1
        timesteps_since_eval += 1

    print("The Best Reward: " + str(Best_reward))

