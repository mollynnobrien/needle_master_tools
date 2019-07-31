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

from .utils import NaivePrioritizedBuffer

Ts, rewards, Best_avg_reward = [], [], -1e5

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, args, policy, T, test_path, result_path):
    global Ts, rewards, Best_avg_reward
    Ts.append(T)
    T_rewards = []
    policy.actor.eval() # set for batchnorm
    for _ in range(args.evaluation_episodes):
        reward_sum = 0
        done = False
        state = env.reset()
        while not done:
            action = policy.select_action(state)
            state, reward, done = env.step(action)
            reward_sum += reward

        env.render(save_image=True, save_path=test_path)
        T_rewards.append(reward_sum)
    avg_reward = float(sum(T_rewards)) / len(T_rewards)
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
  max_colour, mean_colour, std_colour, transparent = (
      'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)',
      'rgba(0, 0, 0, 0)')

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(),
      line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(),
      line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty',
      fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty',
      fillcolor=std_colour, line=Line(color=transparent),
      name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(),
      line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


def run(args):
    args.policy_name = args.policy_name.lower()

    env_data_name = os.path.splitext(
        os.path.basename(args.filename))[0]

    def make_dirs(args):
        path = pjoin(env_data_name, args.policy_name, args.mode)

        save_p = path + '_out'
        test_p = path + '_test'
        result_p = path + '_results'
        for p in [save_p, test_p, result_p]:
          if not os.path.exists(p):
              os.makedirs(p)
        return save_p, test_p, result_p

    save_path, test_path, result_path = make_dirs(args)
    base_filename = '{}_{}_{}'.format(
        args.env_name, args.policy_name, env_data_name)

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        # Disable nondeterministic ops (not sure if critical but better
        # safe than sorry)
        torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    ## environment setup
    log_f = open('log_' + base_filename + '.txt', 'w')

    """ setting up environment """
    env = Environment(filename = args.filename, mode=args.mode, stack_size = args.stack_size)

    """ setting up PID controller """
    #action_constrain = [10, np.pi/20]
    # parameter = [0.1,0.0009]
    # parameter =  [0.0000001, 0.5]
    #pid = PID( parameter, env.width, env.height )

    """ setting up action bound for RL """
    max_action = 0.25 * math.pi

    """ parameters for epsilon declay """
    greedy_decay_rate = 10000000
    std_decay_rate = 10000000
    epsilon_start = args.expl_noise
    epsilon_final = 0.001
    ep_decay = []

    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000

    # Initialize policy
    action_dim = 1
    state_dim = 0

    if args.mode == 'state':
        state = env.reset()
        state_dim = state.shape[-1]

    if args.policy_name == 'td3':
        from .TD3_image import TD3
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
        # Check if we should add noise
        percent_greedy = 1. - min(1., float(total_timesteps) / greedy_decay_rate)
        epsilon_greedy = args.epsilon_greedy * percent_greedy
        if random.random() < epsilon_greedy:
            noise_std = ((args.expl_noise - epsilon_final) *
                math.exp(-1. * float(total_timesteps) / std_decay_rate))
            ep_decay.append(noise_std)
            # log_f.write('epsilon decay:{}\n'.format(noise_std)) # debug
            noise = np.random.normal(0, noise_std, size=action_dim)
        else:
            noise = zero_noise


        # Evaluate episode
        if (total_timesteps % args.eval_freq == 0
            and total_timesteps != 0):
            print("Greedy={}, std={}. Evaluating policy".format(
              epsilon_greedy, noise_std)) # debug
            best_reward = evaluate_policy(
                env, args, policy, total_timesteps, test_path, result_path)

        """ exploration rate decay """

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
            policy.actor.train() # Set actor to training mode

            beta = min(1.0, beta_start + total_timesteps *
                (1.0 - beta_start) / beta_frames)

            critic_loss = policy.train(replay_buffer, env.t, beta, args)

            print ("Training. episode ", episode_num, "R =", env.total_reward, "L =", critic_loss) # debug

            ## training
            str = 'Total:{}, Episode Num:{}, Step:{}, Reward:{}, Loss:{}'.format(
              total_timesteps, episode_num, env.t, env.total_reward, critic_loss)

            log_f.write(str + '\n')
            if episode_num % 20 == 0:
                print(str)
                env.render(save_image=True, save_path=save_path)


            policy.actor.eval() # set for batchnorm

            # Reset environment
            new_state = env.reset()
            done = False
            episode_num += 1

            # print "Training done" # debug

        state = new_state
        total_timesteps += 1

    print("Best Reward: ", best_reward)

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
    parser.add_argument("--expl_noise", default=1.0, type=float,
        help='Starting std of Gaussian exploration noise')
    parser.add_argument("--epsilon_greedy", default=0.8, type=float,
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
