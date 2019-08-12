import numpy as np
import torch
import random, math
import os, sys, argparse
from os.path import abspath
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter

cur_dir= os.path.dirname(abspath(__file__))
sys.path.append(abspath(pjoin(cur_dir, '..')))
from needlemaster.environment import Environment

from .utils import *

def evaluate_policy(tb_writer, total_times, total_rewards,
        env, args, policy, time, test_path):
    ''' Runs deterministic policy for X episodes and
        @param tb_writer: tensorboard writer
        @returns average_reward
    '''
    #policy.actor.eval() # set for batchnorm
    rewards = []
    actions = []
    for _ in xrange(args.evaluation_episodes):
        reward_sum = 0
        done = False
        state = env.reset(random_needle=args.random_needle)
        while not done:
            action = policy.select_action(state)
            actions.append(action)
            state, reward, done = env.step(action)
            reward_sum += reward

        img = env.render(save_image=True, save_path=test_path)
        rewards.append(reward_sum)
    avg_reward = np.array(rewards, dtype=np.float32).mean()
    actions = np.array(actions, dtype=np.float32)
    avg_action = actions.mean()
    std_action = actions.std()
    min_action = actions.min()
    max_action = actions.max()
    total_times.append(time)
    total_rewards.append(avg_reward)
    fig = plot_line(np.array(total_times), np.array(total_rewards), 'Reward',
        path = test_path)
    tb_writer.add_figure('rewards', fig, global_step=time)
    tb_writer.add_image('run', img.transpose(0, 2, 1), global_step=time)

    print ("In {} episodes, R={:.4f}, A avg={:.2f}, std={:.2f}, "
        "min={:.2f}, max={:.2f}".format(
      args.evaluation_episodes, avg_reward, avg_action, std_action,
      min_action, max_action))
    print ("---------------------------------------")
    return avg_reward

def run(args):
    args.policy_name = args.policy_name.lower()

    env_data_name = os.path.splitext(
        os.path.basename(args.filename))[0]

    times, rewards, best_avg_reward = [], [], -1e5

    base_filename = '{}_{}_{}_{}_dim{}'.format(
        args.env_name, env_data_name, args.policy_name, args.mode, 'bn' if args.batchnorm else 'nobn',
        args.img_dim)

    tb_writer = SummaryWriter(comment=base_filename)

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
    env = Environment(filename = args.filename, mode=args.mode,
            stack_size = args.stack_size, img_dim=args.img_dim)

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
        policy = TD3(state_dim, action_dim, args.stack_size,
            max_action, args.mode, lr=args.lr, lr2=args.lr2,
            actor_lr=args.actor_lr, bn=args.batchnorm, img_dim=args.img_dim,
            load_encoder=args.load_encoder)
    elif args.policy_name == 'ddpg':
        from .DDPG_image import DDPG
        policy = DDPG(state_dim, action_dim, args.stack_size,
            max_action, args.mode, bn=args.batchnorm,
            lr=args.lr, actor_lr=args.actor_lr, img_dim=args.img_dim,
            load_encoder=args.load_encoder)
    else:
        raise ValueError(
            args.policy_name + ' is not recognized as a valid policy')

    ## load pre-trained policy
    #try:
    #    policy.load(result_path)
    #except:
    #    pass

    if args.buffer == 'simple':
        replay_buffer = ReplayBuffer(int(args.max_size))
    elif args.buffer == 'priority':
        replay_buffer = NaivePrioritizedBuffer(int(args.max_size))
    else:
        raise ValueError(args.buffer + ' is not a buffer name')

    state = env.reset()
    total_timesteps = 0
    episode_num = 0
    done = False
    zero_noise = np.zeros((action_dim,))
    ou_noise = OUNoise(action_dim)

    policy.actor.eval() # set for batchnorm

    while total_timesteps < args.max_timesteps:

        # Check if we should add noise
        if args.ou_noise:
            noise = ou_noise.sample()
        else:
            # Epsilon-greedy
            percent_greedy = (1. - min(1., float(total_timesteps) /
                greedy_decay_rate))
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
              print ("---------------------------------------")
              if args.ou_noise:
                  print("Evaluating policy")
              else:
                  print("Greedy={}, std={}. Evaluating policy".format(
                    epsilon_greedy, noise_std)) # debug
              best_reward = evaluate_policy(
                  tb_writer, times, rewards, env, args,
                policy, total_timesteps, test_path)

              ## save model parameters if improved
              if best_reward > best_avg_reward:
                  best_avg_reward = best_reward
                  policy.save(result_path)


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

            '''
            #debug
            if episode_num > 200:
                import pdb
                pdb.set_trace()
            '''

            policy.actor.train() # Set actor to training mode

            beta = min(1.0, beta_start + total_timesteps *
                (1.0 - beta_start) / beta_frames)

            critic_loss, actor_loss = policy.train(
                replay_buffer, total_timesteps, beta, args)

            print ("Training E:{:04d} S:{:03d} R: {:.3f} "
                "CL: {:.3f} AL: {:.3f}".format(
                  episode_num, env.t, env.total_reward,
                  critic_loss, actor_loss)) # debug

            ## training
            str = 'Total:{}, Episode Num:{}, Step:{}, Reward:{}, Loss:{}'.format(
              total_timesteps, episode_num, env.t, env.total_reward, critic_loss)

            log_f.write(str + '\n')
            if episode_num % 20 == 0:
                env.render(save_image=True, save_path=save_path)

            policy.actor.eval() # set for batchnorm

            # Reset environment
            new_state = env.reset(random_needle=args.random_needle)
            ou_noise.reset() # reset to mean
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
    parser.add_argument("--max_timesteps", default=5e7, type=float,
        help='Max time steps to run environment for')
    parser.add_argument("--learning_start", default=0, type=int,
        help='Timesteps before learning')
    parser.add_argument("--save_models", action= "store",
        help='Whether or not models are saved')

    #--- Exploration Noise
    parser.add_argument("--no-ou-noise", default=False, action='store_true',
        help='Use OU Noise process for noise instead of epsilon greedy')
    parser.add_argument("--expl_noise", default=1., type=float,
        help='Starting std of Gaussian exploration noise')
    parser.add_argument("--epsilon_greedy", default=0.3, type=float,
        help='Starting percentage of choosing random noise')
    #---

    #--- Batch size is VERY important ---
    parser.add_argument("--batch-size", default=1024, type=int,
        help='Batch size for both actor and critic')
    #---
    parser.add_argument("--discount", default=0.99, type=float,
        help='Discount factor (0.99 is good)')

    parser.add_argument("--policy_noise", default=0.04, type=float, # was 0.2
        help='TD3 Smoothing noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.1, type=float,
        help='TD3 Range to clip target policy noise') # was 0.5

    # For images, need smaller (1e4)
    parser.add_argument("--max_size", default=1e6, type=float,
        help='Size of replay buffer (bigger is better)')
    parser.add_argument("--stack_size", default=2, type=int,
        help='How much history to use')
    parser.add_argument("--evaluation_episodes", default=1, type=int,
        help='How many times to evaluate actor (1 is enough)')
    parser.add_argument("--profile", default=False, action="store_true",
        help="Profile the program for performance")
    parser.add_argument("--mode", default = 'state',
        help="Choose image or state, options are rgb_array and state")
    parser.add_argument("--buffer", default = 'simple', # 'priority'
        help="Choose type of buffer, options are simple and priority")
    parser.add_argument("--random_needle", default = False, action='store_true',
        help="Choose whether the needle should be random at each iteration")
    parser.add_argument("--batchnorm", default = False,
        action='store_true', help="Choose whether to use batchnorm")
    parser.add_argument("--img-dim", default = 224, type=int,
        help="Size of img (224 is max, 112/56 is optional)")

    parser.add_argument("--policy_freq", default=2, type=int,
        help='Frequency of TD3 delayed actor policy updates')

    #--- Tau: percent copied to target
    parser.add_argument("--tau", default=0.001, type=float,
        help='Target critic network update rate')
    parser.add_argument("--actor-tau", default=0.001, type=float,
        help='Target actor network update rate')
    #---

    #--- Learning rates
    parser.add_argument("--lr", default=1e-3, type=float,
        help="Learning rate for critic optimizer")
    parser.add_argument("--lr2", default=1e-3, type=float,
        help="Learning rate for second critic optimizer")
    parser.add_argument("--actor-lr", default=1e-5, type=float,
        help="Learning rate for actor optimizer")
    #--- Model save/load
    parser.add_argument("--load-encoder", default='', type=str,
        help="File from which to load the encoder model")

    parser.add_argument("filename", help='File for environment')
    parser.add_argument("policy_name", default="TD3", type=str)

    args = parser.parse_args()
    args.ou_noise = not args.no_ou_noise

    # Check for replay buffer that's too big
    if args.mode == 'rgb_array' and args.max_size > 1e4:
        args.max_size = 1e4

    if args.profile:
        import cProfile
        cProfile.run('run(args)', sort='cumtime')
    else:
        run(args)
