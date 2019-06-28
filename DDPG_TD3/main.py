import numpy as np
import torch
import argparse
import os
import random
from environment import Environment
from environment import PID
import utils
import TD3
import math
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('ignore')

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, log_f):
    eval_path = './evaluate/'
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    state = env.reset(log_f)
    done = False
    env.episode_num += 1
    env.episode_reward = 0
    episode_timesteps = 0

    while not done:
        action = policy.select_action(state)
        # print("state: " + str(state))
        # print("action: " + str(action))
        next_state, reward, done = env.step(action, log_f)
        # print("next state: " + str(next_state))
        # print("done: " +str(done))
        env.episode_reward += reward
        state = new_state
        episode_timesteps += 1
        env.total_timesteps += 1

    env.render(save_image=True, save_path=eval_path)

    print ("---------------------------------------")
    print ("Episode_num: %d: %f" % (env.episode_num, env.episode_reward))
    print ("---------------------------------------")
    return env.episode_reward

# class Args():
#     policy_name = "TD3"
#     env_name = "NeedleMaster" # OpenAI gym environment name
#     seed = 1e6  # Sets Gym, PyTorch and Numpy seeds
#     random_interval = 5e2  # How many time steps purely random policy is run for
#     eval_freq =5e3 # How often (time steps) we evaluate
#     random_freq = 2e3  # How often we get back to pure random action
#     max_timesteps = 1e5 # Max time steps to run environment for
#     save_models = "store"  # Whether or not models are saved
#     expl_noise = 0.5  # Std of Gaussian exploration noise
#     batch_size = 1000 # Batch size for both actor and critic
#     discount = 0.99  # Discount factor
#     tau = 0.005 # Target network update rate
#     policy_noise = 0.2  # Noise added to target policy during critic update
#     noise_clip = 0.5  # Range to clip target policy noise
#     policy_freq = 2  # Frequency of delayed policy updates
#     max_size = 2e3  # Frequency of delayed policy updates

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="NeedleMaster")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1e6, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--pid_interval", default=1e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--pid_freq", default=5e3, type=int)  # How often we get back to pure random action
    parser.add_argument("--max_timesteps", default=1e3, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action= "store" )  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=1000, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_size", default=2e3, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    env_name = 'environment_14'
    env_path = 'C:/Users/icage/needle_master_tools-lifan/environments/' + env_name + '.txt'
    file_name = "%s_%s_%s" % (env_name,args.policy_name, args.env_name)
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")


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

    action_dim = 2
    env = Environment(action_dim, log_f,env_path)
    # env.GetTargetPoint()
    state_dim = len(env.gates) + 6
    """"  for PID controller """
    action_constrain = [10, np.pi/20]
    parameter = [0.1,0.0009]
    pid = PID( parameter, env.width, env.height )

    """ [lower bound],[higher bound] """
    env.action_bound = np.array((-1,1)) ## for one dimension action
    # env.action_bound = np.array(([-1, -1],[1, 1]))  ## for two dimension action
    max_action = 1

    ### for plotting
    Reward = []
    save_path = './out/'
    """ start straightly """
    evaluations = []

    # Initialize policy
    policy = TD3.TD3(state_dim, action_dim, max_action)
    replay_buffer = utils.ReplayBuffer(args.max_size)

    # Evaluate untrained policy
    # evaluations = [evaluate_policy(policy)]

    env.total_timesteps = 0
    timesteps_since_eval = 0
    done = True

    while env.total_timesteps < args.max_timesteps:

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(policy, log_f))


            if env.last_reward > 100 and env.episode_num > 100:
                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            continue


        ## finish one episode, and train episode_times
        if done:
            log_f.write('~~~~~~~~~~~~~~~~~~~~~~~~ iteration {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'.format(env.episode_num))

            ## load model
            # policy.load(file_name,"./pytorch_models")

            ## training as usual
            # if env.total_timesteps != 0 and env.episode_reward > 500:
            if env.total_timesteps != 0:
                log_f.write('Total:{}, Episode Num:{}, Eposide:{}, Reward:{}\n'.format(env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                log_f.flush()

                if env.episode_num % 1 == 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                    env.render( save_image=True, save_path=save_path)

            if env.total_timesteps != 0:
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)

            Reward.append(env.episode_reward)

            # Reset environment
            state = env.reset(log_f)

            done = False

            env.episode_num += 1
            env.episode_reward = 0
            episode_timesteps = 0

            # """ exploration rate decay """
            # if env.total_timesteps % 1000 == 0 and args.expl_noise > 0:
            #     args.expl_noise -= 0.05

        """ alternative between pid and policy  """
        if env.total_timesteps % args.pid_freq < args.pid_interval:
        # if env.total_timesteps < args.pid_interval:
            state_pid = state[0:3]
            action = pid.PIDcontroller( state_pid, env.next_gate, env.gates)
            # action = env.sample_action()
            # log_f.write('~~~~~~~~~~~random action~~~~~~~~~~\n')
            # log_f.write('random selected action:{}\n'.format(action))

        else:
            # print("state: " +str(state))
            action = policy.select_action(state)
            # print("select")
            # log_f.write('~~~~~~~~~~~selected action~~~~~~~~~~\n')
            log_f.write('Action based on policy:{}\n'.format(action))
            # print("action based on policy:" + str(action))
            # print("action selected: " +str(action))
        if args.expl_noise != 0:
            noise = np.random.normal(0, args.expl_noise, size=action_dim)
            # print("noise: " + str(noise))
            action = (action + noise).clip(-1, 1)

        """ using PID controller """
        # state_pid = state[0:3]
        # action = pid.PIDcontroller( state_pid, env.next_gate, env.gates)
        # print("action based on PID: " + str(action))

        """ action selected based on pure policy """
        # action = policy.select_action(state)
        # log_f.write('action based on policy:{}\n'.format(action))
        # # print("action based on policy:" + str(action))
        # if args.expl_noise != 0:
        #     noise = np.random.normal(0, args.expl_noise, size=action_dim)
        #     # print("noise: " + str(noise))
        #     action = (action + noise).clip(-1,1)


        # Perform action
        new_state, reward, done = env.step(action, log_f)

        done_bool = 0 if episode_timesteps + 1 == env.max_time else float(done)
        env.episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((state, new_state, action, reward, done_bool))
        # print("state: " + str(state))
        state = new_state

        episode_timesteps += 1
        env.total_timesteps += 1
        timesteps_since_eval += 1

    plt.plot(range(len(Reward)), np.array(Reward), 'b')
    plt.savefig('./results/episode reward.png')

    plt.plot(range(len(policy.actor_loss)), policy.actor_loss, 'b')
    plt.savefig('./results/actor loss.png')

    plt.plot(range(len(policy.critic_loss)), policy.critic_loss, 'b')
    plt.savefig('./results/critic loss.png')

    plt.plot(range(len(evaluations)), np.array(evaluations), 'b')
    plt.savefig('./results/evaluation reward.png')
    print(evaluations)
