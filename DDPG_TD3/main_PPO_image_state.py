
import numpy as np
import torch
import argparse
import os
import random
from environment_PPO_image_state import Environment
from environment_PPO_image_state import PID
from PPO_image_state import PPO
from PPO_image_state import Memory
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="PPO_image_state")  # OpenAI gym environment name
    parser.add_argument("--env_name", default="NeedleMaster")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1e6, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--pid_interval", default=5e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--pid_freq", default=1e4, type=int)  # How often we get back to pure random action
    parser.add_argument("--max_timesteps", default=1e8, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action= "store" )  # Whether or not models are saved
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--update_timestep", default=4000, type=int)  # Update policy every n timesteps
    parser.add_argument("--action_std", default=0.6, type=float)  # Constant std for action distribution
    parser.add_argument("--lr", default=0.0025, type=float)
    parser.add_argument("--betas", default=(0.9, 0.999))
    parser.add_argument("--K_epochs", default=10, type=int)  # update policy for K epochs
    parser.add_argument("--eps_clip", default=0.2, type=float)  # clip parameter for PPO
    parser.add_argument("--gamma", default=0.99, type=float)  # discount factor

    parser.add_argument('filename', help='File for environment')

    # setup
    args = parser.parse_args()

    # env_name = 'environment_14'
    # env_path = 'C:/Users/icage/needle_master_tools-lifan/environments/' + env_name + '.txt'
    # file_name = "%s_%s_%s" % (env_name, args.policy_name, args.env_name)

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    else:
        args.device = torch.device('cpu')

    # run from script
    env_name = args.filename
    # get environment name
    count = 0
    for chr in args.filename:
        if chr == '/':
            start = count
        elif chr == '.':
            end = count
        count += 1
    env_name = args.filename[start+1:end]


    file_name = "%s_%s_%s" % (env_name, args.policy_name, args.env_name)
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    if not os.path.exists("./PPO_results"):
        os.makedirs("./PPO_results")

    ## environment set up

    """ Adding the log file """
    logfile = "%s_%s_%s" % (env_name, args.policy_name, args.env_name)
    log_f = open(logfile,"w+")

    """ setting up environment """
    action_dim = 2
    img_stack = 4
    ## from py
    # env = Environment(action_dim, log_f, img_stack, env_path)

    ## from script
    env = Environment(action_dim, log_f, img_stack, args.filename)
    state_dim = len(env.gates) + 9

    """" setting up PID controller """
    action_constrain = [10, np.pi/20]
    # parameter = [0.1,0.0009]
    # parameter =  [0.0000001, 0.5]
    # pid = PID( parameter, env.width, env.height )


    ### for plotting
    Reward = []
    save_path = './PPO_out/'
    if not os.path.exists("./PPO_out"):
        os.makedirs("./PPO_out")

    """ start straightly """
    evaluations = []
    memory = Memory()
    policy = PPO(img_stack, state_dim, action_dim, args.action_std, args.lr, args.betas, args.gamma, args.K_epochs, args.eps_clip)
    env.total_timesteps = 0
    timesteps_since_eval = 0
    pid_assist = 0
    time_step = 0
    done = True

    while env.total_timesteps < args.max_timesteps:
        time_step += 1

        # # Evaluate episode
        # if timesteps_since_eval >= args.eval_freq:
        #     timesteps_since_eval %= args.eval_freq
        #     evaluations.append(evaluate_policy(policy, log_f))
        #     continue

        ## finish one episode, and train episode_times

        if done:
            log_f.write('~~~~~~~~~~~~~~~~~~~~~~~~ iteration {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'.format(env.episode_num))

            ## training as usual
            if env.total_timesteps != 0:
                log_f.write('Total:{}, Episode Num:{}, Eposide:{}, Reward:{}\n'.format(env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                log_f.flush()

                if env.episode_num % 1 == 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        env.total_timesteps, env.episode_num, episode_timesteps, env.episode_reward))
                if env.episode_reward > -200:
                    env.render( save_image=True, mode = 'gray_array', save_path=save_path)

            Reward.append(env.episode_reward)

            # Reset environment
            ob, state = env.reset(log_f)

            done = False

            env.episode_num += 1
            env.episode_reward = 0
            episode_timesteps = 0


        """ action selected based on pure policy """
        action = policy.select_action(ob, state, memory)
        # log_f.write('action based on policy:{}\n'.format(action))

        # Perform action
        new_ob, new_state, reward, done = env.step(action, log_f)

        done_bool = 0 if episode_timesteps + 1 == env.max_time else float(done)
        env.episode_reward += reward
        # Saving reward:
        memory.rewards.append(reward)

        ob = new_ob
        state = new_state

        episode_timesteps += 1
        env.total_timesteps += 1
        timesteps_since_eval += 1

            # update if its time
        if time_step % args.update_timestep == 0:
            policy.update(memory)
            memory.clear_memory()
            time_step = 0

    plt.plot(range(len(Reward)), np.array(Reward), 'b')
    plt.savefig('./results/episode reward.png')



