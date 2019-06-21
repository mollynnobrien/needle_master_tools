import numpy as np
import torch
import argparse
import os
import random
from context import needlemaster as nm
from environment import Environment
import utils
import OurDDPG
import math
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('ignore')

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes= 3):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset(episode_num)
        # state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi)])
        if env.next_gate != None:
            state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), env.next_gate])
        else:
            state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), 5])
        done = False
        while not done:
            action = policy.select_action(state)
            reward, done = env.step(action, episode_num, 'play', save_image=False)
            # state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi)])
            if env.next_gate != None:
                state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), env.next_gate])
            else:
                state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), 5])
            avg_reward += reward

    avg_reward /= eval_episodes
    env.episode_reward = avg_reward
    frame = env.render(episode_num, save_image=True, save_path=save_path)

    print ("---------------------------------------")
    print ("Episode_num: %d, Evaluation over %d episodes: %f" % (episode_num,eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="OurDDPG")  # Policy name
    parser.add_argument("--env_name", default="Needle Master")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e2, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=6e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action= "store" )  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=1000, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_size", default=2e3, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed),1)
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
    # args.device = torch.device('cuda')
    # torch.cuda.manual_seed(random.randint(1, 10000))
    # torch.backends.cudnn.enabled = False

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    else:
        args.device = torch.device('cpu')


    """ original version """
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)



    ## environment set up
    env_path = 'C:/Users/icage/needle_master_tools-lifan/environments/environment_14.txt'
    # env_path = './environment_14.txt'
    env = Environment(0, env_path)
    env.GetTargetPoint()
    state_dim = 10
    action_dim = 2
    """"  for PID controller """
    action_constrain = [10, np.pi/20]
    parameter = [0.1,0.0009]

    """ [lower bound],[higher bound] """
    env.action_bound = np.array(([-1, -1],[1, 1]))  ## modified lower bound
    max_action = 1.0

    ### for plotting
    Reward = []
    save_path = './out/'

    # Initialize policy
    policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    replay_buffer = utils.ReplayBuffer(args.max_size)

    # Evaluate untrained policy
    # evaluations = [evaluate_policy(policy)]
    """ start straightly """
    evaluations = []

    env.total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while env.total_timesteps < args.max_timesteps:

        ## finish one episode, and train episode_times
        if done:

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))

                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            ## load model
            # policy.load(file_name,"./pytorch_models")

            ## training as usual
            else:
                # if env.total_timesteps != 0 and env.episode_reward > 500:
                if env.total_timesteps != 0:
                    print (("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        env.total_timesteps, episode_num, episode_timesteps, env.episode_reward))
                    frame = env.render(episode_num, save_image=True, save_path=save_path)

                if env.total_timesteps != 0:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)


            Reward.append(env.episode_reward)
            plt.plot(np.arange(1, episode_num), Reward[1:episode_num], 'b')
            plt.savefig('./out/episode reward.png')


            # Reset environment
            state = env.reset(episode_num)


            # state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi)])
            # if env.next_gate != None:
            #     state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi), env.next_gate])
            # else:
            #     state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi), 5])

            # state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi)])
            # state = state/scale_factor

            done = False

            episode_num += 1
            env.episode_reward = 0
            episode_timesteps = 0

        # Select action randomly or according to policy
        if env.total_timesteps % args.max_size < args.start_timesteps:
            action = env.sample_action()
            # print("randomly selected: " + str(action))
            # action = env.PIDcontroller(action_constrain, parameter, env.t)
            # print("PID controller: " +str(action))
        else:
            action = policy.select_action(state)
            # print("action based on polilcy:" + str(action))
            # print("action selected: " +str(action))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=2)).clip(
                    env.action_bound[0,:], env.action_bound[1,:])
            # print("noised action: " +str(action))
                # action = (action + np.random.normal(0, args.expl_noise, size=2)*np.array([10,1])).clip(
                #     env.action_bound[1,:], env.action_bound[0,:])
                # noise = np.random.normal(0, args.expl_noise, size=2)*np.array([10,1])
                # print("noise added: " )
                # print(noise)
                # print("actual action: " +str(action))


        ### select action only based on pure RL
        # action = policy.select_action(state)
        # print("action selected: " +str(action))
        # if args.expl_noise != 0:
        #     action = (np.array([-20,0]) + np.random.normal(0, args.expl_noise, size=2) * np.array([10, 1])).clip(env.action_bound[1, :], env.action_bound[0, :])

            # action = (action + np.random.normal(0, args.expl_noise, size=2) * np.array([10, 1])).clip(
            #     env.action_bound[1, :], env.action_bound[0, :])
        # print("real selected: " + str(action))

        # Perform action
        new_state, reward, done = env.step(action, save_image=True)
        # print("done? "+ str(done))
        # new_state = np.array([- env.needle.x, - env.needle.y, -(env.needle.w - math.pi)])

        # if env.next_gate != None:
        #     new_state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), env.next_gate])
        # else:
        #     new_state = np.array([-env.needle.x, -env.needle.y, -(env.needle.w - math.pi), 5])

        running = env.check_status()

        done_bool = 0 if episode_timesteps + 1 == env.max_time else float(done)
        env.episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((state, new_state, action, reward, done_bool))
        # print("state: " + str(state))
        state = new_state


        episode_timesteps += 1
        env.total_timesteps += 1
        timesteps_since_eval += 1

        plt.plot(range(len(policy.actor_loss)), policy.actor_loss)
        plt.savefig('./results/actor loss.png')

        plt.plot(range(len(policy.critic_loss)), policy.critic_loss)
        plt.savefig('./results/critic loss.png')

    # Final evaluation
    # plt.plot(np.arange(1,episode_num),Reward[0:episode_num],'b')
    # plt.savefig('./out/episode reward.png')

        # if  env.total_timesteps % 1000 == 0:
        #     evaluations.append(evaluate_policy(policy))
        #     policy.save("%s" % (file_name), directory="./pytorch_models")
        #     np.save("./results/%s" % (file_name), evaluations)