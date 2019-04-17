import numpy as np
import torch
import argparse
import os
import random
from context import needlemaster as nm
from needlemaster.environment import Environment
import utils
import OurDDPG

# import warnings
# warnings.filterwarnings('ignore')

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes= 2):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset(episode_num)
        state = np.array([env.needle.x, env.needle.y, env.needle.w])
        done = False
        while not done:
            action = policy.select_action(state)
            obs, reward, done = env.step(action, episode_num, 'play', save_image=True)
            state = np.array([env.needle.x, env.needle.y, env.needle.w])
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="OurDDPG")  # Policy name
    parser.add_argument("--env_name", default="Needle Master")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=450,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=900, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
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
    state_dim = 3
    action_dim = 2
    action_constrain = [10, np.pi/20]
    parameter = [0.1,0.0009]
    """ [high bound],[low bound] """
    env.action_bound = np.array(([12, np.pi/20],[-12, -np.pi/20]))
    max_action = np.array([-10.0,0.05])

    ### for plotting
    Reward = []

    # Initialize policy
    policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    # evaluations = [evaluate_policy(policy)]
    """ start straightly """
    evaluations = []

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        ## finish one episode, and train episode_times
        if done:

            if total_timesteps != 0:
                print (("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, env.episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                print("training")

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))

                if args.save_models: policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset(episode_num)
            state = np.array([env.needle.x, env.needle.y, env.needle.w])
            done = False
            # Reward.append(env.episode_reward)
            episode_num += 1
            env.episode_reward = 0
            episode_timesteps = 0

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            # action = env.sample_action()
            action = env.PIDcontroller(action_constrain, parameter, env.t)
        else:
            action = policy.select_action(state)
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=2)*np.array([10,1])).clip(
                    env.action_bound[1,:], env.action_bound[0,:])

        # Perform action
        new_obs, reward, done = env.step(action,  episode_num, 'play', save_image=True)
        # print("done? "+ str(done))
        new_state = np.array([env.needle.x, env.needle.y, env.needle.w])

        running = env.check_status()

        done_bool = 0 if episode_timesteps + 1 == env.max_time else float(done)
        env.episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((state, new_state, action, reward, done_bool))

        obs = new_obs
        state = new_state

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    # plt.plot(np.arange(1,episode_num),Reward[0:episode_num],'b')
    # plt.savefig('./out/episode reward.png')

    # evaluations.append(evaluate_policy(policy))
    # if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    # np.save("./results/%s" % (file_name), evaluations)