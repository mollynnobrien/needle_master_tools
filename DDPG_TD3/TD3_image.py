import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

feat_size = 56

''' Utilities '''
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.encoder = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 16, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
#             torch.nn.Conv2d(32, 64, 4, 2, padding=1),
#             torch.nn.LeakyReLU(),
            Flatten(),
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(feat_size * feat_size * 64, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 40),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(40, 30),
            torch.nn.LeakyReLU(),
        ])

        self.out_linear = nn.Linear(30, int(action_dim / 2))
        self.out_angular = nn.Linear(30, int(action_dim / 2))

    def forward(self, x):

        for layer in self.encoder:
            x = layer(x)
#             print(x.size())
        for layer in self.linear:
            x = layer(x)
#             print(x.size())

        out_linear = torch.sigmoid(self.out_linear(x))
        out_angular = torch.tanh(self.out_angular(x))

        x = torch.cat((out_linear,out_angular),1)

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.encoder_1 = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 16, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
#             torch.nn.Conv2d(32, 64, 4, 2, padding=1),
#             torch.nn.LeakyReLU(),
            Flatten(),
        ])
        
        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(feat_size * feat_size * 64, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512 + action_dim, 40),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(40, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, 1),
        ])
        
        
        self.encoder_2 = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 16, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
#             torch.nn.Conv2d(32, 64, 4, 2, padding=1),
#             torch.nn.LeakyReLU(),
            Flatten(),
        ])
        
        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(feat_size * feat_size * 64, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512 + action_dim, 40),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(40, 30),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(30, 1),
        ])


    def forward(self, x, u):
        
        x1 = x
        for layer in self.encoder_1:
            x1 = layer(x1)         
        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 3:
                x1 = torch.cat([x1, u], 1)
                x1 = layer(x1)
            else:
                x1 = layer(x1)
        
        x2 = x
        for layer in self.encoder_2:
            x2 = layer(x2)         
        counter = 0
        for layer in self.linear_2:
            counter += 1
            if counter == 3:
                x2 = torch.cat([x2, u], 1)
                x2 = layer(x2)
            else:
                x2 = layer(x2)
        
        return x1, x2

    def Q1(self, x, u):
        
        x1 = x
        for layer in self.encoder_1:
            x1 = layer(x1)         
        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 3:
                x1 = torch.cat([x1, u], 1)
                x1 = layer(x1)
            else:
                x1 = layer(x1)

        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim):

        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.actor_loss = []

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_loss = []

        # self.max_action = max_action

    def select_action(self, state):
        state = state.float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, beta_PER, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d, indices, w = replay_buffer.sample(batch_size, beta = beta_PER)
            state = torch.FloatTensor(x).squeeze(1).to(device)
#             print('state size: ' +str(state.size()))
            u = u.reshape((batch_size, self.action_dim))
            action = torch.FloatTensor(u).to(device)
#             print('action size: ' +str(action.size()))
            next_state = torch.FloatTensor(y).squeeze(1).to(device)
#             print('next state size: ' +str(next_state.size()))
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            w = w.reshape((batch_size,-1))
            weights = torch.FloatTensor(w).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action[0] = torch.clamp(next_action[0],0,1)
            next_action[1] = torch.clamp(next_action[1], -1, 1)
#             print(next_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss =  weights * ((current_Q1 - target_Q).pow(2) + (current_Q2 - target_Q).pow(2))
            prios = critic_loss + 1e-5
            critic_loss = critic_loss.mean()
            self.critic_loss.append(critic_loss)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_loss.append(actor_loss)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
