import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

feat_size = 7
latent_dim = feat_size * feat_size * 256

''' Utilities '''
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Actor(nn.Module):
    def __init__(self, action_dim, img_stack, max_action):
        super(Actor, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[img_stack, 224, 224]
            torch.nn.Conv2d(img_stack*3, 16, 5, 2, padding=2), ## [16, 112, 112]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5 ,2, padding=2),   ## [32, 56, 56]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),   ## [64, 28, 28]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 2, padding=2),   ## [128, 14, 14]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## [256, 7, 7]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            Flatten(),   ## 256*7*7
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim, 400),
            torch.nn.ReLU(),
                torch.nn.BatchNorm1d(400),
                torch.nn.Linear(400, 30),
            torch.nn.ReLU(),
                torch.nn.BatchNorm1d(30),
        ])

        self.out_angular = nn.Linear(30, int(action_dim))
        self.max_action = max_action

    def forward(self, x):
        # print("round.....")
        for layer in self.encoder:
            x = layer(x)
            # print(x.size())
        for layer in self.linear:
            x = layer(x)

        x = self.out_angular(x)
        x = self.max_action * torch.tanh(x)

        return x

class Critic(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Critic, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[224, 224]
            torch.nn.Conv2d(img_stack*3, 16, 5, 2, padding=2),  ## [16, 112, 112]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## [32, 56, 56]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## [64, 28, 28]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 2, padding=2),  ## [128, 14, 14]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## [256, 7, 7]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            Flatten(),  ## output: 256*7*7
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 400),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(400),
            torch.nn.Linear(400, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Linear(100, 1),
        ])

    def forward(self, x, u):
        for layer in self.encoder:
                x = layer(x)
        x = torch.cat([x, u], 1)
        for layer in self.linear:
                x = layer(x)
        return x

class DDPG(object):
    def __init__(self, action_dim, img_stack, max_action):

        self.max_action = max_action
        self.action_dim = action_dim
        self.actor = Actor(action_dim, img_stack, max_action).to(device)
        self.actor_target = Actor(action_dim, img_stack, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.actor_loss = []

        self.critic = Critic(action_dim, img_stack).to(device)
        self.critic_target = Critic(action_dim, img_stack).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_loss = []


    def select_action(self, state):
        state = state.float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, beta_PER, args):

        batch_size = args.batch_size
        discount = args.discount
        tau = args.tau

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d, indices, w = replay_buffer.sample(batch_size, beta=beta_PER)
            state = torch.FloatTensor(x).squeeze(1).to(device)
            u = u.reshape((batch_size, self.action_dim))
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).squeeze(1).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            w = w.reshape((batch_size, -1))
            weights = torch.FloatTensor(w).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = weights * ((current_Q - target_Q).pow(2))
            prios = critic_loss + 1e-5
            critic_loss = critic_loss.mean()
            self.critic_loss.append(critic_loss)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()/100
            #actor_loss.data = -1
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

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path, 'actor_target.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(path, 'critic_target.pth'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth')))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, 'critic_target.pth')))

