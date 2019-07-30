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
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# We have greyscale, and then one RGB
def calc_features(img_stack):
    return img_stack - 1 + 3

class BaseImage(nn.Module):
    def __init__(self, img_stack):
        super(BaseImage, self).__init__()
        self.encoder = nn.Sequential(  ## input size:[img_stack, 224, 224]
            nn.Conv2d(calc_features(img_stack), 16, 5, 2, padding=2), ## [16, 112, 112]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5 ,2, padding=2),   ## [32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, 2, padding=2),   ## [64, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, 2, padding=2),   ## [128, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, 2, padding=2),  ## [256, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            Flatten(),   ## 256*7*7
        )

class ImageToPos(BaseImage):
    ''' Class converting the image to a position of the needle.
        We train on this to accelerate RL training off images
    '''
    def __init__(self, img_stack, out_size=3):
        super(ImageToPos, self).__init__(img_stack)

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, out_size) # x, y, w
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x

class ActorImage(BaseImage):
    def __init__(self, action_dim, img_stack, max_action):
        super(ActorImage, self).__init__(img_stack)

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
        )

        self.out_angular = nn.Linear(100, int(action_dim))
        self.max_action = max_action

    def forward(self, x):
        x = self.encoder(x)
        x = linear(x)
        x = self.out_angular(x)
        x = torch.clamp(x, min=-1., max=1.) * self.max_action
        return x

class ActorState(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorState, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.BatchNorm1d(40),
        )

        self.out_angular = nn.Linear(40, int(action_dim))
        self.max_action = max_action

    def forward(self, x):
        x = self.linear(x)
        x = self.out_angular(x)
        x = torch.clamp(x, min=-1., max=1.) * self.max_action
        return x

class CriticImage(BaseImage):
    def __init__(self, action_dim, img_stack):
        super(CriticImage, self).__init__(img_stack)

        self.linear = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1),
        )

    def forward(self, x, u):
        x = self.encoder(x)
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class CriticState(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticState, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Linear(40, 1),
        )

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, img_stack, max_action, mode):

        self.max_action = max_action
        self.action_dim = action_dim
        self.mode = mode
        if mode == 'rgb_array':
            self.actor = ActorImage(action_dim, img_stack, max_action).to(device)
            self.actor_target = ActorImage(action_dim, img_stack, max_action).to(device)
            self.critic = CriticImage( action_dim, img_stack).to(device)
            self.critic_target = CriticImage( action_dim, img_stack).to(device)
        elif mode == 'state':
            self.actor = ActorState(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorState(state_dim, action_dim, max_action).to(device)
            self.critic = CriticState(state_dim, action_dim).to(device)
            self.critic_target = CriticState(state_dim, action_dim).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):
        # Copy as uint8
        if self.mode == 'rgb_array':
            state = torch.from_numpy(state).unsqueeze(0).to(device).float()
            state /= 255.0
        else:
            state = torch.from_numpy(state).to(device).float()
            # print("state size: " + str(state.size()))
        return self.actor(state).cpu().data.numpy().flatten()

    def copy_sample_to_device(self, x, y, u, r, d, w, batch_size):
        # Copy as uint8
        x = torch.from_numpy(x).squeeze(1).to(device).float()
        y = torch.from_numpy(y).squeeze(1).to(device).float()
        if self.mode == 'rgb_array':
            x /= 255.0 # Normalize
            y /= 255.0 # Normalize
        u = u.reshape((batch_size, self.action_dim))
        u = torch.FloatTensor(u).to(device)
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        w = w.reshape((batch_size, -1))
        w = torch.FloatTensor(w).to(device)
        return x, y, u, r, d, w

    def train(self, replay_buffer, iterations, beta_PER, args):

        batch_size = args.batch_size
        discount = args.discount
        tau = args.tau

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d, indices, w = replay_buffer.sample(batch_size, beta=beta_PER)
            state, next_state, action, reward, done, weights = \
                    self.copy_sample_to_device(x, y, u, r, d, w, batch_size)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = weights * ((current_Q - target_Q).pow(2))
            prios = critic_loss + 1e-5
            critic_loss = critic_loss.mean()

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            #actor_loss.data = -1

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

