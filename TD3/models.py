import numpy as np
import torch.nn as nn
import torch

feat_size = 7
latent_dim = feat_size * feat_size * 256

def calc_features(img_stack):
    return img_stack - 1 + 3

def make_linear(in_size, out_size, bn=False):
    l = []
    l.append(nn.Linear(in_size, out_size))
    l.append(nn.ReLU())
    if bn:
        # normally disable for this algorithm
        l.append(nn.BatchNorm1d(out_size))
    return l

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def init_layers(layers):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data.uniform_(*hidden_init(layer))
    #layers[-1].data.uniform_(-3e-3, 3e-3)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BaseImage(nn.Module):
    def __init__(self, img_stack):
        super(BaseImage, self).__init__()
        self.encoder = nn.Sequential(

            ## input size:[img_stack, 224, 224]

            #---
            nn.Conv2d(calc_features(img_stack), 128, 3, 2, padding=1), ## [16, 112, 112]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 64, 3, 1, padding=1), ## [16, 112, 112]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #---
            nn.Conv2d(64, 128, 3 ,2, padding=1),   ## [32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 32, 3 ,1, padding=1),   ## [32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #---
            nn.Conv2d(32, 256, 3, 2, padding=1),   ## [64, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 64, 3, 1, padding=1),   ## [128, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #---
            nn.Conv2d(64, 512, 3, 2, padding=1),   ## [64, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 128, 3, 1, padding=1),   ## [128, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #---
            nn.Conv2d(128, 1024, 3, 2, padding=1),  ## [512, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 256, 3, 1, padding=1),  ## [256, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #---
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

            #nn.Linear(400, 100),
            #nn.ReLU(),
            #nn.BatchNorm1d(100),

            nn.Linear(400, out_size) # x, y, w
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
        #x = torch.clamp(x, min=-1., max=1.) * self.max_action
        x = torch.tanh(x) * self.max_action
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

class ActorState(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, bn=False):
        super(ActorState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend([nn.Linear(100, action_dim)])

        # init
        #init_layers(ll)

        self.linear = nn.Sequential(*ll)
        self.max_action = max_action

    def forward(self, x):
        x = self.linear(x)
        #x = torch.clamp(x, min=-1., max=1.) * self.max_action
        x = torch.tanh(x) * self.max_action
        return x


class CriticState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=False):
        super(CriticState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend([nn.Linear(100, 1)])

        #init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x
