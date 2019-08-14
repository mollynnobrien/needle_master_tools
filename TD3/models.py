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

def make_conv(in_channels, out_channels, kernel_size, stride, padding, bn=False):
    l = []
    l.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    l.append(nn.ReLU())
    if bn:
        l.append(nn.BatchNorm2d(out_channels))
    return l

class BaseImage(nn.Module):
    def __init__(self, img_stack, bn=False, img_dim=224):
        super(BaseImage, self).__init__()

        ## input size:[img_stack, 224, 224]

        ll = []
        in_f = calc_features(img_stack)
        if img_dim == 224:
            ll.extend(make_conv(in_f,128,  3, 2, 1, bn=bn)), ## 112
            ll.extend(make_conv(128,  64,  3, 1, 1, bn=bn)), ## 112
            ll.extend(make_conv(64,  128,  3, 2, 1, bn=bn)), ## 56
            ll.extend(make_conv(128,  32,  3, 1, 1, bn=bn)), ## 56
            ll.extend(make_conv(32,  256,  3, 2, 1, bn=bn)), ## 28
        elif img_dim == 112:
            ll.extend(make_conv(in_f, 128, 3, 2, 1, bn=bn)),  ## 56
            ll.extend(make_conv(128,  32,  3, 1, 1, bn=bn)),  ## 56
            ll.extend(make_conv(32,  256,  3, 2, 1, bn=bn)),  ## 28
        elif img_dim == 56:
            ll.extend(make_conv(in_f,256,  3, 2, 1, bn=bn)),  ## 28
        else:
            raise ValueError(str(img_dim) + " is not a valid img-dim")

        ll.extend(make_conv(256,  64,  3, 1, 1, bn=bn)),  ## 28
        ll.extend(make_conv(64,  512,  3, 2, 1, bn=bn)),  ## 14
        ll.extend(make_conv(512, 128,  3, 1, 1, bn=bn)),  ## 14
        ll.extend(make_conv(128, 1024, 3, 2, 1, bn=bn)),  ## 7
        ll.extend(make_conv(1024, 256, 3, 1, 1, bn=bn)),  ## 7

        ll.extend([Flatten()])
        self.encoder = nn.Sequential(*ll)

class ImageToPos(BaseImage):
    ''' Class converting the image to a position of the needle.
        We train on this to accelerate RL training off images
    '''
    def __init__(self, img_stack, out_size=3, bn=False, img_dim=224):
        super(ImageToPos, self).__init__(img_stack, bn, img_dim=img_dim)

        ll = []
        ll.extend(make_linear(latent_dim, 400, bn=bn))
        ll.extend([nn.Linear(400, out_size)]) # x, y, w
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x

class ActorImage(BaseImage):
    def __init__(self, action_dim, img_stack, max_action, bn=False, img_dim=224):
        super(ActorImage, self).__init__(img_stack, bn=bn, img_dim=img_dim)

        ll = []
        ll.extend(make_linear(latent_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        self.linear = nn.Sequential(*ll)

        self.out_angular = nn.Linear(100, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.out_angular(x)
        #x = torch.clamp(x, min=-1., max=1.) * self.max_action
        x = torch.tanh(x) * self.max_action
        return x

class CriticImage(BaseImage):
    def __init__(self, action_dim, img_stack, bn=False, img_dim=224):
        super(CriticImage, self).__init__(img_stack, bn=bn, img_dim=img_dim)

        ll = []
        ll.extend(make_linear(latent_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        ll.extend([nn.Linear(100, 1)])
        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = self.encoder(x)
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class QImage(BaseImage):
    def __init__(self, action_steps, img_stack, bn=False, img_dim=224):
        super(QImage, self).__init__(img_stack, bn=bn, img_dim=img_dim)

        ll = []
        ll.extend(make_linear(latent_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend([nn.Linear(300, action_steps)])
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.encoder(x)
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
        init_layers(ll)

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

        init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class QState(nn.Module):
    def __init__(self, state_dim, action_steps, bn=False):
        super(QState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 300, bn=bn))
        ll.extend([nn.Linear(300, action_steps)])

        init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.linear(x)
        return x
