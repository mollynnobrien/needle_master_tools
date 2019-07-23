import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())
    # TODO: y = x.abs().sqrt_()
    # TODO: return x.sign_().mul_(y)

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

# Base class for convolutional and full connected
class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class DQNConv(DQNBase):
  def __init__(self, args, action_space):
    super().__init__(args, action_space)

    self.dim = 224
    self.first_size = args.history_length * args.channels
    self.initial_size = int(16 * self.dim * self.dim / (2 * 2))
    self.reduced_size = int(self.initial_size / (2 ** 5))
    print("initial_size = ", self.initial_size)
    print("reduced_size =", self.reduced_size)

<<<<<<< HEAD
    self.conv1 = nn.Conv2d(args.history_length, 16, 5, stride=2, padding=2)
=======
    # initial size: 16 * dim * dim / (2 * 2)
>>>>>>> 0b468b99146f0e23d2d2773d2938d305054247d5
    # every other convolution halves
    self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
    self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
    self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
    self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 7*7
    self.conv6 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
    self.fc_h_v = NoisyLinear(self.reduced_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.reduced_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
<<<<<<< HEAD
    # x = x.view(-1, self.first_size, 224, 224)
=======
    #print(x.shape)
>>>>>>> 0b468b99146f0e23d2d2773d2938d305054247d5
    x = F.relu(self.conv1(x))
    #print(x.shape)
    x = F.relu(self.conv2(x))
    #print(x.shape)
    x = F.relu(self.conv3(x))
    #print(x.shape)
    x = F.relu(self.conv4(x))
<<<<<<< HEAD
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
=======
    #print(x.shape)
    #print(x.shape)
    #print(x.shape)
>>>>>>> 0b468b99146f0e23d2d2773d2938d305054247d5
    x = x.view(-1, self.reduced_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # Log probabilities with action over second dimension
    q = F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)  
    return q
 def return_niose(self):
     for name, module in self.named_children():
         if 'fc' in name:
             module.reset.noise()

