import numpy as np
import torch
import torch.nn as nn
import os, sys
import torch.nn.functional as F
from models import QState, QImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN


# NOTE: Batchnorm is a problem for these algorithms. We need consistency
# and determinism, especially for the actor. Batchnorm seems to break that.

# We have greyscale, and then one RGB

class DQN(object):
    def __init__(self, state_dim, action_dim, action_steps, img_stack,
            max_action, mode, lr, bn=False, img_dim=224, load_encoder=''):

        self.action_dim = action_dim
        self.max_action = max_action
        self.action_steps = action_steps
        self.action_step_size = float(max_action) * 2 / action_steps
        self.action_offset = self.action_step_size * action_steps / 2

        self.mode = mode
        if self.mode == 'rgb_array':
            self.q = QImage(action_steps, img_stack, bn=bn, img_dim=img_dim).to(device)
            self.q_target = QImage(action_steps, img_stack, bn=bn, img_dim=img_dim).to(device)
        elif self.mode == 'state':
            self.q = QState(state_dim, action_steps, bn=bn).to(device)
            self.q_target = QState(state_dim, action_steps, bn=bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)

        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

        if load_encoder != '':
            print "Loading encoder model..."
            for model in [self.q, self.q_target]:
                    model.encoder.load_state_dict(torch.load(load_encoder))

    def select_action(self, state):
        # Copy as uint8
        if self.mode == 'rgb_array':
            state = torch.from_numpy(state).unsqueeze(0).to(device).float()
            state /= 255.0
        elif self.mode == 'state':
            state = torch.from_numpy(state).to(device).float()
            # print("state size: " + str(state.size()))
        else:
            raise ValueError('Unrecognized mode ' + mode)

        q = self.q(state)
        max_action = torch.argmax(q)
        # Translate action choice to continous domain
        translated_action = ((max_action.cpu().data.numpy() *
                self.action_step_size) - self.action_offset)
        action = np.expand_dims(translated_action, 1)
        #print action.shape # debug
        return action

    def copy_sample_to_device(self, x, y, u, r, d, w, batch_size):
        # Copy as uint8
        x = torch.from_numpy(x).squeeze(1).to(device).float()
        y = torch.from_numpy(y).squeeze(1).to(device).float()
        if self.mode == 'rgb_array':
            x /= 255.0 # Normalize
            y /= 255.0 # Normalize
        u = u.reshape((batch_size, self.action_dim))
        # Convert action to discrete
        u += self.action_offset
        u /= self.action_step_size
        u = torch.LongTensor(u).to(device)
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        w = w.reshape((batch_size, -1))
        w = torch.FloatTensor(w).to(device)
        return x, y, u, r, d, w

    def train(self, replay_buffer, timesteps, beta, args):

        batch_size = args.batch_size
        discount = args.discount
        tau = args.tau

        # Sample replay buffer
        x, y, u, r, d, indices, w = replay_buffer.sample(
                batch_size, beta=beta)

        state, next_state, action, reward, done, weights = \
                self.copy_sample_to_device(x, y, u, r, d, w, batch_size)

        target_Qs = self.q_target(next_state)
        target_Q = torch.max(target_Qs, dim=-1)[0].unsqueeze(-1)
        #print action #debug

        # TODO: only count ones that are max
        # Compute the target Q value
        target_Q = reward + (done * discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.q(state)
        current_Q = torch.gather(current_Q, -1, action)

        # Compute loss
        q_loss = (current_Q - target_Q).pow(2)
        prios = q_loss + 1e-5
        prios = prios.data.cpu().numpy()
        q_loss *= weights
        q_loss = q_loss.mean()

        # debug graph
        '''
        import torchviz
        dot = torchviz.make_dot(q_loss, params=dict(self.q.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

        #print("weights = ", w, "prios = ", prios)

        # Optimize the model
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        replay_buffer.update_priorities(indices, prios)

        # Update the frozen target models
        for param, param_t in zip(self.q.parameters(), self.q_target.parameters()):
            param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)

        return q_loss.item(), None

    def save(self, path):
        torch.save(self.q.state_dict(), os.path.join(path, 'q.pth'))
        torch.save(self.q_target.state_dict(), os.path.join(path, 'q_target.pth'))

    def load(self, path):
        self.q.load_state_dict(torch.load(os.path.join(path, 'q.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth')))
        self.q_target.load_state_dict(torch.load(os.path.join(path, 'q_target.pth')))

