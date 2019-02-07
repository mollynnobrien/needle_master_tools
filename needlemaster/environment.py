# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:30:52 2015

@author: Chris Paxton
"""
import os
import math
import numpy as np
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Poly
from shapely.geometry import Polygon, Point # using to replace sympy
from matplotlib.collections import PatchCollection

from pdb import set_trace as woah

def safe_load_line(name,handle):
    l = handle.readline()[:-1].split(': ')
    assert(l[0] == name)

    return l[1].split(',')

x_moves = [1, 3, 5, 10, 20]
theta_moves = [-0.1, -0.05, 0.05, 0.1]
move_array = [(x, 0.) for x in x_moves] + [(0., y) for y in theta_moves]

# Different behaviors for demo/rl, as they have slightly different requirements
mode_demo = 0
mode_rl = 1
two_pi = 2 * math.pi

class Environment:
    metadata = {'render.modes': ['rgb_array']}

    background_color = np.array([99., 153., 174.]) / 255

    record_interval = 10 # how often to record an episode

    def __init__(self, filename=None, mode=mode_demo, device=torch.device('cpu')):

        self.height   = 0
        self.width    = 0
        self.needle   = None
        ''' TODO: how do we want to constrain the game time? '''
        self.max_time = 300
        self.filename = filename
        if not os.path.exists('./out'):
            os.mkdir('./out')
        self.mode = mode
        self.device = device
        self.episode = 0

        self.reset()

    def train(self):
        ''' Dummy method '''
        pass

    def action_space(self):
        ''' Return the action space size of the environment '''
        return len(move_array)

    def reset(self):
        ''' Create a new environment. Currently based on attached filename '''
        self.done = False
        self.ngates = 0
        self.gates = []
        self.surfaces = []
        self.t = 0
        # environment damage is the sum of the damage to all surfaces
        self.damage = 0
        self.passed_gates = 0
        self.next_gate = None
        self.old_score = 0. # deal with scoring -> reward signal
        self.episode += 1 # next episode
        self.record = (self.episode % self.record_interval == 0)

        if self.filename is not None:
            with open(self.filename, 'r') as file:
                self.load(file)

        self.needle = Needle(self.width, self.height)

        return self.render(save_image=False)


    def render(self, mode='rgb_array', save_image=False, save_path='./out/'):
        # For RL, we want small square images
        if self.mode == mode_demo:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=(2.24,2.24), dpi=100) # 224x224
        plt.ylim(self.height)
        plt.xlim(self.width)
        frame = plt.gca()
        frame.set_facecolor(self.background_color)
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])

        for surface in self.surfaces:
            surface.draw()

        for gate in self.gates:
            gate.draw()

        self.needle.draw()

        if save_image or self.record:
            frame.invert_xaxis()
            plt.savefig(save_path + '{:03d}.png'.format(self.t))

        # Return the figure in a numpy buffer
        if mode == 'rgb_array':
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            plt.close('all')
            arr = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            arr = arr.astype(np.float32)
            arr /= 255.
            return torch.from_numpy(arr).permute(2,0,1).to(device=self.device)
        else:
            plt.close('all')

    @staticmethod
    def parse_name(filename):
        toks = filename.split('/')[-1].split('.')[0].split('_')
        return toks[1]

    '''
    Load an environment file.
    '''
    def load(self, handle):

        D = safe_load_line('Dimensions',handle)
        self.height = int(D[1])
        self.width = int(D[0])
        #print " - width=%d, height=%d"%(self.width, self.height)

        D = safe_load_line('Gates',handle)
        self.ngates = int(D[0])
        #print " - num gates=%d"%(self.ngates)

        for _ in range(self.ngates):
            gate = Gate(self.width, self.height)
            gate.load(handle)
            self.gates.append(gate)

        if self.ngates > 0:
            self.next_gate = 0
            self.gates[0].status = NEXT_GATE

        D = safe_load_line('Surfaces', handle)
        self.nsurfaces = int(D[0])
        #print " - num surfaces=%d"%(self.nsurfaces)

        for i in range(self.nsurfaces):
            s = Surface(self.width,self.height)
            s.load(handle)
            self.surfaces.append(s)

    def step(self, action, save_image=False, save_path='Out/'):
        """
            Move one time step forward
            Returns:
              * state of the world (in our case, an image)
              * reward
              * done
        """
        if self.mode == mode_rl:
            print("action =", action)
            action = move_array[action]

        needle_surface = self._surface_with_needle()
        self.needle.move(action, needle_surface, mode=self.mode)
        new_damage = self._get_new_damage(action, needle_surface)
        gate_status = self._update_and_check_gate_status()
        running = self._can_keep_running()
        self.damage += new_damage
        self.t += 1

        if self.mode == mode_rl:
            reward = self.get_reward(gate_status, damage)
        else:
            reward = self.score()
        #print("reward =", reward) # debug

        return (self.render(save_image=save_image, save_path=save_path), reward, not running)

    def _surface_with_needle(self):
        for s in self.surfaces:
            if self._needle_in_surface(s):
                return s
        return None

    def _needle_in_surface(self, s):
        needle_tip = np.array([self.needle.x, self.height - self.needle.y])
        s_flag = s.poly.contains(Point(needle_tip))
        return s_flag

    def _get_new_damage(self, movement, surface):
        if surface is not None:
            return surface.get_update_damage_and_color(movement)
        else:
            return 0.

    def _update_and_check_gate_status(self):
        """ have we passed a new gate? """
        x, y = self.needle.x, self.needle.y
        status = NOP_GATE
        if self.next_gate is not None:
            status = self.gates[self.next_gate].update_status([x, self.height - y])
            if status == FAILED_GATE or status == PASSED_GATE:
                # increment to the next gate
                self.next_gate += 1
                if self.next_gate < self.ngates:
                    # if we have this many gates, set gate status to be next
                    self.gates[self.next_gate].status = NEXT_GATE
                else:
                    self.next_gate = None
        return status

    def _can_keep_running(self):
        """
            verify if the game is in a valid state and we can
            keep playing
        """
        # is the needle off the screen?
        x = self.needle.x
        y = self.needle.y

        """ are you in a valid game configuration? """
        valid_x = x >= 0 and x < self.width
        valid_y = y >= 0 and y < self.height
        valid_pos = valid_x and valid_y
        if not valid_pos:
            print("Invalid position")

        # have you hit deep tissue?
        valid_deep = not self._deep_tissue_intersect()
        if not valid_deep:
            print("Punctured deep tissue")

        # check if you have caused too much tissue damage
        valid_damage = self.damage < 100
        if not valid_damage:
            print("Caused too much tissue damage")

        # are you out of time?
        valid_t = self.t < self.max_time
        if not valid_t:
            print("Ran out of game time")

        return valid_pos and valid_deep and valid_t and valid_damage

    def _deep_tissue_intersect(self):
        """
            check each surface, does the needle intersect the
            surface? is the surface deep?
        """
        for s in self.surfaces:
            if s.deep and self._needle_in_surface(s):
                return True
        return False

    def _compute_passed_gates(self):
        passed_gates = 0
        # see if thread_points goes through the gate at any points
        for gate in self.gates:
            if gate.status == PASSED_GATE:
                passed_gates += 1

        return passed_gates

    def _gate_score(self):
        '''
                if you pass all gates or there are no gates: 1000 pts
                        don't pass all gates: 1000 * (passed_gates/all_gates) pts
        '''
        MAX_SCORE = 1000.0
        passed_gates = self._compute_passed_gates()
        num_gates = len(self.gates)

        if num_gates == 0:
            gate_score = MAX_SCORE
        else:
            gate_score = MAX_SCORE * float(passed_gates)/num_gates
        return gate_score

    def _time_score(self):
        '''
            if more than 1/3 of the time is left: 1000 pts
            else decrease score linearly s.t. score at t=max_time -> 0pts
        '''
        MAX_SCORE = 1000.0
        t = self.t

        if t <= (1/3.) * self.max_time:
            time_score = MAX_SCORE
        else:
            m = -1.0 * MAX_SCORE/(2* self.max_time/3.)
            time_score = 1500 + m*t
        return time_score

    def _path_score(self):
        '''
            if path_len <= screen_width -> no penalty
            else decrease score linearly s.t. score at screen_width*3 -> -1000pts
        '''
        MIN_SCORE = -1000.0
        w = self.needle.path_length
        W = self.width
        if w <= W:
            path_score = 0
        else:
            # limit the path length to 3*W
            w = max(w, 3*W)
            path_score = MIN_SCORE/(2*W)*(w - W)
        return path_score

    def _damage_score(self):
        '''
                damage from tissue btwn [0, 100]. Scale to between [-1000,0]
                damage from deep tissue is -1000
        '''
        MIN_SCORE = -1000.0

        c = MIN_SCORE / 100 # 100 is the max damage you can get in the game
        damage = c * self.damage
        if self._deep_tissue_intersect():
            print("deep tissue intersect!")
            damage = damage + MIN_SCORE
        damage_score = damage
        return damage_score

    def score(self, print_flag=False):
        """
            compute the score for the demonstration

            gate_score: [0, 1000]
            time_score: [0, 1000]
            path_score: [-1000, 0]
            damage_score: [-2000, 0]

            max score:  2000
            min score: -3000

        """
        gate_score   = self._gate_score()
        time_score   = self._time_score()
        path_score   = self._path_score()
        damage_score = self._damage_score()

        score = gate_score + time_score + path_score + damage_score
        if print_flag:
            print("Score: " + str(score))
            print("-------------")
            print("Gate Score: " + str(gate_score))
            print("Time Score: " + str(time_score))
            print("Path Score: " + str(path_score))
            print("Damage Score: " + str(damage_score))

        return score

    def get_reward(self, gate_status, damage):
        ''' Reward for RL '''
        reward = 0.
        if gate_status == PASSED_GATE:
            reward += 0.5
        if gate_status == FAILED_GATE:
            reward -= 0.5
        if self._deep_tissue_intersect():
            reward -= 2.0
        reward -= damage / 100.
        return reward

NOP_GATE = 0 # Means nothing special
PASSED_GATE = 1
FAILED_GATE = 2
NEXT_GATE = 3

class Gate:
    color_passed = np.array([100., 175., 100.]) / 255
    color_failed = np.array([175., 100., 100.]) / 255
    color1 = np.array([251., 216., 114.]) / 255
    color2 = np.array([255., 50., 12.]) / 255
    color3 = np.array([255., 12., 150.]) / 255

    def __init__(self, env_width, env_height):
        self.x = 0
        self.y = 0
        self.w = 0
        self.top = np.zeros((4,2))
        self.bottom = np.zeros((4,2))
        self.corners = np.zeros((4,2))
        self.width = 0
        self.height = 0
        self.status = NOP_GATE
        ''' NOTE Chris implemented 'partner' gates, I think
        we can ignore this and implement gates that have to
        be hit sequentially for now '''

        self.c1 = self.color1
        self.c2 = self.color2
        self.c3 = self.color3
        self.highlight = None

        self.box = None
        self.bottom_box = None
        self.top_box = None

        self.env_width = env_width
        self.env_height = env_height

    def contains(self, poly, traj):
        return [poly.contains(Point(x)) for x in traj]

    def update_status(self, needle_pos):
        ''' take in current position,
            see if you passed or failed the gate
        '''
        p = Point(needle_pos)
        if self.status == NEXT_GATE and self.box.contains(p):
            self.status = PASSED_GATE
            self.c1 = self.color_passed
            self.c2 = self.color_passed
            self.c3 = self.color_passed
        elif self.status != PASSED_GATE and \
            (self.top_box.contains(p) or self.bottom_box.contains(p)):
            self.status = FAILED_GATE
            self.c1 = self.color_failed
            self.c2 = self.color_failed
            self.c3 = self.color_failed
        return self.status

    def draw(self):
        """
        warning = Color.argb(255, 255, 50, 12);
        """
        axes = plt.gca()
        axes.add_patch(Poly(self.corners, color=self.c1))
        if self.status == NEXT_GATE:
            axes.add_patch(Poly(self.corners,
                facecolor=self.c1, edgecolor='green'))
        axes.add_patch(Poly(self.top, facecolor=self.c2))
        axes.add_patch(Poly(self.bottom, facecolor=self.c3))
        # if next_gate, outline in green

    '''
    Load Gate from file at the current position.
    '''
    def load(self,handle):

        pos = safe_load_line('GatePos',handle)
        cornersx = safe_load_line('GateX',handle)
        cornersy = safe_load_line('GateY',handle)
        topx = safe_load_line('TopX',handle)
        topy = safe_load_line('TopY',handle)
        bottomx = safe_load_line('BottomX',handle)
        bottomy = safe_load_line('BottomY',handle)

        self.x = self.env_width*float(pos[0])
        self.y = self.env_height*float(pos[1])
        self.w = float(pos[2])

        self.top[:,0] = [float(x) for x in topx]
        self.top[:,1] = [float(y) for y in topy]
        self.bottom[:,0] = [float(x) for x in bottomx]
        self.bottom[:,1] = [float(y) for y in bottomy]
        self.corners[:,0] = [float(x) for x in cornersx]
        self.corners[:,1] = [float(y) for y in cornersy]

        # apply corrections to make sure the gates are oriented right
        self.w *= -1
        if self.w < 0:
            self.w = self.w + (np.pi * 2)
        if self.w > np.pi:
            self.w -= np.pi
            self.top = np.squeeze(self.top[np.ix_([2,3,0,1]),:2])
            self.bottom = np.squeeze(self.bottom[np.ix_([2,3,0,1]),:2])
            self.corners = np.squeeze(self.corners[np.ix_([2,3,0,1]),:2])

        self.w -= np.pi / 2

        avgtopy = np.mean(self.top[:,1])
        avgbottomy = np.mean(self.bottom[:,1])

        # flip top and bottom if necessary
        if avgtopy < avgbottomy:
            tmp = self.top
            self.top = self.bottom
            self.bottom = tmp

        # compute gate height and width

        # compute other things like polygon
        self.box        = Polygon(self.corners)
        self.top_box    = Polygon(self.top)
        self.bottom_box = Polygon(self.bottom)


class Surface:

    def __init__(self, env_width, env_height):
        self.deep = False
        self.corners = None
        self.color = None
        self.damage = 0 # the damage to this surface

        self.env_width = env_width
        self.env_height = env_height

        self.poly = None

    def draw(self):
        ''' update damage and surface color '''
        axes = plt.gca()
        axes.add_patch(Poly(self.corners, color=self.color))
    '''
    Load surface from file at the current position
    '''
    def load(self, handle):
        isdeep = safe_load_line('IsDeepTissue',handle)

        sx = [float(x) for x in safe_load_line('SurfaceX',handle)]
        sy = [float(x) for x in safe_load_line('SurfaceY',handle)]
        self.corners = np.array([sx,sy]).transpose()
        self.corners[:,1] = self.env_height - self.corners[:,1]


        self.deep = isdeep[0] == 'true'
        self.deep_color = np.array([207., 69., 32.]) / 255
        self.light_color = np.array([232., 146., 124.]) / 255
        self.color = np.array(self.deep_color if self.deep else self.light_color)

        self.poly = Polygon(self.corners)

    def get_update_damage_and_color(self, movement):
        dw = movement[1]
        if abs(dw) > 0.02:
            new_damage = (abs(dw)/2.0 - 0.01) * 100
            self.damage += new_damage
            if self.damage > 100:
                self.damage = 100
            self._update_color()
            return new_damage
        return 0.

    def _update_color(self):
        alpha = self.damage / 100.
        beta = (100. - self.damage) / 100.
        self.color = beta * self.light_color + alpha * self.deep_color

class Needle:

    def __init__(self, env_width, env_height):
        self.x = 96     # read off from saved demonstrations as start x
        self.y = env_height - 108    # read off from saved demonstrations as start y
        self.w = math.pi
        self.corners = None

        self.max_dXY      = 75
        self.length_const = 0.08
        self.scale        = np.sqrt(env_width**2 + env_height**2)
        self.is_moving    = False

        self.env_width = env_width
        self.env_height = env_height

        self.needle_color  = np.array([134., 200., 188.])/255
        self.thread_color  = np.array([167., 188., 214.])/255

        self.thread_points = [(self.x, self.y)]
        self.path_length = 0.

        self.current_surface = None

        self.load()

    def draw(self):
        self._draw_needle()
        self._draw_thread()

    def _update_corners(self):
        """
            given x,y,w compute needle corners and save
        """
        w = self.w
        x = self.x
        y = self.env_height - self.y

        top_w = w - math.pi/2
        bot_w = w + math.pi/2

        length = self.length_const * self.scale

        top_x = x - (0.01 * self.scale) * math.cos(top_w) + \
                (length * math.cos(w))
        top_y = y - (0.01 * self.scale) * math.sin(top_w) + \
                (length * math.sin(w))
        bot_x = x - (0.01 * self.scale) * math.cos(bot_w) + \
                (length * math.cos(w))
        bot_y = y - (0.01 * self.scale) * math.sin(bot_w) + \
                (length * math.sin(w))

        self.corners = np.array([[x, y], [top_x, top_y], [bot_x, bot_y]])

    def _draw_needle(self):
        axes = plt.gca()
        axes.add_patch(Poly(self.corners, color=self.needle_color))

    def _draw_thread(self):
        if len(self.thread_points) > 0:
            thread_points = np.array(self.thread_points)
            plt.plot(thread_points[:,0],
                    self.env_height - thread_points[:, 1],
                    c=self.thread_color)
    def load(self):
        """
            Load the current needle position
        """
        # compute the corners for the current position
        self._update_corners()
        self.poly = Polygon(self.corners)

    def move(self, movement, surface, mode):
        """
            Given an input, move the needle. Update the position, orientation,
            and thread path in android game movement is specified by touch
            points. last_x, last_y specify the x,y in the previous time step
            and x,y specify the current touch point

            dx = x - last_x
            dy = y - last_y
            w  = atan2(dy/dx)

            right now we assume you take in dx and dy
            (since we can directly pass that)

        """
        dX = movement[0]
        dw = movement[1]

        if surface is not None:
            dw = 0.5 * dw
            if abs(dw) > 0.01:
                dw = 0.02 * np.sign(dw)

        self.w += dw
        if self.w < 0.:
            self.w += two_pi
        if self.w > two_pi:
            self.w -= two_pi

        dx = dX * math.cos(self.w)
        dy = -dX * math.sin(self.w)
        oldx, oldy = self.x, self.y
        self.x += dx
        self.y += dy

        # In RL mode, don't allow to go out of bounds
        if mode == mode_rl:
            if self.x < 0 or self.x >= self.env_width:
                self.x = oldx
            if self.y < 0 or self.y >= self.env_height:
                self.y = oldy

        #print("move = ", movement, "wxy = ", self.w, self.x, self.y) # debug

        self._update_corners()
        self.poly = Polygon(self.corners)

        if self.x != oldx or self.y != oldy:
            self.thread_points.append(np.array([self.x, self.y]))
            dlength = math.sqrt(dx * dx + dy * dy)
            self.path_length += dlength
