# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:30:52 2015

@author: Chris Paxton
"""
import math
import numpy as np
import matplotlib
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

def array_to_tuples(array):
    return zip(array[:,0],array[:,1])

class Environment:

    def __init__(self,filename=None):

        self.height   = 0
        self.width    = 0
        self.ngates   = 0
        self.gates    = []
        self.surfaces = []
        self.t        = 0
        self.damage   = 0 # environment damage is the sum of the damage to all surfaces in the scene
        self.needle   = None
        ''' TODO: how do we want to constrain the game time? '''
        self.game_time = 200

        if not filename is None:
            print 'Loading environment from "%s"...'%(filename)
            handle = open(filename,'r')
            self.load(handle)
            handle.close()

            self.needle = Needle(self.width, self.height)

    def draw(self, save_image=False, gamecolor=True):
        axes = plt.gca()
        plt.ylim(self.height)
        plt.xlim(self.width)
        for surface in self.surfaces:
            surface.draw()
        for gate in self.gates:
            gate.draw()

        self.needle.draw()

        if(save_image):
            plt.gca().invert_xaxis()
            plt.savefig(str(self.t) + '.png')
            plt.close('all')

    def in_gate(self,demo):
        for gate in self.gates:
            print gate.contains(demo.s)
        return False

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
        print " - width=%d, height=%d"%(self.width, self.height)

        D = safe_load_line('Gates',handle)
        self.ngates = int(D[0])
        print " - num gates=%d"%(self.ngates)

        for i in range(self.ngates):
            gate = Gate(self.width,self.height)
            gate.load(handle)
            self.gates.append(gate)

        D = safe_load_line('Surfaces',handle)
        self.nsurfaces = int(D[0])
        print " - num surfaces=%d"%(self.nsurfaces)


        for i in range(self.nsurfaces):
            s = Surface(self.width,self.height)
            s.load(handle)
            self.surfaces.append(s)

    def step(self, action):
        """
            Move one time step forward
        """
        self.needle.move(action)
        #self.update_damage()
        self.t = self.t + 1

    def update_damage(self):
<<<<<<< HEAD
        environment_damage = 0
        for s in self.surfaces:
            ''' todo compute if needle intersects tissue'''
            if self.needle_tissue_intersect(s):
                if(abs(self.w) > 0.01):
=======
        for s in self.surfaces:
            environment_damage = 0
            if needle in surface:
                if abs(self.w) > 0.01:
>>>>>>> 19e20f8e213eeee516f5548cae459a5b36b40310
                    s.damage = s.damage + (abs(self.w) - 0.01)*100
                    s.update_color()
                environment_damage = environment_damage + s.damage
        self.damage = environment_damage

    def check_status(self):
        """
            verify if the game is in a valid state and can
            keep playing
        """
        # is the needle off the screen?
        x = self.needle.x
        y = self.needle.y

        valid_x = (x >= 0) and (x <= self.width)
        valid_y = (y >= 0) and (y <= self.height)
        valid_pos = valid_x and valid_y
        if(not valid_pos):
            print("Invalid position")

        # have you hit deep tissue?
        valid_deep = not self.deep_tissue_intersect()
        if(not valid_deep):
            print("Punctured deep tissue")

        # check if you have caused too much tissue damage
        valid_damage = self.damage < 100
        if(not valid_damage):
            print("Caused too much tissue damage")

        # are you out of time?
        valid_t = self.t < self.game_time
        if(not valid_t):
            print("Ran out of game time")

        return valid_pos and valid_deep and valid_t

    def deep_tissue_intersect(self):
        """
            check each surface, does the needle intersect the
            surface? is the surface deep?
        """
        intersect = False

        for s in self.surfaces:
            if(s.deep): # we only care about intersecting deep tissue
                s_intersect = self.needle_tissue_intersect(s)
                intersect = intersect or s_intersect

        return intersect

<<<<<<< HEAD
    def needle_tissue_intersect(self, s):
        return self.needle.poly.intersection(s.poly).area > 0

=======
>>>>>>> 19e20f8e213eeee516f5548cae459a5b36b40310
    def compute_passed_gates(self):
        passed_gates = 0
        # see if thread_points goes through the gate at any points
        for gate in self.gates:
<<<<<<< HEAD
            pass_gate    = np.sum(gate.contains(self.needle.thread_points)) > 0
=======
            pass_gate    = np.sum(gate.contains(self.thread_points)) > 0
>>>>>>> 19e20f8e213eeee516f5548cae459a5b36b40310
            passed_gates = passed_gates + pass_gate
        return passed_gates

    def gate_score(self):
        passed_gates = self.compute_passed_gates()
        num_gates = len(self.gates)

        if(num_gates == 0):
            gate_score = 1000
        else:
            gate_score = 1000 * float(passed_gates)/num_gates
        return gate_score

    def time_score(self):
        ''' TODO this doesn't make sense right now because we are
            measuring time stamps not milliseconds, we should change
            the threshold
            --- right now I'm changing it to 1/3 of self.game_time because
            orig 5000 was 1/3*15000'''
        time_remaining = self.game_time - self.t
        T = (1/3.0) * self.game_time
        if(time_remaining > T):
            time_score = 1000
        else:
            time_score = 1000 * float(time_remaining)/T
        return time_score

    def path_score(self):
        path_length = self.get_path_len()
        path_score = -50*path_length
        return path_score

    def get_path_len(self):
        """
                Compute the path length using the thread points
        """
        path_len = 0
        thread_points = np.array(self.needle.thread_points)
        for i in range(len(thread_points) - 1):
            pt_1 = thread_points[i, :]
            pt_2 = thread_points[i+1, :]

            dX = np.linalg.norm(pt_1 - pt_2)
            path_len = path_len + dX

        return path_len

    def damage_score(self):
        damage = self.damage
        if(self.deep_tissue_intersect):
            damage = damage + 1000 # will become negative on line 227

        damage_score = -4*damage

        return damage_score

    def score(self):
        """
            compute the score for the demonstration
        """
        gate_score   = self.gate_score()
        time_score   = self.time_score()
        path_score   = self.path_score()
        damage_score = self.damage_score()
        woah()
        return gate_score + time_score + path_score + damage_score

class Gate:

    def __init__(self,env_width,env_height):
        self.x = 0
        self.y = 0
        self.w = 0
        self.top = np.zeros((4,2))
        self.bottom = np.zeros((4,2))
        self.corners = np.zeros((4,2))
        self.width = 0
        self.height = 0

        self.box = None
        self.bottom_box = None
        self.top_box = None

        self.env_width = env_width
        self.env_height = env_height

    def contains(self, traj):
        return [self.box.contains(Point(x)) for x in traj] # changed demo.s to traj

    def features(self,demo):
        return False

    def draw(self,gamecolor=True):
        c1 = [251./255, 216./255, 114./255];
        c2 = [255./255, 50./255, 12./255];
        c3 = [255./255, 12./255, 150./255 ];
        ce = [0,0,0];

        if not gamecolor:
          c1 = [0.95, 0.95, 0.95];
          c2 = [0.75,0.75,0.75];
          c3 = [0.75,0.75,0.75];
          ce = [0.66, 0.66, 0.66];

        axes = plt.gca()
        axes.add_patch(Poly(array_to_tuples(self.corners),color=c1))
        axes.add_patch(Poly(array_to_tuples(self.top),color=c2))
        axes.add_patch(Poly(array_to_tuples(self.bottom),color=c3))

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

    def __init__(self,env_width,env_height):
        self.deep = False
        self.corners = None
        self.color = [0.,0.,0.]
        self.damage = 0 # the damage to this surface

        self.env_width = env_width
        self.env_height = env_height

        self.poly = None

    def draw(self):
<<<<<<< HEAD
=======
        ''' update damage and surface color '''
        #self.compute_damage()
        #self.update_color() # based on the amount of damage
>>>>>>> 19e20f8e213eeee516f5548cae459a5b36b40310
        axes = plt.gca()
        axes.add_patch(Poly(array_to_tuples(self.corners), color=self.color))
    '''
    Load surface from file at the current position
    '''
    def load(self, handle):
        isdeep = safe_load_line('IsDeepTissue',handle)

        sx = [float(x) for x in safe_load_line('SurfaceX',handle)]
        sy = [float(x) for x in safe_load_line('SurfaceY',handle)]
        self.corners = np.array([sx,sy]).transpose()
        self.corners[:,1] = self.env_height - self.corners[:,1]


        self.deep = (isdeep[0] == 'true')

        if not self.deep:
            self.color = [232./255, 146./255, 124./255]
        else:
            self.color = [207./255, 69./255, 32./255]

        self.poly = Polygon(self.corners)



    def update_color(self):
        damage = self.damage
        if(self.damage > 100):
            damage = 100

        r = 232 + ((207.0 - 232.0) * damage / 100.0)
        g = 146 + ((69.0 - 146.0) * damage / 100.0)
        b = 142 + ((32.0 - 142.0) * damage / 100.0)

        self.color = (255/255.0, r/255.0, g/255.0, b/255.0)

"""
        Added by Molly 11/28/2018
"""
class Needle:

    def __init__(self, env_width, env_height):
        self.x = 96     # read off from saved demonstrations as start x
        self.y = env_height - 108    # read off from saved demonstrations as start y
        self.PI = 3.141592654
        self.w = self.PI
        self.corners = None

        self.max_dXY      = 75
        self.length_const = 0.08
        self.scale        = np.sqrt(env_width**2 + env_height**2)
        self.is_moving    = False

        self.env_width = env_width
        self.env_height = env_height

        self.needle_color  = [134./255, 200./255, 188./255]
        self.thread_color  = [167./255, 188./255, 214./255]

        self.thread_points = []

        self.load()

    def draw(self):
        self.draw_needle()
        self.draw_thread()

    def compute_corners(self):
        """
            given x,y,w compute needle corners and save
        """
        w = self.w
        x = self.x
        y = self.env_height - self.y

        top_w = w - self.PI/2
        bot_w = w + self.PI/2

        length = self.length_const * self.scale

        top_x = x - (0.01 * self.scale) * math.cos(top_w) + (length * math.cos(w))
        top_y = y - (0.01 * self.scale) * math.sin(top_w) + (length * math.sin(w))
        bot_x = x - (0.01 * self.scale) * math.cos(bot_w) + (length * math.cos(w))
        bot_y = y - (0.01 * self.scale) * math.sin(bot_w) + (length * math.sin(w))

        corners = np.array([[x, y], [top_x, top_y], [bot_x, bot_y]])

        self.corners = corners

    def draw_needle(self):
        axes = plt.gca()
        axes.add_patch(Poly(array_to_tuples(self.corners), color=self.needle_color))

    def draw_thread(self):
        if(len(self.thread_points) > 0): # only draw if we have points
            thread_points = np.array(self.thread_points)
            plt.plot(thread_points[:,0], self.env_height - thread_points[:, 1], c=self.thread_color)


    def load(self):
        """
            Load the current needle position
        """
        # compute the corners for the current position
        self.compute_corners()
        # save the polygon
        self.poly = Polygon(self.corners)#sympy.Polygon(*[x[:2] for x in self.corners])

    def move(self, movement):
        """
            Given an input, move the needle. Update the position, orientation, and thread path

            in android game movement is specified by touch points. last_x, last_y specify
            the x,y in the previous time step and x,y specify the current touch point

            dx = x - last_x
            dy = y - last_y
            w  = atan2(dy/dx)

            right now we assume you take in dx and dy (since we can directly pass that)

        """
        dX = movement[0]
        dw = movement[1]

        self.w = self.w + dw
        self.x = self.x + dX * math.cos(self.w)
        self.y = self.y - dX * math.sin(self.w)

        self.compute_corners()
        self.thread_points.append([self.x,self.y])
