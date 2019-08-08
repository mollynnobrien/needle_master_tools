# -*- coding: utf-8 -*-
import os
import math
import random
import numpy as np
from shapely.geometry import Polygon, Point # using to replace sympy
import pygame

GREEN = (0, 255, 0)

two_pi = math.pi * 2

VELOCITY = 50

def safe_load_line(name, handle):
    l = handle.readline()[:-1].split(': ')
    assert(l[0] == name)

    return l[1].split(',')

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.expand_dims(gray, 0)
    return gray

class Environment:
    metadata = {'render.modes': ['rgb_array', 'state']}
    background_color = np.array([99., 153., 174.])
    record_interval = 40
    record_interval_t = 3

    def __init__(self, mode, stack_size, log_file=None,
            filename=None, max_time=150):
        self.t = 0
        self.height = 0
        self.width = 0
        self.needle = None
        self.max_time = max_time
        self.next_gate = None
        self.filename = filename
        if not os.path.exists('./out'):
            os.mkdir('./out')
        self.mode = mode
        self.episode = 0
        self.status = None
        self.total_timesteps = 0
        self.Reward = []
        """ create image stack """
        self.stack_size = stack_size
        self.log_file = log_file

        self.is_init = False  # One-time stuff to do at reset
        # Create screen for scaling down
        self.scaled_screen = pygame.Surface((224, 224))
        pygame.font.init()
        self.reset()

    def sample_action(self):
        action = np.array([random.uniform(-1, -1), random.uniform(1, 1)])
        return action

    def reset(self, random_needle=False):
        ''' Create a new environment. Currently based on attached filename '''
        self.done = False
        self.ngates = 0
        self.gates = []
        self.surfaces = []
        self.t = 0
        # environment damage is the sum of the damage to all surfaces
        self.damage = 0
        self.next_gate = None
        self.last_dist = None
        self.episode += 1
        self.record = (self.episode == 1 or
                       self.episode % self.record_interval == 0)
        self.total_reward = 0.
        self.last_reward = 0.

        if self.filename is not None:
            with open(self.filename, 'r') as file:
                self.load(file)

        self.needle = Needle(self.width, self.height,
                self.log_file, random_pos=random_needle)

        # Assume the width and height won't change
        # Save the Surface creation
        if not self.is_init:
            self.is_init = True
            self.screen = pygame.Surface((self.width, self.height))

        if self.mode in ['rgb_array', 'both']:
            frame = self.render(save_image=False)
            # Create image stack
            gray = rgb2gray(frame)
            self.stack = [gray] * (self.stack_size - 1)
            ob = np.concatenate(self.stack + [frame])
            self.stack.append(gray)

        if self.mode in ['state', 'both']:
            state = self._get_state().reshape((1,-1))

        if self.mode == 'rgb_array':
            return ob
        elif self.mode == 'state':
            return state
        elif self.mode == 'both':
            return ob, state

    def render(self, mode='rgb_array', save_image=False, save_path='./out/'):

        self.screen.fill(self.background_color)

        for surface in self.surfaces:
            surface.draw(self.screen)

        for gate in self.gates:
            gate.draw(self.screen)

        self.needle.draw(self.screen)

        # --- Done with drawing ---

        # Scale if needed
        surface = self.screen
        if self.scaled_screen is not None:
            # Precreate surface with final dim and use ~DestSurface
            # Also consider smoothscale
            pygame.transform.scale(self.screen, [224, 224], self.scaled_screen)
            surface = self.scaled_screen

        if save_image:
            # draw text
            myfont = pygame.font.SysFont('Arial', 13)
            debug = False
            if debug:
                if self.next_gate is not None:
                    reward_s = "w:{:.2f}, g:({:.0f}, {:.0f}), " \
                    "n:({:.0f}, {:.0f})".format(
                        self.needle.w,
                        self.gates[self.next_gate].x,
                        self.gates[self.next_gate].y,
                        self.needle.x, self.needle.y)
                else:
                    reward_s = ""
            else:
                reward_s = "TR:{:.5f}, R:{:.5f}".format(
                        self.total_reward, self.last_reward)
                #reward_s = "w:{:.5f}, x:{:.5f}, y:{:.5f}".format(
                #        self.needle.w, self.needle.x, self.needle.y)
            txtSurface = myfont.render(reward_s, False, (0, 0, 0))
            surface.blit(txtSurface, (10, 10))

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file = os.path.join(save_path,
                    '{:06d}_{:03d}.png'.format(self.episode, self.t))
            pygame.image.save(surface, save_file)

        # Return the figure in a numpy buffer
        if mode == 'rgb_array':
            # Return an array of uint8 for efficiency
            frame = pygame.surfarray.array3d(surface).transpose((2, 0, 1))
            #frame = torch.from_numpy(arr).permute(2, 0, 1)
            return frame


    @staticmethod
    def parse_name(filename):
        toks = filename.split('/')[-1].split('.')[0].split('_')
        return toks[1]

    '''
    Load an environment file.
    '''
    def load(self, handle):

        D = safe_load_line('Dimensions', handle)
        self.height = int(D[1])
        self.width = int(D[0])
        #print " - width=%d, height=%d"%(self.width, self.height)

        D = safe_load_line('Gates', handle)
        self.ngates = int(D[0])
        #print " - num gates=%d"%(self.ngates)

        for _ in range(self.ngates):
            gate = Gate(self.width, self.height)
            gate.load(handle)
            self.gates.append(gate)

        if self.ngates > 0:
            self.next_gate = 0
            self.gates[0].status = 'next'

        D = safe_load_line('Surfaces', handle)
        self.nsurfaces = int(D[0])
        #print " - num surfaces=%d"%(self.nsurfaces)

        for i in range(self.nsurfaces):
            s = Surface(self.width, self.height)
            s.load(handle)
            self.surfaces.append(s)

    def _get_state(self):
        ''' Get state in a way the NN can read it '''
        if self.next_gate is not None:
            gate = self.gates[self.next_gate]
            gate_x, gate_y, gate_w = gate.x, gate.y, gate.w
        elif self.next_gate is None:
            gate_x, gate_y, gate_w = 0., 0., 0.

        state = []
        state.append(float(self.needle.x) / self.width)
        state.append(float(self.needle.y) / self.height)
        # Get back of needle
        c = self.needle.corners
        state.append((c[1,0] + c[2,0]) / (2.0 * self.width))
        state.append((c[1,1] + c[2,1]) / (2.0 * self.height))
        state.append(float(self.needle.w) / two_pi)
        state.append(float(self.needle.dx))
        state.append(float(self.needle.dy))
        state.append(float(self.needle.dw))
        for gate in self.gates:
            state.append(1.0 if gate.status == 'passed' else 0.0)
        state.append(float(gate_x) / self.width)
        state.append(float(gate_y) / self.height)
        state.append(float(gate_w) / two_pi)
        state = np.array(state, dtype=np.float32)
        #print "state = ", state # debug
        return state

    def step(self, action):
        """
            Move one time step forward
            Returns:
              * state of the world (in our case, an image)
              * reward
              * done
        """
        needle_surface = self._surface_with_needle()
        self.needle.move(action, needle_surface)
        new_damage = self._get_new_damage(action, needle_surface)
        self.damage += new_damage
        self.t += 1

        # calculate reward and done
        reward = 0
        done = False

        status = self._update_and_get_next_gate_status()
        if status == 'passed':
            self.last_dist = None
            reward += 100
        elif status == 'failed':
            self.last_dist = None
            reward -= 1
        elif status == 'done':
            self.last_dist = None
            done = True

        # Distance reward component
        if self.next_gate is not None:
            x2gate = self.needle.x - self.gates[self.next_gate].x
            y2gate = self.needle.y - self.gates[self.next_gate].y
            dist = np.sqrt(x2gate * x2gate + y2gate * y2gate)
            if self.last_dist is not None:
                delta = (self.last_dist - dist)/1000
                if delta < 0:
                    delta *= 10.
                    #delta = -0.1
                elif delta == 0:
                    delta = -0.5 # no standing still!
                else:
                    pass
                    #delta = 0.05
                reward += delta
            self.last_dist = dist

        # Time penalty
        # Makes circles not rewarding
        if not done:
            reward -= 0.01

        # Check for leaving window
        if (self.needle.x <= 0 or self.needle.x >= self.width or
                self.needle.y <= 0 or self.needle.y >= self.height):
            pass
            #reward -= 0.5
            #done = True

        if self._deep_tissue_intersect():
            reward -= 100.
            done = True

        # Damage component
        reward -= new_damage / 100

        # Check for excessive damage
        if self.damage > 100:
            reward -= 50
            done = True

        if self.t > self.max_time:
            done = True

        self.last_reward = reward
        self.total_reward += reward

        if self.record and self.t % self.record_interval_t == 0:
            self.render(mode='rgb_array', save_image=True)

        if self.mode in ['rgb_array', 'both']:
            """ if from image to action """
            frame = self.render(mode='rgb_array')
            self.stack.pop(0)
            ob = np.concatenate(self.stack + [frame])
            # Add memory as grayscale
            self.stack.append(rgb2gray(frame))
            assert len(self.stack) == self.stack_size

        if self.mode in ['state', 'both']:
            """ else from state to action"""
            state = self._get_state().reshape((1,-1))

        if self.mode == 'rgb_array':
            return ob, reward, done
        elif self.mode == 'state':
            return state, reward, done
        elif self.mode == 'both':
            return (ob, state), reward, done

    def _surface_with_needle(self):
        for s in self.surfaces:
            if self._needle_in_surface(s):
                return s
        return None

    def _get_new_damage(self, movement, surface):
        if surface is not None:
            return surface.get_update_damage_and_color(movement)
        else:
            return 0.

    def _needle_in_surface(self, s):
        return s.poly.contains(self.needle.tip)

    def _update_and_get_next_gate_status(self):
        """
            verify if the game is in a valid state and can
            keep playing
        """
        # have we passed a new gate?
        if self.next_gate is None:
            return 'done'

        status = self.gates[self.next_gate].update_status(self.needle.tip)
        # if you passed or failed the gate
        if status == 'failed' or status == 'passed':
            # increment to the next gate
            self.next_gate += 1
            if self.next_gate < self.ngates:
                self.gates[self.next_gate].status = 'next'
            else:
                self.next_gate = None

        return status


    def _deep_tissue_intersect(self):
        """
            check each surface, does the needle intersect the
            surface? is the surface deep?
        """
        for s in self.surfaces:
            if s.deep and self._needle_in_surface(s):
                return True
        return False

class Gate:
    color_passed = np.array([100., 175., 100.])
    color_failed = np.array([175., 100., 100.])
    color1 = np.array([251., 216., 114.])
    color2 = np.array([255., 50., 12.])
    color3 = np.array([255., 12., 150.])

    def __init__(self, env_width, env_height):
        self.x = 0
        self.y = 0
        self.w = 0
        self.top = np.zeros((4,2))
        self.bottom = np.zeros((4,2))
        self.corners = np.zeros((4,2))
        self.width = 0
        self.height = 0
        self.status = None
        self.c1 = self.color1
        self.c2 = self.color2
        self.c3 = self.color3
        self.highlight = None

        self.box = None
        self.bottom_box = None
        self.top_box = None

        self.env_width = env_width
        self.env_height = env_height

    def update_status(self, p):
        ''' take in current position,
            see if you passed or failed the gate
        '''
        if self.status != 'passed' and \
                (self.top_box.contains(p) or self.bottom_box.contains(p)):
            self.status = 'failed'
            self.c1 = self.color_failed
            self.c2 = self.color_failed
            self.c3 = self.color_failed
        elif self.status == 'next' and self.box.contains(p):
            self.status = 'passed'
            self.c1 = self.color_passed
            self.c2 = self.color_passed
            self.c3 = self.color_passed
        return self.status

    def draw(self, surface):
        pygame.draw.polygon(surface, self.c1, self.corners)
        # If next gate, outline in green
        if self.status == 'next':
            pygame.draw.polygon(surface, GREEN, self.corners, 20)
        pygame.draw.polygon(surface, self.c2, self.top)
        pygame.draw.polygon(surface, self.c3, self.bottom)

    '''
    Load Gate from file at the current position.
    '''
    def load(self, handle):

        pos = safe_load_line('GatePos', handle)
        cornersx = safe_load_line('GateX', handle)
        cornersy = safe_load_line('GateY', handle)
        topx = safe_load_line('TopX', handle)
        topy = safe_load_line('TopY', handle)
        bottomx = safe_load_line('BottomX', handle)
        bottomy = safe_load_line('BottomY', handle)

        self.x = self.env_width * float(pos[0])
        self.y = self.env_height * float(pos[1])
        self.w = float(pos[2])

        self.top[:, 0] = [float(x) for x in topx]
        self.top[:, 1] = [float(y) for y in topy]
        self.bottom[:, 0] = [float(x) for x in bottomx]
        self.bottom[:, 1] = [float(y) for y in bottomy]
        self.corners[:, 0] = [float(x) for x in cornersx]
        self.corners[:, 1] = [float(y) for y in cornersy]

        # apply corrections to make sure the gates are oriented right
        self.w *= -1
        if self.w < 0:
            self.w = self.w + (np.pi * 2)
        if self.w > np.pi:
            self.w -= np.pi
            self.top = np.squeeze(self.top[np.ix_([2, 3, 0, 1]), :2])
            self.bottom = np.squeeze(self.bottom[np.ix_([2, 3, 0, 1]), :2])
            self.corners = np.squeeze(self.corners[np.ix_([2, 3, 0, 1]), :2])

        self.w -= np.pi / 2

        avgtopy = np.mean(self.top[:, 1])
        avgbottomy = np.mean(self.bottom[:, 1])

        # flip top and bottom if necessary
        if avgtopy < avgbottomy:
            tmp = self.top
            self.top = self.bottom
            self.bottom = tmp

        # compute gate height and width

        # compute other things like polygon
        self.box = Polygon(self.corners)
        self.top_box = Polygon(self.top)
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

    def draw(self, surface):
        ''' update damage and surface color '''
        pygame.draw.polygon(surface, self.color, self.corners)

    '''
    Load surface from file at the current position
    '''
    def load(self, handle):
        isdeep = safe_load_line('IsDeepTissue', handle)

        sx = [float(x) for x in safe_load_line('SurfaceX', handle)]
        sy = [float(x) for x in safe_load_line('SurfaceY', handle)]
        self.corners = np.array([sx, sy]).transpose()
        self.corners[:, 1] = self.env_height - self.corners[:, 1]

        self.deep = isdeep[0] == 'true'
        self.deep_color = np.array([207., 69., 32.])
        self.light_color = np.array([232., 146., 124.])
        self.color = np.array(self.deep_color if self.deep else self.light_color)

        self.poly = Polygon(self.corners)

    def get_update_damage_and_color(self, movement):
        # Check for 2 components
        if len(movement) == 1:
            dw = movement[0]
        else:
            dw = movement[1]
        if abs(dw) > 0.02:
            new_damage = (abs(dw) / 2.0 - 0.01) * 100
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

    # Assume w=0 points to the negative x-axis

    def __init__(self, env_width, env_height, log_file, random_pos=False):
        if random_pos:
            self.x = random.randint(0, env_width - 1)
            self.y = random.randint(0, env_height - 1)
            self.w = random.random() * two_pi
        else:
            self.x = 96
            self.y = env_height - 108
            self.w = math.pi # face right
        self.dx = 0.0
        self.dy = 0.0
        self.dw = 0.0
        self.corners = None

        self.length_const = 0.08
        self.scale = np.sqrt(env_width ** 2 + env_height ** 2)
        self.is_moving = False

        self.env_width = env_width
        self.env_height = env_height

        self.needle_color = np.array([134., 200., 188.])
        self.thread_color = np.array([167., 188., 214.])

        # Save adjusted thread points since we don't use them for anything
        self.thread_points = [(self.x, env_height - self.y)]
        self.tip = Point(np.array([self.x, self.env_height - self.y]))
        self.path_length = 0.

        self.log_file = log_file

        self.load()


    def draw(self, surface):
        self._draw_needle(surface)
        self._draw_thread(surface)

    def _compute_corners(self):
        """
            given x,y,w compute needle corners and save
        """
        w = self.w
        x = self.x
        y = self.env_height - self.y

        length = self.length_const * self.scale

        lcosw = length * math.cos(w)
        lsinw = length * math.sin(w)
        scale = 0.01 * self.scale

        # Back of the needle
        top_w = w - math.pi/2
        bot_w = w + math.pi/2

        top_x = x - scale * math.cos(top_w) + lcosw
        top_y = y - scale * math.sin(top_w) + lsinw
        bot_x = x - scale * math.cos(bot_w) + lcosw
        bot_y = y - scale * math.sin(bot_w) + lsinw

        self.corners = np.array([[x, y], [top_x, top_y], [bot_x, bot_y]])

    def _draw_needle(self, surface):
        pygame.draw.polygon(surface, self.needle_color, self.corners)

    def _draw_thread(self, surface):
        if len(self.thread_points) > 1:
            pygame.draw.lines(surface, self.thread_color, False, self.thread_points, 10)
    def load(self):
        """
            Load the current needle position
        """
        # compute the corners for the current position
        self._compute_corners()

    def action2motion(self, action):
        """ API for reinforcement learning, mapping action to real motion
           action = [main engine, up-down engine]
           needle is set to moving from left to right (constant value)
           main engine: control the moving speed of the needle, -1 ~ 0 : off, 0 ~ 1 : on
           up-down engine: control the moving direction: -1 ~ -0.5: up, 0.5 ~ 1: down, -0.5 ~ 0.5: off
           action is clipped to [-1, 1]

           motion = [dX, dw], dX linear velocity should always be +, dw angular velocity
        """
        w = self.w

        """ 1 dimension action """
        dw = action[0]
        dw = math.pi if dw > math.pi else -math.pi if dw < -math.pi else dw
        dx = math.cos(math.pi - w - dw) * VELOCITY
        dy = -math.sin(math.pi - w - dw) * VELOCITY

        if self.log_file:
            self.log_file.write('action:{}\n'.format(action[0]))
            self.log_file.write('dx:{}, dy:{}, dw:{}\n'.format(dx, dy, dw))
            self.log_file.flush()

        return dw, dx, dy

    def move(self, action, needle_in_tissue):
        """
            Given an input, move the needle. Update the position, orientation,
            and thread path in android game movement is specified by touch
            points. last_x, last_y specify the x,y in the previous time step
            and x,y specify the current touch point

        """

        dw, dx, dy = self.action2motion(action)

        if needle_in_tissue:
            dw *= 0.5
            if (abs(dw) > 0.01):
                dw = 0.02 * np.sign(dw)

        self.w += dw
        if abs(self.w) > two_pi:
            self.w -= np.sign(self.w) * two_pi

        oldx, oldy = self.x, self.y
        self.x += dx
        self.y -= dy

        # Constrain x and y
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x > self.env_width:
            self.x = self.env_width
        if self.y > self.env_height:
            self.y = self.env_height

        if self.x != oldx or self.y != oldy:
            self.thread_points.append((self.x, self.env_height - self.y))
            dlength = math.sqrt(dx * dx + dy * dy)
            self.path_length += dlength

        self.dx, self.dy, self.dw = dx, dy, dw
        self.tip = Point(np.array([self.x, self.env_height - self.y]))
        self._compute_corners()

class PID:
    def __init__(self, Parameters, width, height):
        self.parameters = Parameters
        self.width = width
        self.height = height

    """" needle_pos = [needle.x, needle.y, needle.w]"""

    def GetSelfState(self, needle_pos):
        x = needle_pos[0] * self.width  ## x
        y = needle_pos[1] * self.height  ## y
        w = needle_pos[2] * pi - pi  ## w
        state = np.array([x, y, -w])
        # print("needle position: "+ str(x) +" "+ str(y))
        return state

    """ next_gate = env.next_gate, gates = env.gates """

    def GetGoalState(self, needle_pos, next_gate, gates):
        needle = self.GetSelfState(needle_pos)  ## needle state in global framework
        if next_gate is not None:
            gate = gates[next_gate]
            gate_x = gate.x
            gate_y = gate.y
        else:
            gate_x = gates[len(gates) - 1].x + 100
            gate_y = gates[len(gates) - 1].y
        local_x = (gate_x - needle[0]) * np.cos(needle[2]) + (gate_y - needle[1]) * np.sin(needle[2])
        local_y = (gate_x - needle[0]) * np.sin(needle[2]) + (gate_y - needle[1]) * np.cos(needle[2])
        return [local_x, local_y]

    def PIDcontroller(self, needle_pos, next_gate, gates, iteration):
        X = self.GetSelfState(needle_pos)
        X_t = self.GetGoalState(needle_pos, next_gate, gates)

        # """ two dimention action """
        action = np.array(X_t) * np.array(self.parameters)
        # ANG_SCALE = angle_final - (angle_final - angle_start) * math.exp(-1. * iteration / increase_rate)

        # action[0] = (action[0] - CONST) / 50
        # action[0] = action[0] / VEL_SCALE
        # action[1] = action[1] / (1/10 * pi)
        action = action.clip([-1, -1], [1, 1])

        return action


