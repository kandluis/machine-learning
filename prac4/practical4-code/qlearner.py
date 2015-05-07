from learner import Learner
from collections import defaultdict

import numpy as np
import copy

from parameters import game_params

class QLearner(Learner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: 0.15, discount_fn = lambda i: .95,
                 bucket_height = 5., bucket_width = 20, velocity_bucket = 4):
        super(QLearner,self).__init__()
        self.Q = defaultdict(lambda: [0, 0])
        self.iter_num = 0
        self.epoch_iters = 0

        # functions to modify behavior of q learner
        self.learn_fn = learn_fn
        self.discount_fn = discount_fn

        # bucket discretization (width of buckets)
        self.bucket_width = bucket_width
        self.bucket_height = bucket_height
        self.velocity_bucket = velocity_bucket

    def pickle(self):
        d = super(QLearner,self).pickle()
        d.update({
                 'QMatrix' : dict(self.Q)
                 })
        return d

    def save_params(self):
        d = super(QLearner, self).save_params()
        d.update({  'learn_rate'    :   self.learn_fn(0),
                    'discount'      :   self.discount_fn(0),
                    'bucket_height' :   self.bucket_height,
                    'bucket_width'  :   self.bucket_width,
                    'velocity_bucket':   self.velocity_bucket 
                })
        return d

    def width_discreet(self, width):
        '''
        Given the distance from the tree, returns a value specifying the bucket
        into which the distance falls.
        '''
        return int(np.round(float(width) /self.bucket_width))

    def height_discreet(self, height):
        '''
        Given the difference in height from the tree to the top of the monkey, 
        return a discretized value
        '''
        return int(np.round(float(height) / self.bucket_height))

    def velocity_discreet(self, vel):
        return int(np.round(float(vel)/self.velocity_bucket))

    def get_state(self,state):
        '''
        Given returns the state of the game, returns the corresponding discreetized 
        state for our RL algorithms
        '''
        height_diff = state['monkey']['top'] - state['tree']['top']
        tree_dist = state['tree']['dist']
        vel = state['monkey']['vel']

        new_state = (self.height_discreet(height_diff),
                     self.velocity_discreet(vel),
                    self.width_discreet(tree_dist))


        return new_state

    def action_callback(self, state):
        '''
        Simple Q-Learning algorithm
        '''
        # what state are we in?
        new_state = self.get_state(state)

        # increase iteration count and maximize chose action to maximize expected reward
        self.iter_num += 1
        try:
            new_action = np.argmax(self.Q[new_state])
        except TypeError:
            print new_state
            raise Exception(self.Q[new_state])

        # we need update our Q for the last state and action
        if (self.last_state is not None and 
            self.last_action is not None and 
            self.last_reward is not None):
            s = self.last_state
            a = self.last_action
            r = self.last_reward
            sp = new_state

            learn_rate = self.learn_fn(self.iter_num) 
            discount = self.discount_fn(self.iter_num)
            self.Q[s][a] += learn_rate * (r + discount*(max(self.Q[sp])) - self.Q[s][a])
        
        self.last_action = new_action
        self.last_state = new_state

        return new_action
