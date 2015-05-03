from learner import Learner
from collections import defaultdict

import numpy as np

from parameters import game_params

class QLearner(Learner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: 0.15, discount_fn = lambda i: 1,
                 nbuckets_h = 5, nbuckets_w = 5, nbuckets_v = 3):
        super(QLearner,self).__init__()
        self.Q = defaultdict(lambda: [0, 0])
        self.iter_num = 0
        self.epoch_iters = 0

        # functions to modify behavior of q learner
        self.learn_fn = learn_fn
        self.discount_fn = discount_fn

    def reset(self):
        super(QLearner,self).reset()

        self.iter_num = 0

        self.learn_fn = None
        self.discount_fn = None

    def tree_discreet(self, tree_dist):
        '''
        Given the distance from the tree, returns a value specifying the bucket
        into which the distance falls.
        '''
        return np.floor(float(tree_dist) / float((game_params['screen_width']) / 5.0))

    def height_diff_discreet(self, height_diff):
        '''
        Given the difference in height from the tree to the top of the monkey, 
        return a discretized value
        '''
        return np.floor(float(height_diff) / (float(game_params['screen_height']) / 5.0))

    def velocity_diff_discreet(self,vel):
        return np.floor(float(vel)/3.0)

    def action_callback(self, state):
        '''
        Simple Q-Learning algorithm
        '''
        height_diff = state['monkey']['top'] - state['tree']['top']
        floor_diff = state['tree']['top'] - state['monkey']['bot']
        tree_dist = state['tree']['dist']
        monkey_vel = state['monkey']['vel']
        
        new_state = (self.height_diff_discreet(height_diff),
                     self.height_diff_discreet(floor_diff),
                     self.tree_discreet(tree_dist),
                     self.velocity_diff_discreet(monkey_vel))

        self.iter_num += 1
        new_action = np.argmax(self.Q[new_state])

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
