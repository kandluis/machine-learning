import numpy.random as npr
import sys

import numpy as np

from parameters import game_params
from collections import defaultdict

class QLearner:
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self):
        self.Q = defaultdict(lambda: [0,0])
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_q_state = None
        self.iter_num = 0
        self.epsilon = .05

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_q_state = None
        self.iter_num = None


    def tree_discreet(self,tree_dist):
        '''
        Given the distance from the tree, returns a value specifying the bucket
        into which the distance falls.
        '''
        try:
            return int(tree_dist / (game_params['screen_width'] / 20.0))
        except ValueError:
            print tree_dist

    def height_diff_discreet(self,height_diff):
        '''
        Given the difference in height from the tree to the top of the monkey, 
        return a discretized value
        '''
        return int(height_diff / (game_params['screen_height'] / 16.0))

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        height_diff = state['tree']['top'] - state['monkey']['top']
        tree_dist = state['tree']['dist']
        
        new_state = (self.height_diff_discreet(height_diff) , self.tree_discreet(tree_dist))

        self.iter_num += 1
        if self.last_state is not None:
            if npr.rand() > 1 - self.epsilon:
                new_action = np.argmax(self.Q[self.last_state])
            else:
                new_action = npr.rand() < .4
        else:
            new_action = npr.rand() < 0.1

        # we need update our Q for the last state and action
        if self.last_state is not None and self.last_action is not None:
            s = self.last_state
            a = self.last_action
            r = self.last_reward
            sp = new_state
            self.Q[s][a] += (1./self.iter_num)*(r + 0.9*max(self.Q[sp]) - self.Q[s][a])

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward



    
