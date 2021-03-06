import numpy.random as npr
import sys

import numpy as np

from parameters import game_params
from collections import defaultdict

class Learner(object):
    '''
    Base class for a Learner. Other class inherit.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def pickle(self):
        '''
        Defines data to be pickled in dictionary format
        '''
        return {}

    def save_params(self):
        return {}

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
