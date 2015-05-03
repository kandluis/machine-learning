from qlearner import QLearner
from collections import defaultdict

import numpy as np

from parameters import game_params

class QLearner2(QLearner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: 0.1, discount_fn = lambda i: 1,
                 nbuckets_h = 10, nbuckets_w = 10, nbuckets_s = 10):
        super(QLearner2,self).__init__(learn_fn, discount_fn, nbuckets_h, nbuckets_w)
        self.nbuckets_s = 10

    def velocity_discreet(self,vel):
        return np.floor(float(vel)/(game_params['horz_speed']/10))

    def get_state(self,state):
        '''
        New state function which increases our state space for better
        learning? 
        '''
        mpos = self.height_discreet((state['monkey']['top'] + state['monkey']['top'])/2)
        mvel = self.velocity_discreet(state['monkey']['vel'])
        ttop = self.height_discreet(state['tree']['top'])
        tbot = self.height_discreet(state['tree']['bot'])
        tree_dist = state['tree']['dist'] if state['tree']['dist'] > 0 else 0
        tdist = self.height_discreet(tree_dist)

        new_state = (mpos,ttop,tbot,tdist)

        return new_state
