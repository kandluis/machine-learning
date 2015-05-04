from learner import Learner
from collections import defaultdict

import numpy as np
from parameters import game_params

class TDLearner(Learner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: 0.15, discount_fn = lambda i: .95,
                 nbuckets_h = 5., nbuckets_w = 5):
        super(QLearner,self).__init__()
        self.V = defaultdict(lambda: 0.0)
        self.NSA = defaultdict(lambda: 0.0)
        self.RSA = defaultdict(lambda: 0.0)
        self.NSAS = defaultdict(lambda: 0.0)
        self.reachable = defaultdict(lambda: {})
        self.iter_num = 0
        self.epoch_iters = 0
        self.last_state = None

        # functions to modify behavior of q learner
        self.learn_fn = learn_fn
        self.discount_fn = discount_fn

        # bucket discretization
        self.nbuckets_w = nbuckets_w
        self.nbuckets_h = nbuckets_h

    def pssa(self, sp, s, a):
        return float(self.NSAS[(s, a, sp)])/self.NSA[(s, a)]

    def expected_reward(self, s, a):
        return float(self.RSA[(s,a)])/self.NSA[(s, a)]

    def optimal_action_helper(self, s, a):
        res = 0.0
        for sp in self.reachable[s,a]:
            res =+ self.V[sp]*self.pssa(sp, s, a)
        return res

    def optimal_action(self, s):
        jump = self.expected_reward(s, 1) + self.optimal_action_helper(s, 1)
        no_jump = self.expected_reward(s, 0) + self.optimal_action_helper(s, 0)
        return 1 if jump > no_jump else 0

    def width_discreet(self, width):
        '''
        Given the distance from the tree, returns a value specifying the bucket
        into which the distance falls.
        '''
        return np.round(float(width) /20)

    def height_discreet(self, height):
        '''
        Given the difference in height from the tree to the top of the monkey,
        return a discretized value
        '''
        return np.round(float(height) / 5)

    def velocity_discreet(self, vel):
        return np.round(float(vel)/4)

    def get_state(self,state):
        '''
        Given returns the state of the game, returns the corresponding discreetized
        state for our RL algorithms
        '''
        height_diff = state['monkey']['top'] - state['tree']['top']
        floor_diff = state['tree']['bot'] - state['monkey']['bot']
        tree_dist = state['tree']['dist']
        vel = state['monkey']['vel']

        #new_state = (self.height_discreet(height_diff),
        #             self.height_discreet(floor_diff),
        #            self.width_discreet(tree_dist))

        #new_state = (self.height_discreet(floor_diff),
        #            self.width_discreet(tree_dist))

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

        # we need update our Q for the last state and action
        if (self.last_state is not None and
            self.last_action is not None and
            self.last_reward is not None):
            s = self.last_state
            a = self.last_action
            r = self.last_reward
            sp = new_state

            self.NSA[(s, a)] += 1
            self.NSAS[(s, a, sp)] += 1
            self.RSA[(s, a)] += r
            self.reachable[(s, a)] = self.reachable[(s, a)].union(sp)

            learn_rate = self.learn_fn(self.iter_num)
            discount = self.discount_fn(self.iter_num)
            self.V[s] += learn_rate * (r + discount*self.V[sp] - self.V[s])

        new_action = self.optimal_action(sp)

        self.last_action = new_action
        self.last_state = new_state

        return new_action
