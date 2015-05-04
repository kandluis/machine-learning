from learner import Learner
from collections import defaultdict

import numpy as np
from parameters import game_params

class TDLearner(Learner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: 0.1, discount_fn = lambda i: .95,
                 bucket_height = 20., bucket_width = 50, velocity_bucket = 150):
        #super(QLearner,self).__init__()
        self.V = defaultdict(lambda: 0.0)
        self.NSA = defaultdict(lambda: 0.0)
        self.RSA = defaultdict(lambda: 0.0)
        self.NSAS = defaultdict(lambda: 0.0)
        self.Q = None
        self.reachable = defaultdict(lambda: set([]))
        self.iter_num = 0
        self.epoch_iters = 0
        self.last_state = None

        # functions to modify behavior of q learner
        self.learn_fn = learn_fn
        self.discount_fn = discount_fn

        # bucket discretization (width of buckets)
        self.bucket_width = bucket_width
        self.bucket_height = bucket_height
        self.velocity_bucket = velocity_bucket


    def pssa(self, sp, s, a):
        return float(self.NSAS[(s, a, sp)])/self.NSA[(s, a)]

    def expected_reward(self, s, a):
        if self.RSA[(s,a)] != 0:
            return float(self.RSA[(s, a)])/self.NSA[(s, a)]
        else:
            return 0.0

    def optimal_action_helper(self, s, a):
        res = 0.0
        for sp in self.reachable[s,a]:
            res =+ float(self.V[sp])*self.pssa(sp, s, a)
        return res

    def optimal_action(self, s):
        jump = self.expected_reward(s, 1) + self.optimal_action_helper(s, 1)
        no_jump = self.expected_reward(s, 0) + self.optimal_action_helper(s, 0)
        res = 1 if jump > no_jump else 0
        #print jump, no_jump
        return res

    def width_discreet(self, width):
        '''
        Given the distance from the tree, returns a value specifying the bucket
        into which the distance falls.
        '''
        return np.round(float(width) /self.bucket_width)

    def height_discreet(self, height):
        '''
        Given the difference in height from the tree to the top of the monkey,
        return a discretized value
        '''
        return np.round(float(height) / self.bucket_height)

    def velocity_discreet(self, vel):
        return np.round(float(vel)/self.velocity_bucket)

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
        #print new_state

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
            self.reachable[(s, a)].add(sp)

            learn_rate = self.learn_fn(self.iter_num)
            discount = self.discount_fn(self.iter_num)
            self.V[s] += learn_rate * (r + discount*self.V[sp] - self.V[s])
            new_action = self.optimal_action(sp)
            #print self.V[s]
        else:
            new_action = 1

        self.last_action = new_action
        self.last_state = new_state

        return new_action
