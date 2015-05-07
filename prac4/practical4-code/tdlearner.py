from qlearner import QLearner
from collections import defaultdict

import numpy as np
from parameters import game_params

class TDLearner(QLearner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins.
    '''
    def __init__(self, learn_fn = lambda i: .186493, discount_fn = lambda i: 1.0,
                 bucket_height = 236., bucket_width = 530, velocity_bucket = 816):
        super(TDLearner,self).__init__(learn_fn, discount_fn,bucket_height, bucket_width,
                                       velocity_bucket)

        # value function, maps state, s -> value, V(s)
        self.V = defaultdict(lambda: 0.0)

        # state,action count. maps (s,a) -> # times we've been in s and performed action a
        self.NSA = defaultdict(lambda: 0)

        # total reward from (s,a) -> # reward
        self.RSA = defaultdict(float)

        # transition from (s,a) to (s') -> #times transition has occured
        self.NSAS = defaultdict(int)

        # set of reachable states
        self.reachable = defaultdict(lambda: set())

    def pssa(self, sp, s, a):
        return float(self.NSAS[(s, a, sp)])/float(self.NSA[(s, a)])

    def expected_reward(self, s, a):
        if self.NSA[(s,a)] != 0:
            return float(self.RSA[(s, a)])/float(self.NSA[(s, a)])
        else:
            return 0.0

    # returns expected value of future returns
    def optimal_action_helper(self, s, a):
        res = 0.0
        for sp in self.reachable[(s,a)]:
            res += float(self.V[sp])*self.pssa(sp, s, a)
        return res

    def optimal_action(self, s):
        jump = self.expected_reward(s, 1) + self.optimal_action_helper(s, 1)
        no_jump = self.expected_reward(s, 0) + self.optimal_action_helper(s, 0)
        res = 1 if jump > no_jump else 0
        #print jump, no_jump
        return res

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
