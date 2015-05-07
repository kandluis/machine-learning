from tdlearner import TDLearner
from collections import defaultdict

import numpy as np

# finds Euclidean norm distance between two lists a, b
def find_dist(a,b):
    cumsum = 0.0
    for i in xrange(len(a)):
        cumsum += (a[i]-b[i])**2
    return np.sqrt(cumsum/float(len(a)))

class ModelBased(TDLearner):
    '''
    Implements model based learning algorithm with value iteration.
    '''
    def __init__(self, discount_fn = lambda i: 0,bucket_height = 1., bucket_width = 28, velocity_bucket = 1000):
        super(ModelBased,self).__init__(learn_fn=lambda i:0, discount_fn=discount_fn,bucket_height=bucket_height, bucket_width=bucket_width,
                                       velocity_bucket=velocity_bucket)

        # keep track of current optimal policy, maps state s -> action a
        self.optimal_policy = defaultdict(int)

        # keep track of all states seen to iterate over in value_iter
        self.seen_states = set()

    def value_iter(self,discount):
        if len(self.seen_states) != 0:
            while True:
                # store off old value function to test for convergence later
                old_value_fn = []
                for s in self.seen_states:
                    old_value_fn.append(self.V[s])

                for s in self.seen_states:
                    # compute Q function for jump and no jump
                    for a in [0,1]:
                        # print "V states {}".format(self.V)
                        self.Q[s][a] = self.expected_reward(s,a) + discount * self.optimal_action_helper(s,a)
                    # find best action from state s
                    self.optimal_policy[s] = 1 if self.Q[s][1] > self.Q[s][0] else 0
                    # update value function for state s
                    self.V[s] = self.Q[s][self.optimal_policy[s]]

                # update new value function
                new_value_fn = []
                for s in self.seen_states:
                    new_value_fn.append(self.V[s])

                # test for convergence
                # print "Old value {}".format(old_value_fn)
                # print "V value {}".format(self.V)
                # print find_dist(old_value_fn, new_value_fn)
                if find_dist(old_value_fn, new_value_fn) < 0.1:
                    break

    def action_callback(self, state):
        '''
        Simple Q-Learning algorithm
        '''
        # what state are we in?
        new_state = self.get_state(state)
        if self.last_state is not None:
            self.seen_states.add(self.last_state)
            # print "Last state {}".format(self.last_state)

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

            new_action = self.optimal_policy[sp]

        else:
            new_action = 1

        # planning stage - updates optimal policy
        discount = self.discount_fn(self.iter_num)
        self.value_iter(discount)

        self.last_action = new_action
        self.last_state = new_state

        return new_action
