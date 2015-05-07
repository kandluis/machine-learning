from qlearner import QLearner
import numpy as np

from collections import defaultdict

class QLearner2(QLearner):
  '''
  Implements a more intelligent Q-Learning Mechanism. Uses Logistic function for learning learn_rate
  decay. Note that the parameters have been optimized using SpearMint.
  '''
  def __init__(self, k = 2, x0 = 0, bucket_height = 46, bucket_width = 200, velocity_bucket = 100):
    # for more info: http://en.wikipedia.org/wiki/Logistic_function 
    # k is steepness
    # x0 is the value at which f(x0) = 0.5
    def logistic(x,x0,k):
      return 1.0 / (1.0 + np.exp(k*(x - x0)))

    super(QLearner2,self).__init__(learn_fn = None, bucket_width = bucket_width, bucket_height = bucket_height, velocity_bucket= velocity_bucket)

    # overwrite learn_fn and discount_fn into a logistic function
    self.learn_fn = lambda x: logistic(x,x0,k)
    self.k = k
    self.x0 = x0

    # discount has been proven to be essentially 1 if we want to be good
    self.discount_fn = lambda i : 0.923

    # used to keep track of the number of times a state Q(s,a) has been updated
    # so maps (s,a) -> int
    self.QCount = defaultdict(int)

    def pickle(self):
        d = super(QLearner2,self).pickle()
        d.update({
                 'QMatrix' : dict(self.Q),
                 'Qcount'  : dict(self.QCount)
                 })
        return d

    def save_params(self):
        d = super(QLearner2, self).save_params()
        d.update({  'k'    :   self.k,
                    'x0'   :   self.x0
                })
        return d

    def action_callback(self,state):
      # what state are we in
      new_state = self.get_state(state)

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

          # update the count for the previous state
          self.QCount[(s,a)] += 1

          learn_rate = self.learn_fn(self.QCount[(s,a)]) 
          discount = self.discount_fn(self.iter_num)
          self.Q[s][a] += learn_rate * (r + discount*(max(self.Q[sp])) - self.Q[s][a])
      
      self.last_action = new_action
      self.last_state = new_state

      return new_action

