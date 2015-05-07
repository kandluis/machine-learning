from qlearner import QLearner
import numpy as np

from collections import defaultdict

class QLearner3(QLearner):
  '''
  Implements a more intelligent Q-Learning Mechanism. Uses Logistic function for learning learn_rate
  decay. Not that the parameters have been optimized using SpearMint.
  '''
  def __init__(self, escale = 100, lscale = 20, bucket_height = 4, bucket_width = 8, velocity_bucket = 4):
    super(QLearner3,self).__init__(learn_fn = None, discount_fn=None, bucket_width = bucket_width, bucket_height = bucket_height, velocity_bucket= velocity_bucket)

    # overwrite learn_fn and discount_fn into a logistic function
    self.learn_fn = lambda x: 1.0 / (float(x) / float(lscale)) if x > lscale else 1.0
    self.discount_fn = lambda x: 0.923
    self.epsilon_fn = lambda x: 1.0 / (float(x) / float(escale)) if x > escale else 1.0
    self.escale = escale
    self.lscale = lscale

    # used to keep track of the number of times a state Q(s,a) has been updated
    # so maps (s,a) -> int
    self.QCount = defaultdict(int)

    def pickle(self):
        d = super(QLearner3,self).pickle()
        d.update({
                 'QMatrix' : dict(self.Q),
                 'Qcount'  : dict(self.QCount)
                 })
        return d

    def save_params(self):
        d = super(QLearner3, self).save_params()
        d.update({  'escale'    :   self.escale,
                    'lscale'   :   self.lscale
                })
        return d

    def action_callback(self,state):
      # what state are we in
      new_state = self.get_state(state)

      self.iter_num += 1
      try:
          # we're now epsion-greedy
          epsilon = epsilon_fn(self.QCount(s,0) + self.QCount(s,1))
          if np.random.rand() < epsilon:
            new_state = 0 if np.random.rand() < 0.5 else 1
          else:
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

