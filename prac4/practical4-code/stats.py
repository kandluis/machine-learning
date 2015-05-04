from matplotlib import pylab as plt
import numpy as np

class Plots(object):
  def __init__(self,history,learner):
    self.history = history
    self.learner = learner

  def plot_score_by_epoch(self):
    '''
    Generates a scatter plot of score per epoch
    '''
    xs = range(self.history.num_rounds())
    ys = [self.history.epoch(t).scores + 1 for t in xs]

    plt.scatter(xs,np.log(ys))
    plt.xlabel("Training Iteration")
    plt.ylabel("Log of (Score + 1)")
    plt.title("%s Score vs Training Iteration" % (self.learner))
    plt.show()


