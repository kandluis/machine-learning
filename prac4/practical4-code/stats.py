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

    f1 = plt.figure()
    plt.scatter(xs,np.log(ys))
    plt.xlabel("Epoch")
    plt.ylabel("Log of (Score + 1)")
    plt.title("%s Score vs Epoch" % (self.learner))

    f2 = plt.figure()
    plt.scatter(xs,ys)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("%s Score vs Epoch" % (self.learner))
    
    return f1,f2
