from matplotlib import pylab as plt
import numpy as np

import seaborn as sns

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
    plt.title("%s Score vs Epoch (%d Epochs)" % (self.learner,len(xs)))

    f2 = plt.figure()
    plt.scatter(xs,ys)
    plt.xlabel("Epoch")
    plt.ylabel("Score + 1")
    plt.title("%s Score vs Epoch (%d Epochs)" % (self.learner,len(xs)))
    
    return f1,f2

  def plot_distribution(self,test_runs):
    '''
    Plots histogram of last last test_runs runs
    '''
    if test_runs != 0:
        f = plt.figure()
        n = self.history.num_rounds()
        scores = [self.history.epoch(t).scores + 1 for t in xrange(n-test_runs,n)]
        plt.hist(scores)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Score Distribution for Last %d/%d Epochs for %s" % (test_runs, n, self.learner))

        return f
    return None