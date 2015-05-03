from matplotlib import pylab as plt

class Plots(object):
  def __init__(self,history,learner):
    self.history = history
    self.learner = learner

  def plot_score_by_epoch(self):
    '''
    Generates a scatter plot of score per epoch
    '''
    xs = range(self.history.num_rounds())
    ys = [self.history.epoch(t).scores for t in xs]

    plt.scatter(xs,ys)
    plt.show()


