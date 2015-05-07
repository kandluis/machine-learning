from SwingyMonkey import SwingyMonkey

# used for command-line argument parsing
from optparse import OptionParser

# to get command-line arguments
import sys
import copy
import os

# store history on epochs
from history import History

# to make graphs
from stats import Plots
import numpy as np

# helper functions
import helpers

def parse_learners(args):
    """
    Each element is a class name like "Learner".
    Returns an array with a list of class names.
    """
    ans = []
    for c in args:
        s = c.split(' ')
        if len(s) == 1:
            ans.extend(s)
        else:
            raise ValueError("Bad argument: %s\n" % c)
    return ans

def load_modules(modules):
  '''
  Loads module classes. Each class must be in class_name.lower().
  Returns a dictionary of class_name -> class object
  '''
  def load(class_name):
    module_name = class_name.lower()
    module = __import__(module_name)
    learner_class = module.__dict__[class_name]
    return (class_name, learner_class)

  return dict(map(load, modules))

def init_learner(learner, learner_classes):
  '''
  All learners are initialized with default parameters. 
  '''
  return learner_classes[learner]()

def session(learner, options):
  learner_class = init_learner(learner, options.learner_classes)

  # history dictionaries: epoch # -> whatever
  rewards = {}
  scores = {}

  history = History(rewards, scores)

  # save state
  pvideo = options.video

  print "Starting training phase for %s ..." % (learner)
  max_score = 0
  for t in xrange(options.train_iters + options.test_iters):
    prev_score = scores[t-1] if t > 0 else 0
    # print information about the epoch currently being run
    if t == options.train_iters:
      print("Starting testing phase for %s ..." % (learner))
      options.video = (options.test_tick > 0)
    if t < options.train_iters:
      print("======= Training epoch %d / %d." % (t,options.train_iters))
    else:
      print("======= Test epoch %d / %d." % (t - options.train_iters, options.test_iters))
    print("Max score: %d. Previous epoch score: %d" % (max_score, prev_score))

    # Make a new monkey object.
    swing = SwingyMonkey(visual=options.video,
                         sound=False,   
                         tick_length=options.train_tick if t < options.train_iters else options.test_tick,        
                         action_callback=learner_class.action_callback,
                         reward_callback=learner_class.reward_callback)

    # Loop until you hit something.
    episode_rewards = []
    while swing.game_loop():
      if learner_class.last_reward is not None:
        episode_rewards.append(learner_class.last_reward)

    # collect statistics
    rewards[t] = copy.deepcopy(episode_rewards)
    scores[t] = copy.deepcopy(swing.score)

    max_score = max(max_score, scores[t])


  # reset
  options.video = pvideo

  return history, learner_class

def get_score(hist, c):
  '''
  Calculates some statistics of the final c epochs in hist.
  '''
  n = hist.num_rounds()
  rewards = [hist.epoch(t).scores for t in xrange(n-c,n)]

  return {  'mean'      : float(sum(rewards))/ float(len(rewards)),
            'median'    : np.median(rewards)}

def run_session(options, args):
    """
    Runs the training simulation given a parsed set of options and its leftover
    elements. 

    Returns a dictionary of {key: value} pairings with information on the results
    of the simulation.
    """
    # leftover args are class names for training methodology
    if len(args) == 0:
      # default
      learners_to_run = ['Learner']
    else:
      learners_to_run = parse_learners(args)

    options.learner_class_names = learners_to_run
    options.learner_classes = load_modules(options.learner_class_names)

    n = len(learners_to_run)

    options.video = options.train_tick > 0

    learner_histories = {}
    taught_learners = {}
    learner_scores = {}
    # train each class and store results
    for learner, learner_class in options.learner_classes.iteritems():
      hist, learned = session(learner,options)
      learner_histories[learner] = hist
      taught_learners[learner] = learned

      learner_scores[learner] = get_score(hist, options.test_iters if options.test_iters != 0 else min(100,options.train_iters))

    # TODO : Here, we have access to each learner's training history as we
    # as the trained learner. Should do stuff with it.
    # create a plot class for each learner 
    plots = [Plots(learner_histories[learner], learner) for learner in options.learner_class_names]

    # generate plots for each learner
    learner_plots = {}
    for plot in plots:
      learner_plots[plot.learner] = list(plot.plot_score_by_epoch())
      dist = plot.plot_distribution(options.test_iters)
      if dist is not None:
        learner_plots[plot.learner].append(dist)

    # save results if possible
    if not helpers.save_results(options.learner_class_names, 
                                taught_learners, learner_histories,
                                learner_scores,
                                learner_plots,
                                options.outfile):
      print "Failed to save results."


   
def parse_inputs(args):
    usage_msg = "Usage:  %run [options] LearnerClass1 LearnerClass2 ..."
    parser = OptionParser(usage=usage_msg)

    def usage(msg):
        print "Error: %s\n" % msg
        parser.print_help()
        sys.exit()

    parser.add_option("--train-iters",
                      dest="train_iters", default=1024, type="int",
                      help="Set number of training epochs")

    parser.add_option("--test-iters",
                      dest="test_iters", default=3, type="int",
                      help="Set number of testing epochs for model evaluations")

    parser.add_option("--train-tick",
                      dest="train_tick", default=0, type="int",
                      help="Clock tick for training. Not displayed when less than 1.")

    parser.add_option("--test-tick",
                      dest="test_tick", default=1, type="int",
                      help="Clock tick for testing phase. Not displayed when less than 1.")

    parser.add_option("--plots",
                      dest="plots", default="true", type="string",
                      help="Boolean specifying whether to generate plots or not.")

    parser.add_option("--outfile",
                      dest="outfile", default="results", type="string",
                      help="Saves pickled learner class to ./TIME/OUTFILE_CLASS.p\
                      and ./TIME/OUTFILE_CLASS.csv")

    return parser

def main(args):
  # parse input arguments into object
  parser = parse_inputs(args)

  # extract options and arguments from parse object
  (options, args) = parser.parse_args()

  run_session(options, args)


if __name__ == '__main__':
  main(sys.argv)
