from SwingyMonkey import SwingyMonkey

# used for commandline argument parsing
from optparse import OptionParser

# to get commandline arguments
import sys
import copy

# store history on epochs
from history import History

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
  learner = init_learner(learner, options.learner_classes)

  # history dictionaries: epoch # -> whatever
  rewards = {}
  scores = {}
  data = {}

  history = History(rewards, scores, data)

  print "Starting training phase..."
  max_score = 0
  for t in xrange(options.train_iters):
    prev_score = scores[t-1] if t > 0 else 0
    #print("======= Epoch %d / %d. Max score: %d. Previous epoch score: %d" % (t,options.train_iters,max_score, prev_score))
    print t, prev_score
    # Make a new monkey object.
    swing = SwingyMonkey(visual=False,sound=options.video,
                         tick_length=0,#options.live_train,
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    episode_rewards = []
    while swing.game_loop():
      if learner.last_reward is not None:
        episode_rewards.append(learner.last_reward)

    # collect statistics
    rewards[t] = copy.deepcopy(episode_rewards)
    scores[t] = swing.score
    data[t] = {'Qmatrix' : copy.deepcopy(learner.Q)}

    max_score = max(max_score, scores[t])

  return history, learner

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

    options.video = options.live_train > 1

    learner_histories = {}
    taught_learners = {}
    # train each class and store results
    for learner, learner_class in options.learner_classes.iteritems():
      hist, learned = session(learner,options)
      learner_histories[learner] = hist
      taught_learners[learner] = learned

    # TODO : Here, we have access to each learner's training history as we
    # as the trained learner. Should do stuff with it.


def parse_inputs(args):
    usage_msg = "Usage:  %run [options] LearnerClass1 LearnerClass2 ..."
    parser = OptionParser(usage=usage_msg)

    def usage(msg):
        print "Error: %s\n" % msg
        parser.print_help()
        sys.exit()

    parser.add_option("--train_iters",
                      dest="train_iters", default=128, type="int",
                      help="Set number of training epochs")

    parser.add_option("--live_train",
                      dest="live_train", default=1, type="int",
                      help="Clock tick for training. Not displayed if 1.")

    parser.add_option("--plots",
                      dest="plots", default="true", type="string",
                      help="Boolean specifying whether to generate plots or not.")

    return parser

def main(args):
  # parse input arguments into object
  parser = parse_inputs(args)

  # extract options and arguments from parse object
  (options, args) = parser.parse_args()

  run_session(options, args)


if __name__ == '__main__':
  main(sys.argv)
