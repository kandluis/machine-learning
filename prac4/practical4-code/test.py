from qlearner import QLearner
from SwingyMonkey import SwingyMonkey

# parse command line argument
# for example, parse training iterations, etc.

if __name__ == '__main__':
  
  iters = 100   # TODO -should be a commandline parameter
  learner = QLearner() # TODO - should be a commandline parameter

  # training phase for our learner
  for ii in xrange(iters):

      # Make a new monkey object.
      swing = SwingyMonkey(visual=False,           # Don't visualize 
                           sound=False,            # Don't play sounds.
                           text="Epoch %d" % (ii), # Display the epoch on screen.
                           tick_length=1,          # Make game ticks super fast.
                           action_callback=learner.action_callback,
                           reward_callback=learner.reward_callback)

      # Loop until you hit something.
      while swing.game_loop():
          pass

      print "Epoch {}".format(ii)

  
  # we're done training the learner, so play it one more time and actually visualize it
  print "Start Visualization"
  game = SwingyMonkey(action_callback=learner.action_callback,
                       reward_callback=learner.reward_callback)
  # Loop until you hit something.
  while game.game_loop():
    pass

  print "End Visualization"
