import sys

'''
The below is only needed for Bayesian Optimization on Luis' Machine.
Ignore and comment out if it causes problems.
'''
path = "/home/luis/Documents/Harvard_School_Work/Spring_2015/cs181/assignments/practicals/prac4/practical4-code/"
if path not in sys.path:
    sys.path.append(path)
'''
End Bayesian
'''

from SwingyMonkey import SwingyMonkey
from qlearner import QLearner

import numpy as np

TRAIN_ITERS = 3500
TEST_ITERS = 500

def branin(discount, learning_rate, buckets_w, buckets_h, buckets_v):
    def run_game():
        # Make a new monkey object.
        swing = SwingyMonkey(visual=False,      # no video
                             sound=False,       # no audio        
                             action_callback=learner_class.action_callback,
                             reward_callback=learner_class.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
          pass

        return swing

    # make a new learner with the given parameters
    learner_class = QLearner(learn_fn = lambda i: learning_rate,
                             discount_fn = lambda i: discount,
                             bucket_height = buckets_h,
                             bucket_width = buckets_w,
                             velocity_bucket = buckets_v)

    # train the learner
    for t in xrange(TRAIN_ITERS):
        run_game()

    # keep learning, take average over the iterations
    scores = []
    for t in xrange(TEST_ITERS):
         # Make a new monkey object.
        swing = run_game()

        scores.append(swing.score)

    avg_score = float(sum(scores))/ float(TEST_ITERS)
    median_score = np.median(scores)

    # which do we return?
    print "The median is %d and the mean is %f." % (median_score, avg_score)

    # out objective is to minimize the negative of the average score
    return -1*avg_score

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return branin(params['discount'],
                  params['learning_rate'],
                  params['width_size'],
                  params['height_size'],
                  params['velocity_bucket'])
