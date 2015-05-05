from qlearner import QLearner

class QLearnerBayes(QLearner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins. This class
    uses the parameters determined through Bayes Optimization using Spearming open source
    software. For more info, see /qlearner/ directory (contains output of optimization)

    Also see: https://github.com/HIPS/Spearmint
    '''
    def __init__(self, learn_fn = lambda i: 0.03595, discount_fn = lambda i: .920354,
                 bucket_height = 18., bucket_width = 50, velocity_bucket = 18):
        super(QLearnerBayes,self).__init__(learn_fn = learn_fn, discount_fn = discount_fn,
            bucket_height=bucket_height, bucket_width=bucket_width,velocity_bucket=velocity_bucket)