from qlearner import QLearner

class QLearnerBayes(QLearner):
    '''
    Implements a Q-Learning algorithm with discretized pixel bins. This class
    uses the parameters determined through Bayes Optimization using Spearming open source
    software. For more info, see /qlearner/ directory (contains output of optimization)

    Also see: https://github.com/HIPS/Spearmint

    Minimum expected objective value under model is -87.72753 (+/- 5.99425), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                discount      float      1.000000    
                learning_rat  float      0.368372    
                height_size   int        16          
                width_size    int        85          
                velocity_buc  int        1000        

    Minimum of observed values is -157.980000, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                discount      float      1.000000    
                learning_rat  float      0.098125    
                height_size   int        17          
                width_size    int        83          
                velocity_buc  int        200     
    '''
    def __init__(self, learn_fn = lambda i: 0.368372, discount_fn = lambda i: 1.0,
                 bucket_height = 16., bucket_width = 85, velocity_bucket = 1000):
        super(QLearnerBayes,self).__init__(learn_fn = learn_fn, discount_fn = discount_fn,
            bucket_height=bucket_height, bucket_width=bucket_width,velocity_bucket=velocity_bucket)