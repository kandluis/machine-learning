from tdlearner import TDLearner

class TDLearnerBayes(TDLearner):
	'''
	Implements a TD-Leaner using the paraters found after performing bayes optimization
	on our original TDLearner (from which this class ineherits)

	Minimum expected objective value under model is -127.54761 (+/- 4.53392), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                velocity_buc  int        440         
                width_size    int        442         
                discount_rat  float      0.954306    
                height_size   int        227         
                learn_rate    float      0.303168    

	Minimum of observed values is -159.750000, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                velocity_buc  int        179         
                width_size    int        469         
                discount_rat  float      0.877524    
                height_size   int        227         
                learn_rate    float      0.317109    

	'''

	def __init__(self, learn_fn = lambda i: .303168, discount_fn = lambda i: 0.954306, bucket_height = 227., bucket_width = 442, velocity_bucket = 440):

		super(TDLearnerBayes,self).__init__(learn_fn=learn_fn, discount_fn=discount_fn,bucket_height=bucket_height, bucket_width=bucket_width,velocity_bucket=velocity_bucket)