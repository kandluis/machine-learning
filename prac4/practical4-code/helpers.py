# to get time
from datetime import datetime
import csv

import os, errno

from pprint import pprint

# to save results
try:
   import cPickle as pickle
except:
   import pickle

def write_csv(history, outfile):
  with open(outfile,"wb") as f:
    csvwriter = csv.writer(f)
    # additionally, we save the results as a csv file so we can easily load it
    xs = range(history.num_rounds())
    ys = [history.epoch(t).scores for t in xs]

    csvwriter.writerows(zip(xs,ys))


def save_results(learner_class_names,learner_classes, learner_histories, learner_scores, out):
    # save output of each learner
    time = datetime.strftime(datetime.now(), '%Y-%m-%d_%H.%M.%S')
    path = os.path.abspath(os.path.join(os.getcwd(),time))

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    prefix = out + "_%s"
    try:
      for learner in learner_class_names:
        outfile = (prefix % learner) + ".p"
        hist_outfile = (prefix % learner) + "_h.p"
        csv_outfile = (prefix % learner) + ".csv"
        outpath = os.path.join(path,outfile)
        hist_outpath = os.path.join(path,hist_outfile)
        csv_outpath = os.path.join(path, csv_outfile)
        with open(outpath,"wb") as f:
          pickle.dump(learner_classes[learner].pickle(), f)
        
        write_csv(learner_histories[learner], csv_outpath)
        
        with open(hist_outpath, 'wt') as out:
          pprint(learner_scores[learner], stream=out)

      return True

    except:
      print "Unable to store results from learning run."
      raise
      return False
