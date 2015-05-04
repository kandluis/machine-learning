def save_results(learner_class_names,learner_classes, learner_histories, out):
    # save output of each learner
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S')
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
        outpath = os.path.join(path,outfile)
        hist_outpath = os.path.join(path,hist_outfile)
        with (open(outpath), open(hist_outpath)) as (f,hf)
          pickle.dump(taught_learners[learner], f)
          pickle.dump(learner_histories[learner], hf)
      return True

    except:
      print "Unable to store results from learning run."
      return False
