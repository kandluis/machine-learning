from sklearn.ensemble import RandomForestClassifier

def random_forest_model(ffs = None, featurefile="forest_features", outputfile = "forest_predictions.csv", train_dir = "train", test_dir = "test"):
    # do a quick load of feature data 
    X_train, global_feat_dict, t_train, train_ids = save_and_load(featurefile, train_dir, ffs)

    pdFrame = toPandasDataFrame(X_train, global_feat_dict, t_train)
    
    # generate random forest model
    model = RandomForestClassifier(n_estimators = 5, compute_importances = True)

    # build a forest of trees from training set (X,y) where X = feature set, Y = target values
    y = pdFrame['class']
    pdFrame.drop('class', axis=1, inplace=True)
    model.fit(pdFrame, y)

    # extract features from test data
    print "extracting test features..."
    X_test,_, t_ignore, test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print

    testData = toPandasDataFrame(X_test, global_feat_dict)

    # make predictions on test data and write them out
    print "making predictions..."
    preds = model.predict(testData)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

forestffs = [system_call_count_feats, first_last_system_call_feats, tag_counts_feats, bag_of_words_feat]

random_forest_model(ffs = forestffs)