from sklearn import ensemble

def ModelRandomForest(config):
    """Random forest model
    """
    rf = ensemble.RandomForestClassifier(
        n_estimators=int(config.n_estimators), # tree number
        criterion=config.criterion, # efficiency criteria
        max_depth=int(config.max_depth), #max tree depth
        min_samples_leaf = int(config.min_samples_leaf), # minimal number of objects
        random_state=int(config.random_seed) 
    )

    return rf