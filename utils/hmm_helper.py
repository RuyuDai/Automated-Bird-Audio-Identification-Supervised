from hmmlearn import hmm
import copy
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import warnings
import glob
import numpy as np
import pandas as pd
import random
warnings.filterwarnings("ignore", category=DeprecationWarning)


def GHMM_train(dataset, states = 4, trials = 10,cov = "diag",max_iter=1000, randomSeed = 100, min_cov = 1e-3):
    rndNum = 10
    prob = -np.inf
    best_g = None
    score = 0
    np.random.seed(randomSeed)
    random_state = np.random.random(trials)
    for i in range(len(random_state)):
        state = int(random_state[i]*1000)
        g = hmm.GaussianHMM(n_components=states, covariance_type = cov, n_iter=max_iter, min_covar = min_cov)
        g.fit(dataset)
        score = g.score(dataset)
        if score > prob:
            prob = score
            best_g = copy.deepcopy(g)
    return best_g

def GHMM_evaluate(testset, GMMmodel):
    #test set is a list of sets of name-date collections
    #GMMmodel is a dictionary of test data
    truth = []
    pred = []
    for dataset in testset:
        truth.append(dataset[0])
        test_score = -np.inf
        pred_name = None
        for modelName, model in GMMmodel.items():
            score = model.score(dataset[1])
            if score > test_score:
                test_score = score
                pred_name = modelName
        pred.append(pred_name)
    label = []
    for item in truth:
        if item not in label:
            label.append(item)
    matrix = confusion_matrix(truth, pred, labels=label)
    print ("truth: ",truth)
    print ("prediction: ", pred)
    print ("item order in table", label)
    return matrix,truth,pred

def get_HMM(feature, states, cov="diag", clean_sparse=True, train_percent=0.8):
    if clean_sparse:
        # Remove sparse segments per species
        no_sparse_seg = feature[['segment', 'species']].drop_duplicates().groupby('species').size()
        no_sparse_seg = pd.DataFrame(no_sparse_seg).reset_index().rename(columns={0: 'Count'})
        no_sparse_seg = no_sparse_seg[no_sparse_seg['Count'] > 50]['segment'].to_list()
        feature = feature[feature['segment'].isin(no_sparse_seg)]

        # Remove sparse sample size per segment
        no_sparse_seg = feature.groupby(['species', 'segment']).size()
        no_sparse_seg = pd.DataFrame(no_sparse_seg).reset_index().rename(columns={0: 'Count'})
        no_sparse_seg = no_sparse_seg[no_sparse_seg['Count'] > 59]['segment'].to_list()
        feature = feature[feature['segment'].isin(no_sparse_seg)]

    test_feature = feature.drop(columns=['low_freq', 'high_freq'])
    segment_list = test_feature['segment'].drop_duplicates().to_list()

    train_list = []
    train_segment_list = random.sample(segment_list, int(train_percent * len(segment_list)))
    selected_feature = test_feature[test_feature['segment'].isin(train_segment_list)]

    for seg in train_segment_list:
        selected_segment = selected_feature[selected_feature['segment'] == seg]
        species = str(selected_segment['species'].drop_duplicates().values)
        data = selected_segment.drop(columns=['species', 'segment'])
        train_list.append((species, data))

    modelDict = {}
    for dataset in train_list:
        model = GHMM_train(dataset=dataset[1], states=states, trials=1, cov=cov, max_iter=100, randomSeed=120,
                           min_cov=1e-5)
        modelDict[dataset[0]] = model

    test_list = []
    selected_feature = test_feature[~test_feature['segment'].isin(train_segment_list)]
    test_segment_list = selected_feature['segment'].drop_duplicates().to_list()

    for seg in test_segment_list:
        selected_segment = selected_feature[selected_feature['segment'] == seg]
        species = str(selected_segment['species'].drop_duplicates().values)
        data = selected_segment.drop(columns=['species', 'segment'])
        test_list.append((species, data))

    return GHMM_evaluate(test_list, modelDict)


"""
def get_HMM(feature, states,cov = "diag", clean_sparse=True, train_percent=0.8,):
    if clean_sparse:
        # sparse segment per species
        no_sparse_seg = feature[['segment', 'species']].drop_duplicates().groupby('species').size()
        no_sparse_seg = pd.DataFrame(no_sparse_seg).reset_index().rename(columns={0: 'Count'})
        no_sparse_seg = no_sparse_seg[no_sparse_seg['Count'] > 50]['segment'].to_list()
        feature = feature[feature['segment'].isin(no_sparse_seg)]

        # sparse sample size per segment
        no_sparse_seg = feature.groupby(['species', 'segment']).size()
        no_sparse_seg = pd.DataFrame(no_sparse_seg).reset_index().rename(columns={0: 'Count'})
        no_sparse_seg = no_sparse_seg[no_sparse_seg['Count']>59]['segment'].to_list()
        feature = feature[feature['segment'].isin(no_sparse_seg)]

    test_feature = feature.drop(columns=['low_freq', 'high_freq'])
    segment_list = test_feature['segment'].drop_duplicates().to_list()

    train_list = []
    train_percent = train_percent
    train_segment_list = random.sample(segment_list, int(train_percent * len(segment_list)))
    selected_feature = test_feature[test_feature['segment'].isin(train_segment_list)]
    for seg in train_segment_list:
        selected_segment = selected_feature[selected_feature['segment'] == seg]
        train_list.append((str(selected_segment['species'].drop_duplicates().values),
                           selected_segment.drop(columns=['species', 'segment'])))

    modelDict = dict()
    for dataset in train_list:
        model = GHMM_train(dataset=dataset[1], states=states, trials=1, cov=cov, max_iter=100, randomSeed=120,
                           min_cov=1e-5)
        modelDict[dataset[0]] = model

    test_list = []
    selected_feature = test_feature[~test_feature['segment'].isin(train_segment_list)]
    test_segment_list = selected_feature['segment'].drop_duplicates().to_list()
    for seg in test_segment_list:
        selected_segment = selected_feature[selected_feature['segment'] == seg]
        test_list.append((str(selected_segment['species'].drop_duplicates().values),
                          selected_segment.drop(columns=['species', 'segment'])))

    return GHMM_evaluate(test_list,modelDict)
"""
