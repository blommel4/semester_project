'''
Imports for this program are here. Need installs of:

xgboost, pandas, numpy, matplotlib, graphviz, sklearn
'''


import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline # only for notebooks!!!
import graphviz
import numpy as np
from sklearn import preprocessing


'''
Need this data from KenPom website that we have already downloaded
in our grithub repository. Import the data and split it into testing
and training data based on year. The goal of this split is to use
past data (years before 2016) to predict 'future' data (years after 2016).
'''

full_results = pd.read_csv('full_results.csv')
full_stats = pd.read_csv('FullKenPom_pt.csv')

###mask test and train data###
res_test_mask = (full_results['Year'] == 2016) | (full_results['Year'] == 2017) | (full_results['Year'] == 2018)
stats_test_mask = (full_stats['Season'] == 2016) | (full_stats['Season'] == 2017) | (full_stats['Season'] == 2018)
res_train_mask = (full_results['Year'] < 2016)
stats_train_mask = (full_stats['Season'] < 2016)

res_data_test = full_results[res_test_mask]
stats_data_test = full_stats[stats_test_mask]
res_data_train = full_results[res_train_mask]
stats_data_train = full_stats[stats_train_mask]

#reindex all of them
res_data_test = res_data_test.reset_index()
stats_data_test = stats_data_test.reset_index()
res_data_train = res_data_train.reset_index()
stats_data_train = stats_data_train.reset_index()

for i in range(len(res_data_test)):
    if res_data_test['Region Name'][i] == "First Four":
        res_data_test = res_data_test.drop(i)
for i in range(len(res_data_train)):
    if res_data_train['Region Name'][i] == "First Four":
        res_data_train = res_data_train.drop(i)
        
#reindex all of them
res_data_test = res_data_test.reset_index()
stats_data_test = stats_data_test.reset_index()
res_data_train = res_data_train.reset_index()
stats_data_train = stats_data_train.reset_index()


#these names match the kenpom stats csv
stats_vec = ["AdjTempo",
            "AdjOE",
            "AdjDE",
            "AdjEM",
            "seed",
            "ConfTournament",
            "SOSAdjEM",
            "NCSOSAdjEM",
            "O-D_eFG_Pct",
            "D-O_TO_Pct",
            "O-D_OR_Pct",
            "O-D_FT_Rate",
            "LastTenRecord"]

# Move training data into 2 numpy arrays - data and labels (results)
N = len(res_data_train)

training_data = np.zeros((N,13))
training_labels = np.zeros((N,1))

for i in range(len(res_data_train)):
    year = res_data_train['Year'][i]
    teamA = res_data_train['TeamA'][i]
    teamB = res_data_train['TeamB'][i]
    score_diff = res_data_train['ScoreA'][i] - res_data_train['ScoreB'][i]
    for k in range(len(stats_data_train)):
        if ((stats_data_train['Season'][k] == year) and (stats_data_train['TeamName'][k] == teamA)):
            indexA = k
            break
            
    for k in range(len(stats_data_train)):
        if ((stats_data_train['Season'][k] == year) and (stats_data_train['TeamName'][k] == teamB)):
            indexB = k
            break
    for s in range(len(stats_vec)):
        stat = stats_vec[s]
        training_data[i][s] = stats_data_train[stat][indexA] - stats_data_train[stat][indexB]
    
    if (score_diff > 0):
        training_labels[i][0] = 1
    else:
        training_labels[i][0] = 0

# Move testing data into 2 numpy arrays - data and labels (results)
N = len(res_data_test)

testing_data = np.zeros((N,13))
testing_labels = np.zeros((N,1))

for i in range(len(res_data_test)):
#for i in range(5):
    year = res_data_test['Year'][i]
    teamA = res_data_test['TeamA'][i]
    teamB = res_data_test['TeamB'][i]
    score_diff = res_data_test['ScoreA'][i] - res_data_test['ScoreB'][i]
    for k in range(len(stats_data_test)):
        if ((stats_data_test['Season'][k] == year) and (stats_data_test['TeamName'][k] == teamA)):
            indexA = k
            break
            
    for k in range(len(stats_data_test)):
        if ((stats_data_test['Season'][k] == year) and (stats_data_test['TeamName'][k] == teamB)):
            indexB = k
            break
    for s in range(len(stats_vec)):
        stat = stats_vec[s]
        testing_data[i][s] = stats_data_test[stat][indexA] - stats_data_test[stat][indexB]
    
    if (score_diff > 0):
        testing_labels[i][0] = 1
    else:
        testing_labels[i][0] = 0



'''
This next section of code is for normalizing the data (if you want
since xgboost does not need the data to be normalized)
'''


#Time to normalize the data
training_data = preprocessing.normalize(training_data, axis=0, norm='max')
testing_data = preprocessing.normalize(testing_data, axis=0, norm='max')

evallist = [(testing_data, 'eval'), (training_data, 'train')]

param = {'objective': 'multi:softprob'}
param['eval_metric'] = "merror"
param['num_class'] = 2

dtrain = xgb.DMatrix(training_data, label=training_labels,
                     feature_names=stats_vec)
dtest = xgb.DMatrix(testing_data, label=testing_labels,
                    feature_names=stats_vec)


'''
The next three parts (labeled 'Phases') are for tuning the model
for creating the best possible paramters for the final model.
'''

#Phase 1: Tuning Max depth and min_child_weight
param = {'objective': 'multi:softprob'}
param['eval_metric'] = "merror"
param['num_class'] = 2  # 2 classes - win or loss

#evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 999 #looks like it levels off at around 200

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(1,8)
    for min_child_weight in range(1,6)
]
min_merror = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    # print("CV with max_depth={}, min_child_weight={}".format(
    #                         max_depth,
    #                         min_child_weight))
    
    # Update Parameters
    param['max_depth'] = max_depth
    param['min_child_weight'] = min_child_weight
    
    #Run CV
    cv_results = xgb.cv(param,
                        dtrain,
                        num_boost_round=num_round, #maybe wrong
                        seed=42,
                        nfold=3,
                        metrics={'merror'},
                        early_stopping_rounds=10)
    
    #Update best MError
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].idxmin()
    # boost_rounds = cv_results['test-merror-mean'].argmin()
    # print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (max_depth, min_child_weight)
    

param['max_depth'] = best_params[0]
param['min_child_weight'] = best_params[1]

#Phase 2: Subsample and Colsample_bytree
#tune subsample,colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(1,11)]
    for colsample in [i/10. for i in range(1,11)]
]
min_merror = float("Inf")
best_params = None
for subsample, colsample in reversed(gridsearch_params):
    # print("CV with subsample={}, colsample={}".format(
    #                         subsample,
    #                         colsample))
    # Update our parameters
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )
    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].idxmin()
    # boost_rounds = cv_results['test-merror-mean'].argmin()
    # print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (subsample,colsample)
        
        
param['subsample'] = best_params[0]
param['colsample_bytree'] = best_params[1]

#Phase 3: eta
min_merror = float("Inf")
best_params = None
for eta in [0.5,0.3, 0.03, .003,0.0003]:
    # print("CV with eta={}".format(eta))
    # Update our parameters
    param['eta'] = eta
    # Run CV
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )
    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].idxmin()
    # boost_rounds = cv_results['test-merror-mean'].argmin()
    # print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = eta
        
param['eta'] = best_params


'''
This section is for the final model for predicting results. Here we test the
model (trained against the data between 2006-2015) with the known results of
the 2016-2018 seasons. The plot showing the best parameters is commented out
for your convience, as well as the results of the correct values.
'''

final_gb = xgb.train(param,dtrain,num_boost_round=num_round,
                   early_stopping_rounds=5,evals=[(dtest, "Test")])

# xgb.plot_importance(final_gb)

ypred = final_gb.predict(dtest)
ypred

correct_list = np.zeros_like(testing_labels)
for i in range(len(ypred)):
    if ypred[i][0] < ypred[i][1]:
        metric = 1 # team A predicted to win
    else:
        metric = 0
    correct_list[i][0] = metric
    # print("TeamA Win% {:.4}  SeedDiff {}".format(ypred[i][1]*100,testing_data[i][4]*-1))

print("% Correct: ", (correct_list.sum() / len(correct_list))*100)

# xgb.plot_tree(final_gb)
# fig = plt.gcf()
# fig.set_size_inches(18,25)
# plt.show()




