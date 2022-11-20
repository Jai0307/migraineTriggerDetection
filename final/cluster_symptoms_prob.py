import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from scipy.stats import chi2_contingency

foldername = 'data/results/kmean_migraine_data'

def chi_test(ds, labels):
    ct = pd.crosstab(ds, labels)
    test = chi2_contingency(ct)
    # print(np.array(test[:-1]))
    return np.array(test[:-1])

TRIAL = 3
foldername = 'data/results/kmean_migraine_data/'
filename = 'data/total_migraine_diary.csv'
data = pd.DataFrame(pd.read_csv(filename, header=0))
data.fillna(0, inplace=True)
# dropcols = ['ID','Unnamed: 0','Study number', " area","Headache intensity (VAS)", "age" ]

all_columns = [
    'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'smell', 'light', 'noise', 'light_noise', 
]
#'migraine', 'headache', 'duration_min', 'intensity', 'intensity_index', 'treatment'
relavent_symptoms = [ 'middle', 'both_sides', 'around_eyes', 'back_of_neck', ]

exclude_symptoms = [ 'light', 'noise', 'light_noise', 'nausea_vomit', 'throw_up', 'unilateral']
maincluster = 4

symptoms = data[all_columns]

cols = list(relavent_symptoms + exclude_symptoms)
# cols.append('prob')

cluster_stats = pd.DataFrame(columns=(cols), dtype='float')
# cluster_stats.insert(0, column='count', value=0)
# print(cluster_stats.columns)

clusters_filename = foldername + str(TRIAL) + '_clusters.csv'
# clusters_filename = foldername + str(TRIAL) + '_clusters.csv'
clusters = pd.DataFrame(pd.read_csv(clusters_filename, header=0))
clusters_ids = np.array(clusters['Cluster'].unique())
clusters_ids.sort(axis=0)
for cluster in clusters_ids:
    # print('cluster: ', cluster)
    idx = clusters[clusters['Cluster']==cluster].index
    cluster_symptom = symptoms.loc[idx]
    for symptom in relavent_symptoms:
        if(cluster_symptom.shape[0]>0):
            try:
                cluster_symptom = cluster_symptom[cluster_symptom[symptom]==1]
            except:
                print('----------except------------------')
                print(cluster_symptom)
    
    for symptom in exclude_symptoms:
        if(cluster_symptom.shape[0]>0):
            try:
                cluster_symptom = cluster_symptom[cluster_symptom[symptom]==0]
            except:
                print('----------except------------------')
                print(cluster_symptom)
    
    row = cluster_symptom.sum(axis=0)
    # row['count'] = idx.shape[0]
    cluster_stats.loc[cluster] = row

probs = cluster_stats.div(cluster_stats.sum(axis=0))*100
name = '_'.join(relavent_symptoms)
name = name + '_exclude_' + '_'.join(exclude_symptoms) + '.csv'
probs.to_csv(foldername + 'symptom_probs/trial_' + str(TRIAL) + '_cluster_' + str(maincluster) + '_.csv')