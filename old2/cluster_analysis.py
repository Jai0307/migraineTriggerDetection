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

TRIAL = 1
foldername = 'data/results/kmean_migraine_data/'
filename = 'data/total_migraine_diary.csv'
data = pd.DataFrame(pd.read_csv(filename, header=0))
# dropcols = ['ID','Unnamed: 0','Study number', " area","Headache intensity (VAS)", "age" ]

columns = ['migraine', 'headache',
    'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise', 'smell', 'light', 'noise', 'light_noise', 'duration_min', 'intensity', 'intensity_index', 'treatment'
]

symptoms = data[columns]

cols = list(columns)
cols.append('count')

cluster_stats = pd.DataFrame(columns=(cols), dtype='float')
# cluster_stats.insert(0, column='count', value=0)
# print(cluster_stats.columns)

clusters_filename = foldername + str(TRIAL) + '_clusters.csv'
clusters = pd.DataFrame(pd.read_csv(clusters_filename, header=0))
clusters_ids = np.array(clusters['Cluster'].unique())
clusters_ids.sort(axis=0)
for cluster in clusters_ids:
    idx = clusters[clusters['Cluster']==cluster].index
    cluster_symptom = symptoms.loc[idx]
    row = cluster_symptom.sum(axis=0)
    row['count'] = idx.shape[0]
    cluster_stats.loc[cluster] = row

cluster_stats.to_csv(foldername + str(TRIAL) + '_cluster_symptoms.csv')