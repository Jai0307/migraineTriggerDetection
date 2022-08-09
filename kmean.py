import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics

TRIAL = 1

filename = 'patientavgs.csv'
pddata = pd.DataFrame(pd.read_csv(filename, header=0))
dropcols = ['ID', 'height', 
       'oversleeping', 'lack_of_sleep', 'fatigue',
       'menstrual_cycle', 'weather_temp_change',
       'noise', 'specific_smell',
       'irregular_meals', 
       'other_triggers1', 'other_triggers2',
       'help_sleep', 'help_rest', 'help_massage_or_stretching',
       'help_exercise', 'help_other',  'gourmet', 'incapacitated_degree',  'throbbing1', 'no_housework', 
        'left_side', 'right_side', 'middle', 'both_sides','sensitivity_to_light', 'light_noise_sensitivity',  'sensitive_to_sound', 'worse_with_movement',   'back_of_the_neck', 'throw_up', 'nausea_vomiting','stress','around_the_eyes','sex',  'weight',]
# 'age',  'tight_headache', 'throbbing2', 'dull_heavy_headache',  'unilateral', 'duration_min', 'pain_intensity', 'headache_freq','bmi',

ids = pddata['ID']
pddata = pddata.drop(dropcols, axis=1)
# pddata = pddata.drop(pddata[pddata['sex']==0].index)
# pddata = pddata.drop(['sex', 'age'], axis=1)
pddata.fillna(0, inplace=True)
print(pddata.index)

# print(pddata.sum(['left_side', 'right_side', 'middle',
    #    'both_sides', 'around_the_eyes', 'back_of_the_neck', 'unilateral',
    #    'tight_headache', 'throbbing2', 'dull_heavy_headache'], axis=1))

# print(pddata.columns)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(pddata)
pddata_scaled = pd.DataFrame(scaled, columns=pddata.columns)
# print(pddata_scaled.sum(axis=0))

K = range(2, 50)
wss = []
silhouette_score = []

print(pddata.columns)

###########################################################
###########################################################
for k in K:
    km = KMeans(n_clusters=k, init='k-means++')
    km=km.fit(pddata_scaled)
    wss_iter = km.inertia_
    wss.append(wss_iter)
    labels=cluster.KMeans(n_clusters=k,random_state=200).fit(pddata_scaled).labels_
    silhouttescore = metrics.silhouette_score(pddata_scaled,labels,metric="euclidean",sample_size=1000,random_state=0)
    print ("Silhouette score for k(clusters) = "+str(k)+" is "
    +str(silhouttescore))
    silhouette_score.append(silhouttescore)

sil_data = np.array(silhouette_score)
print('clusters:', 2+np.argmin(-1*sil_data))

# print(np)

# plt.xlabel('K')
# plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
# plt.plot(K,wss)
# plt.show()
###########################################################
###########################################################
    
###########################################################
###########################################################
# 'age', 'bmi', 'tight_headache', 'throbbing2', 'dull_heavy_headache', 

# k = 5
# km = KMeans(n_clusters=k, init='k-means++')
# result = km.fit_predict(pddata_scaled)
# pddata['cluster'] = km.labels_
# # print(km.labels_)
# all_cols = ['cluster'] + list(pddata.columns)
# results = pd.DataFrame(columns=all_cols, dtype='float')
# print(pddata.columns)
# pddata.to_csv('kmean_results/clustering_result.csv')

# plt.scatter(pddata['bmi'], pddata['age'], c=km.labels_, s=50, alpha=0.5)
# plt.show()
###########################################################
###########################################################

# results['cluster'] = km.labels_
# for col in pddata.columns:
#     results[col] = pddata[col]