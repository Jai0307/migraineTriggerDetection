import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics

TRIAL = 1

filename = 'patient_data.csv'
pddata_file = pd.DataFrame(pd.read_csv(filename, header=0))
patient_info_cols = ['sex', 'age', 'height', 'weight', 'bmi', 'headache_freq']
migraine_identifiers = [
    'duration_min',
    'pain_intensity',
    'left_side',
    'right_side',
    'middle',
    'both_sides',
    'around_the_eyes',
    'back_of_the_neck',
    'unilateral',
    'throbbing1',
    'tight_headache',
    'throbbing2',
    'dull_heavy_headache',
    'worse_with_movement',
    'gourmet',
    'sensitive_to_sound',
    'throw_up',
    'nausea_vomiting',
    'Sensitivity_to_odors',
    'sensitivity_to_light',
    'light_noise_sensitivity',
    'Missed_school_work',
    'Work_decreased_less_than_half',
    'no_housework',
    'Housework_reduced_to_less_than_half',
    'incapacitated_degree',
]
migraine_triggers = [
    'stress',
    'oversleeping',
    'lack_of_sleep',
    'exercise',
    'no_exercise',
    'fatigue',
    'menstrual_cycle',
    'ovulation',
    'emotion',
    'weather_temp_change',
    'excessive_sunlight',
    'noise',
    'improper_lighting',
    'specific_smell',
    'drinking',
    'irregular_meals',
    'overeating',
    'caffeine',
    'smoking',
    'cheese_chocolate',
    'travel',
    'other_triggers1',
    'other_triggers2',
]
helpful_actions = [
    'help_sleep',
    'help_rest',
    'help_massage_or_stretching',
    'help_exercise',
    'help_other',
]
# 'age',  'tight_headache', 'throbbing2', 'dull_heavy_headache',  'unilateral', 'duration_min', 'pain_intensity', 'headache_freq','bmi',

usecols = [
    'sex',
    'age',
    # 'pain_intensity', 
    'bmi',
    'left_side',
    'right_side',
    'middle',
    'both_sides',
    'around_the_eyes',
    'back_of_the_neck',
    'unilateral',
    'throbbing1',
    'tight_headache',
    'throbbing2',
    'dull_heavy_headache',
    'worse_with_movement',
    # 'gourmet',
    'sensitive_to_sound',
    'throw_up',
    'nausea_vomiting',
    'Sensitivity_to_odors',
    'sensitivity_to_light',
    'light_noise_sensitivity',
]

print(pddata_file.shape)

pddata_file = pddata_file[pddata_file['sex'].notna()]
pddata_file = pddata_file[pddata_file['age'].notna()]
pddata_file = pddata_file[pddata_file['height'].notna()]
pddata_file = pddata_file[pddata_file['weight'].notna()]
pddata_file = pddata_file[pddata_file['bmi'].notna()]
pddata_file = pddata_file[pddata_file['headache_freq'].notna()]
print(pddata_file.shape)

pddata = pd.DataFrame(columns=usecols, dtype='float')
for c in pddata.columns:
    pddata[c] = pddata_file[c]


ids = pddata_file['ID']
pddata.fillna(0, inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(pddata)
pddata = pd.DataFrame(scaled, columns=pddata.columns)

K = range(2, 20)
wss = []
silhouette_score = []

print(pddata.columns)

###########################################################
###########################################################
for k in K:
    km = KMeans(n_clusters=k, init='k-means++')
    km=km.fit(pddata)
    wss_iter = km.inertia_
    wss.append(wss_iter)
    labels=cluster.KMeans(n_clusters=k,random_state=200).fit(pddata).labels_
    silhouttescore = metrics.silhouette_score(pddata,labels,metric="euclidean",sample_size=1000,random_state=0)
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
# result = km.fit_predict(pddata)
# pddata['cluster'] = km.labels_
# # print(km.labels_)
# all_cols = ['cluster'] + list(pddata.columns)
# results = pd.DataFrame(columns=all_cols, dtype='float')
# print(pddata.columns)
# pddata.to_csv('kmean2_results/clustering_result.csv')

# plt.scatter(pddata['bmi'], pddata['age'], c=km.labels_, s=50, alpha=0.5)
# plt.show()
###########################################################
###########################################################

# results['cluster'] = km.labels_
# for col in pddata.columns:
#     results[col] = pddata[col]