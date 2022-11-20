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

filename = 'data/total_migraine_diary.csv'
data = pd.DataFrame(pd.read_csv(filename, header=0))
# dropcols = ['ID','Unnamed: 0','Study number', " area","Headache intensity (VAS)", "age" ]

symptoms_labels = ['migraine', 'headache',
    'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise', 'smell', 'light', 'noise', 'light_noise', 'duration_min', 'intensity', 'intensity_index'
]

# symptoms_labels = ['migraine', 'headache',
#     'left side', 'right side', 'middle', 'both sides', 'around the eyes', 'around the back of the neck', 'unilateral', 'throbbing headache1', 'a tight headache', 'throbbing headache2', 'dull heavy headache', 'Headache worsens with movement', 'gourmet', 'sensitive to sound', 'throw up', 'nausea-vomiting', 'Sensitivity to odors', 'sensitivity to light', 'light-noise sensitivity', 'Duration (minutes)',	'Headache Intensity',	'Headache Intensity Index'
# ]

symptoms = data.copy()
# symptoms = data[data['migraine']==1]

ids = symptoms['Patient']
symptoms = symptoms[symptoms_labels]
# print(symptoms.index)

symptoms.fillna(0, inplace=True)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(symptoms)
symptoms = pd.DataFrame(scaled, columns=symptoms.columns)

# print(symptoms.head(4))


K = range(2, 20)
wss = []
silhouette_score = []

# print(symptoms.columns)

findclustering = False

###########################################################
###########################################################
if findclustering:
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++')
        km=km.fit(symptoms)
        wss_iter = km.inertia_
        wss.append(wss_iter)
        labels=cluster.KMeans(n_clusters=k,random_state=200).fit(symptoms).labels_
        silhouttescore = metrics.silhouette_score(symptoms,labels,metric="euclidean",sample_size=1000,random_state=0)
        print ("WSS score for k(clusters) = "+str(k)+" is " + str(wss_iter))
        # print ("Silhouette score for k(clusters) = "+str(k)+" is " + str(silhouttescore))
        silhouette_score.append(silhouttescore)

    sil_data = np.array(silhouette_score)
    print('clusters:', 2+np.argmin(-1*sil_data))

    # print(np)

    plt.xlabel('K')
    plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
    plt.plot(K, wss, 'bx-')
    plt.show()
###########################################################
###########################################################
    
###########################################################
###########################################################
# 'age', 'bmi', 'tight_headache', 'throbbing2', 'dull_heavy_headache', 
else:
    k = 3
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    result = km.fit_predict(symptoms)
    
    result = symptoms.apply(lambda x:chi_test(x, km.labels_)).T
    result.columns = ["Chi2", "P-value", 'dfreedom']
    result = result.sort_values("P-value")

    result.to_csv(foldername + '/'+ str(TRIAL) + '_clustering-results.csv')
    
    cols = symptoms.columns
    symptoms['ID'] = ids
    symptoms['Cluster'] = km.labels_
    symptoms = symptoms.drop(cols, axis=1)

    # print(km.labels_)
    # all_cols = ['cluster'] + list(symptoms.columns)
    # results = pd.DataFrame(columns=all_cols, dtype='float')
    # print(symptoms.columns)
    symptoms.to_csv(foldername+ '/'+ str(TRIAL) + '_clusters.csv')

# plt.scatter(symptoms['bmi'], symptoms['age'], c=km.labels_, s=50, alpha=0.5)
# plt.show()
###########################################################
###########################################################

# results['cluster'] = km.labels_
# for col in symptoms.columns:
#     results[col] = symptoms[col]