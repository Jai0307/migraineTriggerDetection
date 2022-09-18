import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics

TRIAL = 1

filename = 'data/formatted-symptoms.csv'
symptoms = pd.DataFrame(pd.read_csv(filename, header=0))
# dropcols = ['ID','Unnamed: 0','Study number', " area","Headache intensity (VAS)", "age" ]
keepcols = ['Headache frequency (day/month)',
       'headache intensity', 'unilateral',
       'daily exercise concert ', 'throw up', ]

ids = symptoms['ID']
symptoms = symptoms[keepcols]
print(symptoms.index)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(symptoms)
symptoms = pd.DataFrame(scaled, columns=symptoms.columns)

print(symptoms.head(4))

exit


K = range(2, 10)
wss = []
silhouette_score = []

print(symptoms.columns)

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
        print ("Silhouette score for k(clusters) = "+str(k)+" is "
        +str(silhouttescore))
        silhouette_score.append(silhouttescore)

    sil_data = np.array(silhouette_score)
    print('clusters:', 2+np.argmin(-1*sil_data))

    # print(np)

    plt.xlabel('K')
    plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
    plt.plot(K,wss)
    plt.show()
###########################################################
###########################################################
    
###########################################################
###########################################################
# 'age', 'bmi', 'tight_headache', 'throbbing2', 'dull_heavy_headache', 
else:
    k = 4
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    result = km.fit_predict(symptoms)
    cols = symptoms.columns
    symptoms['ID'] = ids
    symptoms['cluster'] = km.labels_
    symptoms = symptoms.drop(cols, axis=1)
    # print(km.labels_)
    # all_cols = ['cluster'] + list(symptoms.columns)
    # results = pd.DataFrame(columns=all_cols, dtype='float')
    print(symptoms.columns)
    symptoms.to_csv('data/clustering_result.csv')

# plt.scatter(symptoms['bmi'], symptoms['age'], c=km.labels_, s=50, alpha=0.5)
# plt.show()
###########################################################
###########################################################

# results['cluster'] = km.labels_
# for col in symptoms.columns:
#     results[col] = symptoms[col]