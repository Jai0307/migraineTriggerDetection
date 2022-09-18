import pandas as pd
import numpy as np;
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from scipy.stats import chi2_contingency

def chi_test(ds, labels):
    ct = pd.crosstab(ds, labels)
    test = chi2_contingency(ct)
    # print(np.array(test[:-1]))
    return np.array(test[:-1])

TRIAL = 1

filename = 'data/formatted-symptoms.csv'
symptoms = pd.DataFrame(pd.read_csv(filename, header=0))
dropcols = ['ID','Unnamed: 0','Study number', " area", "age","kidney","weight","bmi","Headache frequency (day/month)","Duration (hours/times)", "Headache intensity (VAS)","daily exercise concert ","unilateral" ]

keepcols = ['Headache frequency (day/month)',
       'headache intensity', 'unilateral',
       'daily exercise concert ', 'throw up', ]

ids = symptoms['ID']
symptoms = symptoms[keepcols]
# symptoms = symptoms.drop(dropcols, axis=1)
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
    K = range(2, 15)
    cost = []
    for k in K:
        kmode = KModes(n_clusters=k, init = "random", n_init = 10, verbose=1)
        kmode.fit_predict(symptoms)
        cost.append(kmode.cost_)

    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()
       
###########################################################
###########################################################
    
###########################################################
###########################################################
# 'age', 'bmi', 'tight_headache', 'throbbing2', 'dull_heavy_headache', 
else:
    k = 4
    
    kmode = KModes(n_clusters=k, init="huang", n_init=100, verbose=0)
    clusters = kmode.fit_predict(symptoms)

    result = symptoms.apply(lambda x:chi_test(x, kmode.labels_)).T
    result.columns = ["Chi2", "P-value", 'dfreedom']
    result = result.sort_values("P-value")

    result.to_csv('data/results/kmode/clustering-results.csv')
    cols = symptoms.columns
    symptoms.insert(0, "Cluster", clusters, True)
    symptoms.insert(0, "ID", ids, True)
    symptoms = symptoms.drop(cols, axis=1)
    symptoms.to_csv('data/results/kmode/clusters.csv')
###########################################################
###########################################################

# results['cluster'] = km.labels_
# for col in symptoms.columns:
#     results[col] = symptoms[col]