import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def chi_test(ds, labels):
    ct = pd.crosstab(ds, labels)
    test = chi2_contingency(ct)
    # print(np.array(test[:-1]))
    return np.array(test[:-1])

pddata_file = pd.DataFrame(pd.read_csv("patient_data.csv"))

columns = list(pddata_file.columns)
columns[0] = 'ID'
pddata_file.columns = columns
# pddata_file = pddata_file.drop(columns[0], axis=1)
# print(pddata_file.columns)

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

usecols = [
    'sex',
    # 'age',
    'pain_intensity', 
    'bmi',
    # 'headache_freq',
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
    'sensitivity_to_light',
    'light_noise_sensitivity',
]

pddata_file = pddata_file[pddata_file['sex'].notna()]
# pddata_file = pddata_file[pddata_file['age'].notna()]
# pddata_file = pddata_file[pddata_file['height'].notna()]
# pddata_file = pddata_file[pddata_file['weight'].notna()]
# pddata_file = pddata_file[pddata_file['bmi'].notna()]
# pddata_file = pddata_file[pddata_file['headache_freq'].notna()]
# print(pddata_file.loc[pddata_file['ID']=='em32'].index)
pddata_file = pddata_file.drop(pddata_file.loc[pddata_file['ID']=='em32'].index)
pddata_file = pddata_file.drop(pddata_file.loc[pddata_file['ID']=='em60'].index)
# pddata_file = pddata_file.drop(pddata_file['ID']=='em60')

pddata = pd.DataFrame(columns=usecols, dtype='float')
for c in pddata.columns:
    pddata[c] = pddata_file[c]


ids = pddata_file['ID']
pddata.fillna(0, inplace=True)

K = range(2, 15)
cost = []
# for k in K:
#     kmode = KModes(n_clusters=k, init = "random", n_init = 5, verbose=1)
#     kmode.fit_predict(pddata)
#     cost.append(kmode.cost_)

# plt.plot(K, cost, 'bx-')
# plt.xlabel('No. of clusters')
# plt.ylabel('Cost')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# pddata.index = ids



k=3

kmode = KModes(n_clusters=k, init="huang", n_init=100, verbose=0)
clusters = kmode.fit_predict(pddata)

result = pddata.apply(lambda x:chi_test(x, kmode.labels_)).T
result.columns = ["Chi2", "P-value", 'dfreedom']
result = result.sort_values("P-value")

result.to_csv('kmodes_results/stats2_k_3.csv')
pddata.insert(0, "Cluster", clusters, True)

cluster_averages = pd.DataFrame(columns=pddata.columns, dtype='float')

for id in range(0, k):
    means = pddata.loc[pddata['Cluster']==id].mean()
    cluster_averages.loc[id] = means


cluster_averages.to_csv('kmodes_results/cluster_averages2_k_3.csv')


pddata.insert(0, "ID", ids, True)
pddata.to_csv("kmodes_results/trial2_k_3.csv")

clusterid = pddata['Cluster'].loc[0]
pddata = pddata.loc[pddata['Cluster']==clusterid]
print(pddata.head(1))

for k in K:
    kmode = KModes(n_clusters=k, init = "random", n_init = 5, verbose=0)
    kmode.fit_predict(pddata)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()
