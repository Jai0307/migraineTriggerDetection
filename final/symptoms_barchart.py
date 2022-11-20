import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.DataFrame(pd.read_csv('data/total_migraine_diary.csv', header=0))
data.fillna(0, inplace=True)
# symptoms_labels = [
#     'left_side', 'right_side', 'middle', 'both_sides', 'around_the_eyes', 'around_the back of the neck', 'unilateral', 'throbbing headache1', 'a tight headache', 'throbbing headache2', 'dull heavy headache', 'Headache worsens with movement', 'gourmet', 'sensitive to sound', 'throw up', 'nausea-vomiting', 'Sensitivity to odors', 'sensitivity to light', 'light-noise sensitivity'
# ]

symptoms_labels = ['left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'tight_head', 'throbbing_2', 'dull_heavy', 'worsen_move', 'food', 'noise_sensitivity', 'throw_up', 'nausea_vomit', 'smell', 'light', 'light_noise']

symptoms_labels_desc = ['headache on left side', 'headache on right side', 'headache in middle', 'headache on both sides', 'headache on around eyes', 'ache on back of neck', 'unilateral headache', 'throbbing 1 headache', 'tight headache', 'throbbing 2 headache', 'dull heavy headache', 'headache worse with movemenet', 'food sensitivity', 'noise sensitiviy', 'throw up', 'nausea and vomiting', 'sensitivity to smell', 'sensitivity to light', 'light and noise sensitivity']

migraine = data['migraine_headache']
print(migraine.sum(axis=0))
print(migraine.shape)
condition = np.array(['Migraine', 'Headache'])
index = np.array(['Migraine', 'Headache', 'p-value'])
symptoms_table = pd.DataFrame(dtype=float, index=index)
idx = 0
for symptom in symptoms_labels:
    symptoms = data[symptom]
    table = pd.crosstab(symptoms, migraine)
    pivtable = table.drop(0, axis=1)
    table = table.div(table.sum(1).astype(float), axis=0)
    table = table.drop(0, axis=1)
    csq_test = stats.chi2_contingency(pivtable)
    # print("pvalue: ", csq_test[1])

    # print(table)
    # plt.bar(condition, round(table.loc[1]*100))
    # plt.xticks(condition)
    # plt.title( 'Symptom:' + symptoms_labels_desc[idx] )
    # plt.ylabel('% of Symptoms')
    # filename = 'plots/migraine_headache/symptoms/s_'+ symptom
    # plt.savefig(filename)
    # # plt.show()
    # plt.cla()
    # print(table)
    val = np.array(table.loc[1]*100)
    val = np.insert(val, 2, csq_test[1])
    # print("val: ", val)
    symptoms_table.insert(0, column=symptoms_labels[idx], value=val)
    # symptoms_table.insert(0, column=symptoms_labels_desc[idx], value=np.array(table.loc[1]*100))
    idx = idx + 1
    # break

symptoms_table = symptoms_table[symptoms_table.columns[::-1]]
symptoms_table = symptoms_table.transpose().round(2)
# print(symptoms_table)

symptoms_table.to_latex('plots/migraine_headache/symptoms/symptom_table')

plt.bar(symptoms_table.index, symptoms_table['Headache'])
plt.bar(symptoms_table.index, symptoms_table['Migraine'], bottom=symptoms_table['Headache'], color='r')
plt.legend(['Headache', 'Migraine'], loc = "upper center")
plt.xticks(rotation=80, fontsize=8)
# plt.title('Headache vs Migraine')
plt.show()