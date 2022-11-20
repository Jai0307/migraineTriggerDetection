from re import L
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('total_migraine_diary.csv', header=0))
data.fillna(0, inplace=True)
symptoms_labels = [
    'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise', 'smell', 'light', 'noise', 'light_noise'
]

symptoms = data[symptoms_labels]
aggregate_symptoms = symptoms.sum(axis=1)
migraine = data['migraine_headache']

all_symptoms_labels = pd.DataFrame(columns=symptoms_labels, dtype='float')
idx = 0
for symptom in symptoms_labels:
    symptom_col = data[symptom]
    table = pd.crosstab(symptom_col, migraine)
    table = table.div(table.sum(1).astype(float), axis=0)
    # print(table)
    all_symptoms_labels[symptom] = table.loc[1]
    plt.bar(symptom, all_symptoms_labels[symptom].loc[1]*100, color='b')
    plt.bar(symptom, all_symptoms_labels[symptom].loc[2]*100, bottom= all_symptoms_labels[symptom].loc[1]*100, color='r')
    # break
plt.legend(['Migraine','Headache'], loc = "lower center")
plt.xticks(rotation=80, fontsize=10)
plt.ylabel('Probability given a symptom')
plt.title('Migraine vs Headache')
plt.tight_layout()
plt.savefig('plots/migraine_headache/symptoms_barchart')
plt.show()

# stacked bar chart for total triggers vs migraine
# print(table)
# table = table.div(table.sum(1).astype(float), axis=0)
# plt.bar(table.index, table[0]*100, color='g')
# plt.bar(table.index, table[1]*100, bottom=table[0]*100, color='r')
# plt.bar(table.index, table[2]*100, bottom=table[0]*100, color='b')
# plt.xticks(range(1, table.index.shape[0]))
# plt.title('Number of Symptoms causing Headache')
# plt.xlabel('Number of Symptoms')
# plt.ylabel('Total Observations %')
# plt.legend(['Migraine','Headache'], loc = "lower center")
# plt.savefig('plots/migraine_headache/symptoms_headache')
# plt.show()

