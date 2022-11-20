import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('total_migraine_diary.csv', header=0))
data.fillna(0, inplace=True)
symptoms_labels = [
    'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise', 'smell', 'light', 'noise', 'light_noise'
]

migraine = data['migraine_headache']
print(migraine.sum(axis=0))
print(migraine.shape)
for symptom in symptoms_labels:
    symptoms = data[symptom]
    # stacked bar chart for total triggers vs migraine
    table = pd.crosstab(symptoms, migraine)
    table = table.div(table.sum(1).astype(float), axis=0)
    table.index = ['NO', 'YES']
    # print(table)
    plt.bar(table.index, table[0]*100, color='c')
    plt.bar(table.index, table[1]*100, bottom=table[0]*100, color='r')
    plt.bar(table.index, table[2]*100, bottom=table[0]*100, color='g')
    plt.xticks(table.index)
    plt.title( 'Symptom:' + symptom )
    plt.xlabel('Symptom present')
    plt.ylabel('Total Observations %')
    plt.legend(['No Headache','Migraine', 'Headache'], loc = "lower center")
    filename = 'plots/migraine_headache/'+ symptom
    plt.savefig(filename)
    # plt.show()
    # break