import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.DataFrame(pd.read_csv('data/total_migraine_diary.csv', header=0))
data.fillna(0, inplace=True)

triggers = [
    'stress', 'oversleeping', 'lack_of_sleep', 'exercise', 'no_exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional_change', 'weather_temp_change', 'excessive_sunlight', 'noise', 'inadequate_lighting', 'odors', 'drinking', 'irregular_meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other_triggers'
]

triggers_desc = [
    'stress', 'oversleeping', 'lack of sleep', 'exercise', 'no exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional change', 'weather temp change', 'excessive sunlight', 'noise', 'inadequate lighting', 'odors', 'drinking', 'irregular meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other triggers'
]

migraine = data['migraine_headache']
print(migraine.sum(axis=0))
print(migraine.shape)
condition = np.array(['Migraine', 'Headache'])
index = np.array(['Migraine', 'Headache', 'p-value'])
triggers_table = pd.DataFrame(dtype=float, index=index)
idx = 0
for symptom in triggers:
    symptoms = data[symptom]
    table = pd.crosstab(symptoms, migraine)
    print(table)
    pivtable = table
    csq_test = stats.chi2_contingency(pivtable)
    table = table.div(table.sum(0).astype(float), axis=1)
    # print(table)
    table = table.drop(0, axis=1)
    # plt.bar(condition, (table.loc[1]*100))
    # plt.xticks(condition)
    # plt.title( 'Trigger:' + triggers[idx] )
    # plt.ylabel('% of Condition associated with the triggers')
    # filename = 'plots/migraine_headache/triggers/t_'+ symptom
    # plt.savefig(filename)
    # plt.show()
    plt.cla()
    val = np.array(table.loc[1]*100)
    val = np.insert(val, 2, csq_test[1])
    triggers_table.insert(0, column=triggers_desc[idx], value=val)
    idx = idx + 1
    # break

triggers_table = triggers_table[triggers_table.columns[::-1]]
triggers_table = triggers_table.transpose().round(2)
# print(triggers_table)
triggers_table.to_latex('plots/migraine_headache/triggers/triggers_table')

plt.bar(triggers_table.index, triggers_table['Headache'])
plt.bar(triggers_table.index, triggers_table['Migraine'], bottom=triggers_table['Headache'], color='r')
plt.legend(['Headache', 'Migraine'], loc = "upper center")
plt.xticks(rotation=80, fontsize=8)
# plt.title('Headache vs Migraine')
plt.show()