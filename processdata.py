from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

patient_info = pd.DataFrame(pd.read_csv('patient_info.csv', header=0))

patient_info['sex'].replace(['F','M'],[0,1], inplace=True)

patients = patient_info['patient']
patient_info = patient_info.drop(['patient'], axis = 1)
patient_info.index= patients

for p in patients:
    if(pd.isna(patient_info['weight'].loc[p]) or pd.isna(patient_info['height'].loc[p]) or pd.isna(patient_info['bmi'].loc[p])):
        patient_info = patient_info.drop([p])     


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


allfields = patient_info_cols + migraine_identifiers + migraine_triggers + helpful_actions

migraine_data = pd.DataFrame(pd.read_csv('migraine_data.csv',header=0))
idx_delete = migraine_data.index[migraine_data['duration_min']>3000]
print(migraine_data.shape)
migraine_data = migraine_data.drop(idx_delete)

print(migraine_data.shape)
uids = migraine_data['ID'].unique()

patient_averages = pd.DataFrame(columns=allfields, dtype='float')

for id in uids:
    try:
        patient_data_i = patient_info.loc[id]

        patient_i = migraine_data.loc[migraine_data['ID']==id]
        
        pda = patient_i[patient_i.columns.intersection(allfields)]
        means = pda.mean()
        patient_averages.loc[id] = means
        patient_averages['sex'].loc[id] = patient_info['sex'].loc[id]
        patient_averages['age'].loc[id] = patient_info['age'].loc[id]
        patient_averages['height'].loc[id] = patient_info['height'].loc[id]
        patient_averages['weight'].loc[id] = patient_info['weight'].loc[id]
        patient_averages['bmi'].loc[id] = patient_info['bmi'].loc[id]
        patient_averages['headache_freq'].loc[id] = patient_info['headache_freq'].loc[id]

    except:
        print('id not found', id)

# for c in patient_averages.columns:
#     if c!='height' and c!='bmi' and c!='headache_freq' and c!='pain_intensity' and c !='duration_min' and c!='weight' and  c!='incapacitated_degree' and c!='age':
#         for idx in patient_averages.index:
#             if patient_averages[c].loc[idx]<=0.33:
#                 patient_averages[c].loc[idx]=0
#             elif patient_averages[c].loc[idx]<0.67:
#                  patient_averages[c].loc[idx]=1
#             else:
#                 patient_averages[c].loc[idx]=2
            

        # patient_averages[c].loc[patient_averages[c]<=0.3]=0
        # patient_averages[c].loc[patient_averages[c]>0.3 & patient_averages[c]<=0.67]=1
        # patient_averages[c].loc[patient_averages[c]>0.67]=2

patient_averages = patient_averages.drop(['exercise',
'no_exercise',
'ovulation',
'emotion',
'excessive_sunlight',
'improper_lighting',
'drinking',
'overeating',
'caffeine',
'smoking',
'cheese_chocolate',
'travel',
'Missed_school_work',
'Work_decreased_less_than_half',
'Housework_reduced_to_less_than_half',
'Sensitivity_to_odors',

], axis=1)

print(patient_averages.columns)
patient_averages.to_csv('patientavgs.csv')

# print(patient_averages)