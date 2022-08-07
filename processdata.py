import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(pd.read_csv('patient_info.csv', header=0))

data['sex'].replace(['F','M'],[0,1], inplace=True)

patients = data['patient']
data = data.drop(['patient'], axis = 1)
data.index= patients

for p in patients:
    if(pd.isna(data['weight'].loc[p]) or pd.isna(data['height'].loc[p]) or pd.isna(data['bmi'].loc[p])):
        data = data.drop([p])     


patient_data = ['sex', 'age', 'height', 'weight', 'bmi', 'headache_freq']
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

print(data)
# print(data.to_string())