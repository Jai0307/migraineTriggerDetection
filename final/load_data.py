import pandas as pd
import numpy as np

def loadData():

    data = pd.DataFrame(pd.read_csv('data/total_migraine_diary.csv', header=0))
    data.fillna(0, inplace=True)

    columns = ['migraine', 'headache',
        'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise', 'smell', 'light', 'light_noise', 'duration_min', 'intensity', 'intensity_index', 'treatment',
        'tylenol', 'complex', 'triptans', 'Other',
        'no_first_aid_effect', 'first_aid_effective_by_half', 'first_aid_effect_migraine_disappears',
        'stress', 'oversleeping', 'lack_of_sleep', 'exercise', 'no_exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional_change', 'weather_temp_change', 'excessive_sunlight', 'noise', 'inadequate_lighting', 'odors', 'drinking', 'irregular_meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other_triggers'
    ]
    triggers = [
        'stress', 'oversleeping', 'lack_of_sleep', 'exercise', 'no_exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional_change', 'weather_temp_change', 'excessive_sunlight', 'noise', 'inadequate_lighting', 'odors', 'drinking', 'irregular_meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other_triggers'
    ]

    label_symptoms = {
        0: {"required": [], "ignore": ['headache', 'migraine']},
        1: {"required": ['migraine', 'noise', 'nausea_vomit'], "ignore":[]},
        2: {"required": ['migraine', 'noise'], "ignore":['nausea_vomit']},
        3: {"required": ['migraine', 'nausea_vomit'], "ignore":['noise']},
        4: {"required": ['headache', 'unilateral'], "ignore":[]},
        5: {"required": ['headache'], "ignore":['unilateral']},
    }

    labels = np.zeros(shape=[data.shape[0], 1])
    print(labels.shape)

    for key in label_symptoms:
        temp_data = data.copy()
        for req in label_symptoms[key]["required"]:
            temp_data = temp_data[temp_data[req]==1]
        for ig in label_symptoms[key]["ignore"]:
            temp_data = temp_data[temp_data[ig]==0]
        labels[temp_data.index] = key

    return labels, np.array(data[triggers]), triggers