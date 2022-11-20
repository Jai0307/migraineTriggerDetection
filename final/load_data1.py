import pandas as pd
import numpy as np

def loadData1():

    data = pd.DataFrame(pd.read_csv('data/total_migraine_diary.csv', header=0))
    data.fillna(0, inplace=True)

    columns = ['migraine', 'headache',
        'left_side', 'right_side', 'middle', 'both_sides', 'around_eyes', 'back_of_neck', 'unilateral', 'throbbing_1', 'throbbing_2', 'tight_head', 'dull_heavy', 'worsen_move', 'food', 'nausea_vomit', 'throw_up', 'noise_sensitivity', 'smell', 'light', 'light_noise', 'duration_min', 'intensity', 'intensity_index', 'treatment',
        'tylenol', 'complex', 'triptans', 'Other',
        'no_first_aid_effect', 'first_aid_effective_by_half', 'first_aid_effect_migraine_disappears',
        'stress', 'oversleeping', 'lack_of_sleep', 'exercise', 'no_exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional_change', 'weather_temp_change', 'excessive_sunlight', 'noise', 'inadequate_lighting', 'odors', 'drinking', 'irregular_meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other_triggers'
    ]
    triggers = [
        'stress', 'oversleeping', 'lack_of_sleep', 'exercise', 'no_exercise', 'fatigue', 'menstruation', 'ovulation', 'emotional_change', 'weather_temp_change', 'excessive_sunlight', 'noise', 'inadequate_lighting', 'odors', 'drinking', 'irregular_meals', 'surfeit', 'caffeine', 'smoking', 'cheese_chocolate', 'travel', 'other_triggers'
    ]

    labels = data['migraine_headache']

    labels = abs(labels)
    num_labels = 3
# 2 no symtom, 1 migraine, 0 headache
    return labels, np.array(data[triggers]), triggers, num_labels