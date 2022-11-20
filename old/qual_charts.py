import pandas as pd
import numpy as np;
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency


filename = 'data/formatted-symptoms.csv'
symptoms = pd.DataFrame(pd.read_csv(filename, header=0))

all_columns = ['ID', 'Study number', 'gender', 'age', 'kidney', 'weight',
       'bmi', 'Headache frequency (day/month)', 'Duration (hours/times)',
       'headache intensity', 'Headache intensity (VAS)', 'unilateral',
       'daily exercise concert ', ' area', 'throw up', 'photosensitivity',
       'noise sensitivity', ' odor sensitivity']

plt.hist(symptoms['gender'], align='mid')
plt.xticks([0,1])
plt.xlabel('Gender(F=0, M=1)')
plt.ylabel('# of patients')
plt.savefig('charts/gender.png')
plt.cla()

plt.hist(symptoms['age'], bins=5, align='mid')
plt.xlabel('Patient age')
plt.ylabel('# of patients')
plt.savefig('charts/age.png')
plt.cla()

plt.hist(symptoms['Headache frequency (day/month)'], align='mid')
plt.xlabel('Headache frequency (day/month)')
plt.ylabel('# of patients')
plt.savefig('charts/headache_freq.png')
plt.cla()

plt.hist(symptoms['Duration (hours/times)'], align='mid')
plt.xlabel('Duration (hours/times)')
plt.ylabel('# of patients')
plt.savefig('charts/headache_duration.png')
plt.cla()

plt.hist(symptoms['noise sensitivity'], align='mid')
plt.xlabel('Noise sensitivity (0=NO, 1: YES)')
plt.xticks([0,1])
plt.savefig('charts/noisesensitivity.png')
plt.cla()

plt.hist(symptoms['photosensitivity'], align='mid')
plt.xlabel('Photosensitivity (0=NO, 1: YES)')
plt.ylabel('# of patients')
plt.xticks([0,1])
plt.savefig('charts/photosensitivity.png')
plt.cla()

plt.hist(symptoms['throw up'], align='mid')
plt.xlabel('Throw up (0=NO, 1: YES)')
plt.ylabel('# of patients')
plt.xticks([0,1])
plt.savefig('charts/throwup.png')
plt.cla()

plt.hist(symptoms[' odor sensitivity'], align='mid')
plt.xlabel('Odor sensitivity (0=NO, 1: YES)')
plt.ylabel('# of patients')
plt.xticks([0,1])
plt.savefig('charts/odorsensitivity.png')
plt.cla()

plt.hist(symptoms['Headache intensity (VAS)'], align='mid', bins=7)
plt.xlabel('Headache intensity (VAS)')
plt.ylabel('# of patients')
plt.savefig('charts/headache_intensity_VAS.png')
plt.cla()


plt.hist(symptoms['weight'], align='mid', bins=8)
plt.xlabel('Weight (kgs)')
plt.ylabel('# of patients')
plt.savefig('charts/weight.png')
plt.cla()