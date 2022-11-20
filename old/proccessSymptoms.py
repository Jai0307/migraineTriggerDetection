import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

patient_symptoms = pd.DataFrame(pd.read_csv("data/patient-symptoms.csv", header=0))

print(patient_symptoms.head(1))

patient_symptoms['gender'].replace(['F', 'M'], [0,1], inplace=True)


bmiMean = round(patient_symptoms["bmi"].mean(),1)
patient_symptoms['bmi'].fillna(value=bmiMean,inplace=True)
kidneyMean = round(patient_symptoms["kidney"].mean())
patient_symptoms['kidney'].fillna(value=kidneyMean,inplace=True)
weightMean = round(patient_symptoms["weight"].mean())
patient_symptoms['weight'].fillna(value=weightMean,inplace=True)

print(patient_symptoms["bmi"].to_string())
print("----------------------------")
print(patient_symptoms["kidney"].to_string())
print("----------------------------")
print(patient_symptoms["weight"].to_string())

patient_symptoms.fillna(value=0,inplace=True)


patient_symptoms.to_csv("data/formatted-symptoms.csv")