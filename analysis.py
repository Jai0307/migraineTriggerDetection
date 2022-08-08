import pandas as pd
import matplotlib.pyplot as plt

patient_avgs = pd.DataFrame(pd.read_csv('patientavgs.csv', header=0))

plt.scatter(patient_avgs['nausea_vomiting'], patient_avgs['pain_intensity'])
plt.show()