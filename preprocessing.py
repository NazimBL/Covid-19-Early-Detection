# -*- coding: utf-8 -*-
"""Copy of CovidPreProcess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h0nW-N9qg_JHS9RuQB8EEBu0jXPb3fuB
"""

import pandas as pd
import numpy as np

covid_path = "/home/AV2GF3B_COVID.csv"
healthy_path = "/home/AFYLHG4.csv"
others_path = "/home/AA0HAI1_1_hros.csv"

"""#Covid"""

df = pd.read_csv(covid_path)

df2 = df[['Time', 'Var1']]
time_df = df['Time']

arr = np.array_split(df2, 132)
arr_time = np.array_split(time_df, 132)

np.shape(arr_time[1])

arr_time[0]

arr[0]

start_date = df['start date'][0]
recovery_date = df['recovery date'][0]

start_query = time_df[time_df == start_date]
recovery_query = time_df[time_df == recovery_date]

start_query

start_index = df.index[df["Time"]==start_date].tolist()[0]
recovery_index = df.index[df["Time"]==recovery_date].tolist()[0]

recovery_index

df = df.loc[start_index:recovery_index]

df = df[df['Var1'].notnull()]

df = df.rename(columns={"Var1": "HROS",})
df = df.reset_index(drop=True)

dataset_ROI = df
nbr_of_samples = len(dataset_ROI)

dataset_ROI["label"] = '1'
dataset_ROI.to_csv("COVID_VECTOR2.csv", index=False)

nbr_of_samples = len(dataset_ROI)

nbr_of_samples

"""#Healthy"""

df = pd.read_csv(healthy_path)

df = df[df['Var1'].notnull()]

len(df)/720

df = df.rename(columns={"Var1": "HROS","diagnosis":"label"})
df = df.reset_index(drop=True)

dataset_ROI = df.loc[:nbr_of_samples-1]

dataset_ROI

len(dataset_ROI)

dataset_ROI["label"] = '0'
dataset_ROI.to_csv("HEALTHY_VECTOR2.csv", index=False)

"""#Other"""

df = pd.read_csv(others_path)



start_date = df['start date'][0]
#recovery_date = df['recovery date'][0]

start_date

#recovery_date

start_index = df.index[df["Time"]==start_date].tolist()[0]
#recovery_index = df.index[df["Time"]==recovery_date].tolist()[0]

#df = df.loc[start_index:recovery_index]
df = df.loc[start_index:]

len(df)

df = df[df['Var1'].notnull()]

len(df)

df = df.rename(columns={"Var1": "HROS",})
df = df.reset_index(drop=True)

dataset_ROI = df.loc[:nbr_of_samples-1]

len(dataset_ROI)



dataset_ROI["label"] = '0'
del dataset_ROI["start date"]
#del dataset_ROI["recovery date"]
dataset_ROI.to_csv("OTHER_VECTOR2.csv", index=False)

