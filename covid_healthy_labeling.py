import os
import pandas as pd
import numpy as np
from pathlib import Path

directory_path = '/home/nazim/Downloads/HROS'
df = pd.read_csv('covid_table.csv')

#print(df['ParticipantID'])
arr = np.array(df['ParticipantID'].values)


# iterate over files in
# that directory

i=j=0
print("Files and directories in a specified path:")
for filename in os.listdir(directory_path):
    f = os.path.join(directory_path,filename)
    if os.path.isfile(f):
        f_name, f_ext = os.path.splitext(f)
        f_name = os.path.basename(f_name)
        for x in arr:
            print(x)
            print(f_name)
            print('---------------------------')
            if(f_name == x):
                i+=1
                os.rename(f, f_name + '_COVID' + f_ext)
                j+=1

print(i)
print(j)
