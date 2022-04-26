
import pandas as pd

df = pd.read_csv('A0KX894_hr.csv')

# create datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df['ts'] = df[['datetime']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

tmp = df.at[0, 'ts']
print(tmp)

for i in df.index:

    if df.at[i, 'ts'] - tmp < 60:
        df.drop([i],axis=0,inplace=True)
    else:
        tmp = df.at[i, 'ts']

print(df)
