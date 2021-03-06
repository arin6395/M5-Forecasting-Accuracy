import pandas as pd
import numpy as np
import json
import gc
import lightgbm as lgb

df_predicted = pd.read_csv("predicted.csv")
dropcols = df_predicted.columns.tolist()
dropcols.remove('id')
dropcols.remove('d')
dropcols.remove('sales')


df_predicted.drop(dropcols, axis=1, inplace=True)
df_predicted['d'] = df_predicted['d'].apply(
    lambda x: "F" + str(int(x.split('_')[1]) - 1913))
print(df_predicted.head())
df = df_predicted.pivot(index='id', columns='d')['sales']
df2 = df.copy()
df2.index = df.index.str.replace("validation$", "evaluation")
df = pd.concat([df, df2], axis=0, sort=False)
df = df.round(decimals=3)
print(df.head())

df.to_csv('submission13r.csv')
