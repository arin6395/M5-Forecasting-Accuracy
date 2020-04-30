import pandas as pd
import numpy as np
import json
import gc
import lightgbm as lgb
from datetime import datetime, timedelta


# Basic Project Settings
h = 28
max_lags = 57
tr_last = 1913

# working on calendar.csv

cal_dtypes = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
              "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
              "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
df_cal = pd.read_csv("calendar.csv", dtype=cal_dtypes)
# converting date to datetime
df_cal['date'] = pd.to_datetime(df_cal['date'])

# dropping weekday as wday is redundant
dropcols = ['weekday']

df_cal.drop(dropcols, axis=1, inplace=True)
# print(df_cal.shape)
# print(df_cal.info())


# working on sell_prices.csv

sp_dtypes = {"store_id": "category", "item_id": "category",
             "wm_yr_wk": "int16", "sell_price": "float32"}
df_sp = pd.read_csv("sell_prices.csv", dtype=sp_dtypes)
# print(df_sp.head())
# print(df_sp.info())


# working on sales_train_validation.csv
is_train = True
start_day = max(1 if is_train else tr_last - max_lags, 1)
numcols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
dtype = {numcol: "float32" for numcol in numcols}
dtype.update({col: "category" for col in catcols if col != "id"})
df = pd.read_csv("sales_train_validation.csv",
                 usecols=catcols + numcols, dtype=dtype, nrows=None)
for day in range(tr_last + 1, tr_last + 28 + 1):
    df[f"d_{day}"] = np.nan
# print(df.shape)
# print(df.info())

cols = ["id", "item_id", "dept_id",
        "cat_id", "store_id", "state_id"]
df_new = pd.melt(df,
                 id_vars=cols,
                 value_vars=[
                     col for col in df.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

# print(df_new.head(5))
# print(df_new.info())

del df
gc.collect()
# df_new.to_csv('transactional.csv')


# Merging All tables
df_new = df_new.merge(df_cal, how='left', on="d", copy=False)
# print(df_new.head(20))
# print(df_new.info())


df_new = df_new.merge(
    df_sp, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
# print(df_new.info())
# print(df_new.shape)


# Till now we have cleaned and proccessed the data in different tables
# Different Tables have been merged.

for col in df_new.columns:
    if str(df_new[col].dtype) == 'category':
        df_new[col] = df_new[col].cat.codes.astype("int16")
        df_new[col] -= df_new[col].min()

# print(df_new.info())
# print(df_new.shape)


# Feature engineering
df_new['is_weekend'] = np.where(
    (df_new['wday'] == 1) | (df_new['wday'] == 2), 1, 0)
df_new['mday'] = getattr(df_new['date'].dt, "day")
df_new['is_month_start'] = np.where(df_new['mday'] <= 9, 1, 0)
df_new['is_month_mid'] = np.where(
    (df_new['mday'] >= 10) & (df_new['mday'] <= 19), 1, 0)
df_new['is_month_end'] = np.where(df_new['mday'] >= 20, 1, 0)
df_new['quarter'] = getattr(df_new['date'].dt, "quarter")
df_new['week'] = getattr(df_new['date'].dt, "weekofyear")

# df_train = df_new[df_new['date'] < '25-04-2016']
# df_predict = df_new[df_new['date'] >= '25-04-2016']


useless_cols = ["id", "date", "sales", "d", "wm_yr_wk"]
fday = datetime(2016, 4, 25)
# df_new = pd.concat([df_train, df_predict], sort=False)

df_new = df_new[(df_new['date'] >= fday - timedelta(days=60))]
gc.collect()
print(df_new.shape)
# print(df_new.info)
lags = [1, 7, 14, 28]
lagcols = [f"lag_{lag}" for lag in lags]
model = lgb.Booster(model_file='model_tweedie_5.lgb')
for x in range(0, 28):
    curday = fday + timedelta(days=x)
    df_temp = df_new[(df_new['date'] >= curday - timedelta(days=57))
                     & (df_new['date'] <= curday)]
    for lag, lagcol in zip(lags, lagcols):
        df_temp[lagcol] = df_temp[["id", "sales"]].groupby("id")[
            "sales"].shift(lag)

    windows = [7, 14, 28]
    for window in windows:
        for lag, lagcol in zip(lags, lagcols):
            df_temp[f"rmean_{lag}_{window}"] = df_temp[["id", lagcol]].groupby(
                "id")[lagcol].transform(lambda x: x.rolling(window).mean())

    df_temp_predict = df_temp[df_temp['date'] == curday]
    print(x, curday)
    df_temp_predict.drop(useless_cols, axis=1, inplace=True)
    temp_prediction = model.predict(df_temp_predict)
    print(temp_prediction)
    df_new.loc[df_new['date'] == curday, 'sales'] = temp_prediction

df_predict = df_new[df_new['date'] >= '25-04-2016']
# df_predict.drop(useless_cols, axis=1, inplace=True)
df_predict.to_csv('predicted.csv')
