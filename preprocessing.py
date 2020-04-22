# preprocessing
import pandas as pd
import numpy as np
import json
import gc
import lightgbm as lgb

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
df_new['mday'] = getattr(df_new['date'], "day")
df_new['is_month_start'] = np.where(df_new['mday'] <= 9, 1, 0)
df_new['is_month_mid'] = np.where(
    (df_new['mday'] >= 10) & (df_new['mday'] <= 19), 1, 0)
df_new['is_month_end'] = np.where(df_new['mday'] >= 20, 1, 0)
df_new['quarter'] = getattr(df_new['date'], "quarter")
df_new['week'] = getattr(df_new['date'], "weekofyear")
print(df_new.columns)

print(df_new.head(20))

df_train = df_new[df_new['date'] < '25-04-2016']
df_predict = df_new[df_new['date'] >= '25-04-2016']
# print(df_train.shape)
# print(df_predict.shape)

categorical_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + \
    ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]

# Features created


'''
# Starting Training models

# trying the LGB Model
train_cols = df_train.columns[~df_train.columns.isin(useless_cols)]
X_train = df_train[train_cols]
y_train = df_train["sales"]


test_inds = np.random.choice(X_train.index.values, 2_000_000, replace=False)
train_inds = np.setdiff1d(X_train.index.values, test_inds)

train_data = lgb.Dataset(
    X_train.loc[train_inds], label=y_train.loc[train_inds], categorical_feature=categorical_feats)
test_data = lgb.Dataset(
    X_train.loc[test_inds], label=y_train.loc[test_inds], categorical_feature=categorical_feats)

del X_train, y_train, train_inds, test_inds
gc.collect()


params = {
    "objective": "poisson",
    "metric": "rmse",
    "force_row_wise": True,
    "learning_rate": 0.075,
    #    "sub_feature" : 0.8,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    #    "nthread" : 4
    "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations': 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}

m_lgb = lgb.train(params, train_data, valid_sets=[test_data], verbose_eval=20)
m_lgb.save_model("model.lgb")
'''
