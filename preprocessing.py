# preprocessing
import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import pickle


def reduce_mem_usage(df, verbose=True):
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  start_mem = df.memory_usage().sum() / 1024**2
  for col in df.columns:
    col_type = df[col].dtypes
    if col_type in numerics:
      c_min = df[col].min()
      c_max = df[col].max()
      if str(col_type)[:3] == 'int':
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
          df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
          df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
          df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
          df[col] = df[col].astype(np.int64)
      else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
          df[col] = df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
          df[col] = df[col].astype(np.float32)
        else:
          df[col] = df[col].astype(np.float64)
  end_mem = df.memory_usage().sum() / 1024**2
  if verbose:
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))
  return df


# Basic Project Settings
h = 28
max_lags = 57
tr_last = 1913

# working on calendar.csv

cal_dtypes = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
              "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
              "month": "int16", "year": "int16", "snap_CA": "category", 'snap_TX': 'category', 'snap_WI': 'category'}
df_cal = pd.read_csv("calendar.csv", dtype=cal_dtypes)
# converting date to datetime
df_cal['date'] = pd.to_datetime(df_cal['date'])


# Creating feature for upcoming events
# df_cal['is_event'] = np.where((df_cal['event_name_1'].notnull()) | (
#     df_cal['event_name_2'].notnull()), 1, 0).astype('int8')
# f_lags = [1]
# for lag in f_lags:
#   df_cal[f"n_event_in_{lag}"] = df_cal[['date', 'is_event']
#                                        ]['is_event'].transform(lambda x: x.rolling(lag).sum()).shift(-1 * lag)
df_cal['event_1_tmr'] = df_cal['event_name_1'].shift(-1)
df_cal['event_2_tmr'] = df_cal['event_name_2'].shift(-1)

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
dtype = {numcol: "int32" for numcol in numcols}
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

# time features
df_new['is_weekend'] = np.where(
    (df_new['wday'] == 1) | (df_new['wday'] == 2), 1, 0).astype("int16")
df_new['mday'] = getattr(df_new['date'].dt, "day").astype("int16")
df_new['is_month_start'] = np.where(df_new['mday'] <= 7, 1, 0).astype("int16")
df_new['is_month_mid'] = np.where(
    (df_new['mday'] >= 8) & (df_new['mday'] <= 21), 1, 0).astype("int16")
df_new['is_month_end'] = np.where(df_new['mday'] >= 22, 1, 0).astype("int16")
df_new['quarter'] = getattr(df_new['date'].dt, "quarter").astype("int16")
df_new['week'] = getattr(df_new['date'].dt, "weekofyear").astype("int16")


df_new = reduce_mem_usage(df_new)


df_train = df_new[df_new['date'] < '25-04-2016']


# print(df_train.shape)
# print("id grouping")
# print(df_train[["id", "sales"]].groupby("id")['sales'].shift(1).shape)
# print("store grouping")
# print(df_train[["store_id", "sales"]].groupby(
#     "store_id")['sales'].shift(1).shape)
# print("item grouping")
# print(df_train[["item_id", "sales"]].groupby(
#     "item_id")['sales'].shift(1).shape)


# print(df_train.shape)
# print(df_predict.shape)

# Lag features

# id grouping
lags = [1, 7, 15, 30, 60, 90, 180, 360]
lagcols = [f"lag_id_{lag}" for lag in lags]
for lag, lagcol in zip(lags, lagcols):
  print(lagcol)
  df_train.loc[:, lagcol] = df_train[["id", "sales"]].groupby("id")[
      "sales"].shift(lag).astype('float32')


windows = [7, 15, 30, 90]
for window in windows:
  for lag, lagcol in zip(lags, lagcols):
    print(f"rmean_id_{lag}_{window}")
    df_train.loc[:, f"rmean_id_{lag}_{window}"] = df_train[["id", lagcol]].groupby(
        "id")[lagcol].transform(lambda x: x.rolling(window).mean()).astype('float32')


# growth features
growths = {
    '7': [30, 90, 180, 360],
    '30': [90, 180, 360],
    '90': [180, 360]
}

for x in growths:
  for y in growths[x]:
    print(x, y)
    growthcol = f"growth_id_time_{x}_period_{y}"
    print(growthcol)
    df_train.loc[:, growthcol] = df_train[f"rmean_id_1_{x}"].pct_change(y)
df_train = reduce_mem_usage(df_train)


df_train.to_pickle('df_train.pkl')


categorical_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'snap_WI', 'snap_TX', 'snap_CA'] + ["event_name_1", "event_name_2",
                                                                                                                 "event_type_1", "event_type_2"] + ['is_month_end', 'is_month_end', 'is_month_mid', 'is_weekend']
useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]

# Features created
df_train = reduce_mem_usage(df_train)
print(df_train.shape)

print(df_train.dtypes)


# print(df_train[["store_id",'date',"sales",'lag_store_1']].head(20))

# print(df_train[["item_id",'date',"sales",'lag_item_1']].head(20))


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
    "objective": "tweedie",
    'tweedie_variance_power': 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    #"sub_feature" : 0.8,
    "force_row_wise": True,
    "sub_row": 0.8,
    "bagging_freq": 10,
    "lambda_l2": 0.1,
    #"nthread" : 4
    'verbosity': 1,
    'num_iterations': 1500,
    'num_leaves': 1023,
    "min_data_in_leaf": 2047,
    # 'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0,
    # 'save_binary': True,
    # 'gpu_use_dp': False
}


print("Start Training")
m_lgb = lgb.train(params, train_data, valid_sets=[
    test_data], verbose_eval=100, early_stopping_rounds=200)


with open('model_tweedie_12.pkl', 'wb') as fout:
  pickle.dump(m_lgb, fout)
