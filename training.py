# training
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


df_train = pd.read_pickle('df_train.pkl')

categorical_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'snap_WI', 'snap_TX', 'snap_CA'] + ["event_name_1", "event_name_2",
                                                                                                                 "event_type_1", "event_type_2"] + ['is_month_end', 'is_month_end', 'is_month_mid', 'is_weekend']
useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]

# Features created

print(df_train.shape)
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = reduce_mem_usage(df_train)

# print(df_train['growth_id_time_7_period_30'].dtypes)
# if df_train['growth_id_time_7_period_30'].min() > np.finfo(np.float16).min and df_train['growth_id_time_7_period_30'].max() < np.finfo(np.float16).max:
#     print('yes')

# print(df_train.dtypes)
# for x in df_train.columns:
#     print(x + str("::::") + str(df_train[x].dtype))


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
