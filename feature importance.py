import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


model = lgb.Booster(model_file='model_tweedie_5.lgb')

feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(
), model.feature_name())), columns=['Value', 'Feature'])


plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
    by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
