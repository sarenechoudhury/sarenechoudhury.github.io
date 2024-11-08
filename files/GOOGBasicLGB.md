---
layout: page
title: Basic LightGBM Model
subtitle: Python
---

#### This model uses a dataset of Google's stock from 2004-2024. 

Import necessary packages.
```python
import sys
!{sys.executable} -m pip3 install lightgbm
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import sklearn
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
```

Read in the data, which can be found on my Github [here](https://github.com/sarenechoudhury/sarenechoudhury/blob/7869c453d81eaed3fe84d85cb441618483a9a9da/assets/GOOGLE.csv) 
```python
df = pd.read_csv("../assets/GOOGLE.csv")
df
```

Set the x and y values.
```python
x = list(df.drop(['Close', 'Date', 'Adj Close'], axis=1).columns)
y = ['Close']
```

Split the data into training and predicting sets
```python
train=df.sample(frac=0.8,random_state=200)
predict=df.drop(train.index)
predict[x]
```

Transform those datasets into LightGBM Dataset objects
```python
new_train = lgb.Dataset(train[x], train[y])
new_predict = lgb.Dataset(predict[x], predict[y])
```

Set the LightGBM parameters (a list of which can be found [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html)) and train the model.
```python
params = {
    "boosting_type": "gbdt",
    "num_leaves": 5,
    "force_row_wise": True,
    "learning_rate": 0.5,
    "metric": "binary_logloss",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 15,
    "use_quantized_grad": True,
    "num_grad_quant_bins": 4,
    "quant_train_renew_leaf": True,

}
Model = lgb.train(params, new_train, num_boost_round = 100)
```

Predict using the trained model.
```python
y_pred = Model.predict(predict[x])
```

Combine the predicted and true values into one dataframe 
```python
result = pd.concat([pd.DataFrame(y_pred), predict[y].reset_index().drop('index', axis = 1)], axis = 1, ignore_index = True)
result = result.set_axis(['y', 'y_pred'], axis = 1)
result
```

Find the correlation between the predicted and true values. This model had almost perfect accuracy due to the small dataset (4918x6) of one company's stock, but with a more comprehensive dataset would perform regularly. 
```python
result.corr()
```




