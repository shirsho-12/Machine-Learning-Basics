import pandas as pd
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5,6.0,,8.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull().sum())
# print(df.values)
"""
print(df.dropna())    
Arguments:
    axis=1 drops rows with at least one NaN
    how='all' drops columns with all NaNs 
    subset=['x'] drops rows with NaN's in specific columns
"""

imr = SimpleImputer(missing_values=np.nan, strategy='mean').fit(df)    # works with columns
# row info found here: https://github.com/scikit-learn/scikit-learn/issues/10636
imputed_data = imr.transform(df.values)
print(imputed_data)