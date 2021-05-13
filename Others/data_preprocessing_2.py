import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# T-shirt sizes
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

size_mapping = {
    'XL': 3,
    'L' : 2,
    'M' : 1
}

df['size'] = df['size'].map(size_mapping)  # replaces ordinal feature size to numerical values
# print(df)

"""
Reverts the numerical values back to ordinal features

inv_size_mapping = {i:j for j, i in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print(df)
"""

"""
Converts class labels to values (Alternative value used below multi-line comment

class_mapping = {label:index for index, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)           # Same way to revert values back to class names
"""

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# print(y)
df['classlabel'] = y
# print(df)

"""
Reverse method
y = class_le.inverse_transform(y)
print(y)
"""

# fit_transform gives different values for different data, which is unsuitable for values for colors, since
# one color is given preference over another because of it's higher value. Hence OneHotEncoder makes columns of
# 0's and 1's representing a color

"""
ohe = OneHotEncoder(categorical_features=[0])     # Deprecated method, get_dummies method preferred 
x = df[['color', 'size', 'price']].values
x[:, 0] = class_le.fit_transform(x[:, 0])
print(x)
print(ohe.fit_transform(x).toarray())
"""
pd = pd.get_dummies(df[['price', 'color', 'size']])