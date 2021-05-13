"""
Regularized models shrink the number of parameters of the model to induce a penalty against complexity,
thereby decreasing it and reducing overfitting.
Popular approaches:
    Ridge Regression - L2 regularization    # intercept not regularized
    Least Absolute Shrinkage and Selection Operator (LASSO) - L1 regularization(sparse method)
    Elastic Net- Combination of LASSO and Ridge Regression 
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
e_net = ElasticNet(alpha=1.0, l1_ratio=0.5)          # if l1_ratio = 1.0, e_net = lasso
