"""
Comparing Decision Tree Classifiers to bootstrap techniques to increase accuracy by means of
bagging and AdaBoost. These techniques make use of ensemble methods (majority voting) to increase accuracy
with the expense of an increase in computational cost.
The accuracy of the different methods are compared and a decision region graphs are plotted using
Alcohol and Hue features of 2 of the 3 labels of the wine dataset.
"""
from wine_ensemble_common_funcs import tree_compare_plot
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
tree_compare_plot(classifiers=[tree, bag], labels=['Decision Tree', 'Bagging'])

print('')
tree_2 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree_2, n_estimators=500, learning_rate=0.1, random_state=0)
tree_compare_plot(classifiers=[tree_2, ada], labels=["Decision Tree", 'AdaBoost'])
