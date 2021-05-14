import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

def rf_importance(X_train, y_train, max_depth=10, class_weight=None, top_n=15, n_estimators=50, random_state=0):
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    print("Xếp hạng các feature:")
    for f in range(X_train.shape[1]):
        print("%d. feature no: %d. feature name: %s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))

    # visualize
    indices = indices[0:top_n]
    plt.figure()
    plt.title("Feature Importance top %d" % top_n)
    plt.bar(range(top_n), importances[indices],
    color='r', yerr=std[indices], align='center')
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show
    return model  

def dt_importance(X_train, y_train, max_depth=10, top_n=20, random_state=0):
    model = DecisionTreeRegressor(max_depth=max_depth, n_jobs=-1, random_state=random_state)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    print("Xếp hạng các feature:")
    for f in range(X_train.shape[1]):
        print("%d. feature no: %d. feature name: %s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))

    # visualize
    indices = indices[0:top_n]
    plt.figure()
    plt.title("Feature Importance top %d" % top_n)
    plt.bar(range(top_n), importances[indices],
    color='r', yerr=std[indices], align='center')
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show
    return model  