import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Pre-process the data
def pre_process(filename):
    # load data and replace ? by np.nan
    df = pd.read_csv(filepath_or_buffer=filename)
    df_rplc = df.replace('?', np.nan).replace('class1', 0).replace('class2', 1).values
    X = df_rplc[:, :len(df_rplc[1]) - 1]
    y = df_rplc[:, len(df_rplc[1]) - 1:].ravel()
    # meta data of the data frame
    # df_rplc.info()
    # segment view
    # df_rplc
    # deal with missing values using SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_data = imp_mean.fit_transform(X)
    # print(imp_data)
    # normalization using MinMaxScaler and using np.around to format matrix in .4f style
    scaler = MinMaxScaler()
    sclr_imp_data = scaler.fit_transform(imp_data)
    sclr_imp_data = np.around(sclr_imp_data, 4)
    for i in range(0, sclr_imp_data.shape[0]):
        for j in range(0, sclr_imp_data.shape[1]):
            print(format(sclr_imp_data[i][j], '.4f'), end=",")
        print(y[i])
    # print(pd.DataFrame(np.hstack((sclr_imp_data, y))).to_csv(index=False, header=False, float_format='%.4f'))
    return sclr_imp_data
    # print(X)
    # print(y)


########################################################################################################################
# KNN Algorithm
def kNNClassifier(X, y, K):
    scores = []
    knn = KNeighborsClassifier(n_neighbors=K)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# kNNClassifier(X, y, 5)

# Logistic Regression Algorithom
def logregClassifier(X, y):
    scores = []
    logreg = LogisticRegression(random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logreg.fit(X_train, y_train)
        acc = logreg.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# logregClassifier(X,y)

# Naïve Bayes Algorithom
def nbClassifier(X, y):
    scores = []
    nb = GaussianNB()
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nb.fit(X_train, y_train)
        acc = nb.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# nbClassifier(X,y)

# Decision Tree Algorithom
def dtClassifier(X, y):
    scores = []
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt.fit(X_train, y_train)
        acc = dt.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# dtClassifier(X,y)

# Ensembles Algorithms

# bagging
def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):
    scores = []
    bag_clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0),
                                n_estimators=n_estimators,
                                max_samples=max_samples, bootstrap=True, random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        bag_clf.fit(X_train, y_train)
        acc = bag_clf.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# bagDTClassifier(X,y,500,100,1)

# adaboost
def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    scores = []
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0),
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate, random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ada_clf.fit(X_train, y_train)
        acc = ada_clf.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# adaDTClassifier(X, y, 500, 0.5, 1)

# gradient boosting
def gbClassifier(X, y, n_estimators, learning_rate):
    scores = []
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in cvKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gb_clf.fit(X_train, y_train)
        acc = gb_clf.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    print("%.4f" % scores.mean(), end=" ")
    return scores, np.around(scores.mean(), 4)


# gbClassifier(X, y, 500, 0.5)

# Linear SVM Algorithm
def bestLinClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(SVC(kernel='linear', random_state=0), param_grid, cv=cvKFold, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['C'])
    print(grid_search.best_params_['gamma'])
    print(np.around(grid_search.best_score_, 4))
    print(np.around(grid_search.score(X_test, y_test), 4))


# bestLinClassifier(X,y)

# Random Forest Algorithm
def bestRFClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = {
        'n_estimators': [10, 30],
        'max_leaf_nodes': [4, 16]
    }
    grid_search = GridSearchCV(RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=0),
                               param_grid, cv=cvKFold, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_['n_estimators'])
    print(grid_search.best_params_['max_leaf_nodes'])
    print(np.around(grid_search.best_score_, 4))
    print(np.around(grid_search.score(X_test, y_test), 4))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        if sys.argv[2] == 'P':
            pre_process(sys.argv[1])
        else:
            df = pd.read_csv(sys.argv[1]).values
            X = df[:, :len(df[1]) - 1]
            y = df[:, len(df[1]) - 1:].ravel()
            if sys.argv[2] == "LR":
                logregClassifier(X, y)
            elif sys.argv[2] == "NB":
                nbClassifier(X, y)
            elif sys.argv[2] == "DT":
                dtClassifier(X, y)
            elif sys.argv[2] == "RF":
                bestRFClassifier(X, y)
            elif sys.argv[2] == "SVM":
                bestLinClassifier(X, y)
    elif len(sys.argv) == 4:
        df = pd.read_csv(sys.argv[1]).values
        X = df[:, :len(df[1]) - 1]
        y = df[:, len(df[1]) - 1:].ravel()
        if sys.argv[2] == "NN":
            k = int(pd.read_csv(filepath_or_buffer=sys.argv[3])['K'])
            kNNClassifier(X, y, k)
        elif sys.argv[2] == "BAG":
            params = pd.read_csv(filepath_or_buffer=sys.argv[3])
            n_estimators, max_samples, max_depth = int(params['n_estimators']), int(params['max_samples']), int(
                params['max_depth'])
            bagDTClassifier(X, y, n_estimators, max_samples, max_depth)
        elif sys.argv[2] == "ADA":
            params = pd.read_csv(filepath_or_buffer=sys.argv[3])
            n_estimators, learning_rate, max_depth = int(params['n_estimators']), float(params['learning_rate']), int(
                params['max_depth'])
            adaDTClassifier(X, y, n_estimators, learning_rate, max_depth)
        elif sys.argv[2] == "GB":
            params = pd.read_csv(filepath_or_buffer=sys.argv[3])
            n_estimators, learning_rate = int(params['n_estimators']), float(params['learning_rate'])
            gbClassifier(X, y, n_estimators, learning_rate)
