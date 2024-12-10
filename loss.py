import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import torch

LFTr = pd.read_csv("LF_train.csv")
LHTr = pd.read_csv("LH_train.csv")
RFTr = pd.read_csv("RF_train.csv")
RHTr = pd.read_csv("RH_train.csv")


def cleanTrain(Tr, val):
    Tr.drop('dob', axis=1, inplace=True)
    Tr.drop('age', axis=1, inplace=True)
    Tr.drop('forceplate_date', axis=1, inplace=True)
    Tr.drop('gait', axis=1, inplace=True)
    Tr.drop('Gait', axis=1, inplace=True)
    Tr.drop('speed', axis=1, inplace=True)
    Tr.drop('Speed', axis=1, inplace=True)
    if val == 1:
        yTr = Tr['LF']
        Tr.drop('LF', axis=1, inplace=True)
    elif val == 2:
        yTr = Tr['LH']
        Tr.drop('LH', axis=1, inplace=True)
    elif val == 3:
        yTr = Tr['RF']
        Tr.drop('RF', axis=1, inplace=True)
    elif val == 4:
        yTr = Tr['RH']
        Tr.drop('RH', axis=1, inplace=True)
    xTr = Tr

    # xTr['speed'] = pd.to_numeric(xTr['speed'], errors="coerce")
    # xTr['Speed'] = pd.to_numeric(xTr['Speed'], errors="coerce")

    cols = []
    for col in xTr.columns:
        cols.append(col)
    # xTr[cols[3]] = xTr[cols[3]].astype(float)
    # colmean = xTr[cols[3]].mean()
    # xTr[cols[3]] = xTr[cols[3]].fillna(colmean)
    # xTr[cols[184]] = xTr[cols[184]].astype(float)
    # colmean = xTr[cols[184]].mean()
    # xTr[cols[184]] = xTr[cols[184]].fillna(colmean)
    fcount = 1
    for i in range(3, 123):
        if fcount > 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            colmean = xTr[cols[i]].mean()
            xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
        if fcount < 8:
            fcount += 1
        else:
            fcount = 1
    scount = 1
    for i in range(183, 303):
        if scount > 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            colmean = xTr[cols[i]].mean()
            xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
        if scount < 8:
            scount += 1
        else:
            scount = 1
    fscount = 1
    arr = [-1, 1]
    for i in range(3, 123):
        if fscount <= 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            xTr[cols[i+4]] = xTr[cols[i+4]].astype(float)
            colmean = xTr[cols[i]].mean()
            col4mean = xTr[cols[i+4]].mean()
            # random_choice = np.random.choice(arr)
            t = colmean + col4mean
            xTr[cols[i]] = xTr[cols[i]].fillna(t)
        if fscount < 8:
            fscount += 1
        else:
            fscount = 1
    sscount = 1
    for i in range(183, 303):
        if sscount <= 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            xTr[cols[i+4]] = xTr[cols[i+4]].astype(float)
            colmean = xTr[cols[i]].mean()
            col4mean = xTr[cols[i+4]].mean()
            # random_choice = np.random.choice(arr)
            t = colmean + col4mean
            xTr[cols[i]] = xTr[cols[i]].fillna(t)
        if sscount < 8:
            sscount += 1
        else:
            sscount = 1

    for i in range(123, 183):
        xTr[cols[i]] = xTr[cols[i]].astype(float)
        colmean = xTr[cols[i]].mean()
        xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    for i in range(303, len(cols)):
        xTr[cols[i]] = xTr[cols[i]].astype(float)
        colmean = xTr[cols[i]].mean()
        xTr[cols[i]] = xTr[cols[i]].fillna(colmean)

    # count = 1
    # for i in range(3, len(cols)):
    #     if cols[i] == 'speed' or cols[i] == 'Speed':
    #         xTr[cols[i]] = xTr[cols[i]].astype(float)
    #         colmean = xTr[cols[i]].mean()
    #         xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    #     else:
    #         if count > 4:
    #             xTr[cols[i]] = xTr[cols[i]].astype(float)
    #             colmean = xTr[cols[i]].mean()
    #             xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    #         elif count > 8:
    #             count = 1

    #         count += 1
    # scount = 1
    # for i in range(3, len(cols)):
    #     if cols[i] != 'speed' or cols[i] != 'Speed':
    #         if scount <= 4 and i < 124:
    #             xTr[cols[i]] = xTr[cols[i]].astype(float)
    #             colmean = xTr[cols[i+4]].mean()
    #             colmean = colmean + colmean/2
    #             xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    #         elif i >= 124:
    #             xTr[cols[i]] = xTr[cols[i]].astype(float)
    #             colmean = xTr[cols[i]].mean()
    #             xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    #         elif scount > 8:
    #             scount = 1
    #         scount += 1

    X_train, X_test, y_train, y_test = train_test_split(
        xTr, yTr, test_size=0.33, random_state=0)
    # maxdepth 1 nestimators150
    # 225
    dclf = RandomForestClassifier(
        max_samples=30, n_estimators=60, min_samples_leaf=2, max_depth=1, max_features=1)
    clf = AdaBoostClassifier(estimator=dclf, n_estimators=40)

    clf = clf.fit(X_train, y_train)

    return clf, X_test, y_test, X_train, y_train


LFclf, lfx, lfy, xlf, ylf = cleanTrain(LFTr, 1)
LHclf, lhx, lhy, xlh, ylh = cleanTrain(LHTr, 2)
RFclf, rfx, rfy, xrf, yrf = cleanTrain(RFTr, 3)
RHclf, rhx, rhy, xrh, yrh = cleanTrain(RHTr, 4)
print("avg accuracy " +
      str((accuracy_score(lfy, LFclf.predict(lfx)) +
          accuracy_score(lhy, LHclf.predict(lhx)) +
          accuracy_score(rfy, RFclf.predict(rfx)) +
          accuracy_score(rhy, RHclf.predict(rhx)))/4))
print("training avg accuracy " +
      str((accuracy_score(ylf, LFclf.predict(xlf)) + accuracy_score(ylh, LHclf.predict(xlh)) + accuracy_score(yrf, RFclf.predict(xrf)) + accuracy_score(yrh, RHclf.predict(xrh)))/4))
